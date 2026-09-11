"""Local watchlist matching.

The watchlist lives in SQLite and is edited from the GUI. Nothing leaves the
machine and no external service is consulted.

A match is only ever raised against a plate string that OCR actually produced.
There is no fuzzy "close enough" mode by default, because a false watchlist
hit is the most damaging kind of false positive this system can produce. An
optional near-match mode exists for operator-driven review and is reported
separately, never as a confirmed match.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WatchlistEntry:
    """One watched identifier."""

    entry_id: int
    kind: str            # "plate" | "tag"
    value: str
    label: Optional[str] = None
    severity: str = "HIGH"
    notes: Optional[str] = None
    active: bool = True

    @property
    def display(self) -> str:
        return f"{self.value} - {self.label}" if self.label else self.value


@dataclass
class WatchlistMatch:
    """A watchlist hit, exact or near."""

    entry: WatchlistEntry
    observed: str
    exact: bool
    distance: int = 0

    def describe(self) -> str:
        if self.exact:
            return f"Watchlist match: {self.observed}"
        return (f"Possible watchlist match: read '{self.observed}', "
                f"watched '{self.entry.value}' ({self.distance} characters "
                f"differ) - needs operator confirmation")


def _levenshtein(a: str, b: str, cap: int = 3) -> int:
    """Edit distance, abandoned once it exceeds *cap*."""
    if a == b:
        return 0
    if abs(len(a) - len(b)) > cap:
        return cap + 1

    previous = list(range(len(b) + 1))
    for i, ch_a in enumerate(a, start=1):
        current = [i]
        for j, ch_b in enumerate(b, start=1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (ch_a != ch_b),
            ))
        if min(current) > cap:
            return cap + 1
        previous = current
    return previous[-1]


class WatchlistMatcher:
    """Match observed identifiers against the active watchlist.

    Holds an in-memory copy so the pipeline does not query SQLite for every
    plate. Call :meth:`refresh` after the operator edits the list.
    """

    def __init__(
        self,
        entries: Iterable[WatchlistEntry] = (),
        near_match_distance: int = 0,
    ) -> None:
        """
        Parameters
        ----------
        near_match_distance:
            0 disables near matching, which is the default and the safe
            choice. 1 flags single-character differences for review.
        """
        self._entries: list[WatchlistEntry] = []
        self._exact: dict[tuple[str, str], WatchlistEntry] = {}
        self.near_match_distance = max(0, int(near_match_distance))
        self.refresh(entries)

    def refresh(self, entries: Iterable[WatchlistEntry]) -> None:
        self._entries = [e for e in entries if e.active]
        self._exact = {
            (e.kind, self.normalise(e.value)): e for e in self._entries
        }
        logger.info("Watchlist loaded: %d active entries", len(self._entries))

    @staticmethod
    def normalise(value: str) -> str:
        """Upper-case and strip separators so 'DL 01 AB 1234' matches."""
        return "".join(
            ch for ch in str(value).upper() if ch.isalnum()
        )

    def match(self, observed: str, kind: str = "plate") -> Optional[WatchlistMatch]:
        """Look up one observed value. Returns ``None`` when nothing matches."""
        if not observed:
            return None
        needle = self.normalise(observed)
        if not needle:
            return None

        entry = self._exact.get((kind, needle))
        if entry is not None:
            return WatchlistMatch(entry=entry, observed=needle, exact=True)

        if self.near_match_distance > 0:
            best: Optional[tuple[int, WatchlistEntry]] = None
            for candidate in self._entries:
                if candidate.kind != kind:
                    continue
                dist = _levenshtein(
                    needle, self.normalise(candidate.value),
                    cap=self.near_match_distance,
                )
                if dist <= self.near_match_distance:
                    if best is None or dist < best[0]:
                        best = (dist, candidate)
            if best is not None:
                return WatchlistMatch(
                    entry=best[1], observed=needle, exact=False, distance=best[0]
                )
        return None

    @property
    def entries(self) -> list[WatchlistEntry]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)


def entries_from_rows(rows: Iterable) -> list[WatchlistEntry]:
    """Build entries from ``watchlist`` table rows."""
    out: list[WatchlistEntry] = []
    for row in rows:
        try:
            out.append(WatchlistEntry(
                entry_id=int(row["id"]),
                kind=str(row["kind"]),
                value=str(row["value"]),
                label=row["label"],
                severity=str(row["severity"] or "HIGH"),
                notes=row["notes"],
                active=bool(row["active"]),
            ))
        except (KeyError, TypeError, ValueError) as exc:  # pragma: no cover
            logger.warning("Skipping malformed watchlist row: %s", exc)
    return out
