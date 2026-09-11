"""Unsupervised behavioural anomaly detection.

An Isolation Forest learns what ordinary movement looks like *in this scene*
from the behavioural features of completed tracks, then scores new tracks
against that baseline.

What this engine will and will not do
-------------------------------------
It will tell you that a track's movement pattern is unlike the others it has
seen. It will not tell you that someone is hostile: the model has no concept
of intent, and no labelled examples of what a real intrusion looks like.
Treat a high score as "worth a human look", which is exactly how the threat
engine weights it.

Before enough tracks exist to define "ordinary", the engine returns
:data:`STATUS_INSUFFICIENT` and a score of ``None``. It never invents a number
to fill the gap. That behaviour is the whole point: the previous generation of
this project shipped a model fitted on a different video, which scored 92% of
frames as anomalous on new footage.

Feature order comes from :data:`core.behaviour_engine.FEATURE_NAMES`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from core.behaviour_engine import FEATURE_DIM, FEATURE_NAMES

logger = logging.getLogger(__name__)

STATUS_INSUFFICIENT = "INSUFFICIENT_DATA"
STATUS_NORMAL = "NORMAL"
STATUS_ANOMALOUS = "ANOMALOUS"
STATUS_DISABLED = "DISABLED"


@dataclass
class AnomalyResult:
    """One track's anomaly assessment."""

    status: str
    score: Optional[float] = None          # Isolation Forest decision value
    confidence: Optional[float] = None     # 0-1, how far past the threshold
    samples_seen: int = 0
    samples_needed: int = 0
    contributions: dict[str, float] = field(default_factory=dict)

    @property
    def is_anomalous(self) -> bool:
        return self.status == STATUS_ANOMALOUS

    @property
    def is_decided(self) -> bool:
        """Whether the engine actually made a call."""
        return self.status in (STATUS_NORMAL, STATUS_ANOMALOUS)

    def describe(self) -> str:
        if self.status == STATUS_INSUFFICIENT:
            return (f"Not enough data yet ({self.samples_seen}/"
                    f"{self.samples_needed} tracks observed)")
        if self.status == STATUS_DISABLED:
            return "Anomaly detection is disabled in settings"
        return f"{self.status} (score {self.score:+.3f})"

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "score": None if self.score is None else round(self.score, 4),
            "confidence": (None if self.confidence is None
                           else round(self.confidence, 3)),
            "samples_seen": self.samples_seen,
            "samples_needed": self.samples_needed,
            "contributions": self.contributions,
        }


class AnomalyEngine:
    """Isolation Forest over behavioural features, with honest cold-start."""

    def __init__(
        self,
        min_samples: int = 25,
        contamination: float = 0.10,
        n_estimators: int = 150,
        random_state: int = 42,
        refit_every: int = 25,
        decision_threshold: float = 0.0,
        model_path: Optional[Path] = None,
        enabled: bool = True,
    ) -> None:
        self.min_samples = max(5, int(min_samples))
        self.contamination = float(contamination)
        self.n_estimators = int(n_estimators)
        self.random_state = int(random_state)
        self.refit_every = max(1, int(refit_every))
        self.decision_threshold = float(decision_threshold)
        self.model_path = Path(model_path) if model_path else None
        self.enabled = bool(enabled)

        self._samples: list[np.ndarray] = []
        self._model = None
        self._scaler = None
        self._fitted = False
        self._since_fit = 0
        self._fit_count = 0
        self._score_range: tuple[float, float] = (-0.5, 0.5)

    # -- state ------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    def status_text(self) -> str:
        if not self.enabled:
            return "Disabled"
        if self._fitted:
            return f"Fitted on {self.sample_count} tracks ({self._fit_count} refits)"
        return f"Learning: {self.sample_count}/{self.min_samples} tracks"

    # -- learning ---------------------------------------------------------

    def add_sample(self, features: Sequence[float]) -> bool:
        """Add one completed track's features to the baseline.

        Returns True when the model was (re)fitted as a result.
        """
        if not self.enabled:
            return False
        vector = self._coerce(features)
        if vector is None:
            return False

        self._samples.append(vector)
        self._since_fit += 1

        # Bound memory. 5000 tracks is far more than any single session.
        if len(self._samples) > 5000:
            self._samples = self._samples[-5000:]

        if len(self._samples) < self.min_samples:
            return False
        if self._fitted and self._since_fit < self.refit_every:
            return False
        return self.fit()

    def add_samples(self, rows: Iterable[Sequence[float]]) -> int:
        """Bulk-load a baseline, e.g. from previous sessions in SQLite."""
        added = 0
        for row in rows:
            vector = self._coerce(row)
            if vector is not None:
                self._samples.append(vector)
                added += 1
        if added:
            self._since_fit += added
            logger.info("Loaded %d historical tracks into the anomaly baseline",
                        added)
        return added

    def fit(self) -> bool:
        """Fit the Isolation Forest. Returns True on success."""
        if not self.enabled:
            return False
        if len(self._samples) < self.min_samples:
            return False

        try:
            from sklearn.ensemble import IsolationForest
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            logger.error(
                "scikit-learn is not installed; anomaly detection is disabled"
            )
            self.enabled = False
            return False

        matrix = np.vstack(self._samples)
        try:
            self._scaler = StandardScaler()
            scaled = self._scaler.fit_transform(matrix)

            self._model = IsolationForest(
                n_estimators=self.n_estimators,
                contamination=self.contamination,
                random_state=self.random_state,
                n_jobs=-1,
            )
            self._model.fit(scaled)

            scores = self._model.decision_function(scaled)
            self._score_range = (float(scores.min()), float(scores.max()))

            self._fitted = True
            self._since_fit = 0
            self._fit_count += 1
            logger.info(
                "Anomaly model fitted on %d tracks (refit #%d), "
                "score range %.3f to %.3f",
                len(self._samples), self._fit_count, *self._score_range,
            )
            return True
        except Exception as exc:
            logger.error("Anomaly model fit failed: %s", exc, exc_info=True)
            self._fitted = False
            return False

    # -- scoring ----------------------------------------------------------

    def score(self, features: Sequence[float]) -> AnomalyResult:
        """Score one track. Never fabricates a value."""
        if not self.enabled:
            return AnomalyResult(status=STATUS_DISABLED)

        if not self._fitted:
            return AnomalyResult(
                status=STATUS_INSUFFICIENT,
                samples_seen=len(self._samples),
                samples_needed=self.min_samples,
            )

        vector = self._coerce(features)
        if vector is None:
            return AnomalyResult(
                status=STATUS_INSUFFICIENT,
                samples_seen=len(self._samples),
                samples_needed=self.min_samples,
            )

        try:
            scaled = self._scaler.transform(vector.reshape(1, -1))
            raw = float(self._model.decision_function(scaled)[0])
        except Exception as exc:
            logger.error("Anomaly scoring failed: %s", exc)
            return AnomalyResult(
                status=STATUS_INSUFFICIENT,
                samples_seen=len(self._samples),
                samples_needed=self.min_samples,
            )

        anomalous = raw < self.decision_threshold
        low, high = self._score_range
        span = max(1e-6, high - low)
        confidence = float(np.clip(abs(raw - self.decision_threshold) / span, 0.0, 1.0))

        return AnomalyResult(
            status=STATUS_ANOMALOUS if anomalous else STATUS_NORMAL,
            score=raw,
            confidence=confidence,
            samples_seen=len(self._samples),
            samples_needed=self.min_samples,
            contributions=self._contributions(vector),
        )

    def _contributions(self, vector: np.ndarray) -> dict[str, float]:
        """Per-feature z-scores, so an anomaly can be explained.

        The Isolation Forest itself is not additive, so these are not exact
        attributions. They answer a narrower, still useful question: which
        features are furthest from the baseline mean?
        """
        if self._scaler is None:
            return {}
        try:
            z = (vector - self._scaler.mean_) / np.sqrt(
                np.maximum(self._scaler.var_, 1e-9)
            )
            ranked = sorted(
                zip(FEATURE_NAMES, z), key=lambda kv: abs(kv[1]), reverse=True
            )
            return {name: round(float(value), 2) for name, value in ranked[:4]}
        except Exception:  # pragma: no cover - defensive
            return {}

    # -- persistence ------------------------------------------------------

    def save(self, path: Path | None = None) -> Optional[Path]:
        """Persist the fitted model so the next session starts warm."""
        target = Path(path) if path else self.model_path
        if target is None or not self._fitted:
            return None
        try:
            import joblib
            target.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {
                    "model": self._model,
                    "scaler": self._scaler,
                    "samples": self._samples[-2000:],
                    "score_range": self._score_range,
                    "feature_names": list(FEATURE_NAMES),
                    "min_samples": self.min_samples,
                },
                target,
            )
            logger.info("Anomaly model saved to %s", target)
            return target
        except Exception as exc:
            logger.warning("Could not save anomaly model: %s", exc)
            return None

    def load(self, path: Path | None = None) -> bool:
        """Load a previously fitted model, if it matches this feature set."""
        target = Path(path) if path else self.model_path
        if target is None or not target.exists():
            return False
        try:
            import joblib
            payload = joblib.load(target)
        except Exception as exc:
            logger.warning("Could not load anomaly model from %s: %s", target, exc)
            return False

        names = payload.get("feature_names")
        if names is not None and list(names) != list(FEATURE_NAMES):
            logger.warning(
                "Ignoring %s: it was trained on a different feature set (%s). "
                "A fresh baseline will be learned.",
                target.name, names,
            )
            return False

        self._model = payload.get("model")
        self._scaler = payload.get("scaler")
        self._samples = list(payload.get("samples") or [])
        self._score_range = tuple(payload.get("score_range", (-0.5, 0.5)))
        self._fitted = self._model is not None and self._scaler is not None
        if self._fitted:
            logger.info(
                "Anomaly model loaded from %s (%d baseline tracks)",
                target.name, len(self._samples),
            )
        return self._fitted

    def reset(self) -> None:
        self._samples.clear()
        self._model = None
        self._scaler = None
        self._fitted = False
        self._since_fit = 0

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _coerce(features: Sequence[float]) -> Optional[np.ndarray]:
        try:
            vector = np.asarray(features, dtype=np.float32).ravel()
        except (TypeError, ValueError):
            return None
        if vector.size != FEATURE_DIM:
            logger.debug(
                "Ignoring feature vector of length %d (expected %d)",
                vector.size, FEATURE_DIM,
            )
            return None
        return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
