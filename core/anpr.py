"""ANPR: automatic number plate recognition.

Pipeline
--------
::

    vehicle box (from the detector)
        -> plate region proposal   classical CV, no extra model needed
        -> image preprocessing     upscale, CLAHE, denoise, binarise
        -> OCR                     EasyOCR or Tesseract
        -> validation              confidence threshold + optional regex
        -> PlateRead

The one rule that matters
-------------------------
**A plate number is never invented.** If OCR returns nothing, returns
something below the confidence threshold, or the region does not look like a
plate, the result is ``text=None`` and the GUI shows ``UNREADABLE``. The
cropped image is still saved as evidence so a human can read it.

Honest limitations
------------------
* Region proposal is morphological, not a trained plate detector. It works on
  reasonably sharp, roughly frontal plates. It will miss plates that are
  small, heavily angled, motion-blurred or poorly lit.
* General-purpose OCR is not plate OCR. Expect confusion between O/0, I/1,
  S/5, B/8. The character-substitution pass corrects the common cases in
  positions where the plate format makes the answer unambiguous.
* Accuracy on real traffic footage has not been measured for this build. Do
  not quote a percentage you have not tested yourself.

Installing an OCR backend is optional. With none installed the engine reports
``available=False`` and the rest of the application runs unaffected.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)

#: Characters a plate may contain. Anything else is stripped.
_ALLOWED = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

#: Common OCR confusions, applied only where the plate grammar makes the
#: intended character unambiguous.
_TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1",
             "Z": "2", "S": "5", "B": "8", "G": "6", "T": "7"}
_TO_ALPHA = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G"}


@dataclass
class PlateCandidate:
    """A proposed plate region inside a vehicle box."""

    image: np.ndarray
    x: int
    y: int
    w: int
    h: int
    score: float

    @property
    def aspect(self) -> float:
        return self.w / self.h if self.h else 0.0


@dataclass
class OcrResult:
    """Raw OCR output before validation."""

    text: str
    confidence: float
    backend: str


# ---------------------------------------------------------------------------
# OCR backends
# ---------------------------------------------------------------------------

class OcrBackend:
    """Interface every OCR implementation satisfies."""

    name = "none"
    available = False

    def read(self, image: np.ndarray) -> Optional[OcrResult]:  # pragma: no cover
        raise NotImplementedError


class EasyOcrBackend(OcrBackend):
    """EasyOCR. Better on plates than Tesseract, heavier to install.

    The first run downloads detection and recognition weights (roughly 100 MB)
    and needs internet once. After that it works offline.
    """

    name = "easyocr"

    def __init__(self, gpu: bool = False) -> None:
        self._reader = None
        self.available = False
        try:
            import easyocr
        except ImportError:
            logger.info("EasyOCR is not installed")
            return
        try:
            self._reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)
            self.available = True
            logger.info("ANPR backend: EasyOCR (gpu=%s)", gpu)
        except Exception as exc:
            logger.warning(
                "EasyOCR is installed but failed to initialise: %s. "
                "The first run needs internet to fetch its weights.", exc
            )

    def read(self, image: np.ndarray) -> Optional[OcrResult]:
        if not self.available or self._reader is None:
            return None
        try:
            results = self._reader.readtext(
                image,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                detail=1,
                paragraph=False,
            )
        except Exception as exc:
            logger.debug("EasyOCR read failed: %s", exc)
            return None
        if not results:
            return None

        # Join fragments left to right: plates are often split into two boxes.
        results.sort(key=lambda r: r[0][0][0])
        text = "".join(str(r[1]) for r in results)
        confidences = [float(r[2]) for r in results if len(r) > 2]
        confidence = min(confidences) if confidences else 0.0
        return OcrResult(text=text, confidence=confidence, backend=self.name)


class TesseractBackend(OcrBackend):
    """Tesseract via pytesseract.

    Needs both the Python package and the ``tesseract`` binary. On macOS:
    ``brew install tesseract``.
    """

    name = "tesseract"

    def __init__(self) -> None:
        self.available = False
        self._pytesseract = None
        try:
            import pytesseract
        except ImportError:
            logger.info("pytesseract is not installed")
            return
        try:
            pytesseract.get_tesseract_version()
            self._pytesseract = pytesseract
            self.available = True
            logger.info("ANPR backend: Tesseract")
        except Exception as exc:
            logger.warning(
                "pytesseract is installed but the tesseract binary was not "
                "found: %s. Install it with 'brew install tesseract' on macOS.",
                exc,
            )

    def read(self, image: np.ndarray) -> Optional[OcrResult]:
        if not self.available or self._pytesseract is None:
            return None
        # PSM 7 = treat the image as a single text line, which a plate is.
        config = (
            "--oem 3 --psm 7 "
            "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )
        try:
            data = self._pytesseract.image_to_data(
                image, config=config,
                output_type=self._pytesseract.Output.DICT,
            )
        except Exception as exc:
            logger.debug("Tesseract read failed: %s", exc)
            return None

        pieces: list[str] = []
        confidences: list[float] = []
        for text, conf in zip(data.get("text", []), data.get("conf", [])):
            token = str(text).strip()
            if not token:
                continue
            try:
                value = float(conf)
            except (TypeError, ValueError):
                continue
            if value < 0:
                continue
            pieces.append(token)
            confidences.append(value / 100.0)

        if not pieces:
            return None
        return OcrResult(
            text="".join(pieces),
            confidence=min(confidences) if confidences else 0.0,
            backend=self.name,
        )


def build_backend(preference: str = "auto", gpu: bool = False) -> OcrBackend:
    """Pick an OCR backend, honouring an explicit preference where possible."""
    pref = (preference or "auto").strip().lower()

    if pref == "easyocr":
        backend = EasyOcrBackend(gpu=gpu)
        if backend.available:
            return backend
        logger.warning("EasyOCR was requested but is unavailable")
        return OcrBackend()

    if pref == "tesseract":
        backend = TesseractBackend()
        if backend.available:
            return backend
        logger.warning("Tesseract was requested but is unavailable")
        return OcrBackend()

    if pref in ("none", "off", "disabled"):
        return OcrBackend()

    for factory in (lambda: EasyOcrBackend(gpu=gpu), TesseractBackend):
        backend = factory()
        if backend.available:
            return backend

    logger.warning(
        "No OCR backend is available, so ANPR is switched off. Install one "
        "with 'pip install easyocr' or 'pip install pytesseract' plus the "
        "tesseract binary. Everything else keeps working."
    )
    return OcrBackend()


# ---------------------------------------------------------------------------
# ANPR engine
# ---------------------------------------------------------------------------

class AnprEngine:
    """Find and read licence plates inside vehicle detections."""

    def __init__(
        self,
        enabled: bool = True,
        ocr_backend: str = "auto",
        min_ocr_confidence: float = 0.45,
        min_vehicle_height: int = 90,
        plate_regex: Optional[str] = None,
        accept_unvalidated: bool = True,
        gpu: bool = False,
    ) -> None:
        self.enabled = bool(enabled)
        self.min_ocr_confidence = float(min_ocr_confidence)
        self.min_vehicle_height = int(min_vehicle_height)
        self.accept_unvalidated = bool(accept_unvalidated)

        self._pattern: Optional[re.Pattern[str]] = None
        if plate_regex:
            try:
                self._pattern = re.compile(plate_regex)
            except re.error as exc:
                logger.warning(
                    "anpr.plate_regex is not a valid regular expression (%s); "
                    "plate validation is disabled.", exc
                )

        self.backend: OcrBackend = (
            build_backend(ocr_backend, gpu=gpu) if self.enabled else OcrBackend()
        )
        self.attempts = 0
        self.successes = 0

    # -- state ------------------------------------------------------------

    @property
    def available(self) -> bool:
        return self.enabled and self.backend.available

    def status_text(self) -> str:
        if not self.enabled:
            return "Disabled in settings"
        if not self.backend.available:
            return "No OCR backend installed"
        rate = (self.successes / self.attempts * 100) if self.attempts else 0.0
        return (f"{self.backend.name} | {self.successes}/{self.attempts} read "
                f"({rate:.0f}%)")

    # -- region proposal --------------------------------------------------

    def find_plate_regions(
        self, vehicle_crop: np.ndarray, max_candidates: int = 3
    ) -> list[PlateCandidate]:
        """Propose plate-shaped regions inside a vehicle crop.

        Blackhat morphology highlights dark-on-light text, a Sobel pass finds
        the dense vertical edges typical of characters, and a closing
        operation merges them into one blob per plate. Candidates are then
        filtered on aspect ratio and relative area.
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return []

        height, width = vehicle_crop.shape[:2]
        if height < 20 or width < 40:
            return []

        gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY) \
            if vehicle_crop.ndim == 3 else vehicle_crop

        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
        square_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_kernel)

        grad_x = cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=3)
        grad_x = np.absolute(grad_x)
        span = grad_x.max() - grad_x.min()
        if span < 1e-6:
            return []
        grad_x = (255 * (grad_x - grad_x.min()) / span).astype("uint8")

        grad_x = cv2.GaussianBlur(grad_x, (5, 5), 0)
        grad_x = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, rect_kernel)
        _, thresh = cv2.threshold(grad_x, 0, 255,
                                  cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        thresh = cv2.erode(thresh, square_kernel, iterations=1)
        thresh = cv2.dilate(thresh, square_kernel, iterations=2)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        candidates: list[PlateCandidate] = []
        frame_area = float(width * height)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h < 8 or w < 24:
                continue
            aspect = w / float(h)
            if not 1.8 <= aspect <= 7.0:
                continue
            area_ratio = (w * h) / frame_area
            if not 0.008 <= area_ratio <= 0.45:
                continue

            # Plates sit low on a vehicle. Reward that without requiring it,
            # because camera angles vary.
            vertical_bias = (y + h / 2.0) / height
            score = area_ratio * (0.6 + 0.8 * vertical_bias)

            pad_x = max(2, int(w * 0.03))
            pad_y = max(2, int(h * 0.12))
            x0 = max(0, x - pad_x)
            y0 = max(0, y - pad_y)
            x1 = min(width, x + w + pad_x)
            y1 = min(height, y + h + pad_y)
            crop = vehicle_crop[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            candidates.append(
                PlateCandidate(image=crop, x=x0, y=y0,
                               w=x1 - x0, h=y1 - y0, score=score)
            )

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:max_candidates]

    # -- preprocessing ----------------------------------------------------

    @staticmethod
    def preprocess(plate: np.ndarray, target_height: int = 96) -> np.ndarray:
        """Make a plate crop as readable as possible for OCR."""
        if plate is None or plate.size == 0:
            return plate

        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY) if plate.ndim == 3 else plate

        height, width = gray.shape[:2]
        if height < target_height:
            scale = target_height / float(height)
            gray = cv2.resize(
                gray, (max(1, int(width * scale)), target_height),
                interpolation=cv2.INTER_CUBIC,
            )

        # Local contrast, so a plate half in shadow still reads.
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        # Bilateral filtering removes sensor noise but keeps character edges.
        gray = cv2.bilateralFilter(gray, 9, 60, 60)
        binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )[1]

        # Plates are dark text on light background. If the binarisation came
        # out inverted, flip it: OCR engines expect dark on light.
        if np.mean(binary) < 127:
            binary = cv2.bitwise_not(binary)

        return cv2.copyMakeBorder(binary, 8, 8, 8, 8,
                                  cv2.BORDER_CONSTANT, value=255)

    # -- text handling ----------------------------------------------------

    @staticmethod
    def clean_text(raw: str) -> str:
        """Strip everything that cannot appear on a plate."""
        return "".join(ch for ch in str(raw).upper() if ch in _ALLOWED)

    def correct_confusions(self, text: str) -> str:
        """Fix O/0 style errors where the plate grammar is unambiguous.

        Only applied when a regex is configured and the raw text is the right
        length. Without a known format there is no basis for the correction,
        so the text is returned untouched.
        """
        if not self._pattern or not text:
            return text
        if self._pattern.match(text):
            return text

        # Indian format: LL DD LLL DDDD - letters, digits, letters, digits.
        if len(text) >= 8:
            chars = list(text)
            for i in (0, 1):
                chars[i] = _TO_ALPHA.get(chars[i], chars[i])
            for i in (2, 3):
                chars[i] = _TO_DIGIT.get(chars[i], chars[i])
            for i in range(len(chars) - 4, len(chars)):
                if i >= 4:
                    chars[i] = _TO_DIGIT.get(chars[i], chars[i])
            corrected = "".join(chars)
            if self._pattern.match(corrected):
                logger.debug("Plate corrected: %s -> %s", text, corrected)
                return corrected
        return text

    def validate(self, text: str) -> bool:
        if not text:
            return False
        if self._pattern is None:
            return len(text) >= 4
        return bool(self._pattern.match(text))

    # -- main entry point -------------------------------------------------

    def read_plate(
        self, frame: np.ndarray, box: Sequence[float]
    ) -> dict[str, Any]:
        """Attempt to read a plate on the vehicle inside *box*.

        Parameters
        ----------
        frame:
            Full BGR frame.
        box:
            Vehicle box in pixels, ``(x1, y1, x2, y2)``.

        Returns
        -------
        dict with keys:
            ``text``        the plate string, or ``None`` when unreadable
            ``confidence``  OCR confidence, 0.0 when nothing was read
            ``validated``   whether it matched the configured plate format
            ``backend``     which OCR engine ran
            ``crop``        the plate crop (BGR) for evidence, or ``None``
            ``reason``      why it failed, when it did
        """
        blank: dict[str, Any] = {
            "text": None, "confidence": 0.0, "validated": False,
            "backend": self.backend.name, "crop": None, "reason": "",
        }

        if not self.available:
            blank["reason"] = "no OCR backend available"
            return blank

        x1, y1, x2, y2 = (int(round(v)) for v in box)
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))

        if (y2 - y1) < self.min_vehicle_height:
            blank["reason"] = (
                f"vehicle too small ({y2 - y1}px, need "
                f"{self.min_vehicle_height}px)"
            )
            return blank

        vehicle = frame[y1:y2, x1:x2]
        if vehicle.size == 0:
            blank["reason"] = "empty vehicle crop"
            return blank

        self.attempts += 1
        candidates = self.find_plate_regions(vehicle)
        if not candidates:
            blank["reason"] = "no plate-shaped region found"
            return blank

        best: dict[str, Any] | None = None
        for candidate in candidates:
            prepared = self.preprocess(candidate.image)
            result = self.backend.read(prepared)
            if result is None:
                continue

            text = self.clean_text(result.text)
            if len(text) < 4:
                continue
            text = self.correct_confusions(text)
            validated = self.validate(text)

            payload = {
                "text": text,
                "confidence": result.confidence,
                "validated": validated,
                "backend": result.backend,
                "crop": candidate.image,
                "reason": "",
            }
            # A validated read always wins; otherwise keep the most confident.
            if best is None or validated and not best["validated"] \
                    or (validated == best["validated"]
                        and result.confidence > best["confidence"]):
                best = payload
            if validated and result.confidence >= self.min_ocr_confidence:
                break

        if best is None:
            blank["crop"] = candidates[0].image
            blank["reason"] = "OCR returned no usable text"
            return blank

        if best["confidence"] < self.min_ocr_confidence:
            return {
                **blank,
                "crop": best["crop"],
                "reason": (f"OCR confidence {best['confidence']:.2f} below "
                           f"threshold {self.min_ocr_confidence:.2f}"),
            }

        if not best["validated"] and not self.accept_unvalidated:
            return {
                **blank,
                "crop": best["crop"],
                "reason": f"'{best['text']}' does not match the plate format",
            }

        self.successes += 1
        return best
