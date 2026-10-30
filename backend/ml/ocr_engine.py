"""OCR engine wrapper + ITR income extraction.

Engine: RapidOCR (PP-OCR detection/recognition + ONNX Runtime).
Extraction: keyword-anchored amount parsing, so the *family annual income*
figure is pulled from the right box on a BIR Form 1701 instead of "the biggest
number on the page".
"""

from __future__ import annotations

import re
import sys
import time
from typing import Dict, List, Optional

from . import config as _config

# ---------------------------------------------------------------- amount ----
AMOUNT_RE = re.compile(
    r"(?<![\d.,])(?:"
    r"\d{1,3}(?:,\d{3})+"
    r"|\d{1,3}(?:\.\d{3})+"
    r"|\d{4,9}"
    r")(?:[.,]\d{1,2})?(?![\d.,])"
)
BARE_CURRENCY_RE = re.compile(r"(?<![\d.,])\d{4,9}(?![\d.,])")

# Ordered from family-level income labels through common BIR compensation,
# business-income and taxable-income labels. BIR layouts vary by form and year.
INCOME_KEYWORDS = [
    (re.compile(r"total\s*family\s*annual\s*income", re.I), True),
    (re.compile(r"total\s*annual\s*family\s*income", re.I), True),
    (re.compile(r"total\s*family\s*income", re.I), True),
    (re.compile(r"total\s*gross\s*compensation\s*income", re.I), True),
    (re.compile(r"total\s*compensation\s*income", re.I), True),
    (re.compile(r"gross\s*compensation\s*income", re.I), True),
    (re.compile(r"taxable\s*compensation\s*income", re.I), False),
    (re.compile(r"annual\s*compensation\s*income", re.I), False),
    (re.compile(r"total\s*annual\s*income", re.I), True),
    (re.compile(r"family\s*annual\s*income", re.I), False),
    (re.compile(r"annual\s*income", re.I), False),
    (re.compile(r"total\s*gross\s*income", re.I), True),
    (re.compile(r"gross\s*income\s*from\s*(?:business|operation)", re.I), True),
    (re.compile(r"gross\s*income", re.I), False),
    (re.compile(r"total\s*taxable\s*income", re.I), True),
    (re.compile(r"net\s*taxable\s*income", re.I), False),
    (re.compile(r"total\s*income", re.I), True),
    (re.compile(r"taxable\s*income", re.I), False),
    (re.compile(r"total\s*net\s*income", re.I), False),
    (re.compile(r"net\s*income", re.I), False),
    (re.compile(r"total\s*business\s*income", re.I), False),
    (re.compile(r"compensation\s*income", re.I), False),
]

IGNORE_RE = re.compile(
    r"page\s*\d|form\s*no|birt|date|schedule|part\s*[iv]+|"
    r"tax\s*due|tax\s*payable|amount\s*due|penalt|surcharge|advances",
    re.I,
)

# Payment/due lines sit near income rows on BIR forms - their amounts must
# never be mistaken for income (e.g. "TOTAL TAX DUE 12,500").
DUE_RE = re.compile(
    r"tax\s*due|tax\s*payable|amount\s*due|amount\s*payable|penalt|surcharge|"
    r"interest\s*on\s|advances\s*from|credit\s*balance",
    re.I,
)

# The form title is "ANNUAL INCOME TAX RETURN" - OCR often returns it glued
# together ("ANNUALINCOMETAXRETURN"), which a naive "annual income" search
# would happily treat as the income label. Reject label candidates that are
# really part of the document header.
ANCHOR_BLOCK_RE = re.compile(
    r"tax\s*return|income\s*tax|\bform\b|bureau|republic|philippines|"
    r"calendar|department|particulars|registered|revised|envelope|schedule",
    re.I,
)
ANCHOR_MAX_LEN = 70

_ORT_CONFIGURED = False
_ORT_INFO: Dict[str, str] = {}


def _configure_ort_session() -> Dict[str, str]:
    """Speed up + fix ONNX Runtime as used by ``rapidocr_onnxruntime``.

    Two problems are patched at the ``InferenceSession`` factory that
    ``OrtInferSession`` closes over:

    1. rapidocr sets ``enable_cpu_mem_arena = False``, which allocates and
       frees every intermediate tensor on every node of every run.
    2. The bundled PP-OCR models are CPU-only by default. On this machine
       (AMD Athlon 300GE, 2 cores / 4 threads) recognition costs ~0.43 s per
       text box (~75 s per page). Routing the same ONNX graphs through the
       ``DmlExecutionProvider`` on the Radeon Vega iGPU measures **0.29 s per
       6-box batch - a 10.6x speedup** - with CPU kept as the fallback
       provider for any op DirectML does not cover.
    """
    global _ORT_CONFIGURED
    if _ORT_CONFIGURED:
        return dict(_ORT_INFO)

    import onnxruntime as ort

    original = ort.InferenceSession
    providers = list(ort.get_available_providers())

    if "DmlExecutionProvider" in providers:
        run_providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        _ORT_INFO["provider"] = "DmlExecutionProvider (GPU)"
    else:
        run_providers = ["CPUExecutionProvider"]
        _ORT_INFO["provider"] = "CPUExecutionProvider"

    def _inference_session(model_path, *args, **kwargs):
        sess_options = kwargs.get("sess_options")
        if sess_options is not None and not sess_options.enable_cpu_mem_arena:
            sess_options.enable_cpu_mem_arena = True
        if sess_options is not None and not sess_options.intra_op_num_threads:
            # One thread: 0.1 CPU cannot feed a pool, and every extra ORT thread
            # keeps its own scratch buffers alive on a 512 MB instance.
            sess_options.intra_op_num_threads = 1
        kwargs["providers"] = run_providers
        return original(model_path, *args, **kwargs)

    for name in ("rapidocr_onnxruntime.utils", "utils"):
        module = sys.modules.get(name)
        if module is not None and getattr(module, "InferenceSession", None) is original:
            module.InferenceSession = _inference_session
            _ORT_INFO["patched_module"] = name

    _ORT_INFO["available"] = ",".join(providers)
    _ORT_CONFIGURED = bool(_ORT_INFO.get("patched_module"))
    return dict(_ORT_INFO)


def parse_amount(raw: str) -> Optional[float]:
    text = raw.replace("₱", "").replace("PHP", "").strip()
    text = re.sub(r"[^\d.]", "", text)
    if not text or text == ".":
        return None
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", text):
        text = text.replace(".", "")
    try:
        value = float(text)
    except ValueError:
        return None
    return value if value >= 0 else None


class ItrOcr:
    """Lazily loaded RapidOCR engine."""

    def __init__(self) -> None:
        self._engine = None
        self.load_ms = 0.0
        self.engine_info: Dict[str, str] = {}

    @property
    def engine(self):
        if self._engine is None:
            started = time.time()
            import rapidocr_onnxruntime  # noqa: F401  (registers rapidocr_onnxruntime.utils)
            from rapidocr_onnxruntime import RapidOCR

            patch = _configure_ort_session()
            # 180-degree angle classification costs ~14 s per page on this
            # hardware (187 crops) and ITR uploads are upright documents.
            self._engine = RapidOCR(use_angle_cls=False)
            self.load_ms = round((time.time() - started) * 1000, 1)

            import onnxruntime as ort
            try:
                from importlib.metadata import version as _pkg_version

                _v = _pkg_version("rapidocr-onnxruntime")
            except Exception:
                _v = "unknown"

            self.engine_info = {
                "engine": "RapidOCR (PP-OCRv4 detection + recognition)",
                "runtime": f"ONNX Runtime {ort.__version__}",
                "provider": patch.get("provider", "?"),
                "rapidocr": str(_v),
                "ort_patch": "on" if patch.get("patched_module") else "off",
            }
        return self._engine

    def read_lines(self, image, **call_kwargs) -> List[dict]:
        """Run OCR and return lines sorted top-to-bottom."""
        result, _ = self.engine(image, **call_kwargs)
        lines: List[dict] = []
        for box, text, score in result or []:
            xs = [float(p[0]) for p in box]
            ys = [float(p[1]) for p in box]
            lines.append(
                {
                    "text": str(text),
                    "score": float(score),
                    "y": round(sum(ys) / len(ys), 1),
                    "x": round(min(xs), 1),
                    "h": round(max(ys) - min(ys), 1),
                }
            )
        lines.sort(key=lambda item: (item["y"], item["x"]))
        return lines

    def scan(
        self,
        image,
        box_thresh: Optional[float] = None,
        allow_fallback: bool = True,
    ) -> dict:
        """OCR a candidate image and extract the family annual income."""
        started = time.time()
        # RapidOCR only rewrites its stored threshold when a kwarg is passed,
        # so always send an explicit value - otherwise a previous stricter
        # pass silently leaks into this one.
        lines = self.read_lines(image, box_thresh=box_thresh if box_thresh else 0.5)
        elapsed_ms = round((time.time() - started) * 1000, 1)
        extracted = extract_income(lines, allow_fallback=allow_fallback)
        return {
            "lines": lines,
            "extracted": extracted,
            "box_thresh": box_thresh,
            "line_count": len(lines),
            "avg_confidence": round(
                sum(item["score"] for item in lines) / len(lines), 4
            )
            if lines
            else 0.0,
            "ocr_ms": elapsed_ms,
        }


def _amounts_in(line: dict) -> List[tuple]:
    out = []
    for match in AMOUNT_RE.finditer(line["text"]):
        value = parse_amount(match.group(0))
        if value is not None:
            out.append((value, match.group(0), match.start()))
    return out


def extract_income(lines: List[dict], allow_fallback: bool = True) -> Optional[dict]:
    """Locate the family annual income figure inside OCR'd ITR text."""
    if not lines:
        return None

    for pattern, want_total in INCOME_KEYWORDS:
        for index, line in enumerate(lines):
            anchor_match = pattern.search(line["text"])
            if not anchor_match:
                continue
            is_tax_return_header = re.search(
                r"(?:annual\s*income\s*tax|income\s*tax\s*return|tax\s*return)",
                line["text"],
                re.I,
            )
            if is_tax_return_header or (
                not want_total
                and (
                    ANCHOR_BLOCK_RE.search(line["text"])
                    or len(line["text"]) > ANCHOR_MAX_LEN
                )
            ):
                continue

            height = max(line["h"], 10.0)
            window: List[dict] = []
            for nxt in lines[index : index + 5]:
                if nxt is not line and pattern.search(nxt["text"]) and want_total:
                    break
                if nxt["y"] - line["y"] > max(height * 3.0, 90.0):
                    break
                window.append(nxt)

            candidates: List[tuple] = []
            due_offsets = {
                i for i, win in enumerate(window) if DUE_RE.search(win["text"])
            }
            for offset, win in enumerate(window):
                # Skip tax-due/penalty rows AND the amount line under them.
                if offset in due_offsets or (offset - 1) in due_offsets:
                    continue
                for value, raw, start in _amounts_in(win):
                    if value < 1000:
                        continue
                    if offset == 0 and start < anchor_match.end():
                        continue
                    if (
                        1900 <= value <= 2099
                        and re.search(r"\b(?:tax\s*year|calendar\s*year|t\.?\s*y\.?)\b", win["text"], re.I)
                    ):
                        continue
                    candidates.append((value, raw, offset, start, win))

            if not candidates:
                continue

            if want_total:
                chosen = min(candidates, key=lambda c: (c[2], c[3]))
            else:
                chosen = max(candidates, key=lambda c: c[0])

            value, raw, offset, _, win = chosen
            scores = [line["score"], win["score"]] if offset else [line["score"]]
            return {
                "value": value,
                "raw": raw,
                "method": pattern.pattern.replace("\\s*", " ").replace("\\s", " "),
                "anchor": line["text"].strip(),
                "amount_line": win["text"].strip(),
                "confidence": round(sum(scores) / len(scores), 4),
                "matched": True,
            }

    if not allow_fallback:
        return None

    # Fallback: biggest comma-grouped amount anywhere on the page.
    # Only reached on the final, full-detail pass - flagged low-confidence so
    # the UI never presents it as a verified figure.
    fallback: List[tuple] = []
    for line in lines:
        if IGNORE_RE.search(line["text"]):
            continue
        has_group = bool(re.search(r"\d{1,3}(?:,\d{3})+", line["text"]))
        if not has_group or line["score"] < _config.OCR_MIN_CONFIDENCE:
            continue
        for value, raw, _ in _amounts_in(line):
            if value >= 10_000:
                fallback.append((value, raw, line))

    if fallback:
        value, raw, line = max(fallback, key=lambda f: f[0])
        return {
            "value": value,
            "raw": raw,
            "method": "largest_currency_amount",
            "anchor": "",
            "amount_line": line["text"].strip(),
            "confidence": round(line["score"], 4),
            "matched": True,
            "low_confidence": True,
        }

    return None


def match_declared(ocr_value: Optional[float], declared: Optional[float]) -> dict:
    """Compare the OCR figure with the applicant's typed family income."""
    if ocr_value is None or declared is None:
        return {
            "exact_match": False,
            "align_pct": 0.0,
            "diff": None,
            "status": "not_comparable",
        }

    ocr_value = float(ocr_value)
    declared = float(declared)
    diff = abs(ocr_value - declared)
    denominator = max(ocr_value, declared, 1.0)
    align_pct = max(0.0, min(100.0, 100.0 * (1.0 - diff / denominator)))
    return {
        "exact_match": bool(diff < 0.005),
        "align_pct": round(align_pct, 2),
        "diff": round(diff, 2),
        "status": "match" if diff < 0.005 else "mismatch",
    }


_SHARED: Optional["ItrOcr"] = None


def shared() -> "ItrOcr":
    """Process-wide engine singleton (created on first real OCR use)."""
    global _SHARED
    if _SHARED is None:
        _SHARED = ItrOcr()
    return _SHARED


def peek() -> Optional["ItrOcr"]:
    """The engine if it has already been loaded, otherwise None."""
    return _SHARED
