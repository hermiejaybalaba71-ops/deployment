import sys
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml import pipeline, preproc


class OcrMemorySafetyTests(unittest.TestCase):
    def test_decode_bounds_ocr_dimensions_and_preserves_source_dimensions(self):
        source = Image.new("RGB", (3000, 3600), "white")
        encoded = BytesIO()
        source.save(encoded, format="PNG")

        rgb, metadata = preproc.decode(encoded.getvalue())

        self.assertLessEqual(max(rgb.shape[:2]), preproc.MAX_OCR_SIDE)
        self.assertEqual(metadata["size"], [3000, 3600])
        self.assertEqual(
            metadata["ocr_size"],
            [rgb.shape[1], rgb.shape[0]],
        )

    def test_candidate_generation_is_lazy_after_original_image(self):
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)

        with patch("ml.preproc.upscale", wraps=preproc.upscale) as upscale:
            candidates = preproc.build_candidates(rgb)
            first = next(candidates)

            self.assertEqual(first["name"], "original")
            self.assertIs(first["array"], rgb)
            upscale.assert_not_called()
            candidates.close()

    def test_candidate_generation_preserves_the_configured_variant_limit(self):
        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        gray = np.zeros((8, 8), dtype=np.uint8)

        with (
            patch("ml.preproc._config.OCR_MAX_VARIANTS", 2),
            patch("ml.preproc.upscale", return_value=rgb),
            patch("ml.preproc.to_gray", return_value=gray),
            patch("ml.preproc.normalize", return_value=gray),
            patch("ml.preproc._to_rgb", side_effect=lambda image: np.repeat(image[:, :, None], 3, axis=2)),
            patch("ml.preproc.otsu", return_value=gray),
            patch("ml.preproc.deskew", return_value=(gray, 2.0)) as deskew,
        ):
            candidates = list(preproc.build_candidates(rgb))

        self.assertEqual(
            [candidate["name"] for candidate in candidates],
            ["original", "grayscale+contrast", "grayscale+otsu"],
        )
        deskew.assert_not_called()

    def test_income_fallback_uses_the_last_candidate_without_extra_ocr(self):
        first = {"name": "original", "array": np.zeros((600, 600, 3), dtype=np.uint8), "steps": ["decode"]}
        last = {"name": "grayscale+otsu", "array": np.ones((600, 600, 3), dtype=np.uint8), "steps": ["otsu"]}

        class FakeEngine:
            def __init__(self):
                self.allow_fallback_values = []
                self.scan_count = 0

            def scan(self, image, box_thresh, allow_fallback):
                self.allow_fallback_values.append(allow_fallback)
                self.scan_count += 1
                return {
                    "lines": [{"text": str(self.scan_count), "score": 0.9, "y": 1, "x": 1, "h": 1}],
                    "extracted": None,
                    "line_count": 1,
                    "avg_confidence": 0.9,
                    "ocr_ms": 1.0,
                }

        fake_engine = FakeEngine()
        fallback = {
            "value": 132_000,
            "method": "largest_currency_amount",
            "low_confidence": True,
            "confidence": 0.9,
            "anchor": "",
            "amount_line": "132,000",
        }

        with (
            patch("ml.pipeline.storage.download", return_value=b"x" * 20_000),
            patch("ml.pipeline.preproc.decode", return_value=(np.zeros((600, 600, 3), dtype=np.uint8), {})),
            patch("ml.pipeline.preproc.image_stats", return_value={}),
            patch(
                "ml.pipeline.preproc.build_candidates",
                return_value=(candidate for candidate in [first, last]),
            ),
            patch("ml.pipeline.ocr_engine.peek", return_value=None),
            patch("ml.pipeline.ocr_engine.shared", return_value=fake_engine),
            patch("ml.pipeline.ocr_engine.extract_income", return_value=fallback) as extract_income,
        ):
            result = pipeline._scan_one(
                {
                    "path": "test/itr.png",
                    "application_id": 1,
                    "student_name": "Test applicant",
                    "declared_income": 132_000,
                    "linked": True,
                }
            )

        self.assertEqual(result["ocr_value"], 132_000)
        self.assertEqual(result["variant"], "grayscale+otsu")
        self.assertTrue(result["low_confidence"])
        self.assertEqual(fake_engine.scan_count, 3)
        self.assertEqual(fake_engine.allow_fallback_values, [False, False, False])
        extract_income.assert_called_once_with(
            [{"text": "3", "score": 0.9, "y": 1, "x": 1, "h": 1}],
            allow_fallback=True,
        )


if __name__ == "__main__":
    unittest.main()
