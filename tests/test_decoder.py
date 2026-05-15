from __future__ import annotations

import cv2
import numpy as np
import pytest

from zxing_cpp_sr import DecodeConfig, decode_image

pytest.importorskip("zxingcpp")


def make_qr(payload: str, module_px: int = 8) -> np.ndarray:
    encoder = cv2.QRCodeEncoder_create()
    qr = encoder.encode(payload)
    qr = cv2.resize(
        qr,
        (qr.shape[1] * module_px, qr.shape[0] * module_px),
        interpolation=cv2.INTER_NEAREST,
    )
    return cv2.cvtColor(qr, cv2.COLOR_GRAY2BGR)


def test_decode_generated_qr_original() -> None:
    image = make_qr("barcode=4680140271438&price1=144.99", module_px=6)

    result = decode_image(image, DecodeConfig(scale_factors=(), enable_clahe=False))

    assert result.success
    assert result.payload == "barcode=4680140271438&price1=144.99"
    assert result.variant == "original"
    assert result.barcode is not None
    assert result.barcode.valid
    assert result.barcode.position is not None


def test_decode_uses_scaled_variant_when_original_disabled() -> None:
    image = make_qr("tiny-qr", module_px=3)

    result = decode_image(
        image,
        DecodeConfig(
            scale_factors=(2.0,),
            interpolations=("nearest",),
            try_original_first=False,
            enable_clahe=False,
            enable_sharpen=False,
        ),
    )

    assert result.success
    assert result.payload == "tiny-qr"
    assert result.variant == "nearest-x2"
    assert result.scale == 2.0


def test_decode_short_circuits_before_sr_model_when_original_succeeds() -> None:
    image = make_qr("short-circuit", module_px=6)
    calls = 0

    def sr_model(source: np.ndarray, scale: float) -> np.ndarray:
        nonlocal calls
        calls += 1
        return cv2.resize(source, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    result = decode_image(
        image,
        DecodeConfig(scale_factors=(2.0,), interpolations=("nearest",)),
        sr_model=sr_model,
    )

    assert result.success
    assert result.payload == "short-circuit"
    assert result.variant == "original"
    assert len(result.attempts) == 1
    assert calls == 0


def test_failed_decode_keeps_attempt_diagnostics() -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)

    result = decode_image(
        image,
        DecodeConfig(
            scale_factors=(2.0,),
            interpolations=("nearest",),
            enable_clahe=False,
            enable_sharpen=False,
        ),
    )

    assert not result.success
    assert result.payload is None
    assert len(result.attempts) == 2
    payload = result.to_dict()
    assert payload["backend"] == "zxing-cpp"
    assert payload["attempts"][0]["variant"] == "original"
