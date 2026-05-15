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
    assert result.barcode.backend == "zxing-cpp"
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


def test_decode_short_circuits_before_sr_model_and_wechat_when_original_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = make_qr("short-circuit", module_px=6)
    calls = 0
    wechat_inits = 0

    def sr_model(source: np.ndarray, scale: float) -> np.ndarray:
        nonlocal calls
        calls += 1
        return cv2.resize(source, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    class FakeWechat:
        def detectAndDecode(self, image: np.ndarray) -> tuple[tuple[str, ...], None]:
            raise AssertionError("WeChat fallback should not run after ZXing succeeds")

    def make_wechat() -> FakeWechat:
        nonlocal wechat_inits
        wechat_inits += 1
        return FakeWechat()

    monkeypatch.setattr(cv2, "wechat_qrcode_WeChatQRCode", make_wechat, raising=False)

    result = decode_image(
        image,
        DecodeConfig(backends=("zxing", "wechat"), scale_factors=(2.0,), interpolations=("nearest",)),
        sr_model=sr_model,
    )

    assert result.success
    assert result.payload == "short-circuit"
    assert result.variant == "original"
    assert len(result.attempts) == 1
    assert result.attempts[0].backend == "zxing-cpp"
    assert calls == 0
    assert wechat_inits == 0


def test_failed_decode_keeps_attempt_diagnostics() -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)

    result = decode_image(
        image,
        DecodeConfig(
            backends=("zxing",),
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
    assert payload["backend"] is None
    assert payload["attempts"][0]["backend"] == "zxing-cpp"
    assert payload["attempts"][0]["variant"] == "original"


def test_wechat_fallback_runs_after_zxing_miss(monkeypatch: pytest.MonkeyPatch) -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    calls: list[tuple[int, int, int]] = []

    class FakeWechat:
        def detectAndDecode(self, image: np.ndarray) -> tuple[list[str], np.ndarray]:
            calls.append(image.shape)
            return (
                ["wechat-hit"],
                np.array([[[2.0, 4.0], [6.0, 4.0], [6.0, 8.0], [2.0, 8.0]]], dtype=np.float32),
            )

    monkeypatch.setattr(cv2, "wechat_qrcode_WeChatQRCode", FakeWechat, raising=False)

    result = decode_image(
        image,
        DecodeConfig(
            backends=("zxing", "wechat"),
            scale_factors=(),
            enable_clahe=False,
            enable_sharpen=False,
        ),
    )

    assert result.success
    assert result.payload == "wechat-hit"
    assert result.barcode is not None
    assert result.barcode.backend == "wechat_qrcode"
    assert result.barcode.position == ((2.0, 4.0), (6.0, 4.0), (6.0, 8.0), (2.0, 8.0))
    assert [attempt.backend for attempt in result.attempts] == ["zxing-cpp", "wechat_qrcode"]
    assert calls == [(64, 64, 3)]


def test_wechat_unavailable_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    monkeypatch.setattr(cv2, "wechat_qrcode_WeChatQRCode", None, raising=False)

    result = decode_image(
        image,
        DecodeConfig(
            backends=("wechat",),
            scale_factors=(),
            enable_clahe=False,
            enable_sharpen=False,
        ),
    )

    assert not result.success
    assert result.attempts == ()
    assert any("WeChat QR fallback unavailable" in note for note in result.notes)
