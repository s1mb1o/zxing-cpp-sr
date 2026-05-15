"""ZXing-C++ decode wrapper with deterministic classical SR variants."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

SRModel = Callable[[np.ndarray, float], np.ndarray]
ZXING_BACKEND = "zxing-cpp"
WECHAT_BACKEND = "wechat_qrcode"
PYZBAR_BACKEND = "pyzbar"
OPENCV_QR_BACKEND = "opencv_qr"
_UNAVAILABLE = object()
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DecodedBarcode:
    """A normalized barcode result from one decoder backend."""

    backend: str
    text: str
    bytes_hex: str
    format: str
    symbology: str
    symbology_identifier: str
    valid: bool
    error: str | None
    position: tuple[tuple[float, float], ...] | None
    extra: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "text": self.text,
            "bytes_hex": self.bytes_hex,
            "format": self.format,
            "symbology": self.symbology,
            "symbology_identifier": self.symbology_identifier,
            "valid": self.valid,
            "error": self.error,
            "position": self.position,
            "extra": self.extra,
        }


@dataclass(frozen=True)
class DecodeAttempt:
    """One decoder backend pass on one preprocessing variant."""

    backend: str
    variant: str
    scale: float
    width: int
    height: int
    elapsed_ms: float
    barcodes: tuple[DecodedBarcode, ...]
    error: str | None = None

    @property
    def decoded(self) -> bool:
        return any(item.text and item.valid for item in self.barcodes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "variant": self.variant,
            "scale": self.scale,
            "width": self.width,
            "height": self.height,
            "elapsed_ms": self.elapsed_ms,
            "decoded": self.decoded,
            "barcodes": [item.to_dict() for item in self.barcodes],
            "error": self.error,
        }


@dataclass(frozen=True)
class DecodeResult:
    """Full decode result, including every attempted variant."""

    success: bool
    payload: str | None
    variant: str | None
    scale: float | None
    barcode: DecodedBarcode | None
    attempts: tuple[DecodeAttempt, ...]
    elapsed_ms: float
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "payload": self.payload,
            "backend": self.barcode.backend if self.barcode is not None else None,
            "variant": self.variant,
            "scale": self.scale,
            "barcode": self.barcode.to_dict() if self.barcode is not None else None,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "elapsed_ms": self.elapsed_ms,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class DecodeConfig:
    """Controls decoder backends and preprocessing retries."""

    backends: tuple[str, ...] = ("zxing", "wechat")
    input_color: str = "bgr"  # bgr for cv2.imread/CLI, rgb for main-pipeline crops
    formats: str = "qr"  # qr, retail, all
    scale_factors: tuple[float, ...] = (2.0, 3.0, 4.0)
    interpolations: tuple[str, ...] = ("cubic", "lanczos")
    try_original_first: bool = True
    enable_clahe: bool = True
    enable_sharpen: bool = True
    try_rotate: bool = True
    try_downscale: bool = True
    try_invert: bool = True
    return_errors: bool = True
    accept_checksum_errors: bool = False


def decode_file(
    path: str | Path,
    config: DecodeConfig | None = None,
    *,
    sr_model: SRModel | None = None,
) -> DecodeResult:
    """Decode an image file."""

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return decode_image(image, config=config, sr_model=sr_model)


def decode_image(
    image: np.ndarray,
    config: DecodeConfig | None = None,
    *,
    sr_model: SRModel | None = None,
) -> DecodeResult:
    """Decode an image using a ZXing-first cascade over original and SR/upscale variants."""

    cfg = config or DecodeConfig()
    source = _normalize_image(image, input_color=cfg.input_color)
    backends = _normalize_backends(cfg.backends)
    started = time.perf_counter()
    attempts: list[DecodeAttempt] = []
    notes: list[str] = []
    backend_state: dict[str, Any] = {}

    for variant_name, scale, variant in _iter_variants(source, cfg, sr_model=sr_model):
        for backend in backends:
            attempt_started = time.perf_counter()
            try:
                decoded = _decode_backend(
                    backend,
                    variant,
                    cfg,
                    scale=scale,
                    backend_state=backend_state,
                    notes=notes,
                )
                if decoded is None:
                    continue
                barcodes = tuple(decoded)
                error = None
            except Exception as exc:  # pragma: no cover - environment-dependent binding failures
                barcodes = ()
                error = str(exc)
            elapsed_ms = (time.perf_counter() - attempt_started) * 1000.0
            height, width = variant.shape[:2]
            attempt = DecodeAttempt(
                backend=backend,
                variant=variant_name,
                scale=scale,
                width=width,
                height=height,
                elapsed_ms=elapsed_ms,
                barcodes=barcodes,
                error=error,
            )
            attempts.append(attempt)

            winner = _pick_success(barcodes, accept_checksum_errors=cfg.accept_checksum_errors)
            if winner is not None:
                total_ms = (time.perf_counter() - started) * 1000.0
                return DecodeResult(
                    success=True,
                    payload=winner.text,
                    variant=variant_name,
                    scale=scale,
                    barcode=winner,
                    attempts=tuple(attempts),
                    elapsed_ms=total_ms,
                    notes=tuple(notes),
                )

    total_ms = (time.perf_counter() - started) * 1000.0
    if any(item.barcodes for item in attempts):
        notes.append("Decoder candidates were returned, but none were accepted as valid payloads.")
    return DecodeResult(
        success=False,
        payload=None,
        variant=None,
        scale=None,
        barcode=None,
        attempts=tuple(attempts),
        elapsed_ms=total_ms,
        notes=tuple(notes),
    )


def _normalize_backends(backends: tuple[str, ...]) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in backends:
        key = value.strip().lower().replace("_", "-")
        if key in {"zxing", "zxing-cpp", "zxingcpp"}:
            backend = ZXING_BACKEND
        elif key in {"wechat", "wechat-qr", "wechat-qrcode"}:
            backend = WECHAT_BACKEND
        elif key in {"pyzbar", "zbar"}:
            backend = PYZBAR_BACKEND
        elif key in {"opencv", "opencv-qr", "cv2"}:
            backend = OPENCV_QR_BACKEND
        else:
            raise ValueError("backends must contain only: zxing, wechat, pyzbar, opencv_qr")
        if backend not in normalized:
            normalized.append(backend)
    if not normalized:
        raise ValueError("at least one backend is required")
    return tuple(normalized)


def _decode_backend(
    backend: str,
    image: np.ndarray,
    cfg: DecodeConfig,
    *,
    scale: float,
    backend_state: dict[str, Any],
    notes: list[str],
) -> list[DecodedBarcode] | None:
    if backend == ZXING_BACKEND:
        return _decode_zxing(image, cfg, scale=scale)
    if backend == WECHAT_BACKEND:
        detector = _get_wechat_detector(backend_state, notes)
        if detector is None:
            return None
        return _decode_wechat_qr(image, detector, scale=scale)
    if backend == PYZBAR_BACKEND:
        decode = _get_pyzbar_decode(backend_state, notes)
        if decode is None:
            return None
        return _decode_pyzbar(image, decode, scale=scale)
    if backend == OPENCV_QR_BACKEND:
        detector = _get_opencv_qr_detector(backend_state, notes)
        if detector is None:
            return None
        return _decode_opencv_qr(image, detector, scale=scale)
    raise AssertionError(f"unexpected backend: {backend}")


def _decode_zxing(image: np.ndarray, cfg: DecodeConfig, *, scale: float) -> list[DecodedBarcode]:
    import zxingcpp

    results = zxingcpp.read_barcodes(
        image,
        formats=_resolve_formats(zxingcpp, cfg.formats),
        try_rotate=cfg.try_rotate,
        try_downscale=cfg.try_downscale,
        try_invert=cfg.try_invert,
        return_errors=cfg.return_errors,
    )
    return [_normalize_barcode(item, scale=scale) for item in results]


def _get_wechat_detector(backend_state: dict[str, Any], notes: list[str]) -> Any | None:
    if "wechat_detector" in backend_state:
        detector = backend_state["wechat_detector"]
        return None if detector is _UNAVAILABLE else detector

    constructor = getattr(cv2, "wechat_qrcode_WeChatQRCode", None)
    if constructor is None:
        backend_state["wechat_detector"] = _UNAVAILABLE
        notes.append(
            "WeChat QR fallback unavailable: cv2 has no wechat_qrcode_WeChatQRCode. "
            "Install opencv-contrib-python to enable the CNN/SR fallback."
        )
        return None

    try:
        detector = constructor()
    except Exception as exc:  # pragma: no cover - depends on OpenCV build/model packaging
        backend_state["wechat_detector"] = _UNAVAILABLE
        logger.warning(
            "OpenCV WeChat QR decoder failed to initialize; continuing without it: %s",
            exc,
        )
        notes.append(f"WeChat QR fallback unavailable: {exc}")
        return None

    backend_state["wechat_detector"] = detector
    return detector


def _decode_wechat_qr(image: np.ndarray, detector: Any, *, scale: float) -> list[DecodedBarcode]:
    if image.ndim == 2:
        detector_input = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        # Ported from pricetag_vision.core.qr: WeChat's CNN/SR path expects BGR.
        # decode_image(..., input_color="rgb") converts caller RGB crops to BGR
        # before variants are generated, so all internal 3-channel images are BGR.
        detector_input = image

    texts, points = _unpack_wechat_result(detector.detectAndDecode(detector_input))
    barcodes: list[DecodedBarcode] = []
    for index, raw_text in enumerate(texts):
        text = str(raw_text or "")
        if not text:
            continue
        barcodes.append(
            DecodedBarcode(
                backend=WECHAT_BACKEND,
                text=text,
                bytes_hex=text.encode("utf-8", "replace").hex(),
                format="QRCode",
                symbology="QR Code",
                symbology_identifier="",
                valid=True,
                error=None,
                position=_opencv_points_to_tuple(points, index=index, scale=scale),
                extra={"detector": "cv2.wechat_qrcode_WeChatQRCode"},
            )
        )
    return barcodes


def _get_pyzbar_decode(backend_state: dict[str, Any], notes: list[str]) -> Any | None:
    if "pyzbar_decode" in backend_state:
        decode = backend_state["pyzbar_decode"]
        return None if decode is _UNAVAILABLE else decode

    try:
        from pyzbar.pyzbar import decode
    except ImportError:
        backend_state["pyzbar_decode"] = _UNAVAILABLE
        notes.append("pyzbar fallback unavailable: install pyzbar and libzbar to enable it.")
        return None

    backend_state["pyzbar_decode"] = decode
    return decode


def _decode_pyzbar(image: np.ndarray, decode: Any, *, scale: float) -> list[DecodedBarcode]:
    decoded = decode(image)
    barcodes: list[DecodedBarcode] = []
    for item in decoded:
        raw_bytes = bytes(getattr(item, "data", b"") or b"")
        text = raw_bytes.decode("utf-8", errors="replace")
        symbology = str(getattr(item, "type", "") or "")
        if not text:
            continue
        barcodes.append(
            DecodedBarcode(
                backend=PYZBAR_BACKEND,
                text=text,
                bytes_hex=raw_bytes.hex(),
                format=symbology,
                symbology=symbology,
                symbology_identifier="",
                valid=True,
                error=None,
                position=_pyzbar_polygon_to_tuple(getattr(item, "polygon", None), scale=scale),
                extra={"detector": "pyzbar"},
            )
        )
    return barcodes


def _get_opencv_qr_detector(backend_state: dict[str, Any], notes: list[str]) -> Any | None:
    if "opencv_qr_detector" in backend_state:
        detector = backend_state["opencv_qr_detector"]
        return None if detector is _UNAVAILABLE else detector

    try:
        detector = cv2.QRCodeDetector()
    except Exception as exc:  # pragma: no cover - depends on OpenCV build
        backend_state["opencv_qr_detector"] = _UNAVAILABLE
        notes.append(f"OpenCV QR fallback unavailable: {exc}")
        return None

    backend_state["opencv_qr_detector"] = detector
    return detector


def _decode_opencv_qr(image: np.ndarray, detector: Any, *, scale: float) -> list[DecodedBarcode]:
    try:
        data, points, _straight = detector.detectAndDecode(image)
    except ValueError:
        data, points = "", None
    if not data:
        return []
    text = str(data)
    return [
        DecodedBarcode(
            backend=OPENCV_QR_BACKEND,
            text=text,
            bytes_hex=text.encode("utf-8", "replace").hex(),
            format="QRCode",
            symbology="QRCODE",
            symbology_identifier="",
            valid=True,
            error=None,
            position=_opencv_points_to_tuple(points, index=0, scale=scale),
            extra={"detector": "cv2.QRCodeDetector"},
        )
    ]


def _unpack_wechat_result(result: Any) -> tuple[tuple[Any, ...], Any]:
    if isinstance(result, tuple):
        payloads = result[0] if result else ()
        points = result[1] if len(result) > 1 else None
    else:
        payloads = result
        points = None

    if payloads is None:
        texts: tuple[Any, ...] = ()
    elif isinstance(payloads, str):
        texts = (payloads,)
    else:
        texts = tuple(payloads)
    return texts, points


def _opencv_points_to_tuple(
    points: Any,
    *,
    index: int,
    scale: float,
) -> tuple[tuple[float, float], ...] | None:
    if points is None:
        return None
    array = np.asarray(points, dtype=np.float32)
    if array.size == 0:
        return None
    array = np.squeeze(array)
    if array.ndim == 3:
        if index >= array.shape[0]:
            return None
        quad = array[index]
    elif array.ndim == 2:
        quad = array
    else:
        return None
    flat_quad = np.asarray(quad, dtype=np.float32).reshape(-1, 2)
    if len(flat_quad) < 4:
        return None
    return tuple((float(x) / scale, float(y) / scale) for x, y in flat_quad[:4])


def _pyzbar_polygon_to_tuple(
    polygon: Any,
    *,
    scale: float,
) -> tuple[tuple[float, float], ...] | None:
    if polygon is None:
        return None
    points: list[tuple[float, float]] = []
    for point in polygon:
        x = getattr(point, "x", None)
        y = getattr(point, "y", None)
        if x is None or y is None:
            return None
        points.append((float(x) / scale, float(y) / scale))
    return tuple(points) if points else None


def _resolve_formats(zxingcpp: Any, formats: str) -> Any:
    key = formats.strip().lower()
    if key == "all":
        return None
    if key == "qr":
        return zxingcpp.BarcodeFormat.QRCode
    if key == "retail":
        return [
            zxingcpp.BarcodeFormat.QRCode,
            zxingcpp.BarcodeFormat.EAN13,
            zxingcpp.BarcodeFormat.EAN8,
            zxingcpp.BarcodeFormat.UPCA,
            zxingcpp.BarcodeFormat.UPCE,
        ]
    raise ValueError("formats must be one of: qr, retail, all")


def _normalize_barcode(item: Any, *, scale: float) -> DecodedBarcode:
    raw_bytes = getattr(item, "bytes", b"") or b""
    if isinstance(raw_bytes, str):
        bytes_hex = raw_bytes.encode("utf-8", "replace").hex()
    else:
        bytes_hex = bytes(raw_bytes).hex()
    return DecodedBarcode(
        backend=ZXING_BACKEND,
        text=str(getattr(item, "text", "") or ""),
        bytes_hex=bytes_hex,
        format=str(getattr(item, "format", "")),
        symbology=str(getattr(item, "symbology", "")),
        symbology_identifier=str(getattr(item, "symbology_identifier", "")),
        valid=bool(getattr(item, "valid", False)),
        error=_error_to_str(getattr(item, "error", None)),
        position=_position_to_tuple(getattr(item, "position", None), scale=scale),
        extra=dict(getattr(item, "extra", {}) or {}),
    )


def _error_to_str(error: Any) -> str | None:
    if error is None:
        return None
    return str(error)


def _position_to_tuple(position: Any, *, scale: float) -> tuple[tuple[float, float], ...] | None:
    if position is None:
        return None
    points = (
        getattr(position, "top_left", None),
        getattr(position, "top_right", None),
        getattr(position, "bottom_right", None),
        getattr(position, "bottom_left", None),
    )
    out: list[tuple[float, float]] = []
    for point in points:
        if point is None:
            return None
        out.append((float(point.x) / scale, float(point.y) / scale))
    return tuple(out)


def _pick_success(
    barcodes: tuple[DecodedBarcode, ...],
    *,
    accept_checksum_errors: bool,
) -> DecodedBarcode | None:
    for barcode in barcodes:
        if not barcode.text:
            continue
        if barcode.valid or accept_checksum_errors:
            return barcode
    return None


def _iter_variants(
    image: np.ndarray,
    cfg: DecodeConfig,
    *,
    sr_model: SRModel | None,
) -> Iterator[tuple[str, float, np.ndarray]]:
    if cfg.try_original_first:
        yield ("original", 1.0, image)

    for scale in cfg.scale_factors:
        if scale <= 0:
            raise ValueError("scale factors must be positive")
        if scale == 1.0 and cfg.try_original_first:
            continue
        for interpolation in cfg.interpolations:
            resized = _resize(image, scale, interpolation)
            base_name = f"{interpolation}-x{scale:g}"
            yield (base_name, scale, resized)
            if cfg.enable_clahe:
                yield (f"{base_name}-clahe", scale, _apply_clahe(resized))
            if cfg.enable_sharpen:
                yield (f"{base_name}-sharpen", scale, _sharpen(resized))
        if sr_model is not None:
            modeled = _normalize_image(sr_model(image, scale), input_color="bgr")
            yield (f"model-x{scale:g}", scale, modeled)


def _normalize_image(image: np.ndarray, *, input_color: str = "bgr") -> np.ndarray:
    color = _normalize_input_color(input_color)
    array = np.asarray(image)
    if array.dtype != np.uint8:
        raise TypeError(f"image dtype must be uint8, got {array.dtype}")
    if array.ndim == 2:
        return np.ascontiguousarray(array)
    if array.ndim != 3:
        raise ValueError("image must be grayscale, BGR/RGB, or BGRA/RGBA")
    if array.shape[2] == 3:
        if color == "rgb":
            return np.ascontiguousarray(cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
        return np.ascontiguousarray(array)
    if array.shape[2] == 4:
        code = cv2.COLOR_RGBA2BGR if color == "rgb" else cv2.COLOR_BGRA2BGR
        return np.ascontiguousarray(cv2.cvtColor(array, code))
    raise ValueError("image must be grayscale, BGR/RGB, or BGRA/RGBA")


def _normalize_input_color(value: str) -> str:
    color = str(value).strip().lower()
    if color not in {"bgr", "rgb"}:
        raise ValueError("input_color must be one of: bgr, rgb")
    return color


def _resize(image: np.ndarray, scale: float, interpolation: str) -> np.ndarray:
    interp_table = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
    }
    if interpolation not in interp_table:
        raise ValueError(f"unknown interpolation: {interpolation}")
    height, width = image.shape[:2]
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(image, new_size, interpolation=interp_table[interpolation])


def _apply_clahe(image: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    if image.ndim == 2:
        return clahe.apply(image)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _sharpen(image: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.0)
    return cv2.addWeighted(image, 1.5, blurred, -0.5, 0.0)


def _parse_csv_floats(value: str) -> tuple[float, ...]:
    if not value.strip():
        return ()
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def _parse_csv_strings(value: str) -> tuple[str, ...]:
    if not value.strip():
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Decode QR/barcodes with a cascade + SR retries.")
    parser.add_argument("image", type=Path, help="image to decode")
    parser.add_argument("--json", action="store_true", help="print full JSON diagnostics")
    parser.add_argument(
        "--formats",
        choices=("qr", "retail", "all"),
        default="qr",
        help="barcode formats to ask ZXing to decode",
    )
    parser.add_argument("--scales", default="2,3,4", help="comma-separated SR/upscale factors")
    parser.add_argument(
        "--backends",
        default="zxing,wechat",
        help="comma-separated decoder backends; supported: zxing,wechat,pyzbar,opencv_qr",
    )
    parser.add_argument(
        "--input-color",
        choices=("bgr", "rgb"),
        default="bgr",
        help="channel order for 3/4-channel input arrays; CLI files are read as bgr",
    )
    parser.add_argument(
        "--interpolations",
        default="cubic,lanczos",
        help="comma-separated OpenCV resize methods",
    )
    parser.add_argument("--no-original", action="store_true", help="skip the original-image attempt")
    parser.add_argument("--no-clahe", action="store_true", help="disable CLAHE variants")
    parser.add_argument("--no-sharpen", action="store_true", help="disable sharpen variants")
    parser.add_argument("--no-return-errors", action="store_true", help="do not ask ZXing for errors")
    parser.add_argument(
        "--accept-errors",
        action="store_true",
        help="accept non-valid ZXing results when text is present",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = DecodeConfig(
        backends=_parse_csv_strings(args.backends),
        input_color=args.input_color,
        formats=args.formats,
        scale_factors=_parse_csv_floats(args.scales),
        interpolations=_parse_csv_strings(args.interpolations),
        try_original_first=not args.no_original,
        enable_clahe=not args.no_clahe,
        enable_sharpen=not args.no_sharpen,
        return_errors=not args.no_return_errors,
        accept_checksum_errors=bool(args.accept_errors),
    )
    result = decode_file(args.image, cfg)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    elif result.success:
        print(f"OK payload={result.payload!r}")
        print(
            f"   backend={result.barcode.backend if result.barcode else None} "
            f"variant={result.variant} scale={result.scale} attempts={len(result.attempts)}"
        )
        print(f"   elapsed={result.elapsed_ms:.1f} ms")
    else:
        print(f"FAIL attempts={len(result.attempts)} elapsed={result.elapsed_ms:.1f} ms")
        for note in result.notes:
            print(f"   note: {note}")
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
