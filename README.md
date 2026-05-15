# zxing-cpp-sr

Standalone QR/barcode decoder cascade. It ports the tested decoder-chain
solutions from the main `pricetag_vision` repo into a shareable package: lazy
first-success backend dispatch, WeChat QR init failure handling, and the
approved WeChat RGB-to-BGR fix. The default order tries ZXing-C++ first, then
falls back to OpenCV's WeChat QR detector/SR backend when ZXing misses, and can
retry both over deterministic upscale and lightweight "classical SR" variants.
Optional `pyzbar` and OpenCV QR fallbacks are available for diagnostic sweeps.

This package does not ship a custom learned super-resolution model. The WeChat
fallback uses OpenCV-contrib's bundled QR detector/SR implementation when
available. The package-level variants are OpenCV-based bicubic/Lanczos upscales
plus optional CLAHE and sharpening. A caller can pass a model hook if a trained
SR network becomes available.

## Quickstart

```bash
cd zxing_cpp_sr
uv pip install -e .

uv run zxing-cpp-sr path/to/qr_crop.png
uv run zxing-cpp-sr path/to/qr_crop.png --json
uv run zxing-cpp-sr path/to/qr_crop.png --backends zxing,wechat
uv run zxing-cpp-sr path/to/qr_crop.png --backends zxing,wechat,pyzbar,opencv_qr

uv run python scripts/decode.py path/to/qr_crop.png \
  --scales 2,3,4 \
  --formats qr \
  --json
```

## Public API

```python
from zxing_cpp_sr import DecodeConfig, decode_file, decode_image

cfg = DecodeConfig(
    backends=("zxing", "wechat"),
    input_color="bgr",  # use "rgb" for RGB arrays from the main pipeline
    formats="qr",
    scale_factors=(2.0, 3.0, 4.0),
    interpolations=("cubic", "lanczos"),
    enable_clahe=True,
    enable_sharpen=True,
)

result = decode_file("crop.png", cfg)
result.success       # bool
result.payload       # str | None
result.barcode.backend if result.barcode else None  # "zxing-cpp" or "wechat_qrcode"
result.variant       # e.g. "original", "cubic-x2-clahe"
result.attempts      # per-variant diagnostics
result.to_dict()     # JSON-serializable report
```

Supported backend names and aliases:

- `zxing`, `zxing-cpp`, `zxingcpp`
- `wechat`, `wechat_qr`, `wechat_qrcode`
- `pyzbar`, `zbar` (optional extra: `uv pip install -e ".[pyzbar]"` plus libzbar)
- `opencv`, `opencv_qr`, `cv2`

The package remains standalone and does not import `pricetag_vision`. The
cascade behavior is copied/adapted from the approved main-repo code:

- `pricetag_vision.core.qr.UnionBarcodeDecoder` for WeChat setup, pyzbar/OpenCV
  QR fallbacks, and the WeChat BGR channel-order fix;
- `pricetag_vision.core.qr_pipeline._DecoderChain` for lazy first-success
  backend dispatch.

More details live in [docs/cascade.md](docs/cascade.md).

Optional model hook:

```python
def my_sr_model(image_bgr, scale):
    # Return uint8 grayscale or BGR image.
    return restored_bgr

result = decode_image(image_bgr, cfg, sr_model=my_sr_model)
```

## What It Records

For each attempted variant:

- variant name and scale;
- decoder backend (`zxing-cpp`, `wechat_qrcode`, `pyzbar`, or `opencv_qr`);
- image size handed to the backend;
- elapsed time;
- every barcode candidate returned by the backend;
- validity / error state when `return_errors=True`;
- original-coordinate position quad, scaled back from the variant image.

This makes it usable both as a decoder and as a small diagnostic probe for
partial-content or checksum-error cases.

## Benchmarks

Measured QR parsing speed and recall notes live in
[docs/benchmark_results.md](docs/benchmark_results.md). Current result:
`original_only` gets all decodes that the SR retry stack gets on the local Lenta
fixtures, while staying in the sub-millisecond to low-single-digit millisecond
range for crop-sized inputs. The full retry stack is useful for offline failure
analysis, not the hot path, unless a target crop family proves otherwise.

## Scope

In:

- ZXing-C++ first decoding.
- Lazy WeChat QR fallback via OpenCV-contrib's `wechat_qrcode_WeChatQRCode`
  backend when ZXing misses.
- Optional pyzbar and OpenCV QR fallback backends for parity with the tested
  main-repo cascade.
- Optional retail format mode (`QR`, `EAN-13`, `EAN-8`, `UPC-A`, `UPC-E`).
- Deterministic upscale / CLAHE / sharpen retries.
- Partial/error candidate capture via ZXing-C++ `return_errors=True`.

Out:

- Detection in full video frames.
- Reed-Solomon override or Lenta-specific QR recovery.
- Generic image restoration. Learned SR should be passed as a hook and measured
  separately.

## Validation

```bash
cd zxing_cpp_sr
uv run pytest -q
uv run ruff check .
```
