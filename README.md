# zxing-cpp-sr

Standalone ZXing-C++ wrapper that retries QR/barcode decoding over deterministic
upscale and lightweight "classical SR" variants. It is meant to be easy for
teammates to install, run on a folder of hard crops, and inspect why a crop did
or did not decode.

This is not a learned super-resolution model. The default variants are
OpenCV-based bicubic/Lanczos upscales plus optional CLAHE and sharpening. A
caller can pass a model hook if a trained SR network becomes available.

## Quickstart

```bash
cd zxing_cpp_sr
uv pip install -e .

uv run zxing-cpp-sr path/to/qr_crop.png
uv run zxing-cpp-sr path/to/qr_crop.png --json

uv run python scripts/decode.py path/to/qr_crop.png \
  --scales 2,3,4 \
  --formats qr \
  --json
```

## Public API

```python
from zxing_cpp_sr import DecodeConfig, decode_file, decode_image

cfg = DecodeConfig(
    formats="qr",
    scale_factors=(2.0, 3.0, 4.0),
    interpolations=("cubic", "lanczos"),
    enable_clahe=True,
    enable_sharpen=True,
)

result = decode_file("crop.png", cfg)
result.success       # bool
result.payload       # str | None
result.variant       # e.g. "original", "cubic-x2-clahe"
result.attempts      # per-variant diagnostics
result.to_dict()     # JSON-serializable report
```

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
- image size handed to ZXing-C++;
- elapsed time;
- every barcode candidate returned by ZXing-C++;
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

- QR-first ZXing-C++ decoding.
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
