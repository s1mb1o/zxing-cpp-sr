# zxing-cpp-sr benchmark results

Measured: 2026-05-16

## Setup

Host: local MacBook `uv` environment.

Runtime stack from `uv.lock`:

- Python 3.12.9.
- `zxing-cpp==3.0.0`.
- `opencv-contrib-python==4.13.0.92`.
- `numpy==2.4.5`.

Timing source: `DecodeResult.elapsed_ms`, so image file reads via `cv2.imread`
are excluded. The synthetic sweep uses 20 repeated decodes per point after a
warmup. The real-image sweeps are one pass per file because those inputs are
used here as regression fixtures, not as a microbenchmark corpus.

Important implementation note: this benchmark also fixed the retry iterator to
be lazy. A successful original-image decode now stops before constructing any
upscaled / CLAHE / sharpen variants. `tests/test_decoder.py` covers this with
`test_decode_short_circuits_before_sr_model_when_original_succeeds`.

## Configurations

| Config | Meaning |
|---|---|
| `original_only` | One ZXing-C++ pass on the input image. No package-level resize / CLAHE / sharpen variants. |
| `default_sr_retry` | ZXing-C++ original first, then x2/x3/x4, bicubic/Lanczos, plus CLAHE and sharpen variants. Stops on first valid decode. |
| `x2_nearest_only` | Cheap fallback: original first, then a single x2 nearest-neighbor variant. No CLAHE / sharpen. |

All measured configs used `backends=("zxing",)`, `formats="qr"`, and
`return_errors=True` to isolate the classical upscale/SR retry cost. The current
package default is a cascade, `backends=("zxing", "wechat")`: WeChat's
OpenCV-contrib QR detector/SR backend is initialized lazily only after a ZXing
miss on the current variant.

## Real Lenta Sets

### Summary

| Set | Config | Success | Mean ms | Median ms | P95 ms | Mean attempts |
|---|---:|---:|---:|---:|---:|---:|
| `docs/qr_barcode/lenta_failures/*.png` | `original_only` | 2 / 8 (25.0%) | 0.218 | 0.204 | 0.384 | 1.00 |
| `docs/qr_barcode/lenta_failures/*.png` | `default_sr_retry` | 2 / 8 (25.0%) | 40.178 | 32.099 | 143.943 | 14.50 |
| `docs/qr_barcode/lenta_failures/*.png` | `x2_nearest_only` | 2 / 8 (25.0%) | 0.686 | 0.669 | 1.146 | 1.75 |
| `data/external/variants/rectified/*.jpg` | `original_only` | 8 / 17 (47.1%) | 2.389 | 2.057 | 3.964 | 1.00 |
| `data/external/variants/rectified/*.jpg` | `default_sr_retry` | 8 / 17 (47.1%) | 281.051 | 235.292 | 936.807 | 10.53 |
| `data/external/variants/rectified/*.jpg` | `x2_nearest_only` | 8 / 17 (47.1%) | 6.719 | 4.944 | 19.210 | 1.53 |

### Interpretation

- The SR/upscale retries produced **zero net-new decodes** on these two real
  Lenta sets. This matches the existing QR-recovery diagnosis: the `fail_*`
  samples are dominated by missing modules, clipped finders, or print damage,
  not by simple undersampling.
- `original_only` is the right production default when the crop is already a
  reasonably sized QR or a rectified tag crop.
- `x2_nearest_only` is a safe cheap fallback for diagnostic sweeps. It costs
  single-digit milliseconds on the tag-crop set and did not change recall here.
- `default_sr_retry` is intentionally expensive on failures. It tries 19
  variants when nothing decodes, and large tag crops can reach hundreds of
  milliseconds. Use it as an offline diagnostic, not inside the hot path.

### Successful Real Files

`docs/qr_barcode/lenta_failures/*.png` successful files:

- `ok_PXL_20260420_123511023__b0.png`
- `ok_PXL_20260420_123552804__b1.png`

`data/external/variants/rectified/*.jpg` successful files:

- `PXL_20260420_123511023__b0.jpg`
- `PXL_20260420_123540194__b0.jpg`
- `PXL_20260420_123540194__b1.jpg`
- `PXL_20260420_123552804__b0.jpg`
- `PXL_20260420_123552804__b1.jpg`
- `PXL_20260420_123623002.MP__b0.jpg`
- `PXL_20260420_123704224.MP__b0.jpg`
- `PXL_20260420_124255184__b0.jpg`

## Synthetic Resolution Sweep

Payload:

```text
barcode=4680140271438&price1=144.99&price2=137.79&price4=129.99&aP=
```

OpenCV generated a 37x37-module QR symbol. Each row below is 20/20 successful
decodes. Times are wall-clock milliseconds per decode.

| Module px | Image px | `original_only` mean | `original_only` p95 | `default_sr_retry` mean | `default_sr_retry` p95 |
|---:|---:|---:|---:|---:|---:|
| 1 | 37x37 | 0.050 | 0.065 | 0.044 | 0.046 |
| 2 | 74x74 | 0.062 | 0.066 | 0.060 | 0.063 |
| 3 | 111x111 | 0.080 | 0.084 | 0.081 | 0.085 |
| 4 | 148x148 | 0.112 | 0.120 | 0.110 | 0.112 |
| 6 | 222x222 | 0.189 | 0.193 | 0.184 | 0.190 |
| 8 | 296x296 | 0.283 | 0.291 | 0.328 | 0.362 |
| 12 | 444x444 | 0.591 | 0.617 | 0.610 | 0.628 |
| 16 | 592x592 | 1.143 | 1.238 | 1.329 | 1.889 |

### Synthetic Takeaways

- ZXing-C++ is sub-millisecond on generated QR crops up to about 444x444 px.
- Runtime scales mostly with input pixel area once detection is trivial.
- After the lazy retry fix, `default_sr_retry` has essentially the same cost as
  `original_only` when the original image decodes, because no fallback variants
  are materialized.

## Recommendation

Use a two-tier policy:

1. Run `original_only` first in production. It is fast and gets all decodable
   cases in the current Lenta fixtures.
2. Escalate to the WeChat QR fallback after a ZXing miss when QR recall matters
   more than worst-case latency. It can be enabled with
   `DecodeConfig(backends=("zxing", "wechat"))` or `--backends zxing,wechat`.
3. Escalate to `x2_nearest_only` or `default_sr_retry` only for offline review,
   failure bucketing, or a crop family where a benchmark proves that upscale
   variants add net-new decodes.

For the current QR residual class, spend effort on geometry / finder recovery /
Reed-Solomon-aware recovery rather than generic upscale retries.
