# DG-MMT: TBME open-source code

Dynamic-Gated sEMG-Ultrasound Fusion Transformer for Temporally Consistent Hand Gesture Decoding.

## Release scope

This initial open-source release contains **data preprocessing and the primary
in-house model architecture only**, with contract tests and documentation.
Additional experiment-specific code, configuration files, split definitions,
and de-identified research data will be released following acceptance, subject
to ethics/data-sharing requirements and third-party dataset licenses.

## Installation and checks

Python 3.12 was used for local verification.

```bash
python -m pip install -r requirements.txt
python -m unittest -v test_contract
```

The tests use generated arrays solely to check equations, tensor dimensions,
window counts, segment boundaries, causal independence, and gradient propagation.

## Model

```python
import torch
from model import DG_MMT

model = DG_MMT(num_classes=9).eval()  # or 3 for the focused benchmark
# Replace these interface-demonstration inputs with processed recordings.
semg = torch.zeros(1, 4, 4, 50)      # batch, time, channels, samples
aus = torch.zeros(1, 4, 4, 128)      # batch, time, channels, depth
probabilities = model.predict_proba(semg, aus)
logits, gate = model(semg, aus, return_gating=True)
```

Each branch has two Conv1D–BN–ReLU layers, pooling and a 64-dimensional
projection. Kernels are 3/3 for sEMG and 5/3 for AUS. The gate is exactly
`sigmoid(Linear(concat(mean_time(E_s), mean_time(E_a))))`: 129 parameters,
one scalar per observation. Fusion is `[g E_s; (1-g) E_a]`, with no factor of
two. Eight learned positional tokens enter two pre-LN, four-head Transformer
layers. Flatten → FC64 → dropout 0.3 → class logits follows, without a classifier ReLU. Softmax
is explicit in `predict_proba`; there is no voting or output smoothing.
`dynamic_gate=False` provides the gate-free architecture comparison.

Hidden convolutional widths and Transformer FFN width that are not uniquely
specified in the manuscript are exposed as configurable reference-implementation
parameters: `conv_channels=(16,32)` and `ffn_dim=128`. Encoder dropout defaults
to `0.0` (configurable); classifier dropout is fixed at the documented `0.3`.
These defaults are not asserted to be confirmed historical experiment settings.
The released class targets the primary in-house
4-channel, 20-Hz AUS setup, **not** the task-adapted Ultra-Pro/GRABMyo models.

## Preprocessing contract

1. Supply previously assigned, separate trials or calibration/evaluation blocks.
   Splitting and training are not implemented in this release. The caller must
   enforce the manuscript protocol before window extraction.
2. Supply already synchronized sEMG at 1000 Hz and AUS at 20 Hz, sharing a
   timestamp origin. The API rejects unequal lengths; it cannot infer clock
   offsets or correct probe displacement.
3. Process each trial, calibration block or held-out block separately. The
   5-s calibration and separate 15-s evaluation blocks must never be concatenated.
   `role='monitoring'` is rejected by the dataset builder.
4. A causal 50-Hz notch precedes the Butterworth 20–450 Hz band-pass. SOS
   filter states persist across `filter_semg_chunk` calls. Call
   `reset_semg_state()` at an independent stream boundary; `process_segment`
   does this automatically, including between calibration and held-out blocks.
   Normalize only after filtering and window extraction, independently per
   channel using statistics from the current window. No `filtfilt` is used.
   `notch_q=30` is a configurable reference default because the manuscript
   specifies the notch frequency but not its Q factor. Filtering order,
   zero-state initialization and epsilon=1e-8 are explicit reference choices.
   SciPy `butter(N=4, btype='bandpass')` uses a fourth-order low-pass prototype;
   its band-pass transform has four SOS sections and an eighth-order transfer
   function. We document this distinction rather than claiming total order 4.
5. AUS: ADC offset removal → `exp(alpha*t)` → Hilbert envelope →
   `log(1+a*E)/log(1+a)` → ROI crop → linear interpolation to 128 depths.
   `AUSConfig` requires explicit TGC rate (s^-1), sample interval (seconds), ROI,
   compression factor, and ADC offset. **No unverified acquisition defaults**
   are silently supplied. Time starts at sample index 0.
   No additional RF/Gaussian 4-MHz band-pass is applied.
   `adc_offset` is an optional hardware-specific reference parameter and is not
   part of the manuscript-wide preprocessing specification; setting it to zero
   recovers the reported equation. The current API requires an explicit value.
6. Windows are 200 ms with 50-ms stride; endpoints are inclusive when a full
   window fits. Counts are 157 / 97 / 297 for 8 / 5 / 15 s. Return shapes are
   `[N,4,4,50]` and `[N,4,4,128]`.
7. This preprocessing interface returns signal tensors only; annotation,
   label assignment and training are outside this release.

```python
from preprocessing import AUSConfig, SignalProcessor

# Populate config values from verified acquisition metadata, not the unit tests.
# config = AUSConfig(tgc_rate_per_second=..., depth_sample_period_s=...,
#                    roi_start=..., roi_stop=..., log_compression=..., adc_offset=...)
# proc = SignalProcessor(config)
# semg_windows, aus_windows = proc.process_segment(
#     aligned_semg, aligned_aus, role='calibration')
```

License: MIT.
