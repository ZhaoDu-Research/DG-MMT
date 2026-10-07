from dataclasses import dataclass
import numpy as np
from scipy import signal


@dataclass(frozen=True)
class AUSConfig:

    tgc_rate_per_second: float
    depth_sample_period_s: float
    roi_start: int
    roi_stop: int
    log_compression: float
    adc_offset: float

    def __post_init__(self):
        values = (self.tgc_rate_per_second, self.depth_sample_period_s,
                  self.log_compression, self.adc_offset)
        if not np.isfinite(values).all():
            raise ValueError('AUS parameters must be finite.')
        if self.tgc_rate_per_second < 0 or self.depth_sample_period_s <= 0:
            raise ValueError('Invalid TGC rate or depth sampling period.')
        if self.log_compression <= 0:
            raise ValueError('Log compression factor must be positive.')
        if not (isinstance(self.roi_start, int) and isinstance(self.roi_stop, int)
                and 0 <= self.roi_start < self.roi_stop <= 1000
                and self.roi_stop - self.roi_start >= 2):
            raise ValueError('ROI must contain at least two of the 1000 depth samples.')
        if self.tgc_rate_per_second * 999 * self.depth_sample_period_s > 80:
            raise ValueError('TGC exponent is too large; check units.')


def _array(x, name):
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError(f'{name} contains nonfinite samples.')
    return x


def process_aus_frame(raw_frame, config):

    x = _array(raw_frame, 'AUS')
    if x.shape != (4, 1000):
        raise ValueError('Expected AUS frame [4,1000].')
    time = np.arange(1000) * config.depth_sample_period_s
    compensated = (x - config.adc_offset) * np.exp(config.tgc_rate_per_second * time)
    envelope = np.abs(signal.hilbert(compensated, axis=-1))
    compressed = np.log1p(config.log_compression * envelope) / np.log1p(config.log_compression)
    roi = compressed[:, config.roi_start:config.roi_stop]

    old = np.linspace(0, 1, roi.shape[1])
    new = np.linspace(0, 1, 128)
    return np.stack([np.interp(new, old, ch) for ch in roi]).astype(np.float32)


def normalize_window(semg_window, epsilon=1e-8):

    x = _array(semg_window, 'sEMG')
    if x.shape != (200, 4) or not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('Expected [200,4] and positive finite epsilon.')
    return ((x-x.mean(axis=0)) / (x.std(axis=0, ddof=0)+epsilon)).astype(np.float32)


def window_count(duration_s):
    samples = duration_s * 1000
    if not np.isfinite(samples) or samples < 0 or not np.isclose(samples, round(samples)):
        raise ValueError('Duration must represent an integer number of 1-kHz samples.')
    return max(0, (round(samples)-200)//50+1)


class SignalProcessor:


    def __init__(self, aus_config, *, notch_q=30.0):
        if not np.isfinite(notch_q) or notch_q <= 0:
            raise ValueError('notch_q must be positive and finite.')
        self.aus_config = aus_config


        self.band = signal.butter(4, [20, 450], btype='bandpass', fs=1000, output='sos')
        b, a = signal.iirnotch(50, notch_q, fs=1000)
        self.notch = signal.tf2sos(b, a)
        self.reset_semg_state()

    def reset_semg_state(self):

        self._notch_state = np.zeros((len(self.notch), 2, 4))
        self._band_state = np.zeros((len(self.band), 2, 4))

    def filter_semg_chunk(self, raw_semg):


        x = _array(raw_semg, 'sEMG')
        if x.ndim != 2 or x.shape[1] != 4:
            raise ValueError('Expected sEMG chunk [samples,4].')
        if len(x) == 0:
            return x.copy()
        x, self._notch_state = signal.sosfilt(self.notch, x, axis=0, zi=self._notch_state)
        x, self._band_state = signal.sosfilt(self.band, x, axis=0, zi=self._band_state)
        return x

    def process_segment(self, raw_semg, raw_aus, *, role):


        if role not in {'train','validation','test','calibration','monitoring'}:
            raise ValueError('Explicit segment role is required.')
        if role == 'monitoring':
            raise ValueError('Monitoring is excluded from updating and held-out datasets.')
        semg, aus = _array(raw_semg, 'sEMG'), _array(raw_aus, 'AUS')
        if aus.ndim != 3 or aus.shape[1:] != (4, 1000) or len(aus) < 4:
            raise ValueError('Expected >=4 AUS frames, each [4,1000].')
        if semg.shape != (len(aus)*50, 4):
            raise ValueError('Signals must be aligned; no silent truncation is permitted.')
        self.reset_semg_state()
        filtered = self.filter_semg_chunk(semg)
        processed = np.stack([process_aus_frame(frame, self.aus_config) for frame in aus])
        sw, aw = [], []
        for i in range(len(aus)-4+1):
            window = normalize_window(filtered[i*50:(i+4)*50])
            sw.append(window.reshape(4,50,4).transpose(0,2,1))
            aw.append(processed[i:i+4])
        return np.stack(sw), np.stack(aw)
