import hashlib
import json
import os
from pathlib import Path
import numpy as np
from scipy.io import wavfile
import torch
from torch.utils.data import Dataset


def read_labels(clip, speed_threshold=0.5):
    states = np.load(clip / 'metadata/canonical_state_frames.npy', allow_pickle=False)
    times = np.load(clip / 'metadata/frame_times.npy', allow_pickle=False).reshape(-1)
    if states.ndim != 2 or states.shape != (len(times), 4):
        raise ValueError(f'{clip}: labels must have shape [T,4] in x,vx,y,vy order')
    if len(times) < 2 or not np.isfinite(states).all() or not np.isfinite(times).all():
        raise ValueError(f'{clip}: invalid/nonfinite labels')
    if np.any(np.diff(times) <= 0) or times[0] < 0:
        raise ValueError(f'{clip}: frame times must increase and be WAV-relative seconds')
    moving = np.linalg.norm(states[:, [1, 3]], axis=1) > speed_threshold
    indices = np.flatnonzero(moving)
    if len(indices) < 2:
        raise ValueError(f'{clip}: fewer than 2 moving frames')
    # Remove only leading/trailing stationary pads, never stitch across stops.
    span = slice(indices[0], indices[-1] + 1)
    return states[span].astype(np.float32), times[span].astype(np.float64), moving[span]


class Clips(Dataset):
    def __init__(self, available, names, config, cache=None, stats=None):
        self.available, self.names, self.config = available, names, config
        self.cache = Path(cache) if cache else None
        if self.cache:
            self.cache.mkdir(parents=True, exist_ok=True)
        self.stats = stats

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        clip = self.available[name]
        states, times, moving = read_labels(clip, self.config['speed_threshold'])
        wavs = sorted(clip.glob('*.wav')) + sorted(clip.glob('*.WAV'))
        if len(wavs) != 1:
            raise ValueError(f'{clip}: expected exactly one WAV, found {len(wavs)}')
        wav = wavs[0]
        sources = [wav, clip / 'metadata/canonical_state_frames.npy', clip / 'metadata/frame_times.npy']
        signature = [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in sources]
        token = hashlib.sha256(json.dumps([signature, self.config, 'logpower-v1'],
                                          sort_keys=True).encode()).hexdigest()[:24]
        cached = self.cache / (token + '.npy') if self.cache else None
        if cached and cached.exists():
            features = np.load(cached, allow_pickle=False)
        else:
            sr, audio = wavfile.read(wav)
            if sr != self.config['sample_rate'] or audio.ndim != 1:
                raise ValueError(f'{wav}: expected mono {self.config["sample_rate"]} Hz; got {sr}, {audio.shape}')
            if np.issubdtype(audio.dtype, np.signedinteger):
                audio = audio.astype(np.float32) / float(max(abs(np.iinfo(audio.dtype).min), np.iinfo(audio.dtype).max))
            elif audio.dtype == np.uint8:
                audio = (audio.astype(np.float32) - 128) / 128
            else:
                audio = audio.astype(np.float32)
            if not np.isfinite(audio).all() or len(audio) < 2:
                raise ValueError(f'{wav}: invalid waveform')
            nfft, hop = self.config['n_fft'], self.config['hop']
            with torch.no_grad():
                spec = torch.stft(torch.from_numpy(audio), n_fft=nfft, hop_length=hop,
                                  window=torch.hann_window(nfft), center=True,
                                  pad_mode='constant', return_complex=True)
                raw = torch.log(spec.abs().square().clamp_min(1e-10)).T.numpy()
            stft_times = np.arange(len(raw), dtype=np.float64) * hop / sr
            # Receiver/audio clock only; don't silently clamp gross clock mismatches.
            if times[-1] > len(audio) / sr + hop / sr or times[0] < 0:
                raise ValueError(f'{clip}: label clock exceeds WAV duration; check seconds/emission-time convention')
            features = np.stack([np.interp(times, stft_times, raw[:, i])
                                 for i in range(raw.shape[1])], axis=-1).astype(np.float32)
            if cached:
                temporary = cached.with_name(cached.name + f'.{os.getpid()}.tmp')
                with temporary.open('wb') as f:
                    np.save(f, features)
                os.replace(temporary, cached)
        if features.shape != (len(times), self.config['n_fft'] // 2 + 1):
            raise ValueError(f'{clip}: invalid cached feature shape')
        if self.stats is not None:
            features = (features - self.stats['mean']) / self.stats['std']
        return {'id': name, 'features': features.astype(np.float32),
                'position': states[:, [0, 2]], 'velocity': states[:, [1, 3]],
                'times': times.astype(np.float32), 'moving': moving}


def collate(items):
    lengths = torch.tensor([len(x['times']) for x in items], dtype=torch.long)
    result = {'ids': [x['id'] for x in items], 'lengths': lengths}
    for key in ('features', 'position', 'velocity', 'times', 'moving'):
        tensors = [torch.from_numpy(x[key]) for x in items]
        result[key] = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True)
    result['valid'] = torch.arange(int(lengths.max()))[None, :] < lengths[:, None]
    return result


def training_stats(dataset):
    from tqdm import tqdm
    total = square = None
    count = 0
    for item in tqdm(dataset, desc='Training-only feature statistics', total=len(dataset)):
        x = item['features'].astype(np.float64)
        total = x.sum(0) if total is None else total + x.sum(0)
        square = (x * x).sum(0) if square is None else square + (x * x).sum(0)
        count += len(x)
    mean = total / count
    std = np.sqrt(np.maximum(square / count - mean ** 2, 1e-4))
    return {'mean': mean.astype(np.float32), 'std': std.astype(np.float32), 'frames': count}
