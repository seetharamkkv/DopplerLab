import torch
import torchaudio.transforms as T
import torch.nn.functional as F
import soundfile as sf

class SpectrogramGenerator:
    def __init__(self, sample_rate=16000, n_fft=1024, hop_length=256, n_mels=128):
        """
        Generates multi-channel log-Mel spectrograms with delta features.
        Output channels: [Mel, Delta, Delta-Delta] for tracking Doppler frequency shifts.
        
        Args:
            sample_rate (int): Target sample rate.
            n_fft (int): FFT window size.
            hop_length (int): Hop length (256 for higher temporal resolution).
            n_mels (int): Number of Mel frequency bins.
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        
        self.amplitude_to_db = T.AmplitudeToDB()
        
        # Delta computation (velocity and acceleration of spectral change)
        self.compute_deltas = T.ComputeDeltas()

    def generate(self, audio_path, target_frames=None):
        """
        Loads audio and generates a 3-channel normalized spectrogram.
        Channels: [log-Mel, Delta, Delta-Delta]
        
        Returns:
            torch.Tensor: Shape (3, n_mels, target_frames)
        """
        # Load audio
        wav, sr = sf.read(audio_path)
        waveform = torch.from_numpy(wav).float()
        
        # Ensure (C, L) format: (1, samples)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.shape[0] > 1 and waveform.shape[1] > waveform.shape[0]:
            pass  # Already (C, L)
        else:
            waveform = waveform.transpose(0, 1)
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
            
        # Convert to mono if multi-channel
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Generate Mel Spectrogram
        mel_spec = self.mel_spectrogram(waveform)  # (1, n_mels, frames)
        
        # Convert to log scale (dB)
        log_mel_spec = self.amplitude_to_db(mel_spec)  # (1, n_mels, frames)
        
        # Z-score normalization (more stable than min-max)
        mean = log_mel_spec.mean()
        std = log_mel_spec.std()
        log_mel_spec = (log_mel_spec - mean) / (std + 1e-6)
        
        # Compute Delta (velocity of spectral change) and Delta-Delta (acceleration)
        delta = self.compute_deltas(log_mel_spec)        # (1, n_mels, frames)
        delta_delta = self.compute_deltas(delta)          # (1, n_mels, frames)
        
        # Stack into 3 channels: (3, n_mels, frames)
        multi_channel = torch.cat([log_mel_spec, delta, delta_delta], dim=0)
        
        # Padding/Cropping to target_frames
        if target_frames:
            curr_frames = multi_channel.shape[-1]
            if curr_frames < target_frames:
                pad_amount = target_frames - curr_frames
                multi_channel = F.pad(multi_channel, (0, pad_amount))
            elif curr_frames > target_frames:
                multi_channel = multi_channel[:, :, :target_frames]
                
        return multi_channel

if __name__ == "__main__":
    gen = SpectrogramGenerator()
    path = "/Users/rohith/Desktop/Rohith/CMU/machine-learning/seetharam/Doppler-Based-Single-Channel-Sound-Source-Separation-in-Traffic-Audio-main/dataset/new_test_1/0001/mixed_audio.wav"
    spec = gen.generate(path, target_frames=512)
    print(f"Spectrogram shape: {spec.shape}")  # Expect (3, 128, 512)
