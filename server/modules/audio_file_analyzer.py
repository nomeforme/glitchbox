"""
File-based audio FFT analyzer for headless mode.

Replicates the exact same FFT processing as the frontend_py VideoAudioThread:
- PCM 16-bit audio loaded via ffmpeg
- np.fft.fft (no windowing)
- Linear equal-sized bins
- Raw magnitude means (no normalization)
"""

import subprocess
import numpy as np


class AudioFileAnalyzer:
    def __init__(self, audio_path: str, n_bins: int = 50, sample_rate: int = 44100,
                 fps: float = 10.0):
        self.n_bins = n_bins
        self.sample_rate = sample_rate

        # Load audio as PCM 16-bit (same as frontend)
        self.audio_samples = self._load_audio(audio_path, sample_rate)
        self.duration = len(self.audio_samples) / sample_rate
        self.total_samples = len(self.audio_samples)

        # Chunk size matches frontend: sample_rate / fps
        self.samples_per_chunk = int(sample_rate / fps)
        self.current_sample = 0

    def _load_audio(self, path: str, sample_rate: int) -> np.ndarray:
        """Extract mono PCM 16-bit audio (same as frontend_py VideoAudioThread)."""
        cmd = [
            'ffmpeg', '-i', path,
            '-vn',
            '-acodec', 'pcm_s16le',
            '-ar', str(sample_rate),
            '-ac', '1',
            '-f', 'wav',
            '-v', 'quiet',
            'pipe:1'
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()[:200]}")

        # Skip WAV header (44 bytes) and read as int16
        raw = result.stdout
        if len(raw) > 44:
            audio = np.frombuffer(raw[44:], dtype=np.int16)
        else:
            audio = np.frombuffer(raw, dtype=np.int16)

        if len(audio) == 0:
            raise RuntimeError(f"No audio data extracted from {path}")
        return audio

    def get_next_frame_energies(self) -> np.ndarray:
        """
        Get frequency bin energies for the next frame.
        Replicates frontend_py VideoAudioThread.run() exactly.
        """
        if self.current_sample >= self.total_samples:
            self.current_sample = 0

        # Extract chunk
        end_sample = min(self.current_sample + self.samples_per_chunk, self.total_samples)
        audio_chunk = self.audio_samples[self.current_sample:end_sample]

        # Pad if necessary
        if len(audio_chunk) < self.samples_per_chunk:
            audio_chunk = np.pad(audio_chunk, (0, self.samples_per_chunk - len(audio_chunk)), 'constant')

        # Convert to float and normalize (same as frontend)
        audio_chunk = audio_chunk.astype(np.float32) / 32768.0

        # FFT (no windowing, same as frontend)
        fft_result = np.fft.fft(audio_chunk)
        fft_magnitude = np.abs(fft_result[:len(fft_result) // 2])

        # Linear equal-sized bins (same as frontend)
        bin_size = len(fft_magnitude) // self.n_bins
        binned_fft = np.zeros(self.n_bins)

        for i in range(self.n_bins):
            start_idx = i * bin_size
            end_idx = start_idx + bin_size
            if i == self.n_bins - 1:
                end_idx = len(fft_magnitude)
            binned_fft[i] = np.mean(fft_magnitude[start_idx:end_idx])

        self.current_sample += self.samples_per_chunk

        # Return raw energies (same as frontend's normalized_energies = binned_fft)
        return binned_fft

    def get_progress(self) -> float:
        return self.current_sample / max(self.total_samples, 1)

    def get_current_time(self) -> float:
        return self.current_sample / self.sample_rate
