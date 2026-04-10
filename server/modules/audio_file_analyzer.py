"""
File-based audio FFT analyzer for headless mode.

Loads an audio file, performs FFT analysis frame-by-frame,
and returns normalized frequency bin energies matching the
format expected by BeatZoomController and LoraSoundController.
"""

import subprocess
import numpy as np
from scipy.fft import rfft
from scipy.signal import savgol_filter


class AudioFileAnalyzer:
    def __init__(self, audio_path: str, n_bins: int = 51, sample_rate: int = 44100,
                 window_ms: float = 50, fps: float = 10.0):
        """
        Load an audio file and prepare for frame-by-frame FFT analysis.

        Args:
            audio_path: Path to audio file (mp4, mp3, wav, etc.)
            n_bins: Number of frequency bins (default 51, matches Stream_Analyzer)
            sample_rate: Target sample rate
            window_ms: FFT window size in milliseconds
            fps: Expected frame rate of the pipeline (determines time advance per frame)
        """
        self.n_bins = n_bins
        self.sample_rate = sample_rate
        self.window_size = int(sample_rate * window_ms / 1000)
        self.hop_size = int(sample_rate / fps)  # Advance per frame
        self.current_pos = 0

        # Extract audio using ffmpeg
        self.audio_data = self._load_audio(audio_path, sample_rate)
        self.duration = len(self.audio_data) / sample_rate
        self.total_samples = len(self.audio_data)

        # Precompute log-spaced frequency bin edges
        freq_resolution = sample_rate / self.window_size
        max_freq = sample_rate / 2
        self.bin_edges = np.logspace(
            np.log10(max(20, freq_resolution)),
            np.log10(max_freq),
            n_bins + 1
        )

        # Hamming window
        self.window = np.hamming(self.window_size)

    def _load_audio(self, path: str, sample_rate: int) -> np.ndarray:
        """Extract mono audio from any media file using ffmpeg."""
        cmd = [
            'ffmpeg', '-i', path,
            '-f', 'f32le',     # 32-bit float PCM
            '-acodec', 'pcm_f32le',
            '-ac', '1',        # mono
            '-ar', str(sample_rate),
            '-v', 'quiet',
            'pipe:1'
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()[:200]}")

        audio = np.frombuffer(result.stdout, dtype=np.float32)
        if len(audio) == 0:
            raise RuntimeError(f"No audio data extracted from {path}")
        return audio

    def get_next_frame_energies(self) -> np.ndarray:
        """
        Get normalized frequency bin energies for the next frame.

        Returns:
            numpy array of shape (n_bins,) with values 0.0-1.0,
            or None if audio is exhausted.
        """
        if self.current_pos + self.window_size > self.total_samples:
            # Loop back to start
            self.current_pos = 0

        # Extract window
        chunk = self.audio_data[self.current_pos:self.current_pos + self.window_size]
        self.current_pos += self.hop_size

        # Apply window and FFT
        windowed = chunk * self.window
        fft_data = np.abs(rfft(windowed))

        # Map FFT bins to log-spaced frequency bins
        freqs = np.fft.rfftfreq(self.window_size, 1.0 / self.sample_rate)
        energies = np.zeros(self.n_bins)

        for i in range(self.n_bins):
            low = self.bin_edges[i]
            high = self.bin_edges[i + 1]
            mask = (freqs >= low) & (freqs < high)
            if mask.any():
                energies[i] = np.mean(fft_data[mask])

        # Normalize to 0-1
        max_energy = energies.max()
        if max_energy > 0:
            energies = energies / max_energy

        # Light smoothing
        if len(energies) > 5:
            try:
                energies = savgol_filter(energies, 5, 2)
                energies = np.clip(energies, 0, 1)
            except Exception:
                pass

        return energies

    def get_progress(self) -> float:
        """Return current playback progress 0.0-1.0."""
        return self.current_pos / max(self.total_samples, 1)

    def get_current_time(self) -> float:
        """Return current playback time in seconds."""
        return self.current_pos / self.sample_rate
