import numpy as np
import random
from collections import deque
from .utils import (
    average_frequency_bins,
    get_volume_from_ft,
    get_perceived_loudness,
    convert_to_mel_bins,
    get_perceptual_frequency_ranges,
    convert_to_decibels
)
from typing import List, Tuple

class LoraSoundController:
    """
    Sound-reactive controller that synchronizes LoRA pipe selection 
    with audio frequency analysis using mel-frequency scaling by default.
    """
    
    def __init__(self, 
                 num_pipes=2,  # Number of available pipes in the pipeline
                 num_prompts=2,  # Number of available prompts in the pipeline
                 enabled=True,
                 debug=False,
                 freq_start_idx=0,  # Start index for frequency range
                 freq_end_idx=50,  # End index for frequency range (None means use all)
                 averaging_factor=10,  # Number of bins to average together (used for linear frequency backup)
                 custom_bin_ranges=None,  # List of tuples defining custom ranges: [(start, end), ...]
                 rolling_window_size=5,  # Size of rolling window for percentage change calculation
                 mel_mode=True,  # Use mel-frequency scaling for perceptual frequency analysis (default)
                 sample_rate=44100,  # Sample rate for mel conversion
                 max_freq=10000,  # Maximum frequency for mel conversion
                 min_freq=60,  # Minimum frequency for mel conversion (start at bass range)
                 use_decibel_scale=True,  # Convert mel bins to decibel scale for perceptual analysis
                 frequency_bin_boost_factors=None,  # Array of boost factors for mel bins [bass, low_mids, mids, high_mids, treble]
                 smoothed_mode=True):  # Enable smoothed pipe index transitions (max ±1 change per frame)
        """
        Initialize the sound-reactive LoRA controller.
        
        Args:
            num_pipes (int): Number of available pipes in the pipeline
            num_prompts (int): Number of available prompts in the pipeline
            enabled (bool): Whether the controller is active
            debug (bool): Enable debug printing
            freq_start_idx (int): Start index for frequency range
            freq_end_idx (int): End index for frequency range (None means use all)
            averaging_factor (int): Number of bins to average together (used for linear frequency backup)
            custom_bin_ranges (list): List of tuples defining custom ranges: [(start, end), ...]
                                    e.g. [(0, 10), (11, 15), (20, 25)] creates 3 output bins
            rolling_window_size (int): Size of rolling window for percentage change calculation
            mel_mode (bool): Use mel-frequency scaling for perceptual frequency analysis (default: True)
            sample_rate (int): Sample rate for mel conversion
            max_freq (int): Maximum frequency for mel conversion
            min_freq (int): Minimum frequency for mel conversion
            use_decibel_scale (bool): Convert mel bins to decibel scale for perceptual analysis
            frequency_bin_boost_factors (list): Array of boost factors for mel bins [bass, low_mids, mids, high_mids, treble]
            smoothed_mode (bool): Enable smoothed pipe index transitions (max ±1 change per frame)
        """
        # Store configuration parameters
        self.num_pipes = num_pipes
        self.num_prompts = num_prompts
        self.enabled = enabled
        self.debug = debug
        self.freq_start_idx = freq_start_idx
        self.freq_end_idx = freq_end_idx
        self.averaging_factor = averaging_factor
        self.custom_bin_ranges = custom_bin_ranges or [(0, 5), (6, 11), (12, 17), (18, 23), (24, 29)]  # 5 equally spaced bins from 0-30
        self.rolling_window_size = rolling_window_size
        self.mel_mode = mel_mode
        self.sample_rate = sample_rate
        self.max_freq = max_freq
        self.min_freq = min_freq
        self.use_decibel_scale = use_decibel_scale
        self.frequency_bin_boost_factors = frequency_bin_boost_factors or [1.0, 1.0, 1.5, 2.0, 2.5]

        # self.treble_boost_factors = treble_boost_factors or [1.2, 1.1, 1.0, 1.0, 1.0]
        self.smoothed_mode = smoothed_mode
        
        
        # Current pipe index value
        self.current_pipe_index = 0
        self.current_prompt_index = 0

        # Per-mel-bin running average for normalization (compensates for spectral rolloff)
        # Each bin is compared to its own history, not across bins.
        self._mel_running_avg = None  # initialized on first call
        self._mel_avg_alpha = 0.05   # EMA smoothing factor (lower = slower adaptation)

        # Store previous averaged bins for rolling window percentage change calculation (used for linear frequency backup)
        self.rolling_window = deque(maxlen=self.rolling_window_size)

    def enable_debug(self, enabled: bool = True) -> None:
        """Enable or disable debug output"""
        self.debug = enabled
        if self.debug:
            print(f"[LoraSoundController] Debug mode enabled")
            print(f"[LoraSoundController] Current frequency bin boost factors: {self.frequency_bin_boost_factors}")

    def update_frequency_bin_boost_factors(self, bass_boost: float, low_mids_boost: float, mids_boost: float, high_mids_boost: float, treble_boost: float) -> None:
        """Update the frequency bin boost factors"""
        self.frequency_bin_boost_factors = [bass_boost, low_mids_boost, mids_boost, high_mids_boost, treble_boost]
        if self.debug:
            print(f"[LoraSoundController] Updated frequency bin boost factors: {self.frequency_bin_boost_factors}")

    def _apply_smoothing(self, current_index, target_index, max_indices=None):
        """
        Apply smoothing to pipe index transitions, limiting changes to ±1 per frame.
        
        Args:
            target_pipe_index (int): The desired pipe index from frequency analysis
            
        Returns:
            int: The smoothed pipe index (limited to ±1 change from current)
        """
        current = current_index
        target = target_index
        
        # Calculate the difference
        diff = target - current
        
        # Limit the change to ±1
        if diff > 1:
            smoothed_index = current + 1
        elif diff < -1:
            smoothed_index = current - 1
        else:
            smoothed_index = target
        
        # Ensure the index stays within valid bounds
        if max_indices is not None:
            max_indices = max_indices
        else:
            max_indices = self.num_pipes

        if self.debug:
            print(f"[LoraSoundController] Smoothing: using max indices - {max_indices}")

        smoothed_index = max(0, min(max_indices - 1, smoothed_index))
        
        if self.debug and smoothed_index != target:
            print(f"[LoraSoundController] Smoothing applied: target={target}, current={current}, smoothed={smoothed_index}")
        
        return smoothed_index

    def process_frequency_bins(self, normalized_energies: List[float], debug=True, num_pipes_override=None) -> Tuple[int, int]:
        """
        Process frequency bins to adjust pipe index based on audio input.
        Uses mel-frequency analysis by default with linear frequency analysis as backup.
        
        Args:
            normalized_energies: List containing the normalized energies
            debug (bool): Enable debug output for this call
            num_pipes_override (int): Override the default num_pipes value for this call
            
        Returns:
            int: Selected pipe index (0 to num_pipes-1)
        """
        # Validate input
        if not self.enabled:
            if self.debug:
                print("Controller is disabled, returning current pipe index.")
            return self.current_pipe_index, self.current_prompt_index

        if normalized_energies is None or len(normalized_energies) == 0:
            if self.debug:
                print(f"Invalid frequency bins, returning current pipe index: {self.current_pipe_index}")
            return self.current_pipe_index, self.current_prompt_index
    

        # Use frequency analysis method (mel-frequency by default, linear as backup)
        if self.mel_mode:
            # Convert linear FFT bins to mel-spaced perceptual bins
            mel_bins, mel_centers, bin_mapping = convert_to_mel_bins(
                normalized_energies, 
                sample_rate=self.sample_rate,
                n_fft_bins=len(normalized_energies),
                max_freq=self.max_freq,
                min_freq=self.min_freq,
                n_mel_bins=self.num_pipes,  # Match mel bins to number of pipes
                debug=self.debug or debug
            )
            
            # Store original mel bins for debug output
            original_mel_bins = mel_bins.copy()

            # --- Per-bin running-average normalization ---
            # Each mel bin is divided by its own exponential moving average.
            # This compensates for natural spectral rolloff (bass >> treble in
            # absolute energy) by making selection about *relative spike* not
            # *absolute magnitude*. This matches what the standard frontend's
            # Stream_Analyzer does with bin_mean_values normalization.
            if self._mel_running_avg is None:
                # Seed the running average with the first observation
                self._mel_running_avg = mel_bins.copy()
                # For the very first frame, all normalized bins = 1.0 (no preference)
                normalized_mel = np.ones_like(mel_bins)
            else:
                # Update running average (EMA)
                self._mel_running_avg = (
                    self._mel_avg_alpha * mel_bins
                    + (1 - self._mel_avg_alpha) * self._mel_running_avg
                )
                # Normalize: current energy / running average
                # Bins above their baseline score > 1, below score < 1
                safe_avg = np.maximum(self._mel_running_avg, 1e-10)
                normalized_mel = mel_bins / safe_avg

            # Apply boost factors in linear space to the normalized values
            # Extend boost factors to match number of mel bins (pad with 1.0)
            boost_factors = list(self.frequency_bin_boost_factors)
            if len(boost_factors) < len(normalized_mel):
                boost_factors.extend([1.0] * (len(normalized_mel) - len(boost_factors)))
            boost_arr = np.array(boost_factors[:len(normalized_mel)])
            boosted_bins = normalized_mel * boost_arr

            # Select bin with highest relative energy (after boost)
            raw_audio_index = np.argmax(boosted_bins)

            if self.debug or debug:
                print(f"[LoraSoundController] Mel-frequency analysis mode (per-bin normalized)")
                print(f"[LoraSoundController] Pipe index mode: {'smoothed' if self.smoothed_mode else 'instant'}")
                print(f"[LoraSoundController] Mel bin centers (Hz): {[f'{freq:.0f}' for freq in mel_centers]}")
                print(f"[LoraSoundController] Raw mel energies: {original_mel_bins}")
                print(f"[LoraSoundController] Running avg:      {self._mel_running_avg}")
                print(f"[LoraSoundController] Normalized (cur/avg): {normalized_mel}")
                print(f"[LoraSoundController] Boost factors: {boost_arr}")
                print(f"[LoraSoundController] Boosted normalized: {boosted_bins}")
                print(f"[LoraSoundController] Raw selected audio index: {raw_audio_index}")

                range_names = ['bass', 'low_mids', 'mids', 'high_mids', 'treble']
                for i, (name, value) in enumerate(zip(range_names[:len(boosted_bins)], boosted_bins)):
                    status = "🔥" if i == raw_audio_index else "  "
                    print(f"  {status} Bin {i} ({name}): {value:.3f} (raw={original_mel_bins[i]:.4f}, avg={self._mel_running_avg[i]:.4f}, norm={normalized_mel[i]:.3f})")
                
        else:
            # Use linear frequency analysis method as backup
            averaged_energies, percentage_changes = average_frequency_bins(
                normalized_energies, 
                self.custom_bin_ranges, 
                self.rolling_window,
                debug=self.debug or debug
            )
            raw_audio_index = np.argmax(averaged_energies)

            # Store current values for next iteration
            self.rolling_window.append(averaged_energies.copy())

            if self.debug or debug:
                print(f"[LoraSoundController] Linear frequency analysis mode (backup)")
                print(f"[LoraSoundController] Pipe index mode: {'smoothed' if self.smoothed_mode else 'instant'}")
                print(f"[LoraSoundController] Averaged energies: {averaged_energies}, length: {len(averaged_energies)}")
                print(f"[LoraSoundController] Raw selected audio index: {raw_audio_index}")
            
        # Store the new pipe index for next iteration
        if self.smoothed_mode:
            # Apply smoothing: limit changes to ±1 per frame
            current_pipe_index = self._apply_smoothing(self.current_pipe_index, raw_audio_index, max_indices=self.num_pipes)
            current_prompt_index = self._apply_smoothing(self.current_prompt_index, raw_audio_index, max_indices=self.num_prompts)
        else:
            # Instant mode: use the raw selected index directly
            current_pipe_index = raw_audio_index
            current_prompt_index = raw_audio_index
        
        self.current_pipe_index = current_pipe_index
        self.current_prompt_index = current_prompt_index

        # Show final pipe index in debug output
        if self.debug or debug:
            print(f"[LoraSoundController] Using pipe index: {current_pipe_index}")
            print(f"[LoraSoundController] Raw audio index: {raw_audio_index}")

        return self.current_pipe_index, self.current_prompt_index
    
    def get_pipe_index(self):
        """
        Get the current pipe index value
        
        Returns:
            int: Current pipe index value
        """
        return self.current_pipe_index 
    
    def is_smoothed_mode(self):
        """
        Check if smoothed mode is enabled.
        
        Returns:
            bool: True if smoothed mode is enabled, False for instant mode
        """
        return self.smoothed_mode
    
    def get_current_volume(self):
        """
        Get the current volume level (RMS)
        
        Returns:
            float: Current volume level (0.0 to 1.0)
        """
        return getattr(self, 'current_volume', 0.0)
    
    def get_current_loudness(self):
        """
        Get the current perceived loudness level
        
        Returns:
            float: Current perceived loudness (0.0 to 1.0)
        """
        return getattr(self, 'current_perceived_loudness', 0.0) 