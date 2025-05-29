import numpy as np
import random
from collections import deque
from .utils import (
    update_max_energy,
    calculate_energy_percentage,
    get_pipe_index_from_percentage,
    average_frequency_bins,
    get_volume_from_ft,
    get_perceived_loudness,
    get_volume_quintile,
    get_adaptive_volume_quintile,
    convert_to_mel_bins,
    get_perceptual_frequency_ranges,
    convert_to_decibels
)

class LoraSoundController:
    """
    Sound-reactive controller that synchronizes LoRA pipe selection 
    with audio frequency analysis using dynamic peak tracking.
    """
    
    def __init__(self, 
                 num_pipes=2,  # Number of available pipes in the pipeline
                 enabled=True,
                 debug=False,
                 freq_start_idx=0,  # Start index for frequency range
                 freq_end_idx=50,  # End index for frequency range (None means use all)
                 energy_bin_index=48,  # Index of the energy bin to use
                 averaging_factor=10,  # Number of bins to average together (deprecated if using custom_bin_ranges)
                 custom_bin_ranges=None,  # List of tuples defining custom ranges: [(start, end), ...]
                 decay_factor=1.00,  # How quickly the max decays (closer to 1 = slower decay)
                 min_max_value=0.01,  # Lower bound for the max value
                 reset_interval=5,  # Reset max energy to min every n iterations
                 rolling_window_size=5,  # Size of rolling window for percentage change calculation
                 mel_mode=True,  # Use mel-frequency scaling for perceptual frequency analysis
                 sample_rate=44100,  # Sample rate for mel conversion
                 max_freq=10000,  # Maximum frequency for mel conversion
                 min_freq=201,  # Minimum frequency for mel conversion
                 use_decibel_scale=True,  # Convert mel bins to decibel scale for perceptual analysis
                 treble_boost_factors=None):  # Array of boost factors for mel bins [bass, low_mids, mids, high_mids, treble]
        """
        Initialize the sound-reactive LoRA controller.
        
        Args:
            num_pipes (int): Number of available pipes in the pipeline
            enabled (bool): Whether the controller is active
            debug (bool): Enable debug printing
            freq_start_idx (int): Start index for frequency range
            freq_end_idx (int): End index for frequency range (None means use all)
            averaging_factor (int): Number of bins to average together (deprecated if using custom_bin_ranges)
            custom_bin_ranges (list): List of tuples defining custom ranges: [(start, end), ...]
                                    e.g. [(0, 10), (11, 15), (20, 25)] creates 3 output bins
            decay_factor (float): How quickly the max energy decays over time (0.995-0.999)
            min_max_value (float): Lower bound for the maximum energy value
            reset_interval (int): Reset max energy to min every n iterations
            rolling_window_size (int): Size of rolling window for percentage change calculation
            mel_mode (bool): Use mel-frequency scaling for perceptual frequency analysis
            sample_rate (int): Sample rate for mel conversion
            max_freq (int): Maximum frequency for mel conversion
            min_freq (int): Minimum frequency for mel conversion
            use_decibel_scale (bool): Convert mel bins to decibel scale for perceptual analysis
            treble_boost_factors (list): Array of boost factors for mel bins [bass, low_mids, mids, high_mids, treble]
        """
        # Store configuration parameters
        self.num_pipes = num_pipes
        self.enabled = enabled
        self.debug = debug
        self.freq_start_idx = freq_start_idx
        self.freq_end_idx = freq_end_idx
        self.averaging_factor = averaging_factor
        self.custom_bin_ranges = custom_bin_ranges or [(0, 5), (6, 11), (12, 17), (18, 23), (24, 29)]  # 5 equally spaced bins from 0-30
        self.decay_factor = decay_factor
        self.min_max_value = min_max_value
        self.reset_interval = reset_interval
        self.energy_bin_index = energy_bin_index
        self.rolling_window_size = rolling_window_size
        self.mel_mode = mel_mode
        self.sample_rate = sample_rate
        self.max_freq = max_freq
        self.min_freq = min_freq
        self.use_decibel_scale = use_decibel_scale
        self.treble_boost_factors = treble_boost_factors or [1.0, 1.0, 1.5, 2.0, 2.5]
        
        # Current pipe index value
        self.current_pipe_index = 0

        # Dynamic peak tracking
        self.max_energy = min_max_value  # Start with minimum value
        self.iteration_counter = 0  # Counter for periodic resets
        
        # Store previous averaged bins for rolling window percentage change calculation
        self.rolling_window = deque(maxlen=self.rolling_window_size)
        
        # Percentage-based thresholds for pipe selection (0-4 pipes)
        self.percentage_thresholds = [0.0, 20.0, 40.0, 60.0, 80.0]

    def enable_debug(self, enabled=True):
        """Enable or disable debug printing"""
        self.debug = enabled
        
    def process_frequency_bins(self,
                               normalized_energies,
                               volume_mode=False,
                               adaptive=False,
                               use_perceived_loudness=False,
                               debug=True):
        """
        Process frequency bins to adjust pipe index based on audio input.
        
        Args:
            normalized_energies: List containing the normalized energies
            volume_mode (bool): Use volume-based quintile selection instead of frequency analysis
            adaptive (bool): Use adaptive volume thresholds based on recent history
            use_perceived_loudness (bool): Use perceived loudness for volume calculation
            
        Returns:
            int: Selected pipe index (0-4)
        """
        # Validate input
        if not self.enabled:
            if self.debug:
                print("Controller is disabled, returning current pipe index.")
            return self.current_pipe_index

        if normalized_energies is None or (isinstance(normalized_energies, (list, tuple, np.ndarray)) and len(normalized_energies) < 26):
            if self.debug:
                print(f"Invalid frequency bins, returning current pipe index: {self.current_pipe_index}")
            return self.current_pipe_index
        
        # Calculate current volume from frequency data
        self.current_volume = get_volume_from_ft(normalized_energies, method='rms', debug=self.debug)
        self.current_perceived_loudness = get_perceived_loudness(normalized_energies, debug=self.debug)

        if volume_mode:
            # Use volume-based quintile selection
            if adaptive:
                new_pipe_index, self.max_energy, self.iteration_counter = get_adaptive_volume_quintile(
                    normalized_energies, 
                    self.max_energy,
                    self.decay_factor,
                    self.min_max_value,
                    self.iteration_counter,
                    self.reset_interval,
                    method='rms', 
                    use_perceived_loudness=use_perceived_loudness,
                    debug=self.debug
                )
            else:
                new_pipe_index = get_volume_quintile(
                    normalized_energies, 
                    method='rms', 
                    use_perceived_loudness=use_perceived_loudness,
                    debug=self.debug
                )
                
            if self.debug:
                print(f"[LoraSoundController] Volume mode: {'adaptive' if adaptive else 'fixed'}")
                print(f"[LoraSoundController] Using {'perceived loudness' if use_perceived_loudness else 'RMS volume'}")
                print(f"[LoraSoundController] Current volume (RMS): {self.current_volume:.4f}")
                print(f"[LoraSoundController] Perceived loudness: {self.current_perceived_loudness:.4f}")
                print(f"[LoraSoundController] Selected pipe index: {new_pipe_index}")
        else:
            # Use frequency analysis method (with optional mel-frequency conversion)
            if self.mel_mode:
                # Convert linear FFT bins to mel-spaced perceptual bins
                mel_bins, mel_centers, bin_mapping = convert_to_mel_bins(
                    normalized_energies, 
                    sample_rate=self.sample_rate,
                    n_fft_bins=len(normalized_energies),
                    max_freq=self.max_freq,
                    min_freq=self.min_freq,
                    n_mel_bins=5,  # Always use 5 mel bins for consistency
                    debug=self.debug or debug
                )
                
                # Store original mel bins for debug output
                original_mel_bins = mel_bins.copy()
                
                # Convert to decibel scale if enabled (standard practice for perceptual audio analysis)
                if self.use_decibel_scale:
                    mel_bins_db = convert_to_decibels(
                        mel_bins,
                        reference_energy=1e-10,
                        min_db=-80.0,
                        debug=self.debug or debug
                    )
                    # Use decibel values for further processing
                    processing_bins = mel_bins_db
                else:
                    # Use linear energy values
                    processing_bins = mel_bins
                
                # Apply treble boost to higher frequency bins
                if self.use_decibel_scale:
                    # In dB space: add boost in dB (10*log10(boost_factor))
                    boost_factors_db = 10 * np.log10(np.array(self.treble_boost_factors[:len(processing_bins)]))
                    boosted_bins = processing_bins + boost_factors_db
                else:
                    # In linear energy space: multiply by boost factor
                    boosted_bins = processing_bins * self.treble_boost_factors[:len(processing_bins)]
                
                # Select bin with highest value (correctly boosted in appropriate space)
                new_pipe_index = np.argmax(boosted_bins)
                
                if self.debug or debug:
                    print(f"[LoraSoundController] Mel-frequency analysis mode")
                    print(f"[LoraSoundController] Using {'decibel' if self.use_decibel_scale else 'linear'} scale for bin selection")
                    print(f"[LoraSoundController] Current volume (RMS): {self.current_volume:.4f}")
                    print(f"[LoraSoundController] Perceived loudness: {self.current_perceived_loudness:.4f}")
                    print(f"[LoraSoundController] Mel bin centers (Hz): {[f'{freq:.0f}' for freq in mel_centers]}")
                    print(f"[LoraSoundController] Raw mel energies: {original_mel_bins}")
                    
                    if self.use_decibel_scale:
                        print(f"[LoraSoundController] Mel energies (dB): {mel_bins_db}")
                        print(f"[LoraSoundController] Treble boost factors (dB): {boost_factors_db}")
                        print(f"[LoraSoundController] Boosted mel bins (dB): {boosted_bins}")
                    else:
                        print(f"[LoraSoundController] Treble boost factors: {self.treble_boost_factors[:len(processing_bins)]}")
                        print(f"[LoraSoundController] Boosted mel energies: {boosted_bins}")
                    
                    print(f"[LoraSoundController] Selected pipe index: {new_pipe_index}")
                    
                    # Show perceptual frequency ranges
                    freq_ranges = get_perceptual_frequency_ranges()
                    range_names = ['bass', 'low_mids', 'mids', 'high_mids', 'treble']
                    for i, (name, value) in enumerate(zip(range_names[:len(boosted_bins)], boosted_bins)):
                        status = "🔥" if i == new_pipe_index else "  "
                        unit = "dB" if self.use_decibel_scale else ""
                        print(f"  {status} Bin {i} ({name}): {value:.2f} {unit}")
                    
            else:
                # Use original linear frequency analysis method
                averaged_energies, percentage_changes = average_frequency_bins(
                    normalized_energies, 
                    self.custom_bin_ranges, 
                    self.rolling_window,
                    debug=self.debug or debug
                )
                new_pipe_index = np.argmax(averaged_energies)

                # Store current values for next iteration
                self.rolling_window.append(averaged_energies.copy())

                if self.debug or debug:
                    print(f"[LoraSoundController] Linear frequency analysis mode")
                    print(f"[LoraSoundController] Current volume (RMS): {self.current_volume:.4f}")
                    print(f"[LoraSoundController] Perceived loudness: {self.current_perceived_loudness:.4f}")
                    print(f"[LoraSoundController] Averaged energies: {averaged_energies}, length: {len(averaged_energies)}")
                    print(f"[LoraSoundController] Selected pipe index: {new_pipe_index}")
            
        # Store the new pipe index for next iteration
        self.current_pipe_index = new_pipe_index
        
        return self.current_pipe_index
    
    def get_pipe_index(self):
        """
        Get the current pipe index value
        
        Returns:
            int: Current pipe index value
        """
        return self.current_pipe_index 
    
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