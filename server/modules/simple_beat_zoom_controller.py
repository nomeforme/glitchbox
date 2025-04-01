import numpy as np
from collections import deque

class BeatZoomController:
    """
    Simple beat-reactive zoom controller that synchronizes zoom level 
    with the energy in the low frequency band (bass).
    """
    
    def __init__(self, 
                 baseline_window_size=30,
                 baseline_avg_pct=0.3,
                 min_zoom=0.8,
                 max_zoom=1.5,
                 smoothing_factor=0.3,
                 amplifying_factor=1000,
                 energy_amplifier=1.00,
                 use_baseline=False,
                 max_bin_decay_rate=0.995,
                 enabled=True,
                 debug=False):
        """
        Initialize the beat-reactive zoom controller.
        
        Args:
            baseline_window_size (int): Number of samples to keep for baseline calculation
            min_zoom (float): Minimum allowed zoom factor
            max_zoom (float): Maximum allowed zoom factor
            smoothing_factor (float): Controls the smoothness of transitions (0.0-1.0)
            amplifying_factor (float): Factor to amplify the raw frequency bin values
            energy_amplifier (float): Multiplier for the energy value 
            enabled (bool): Whether the controller is active
            use_baseline (bool): Whether to use baseline for energy calculation
            max_bin_decay_rate (float): Rate at which max_low_bin decays (0.0-1.0)
            debug (bool): Enable debug printing
        """
        # Store configuration parameters
        self.baseline_window_size = baseline_window_size
        self.baseline_avg_pct = baseline_avg_pct
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom
        self.smoothing_factor = smoothing_factor
        self.amplifying_factor = amplifying_factor
        self.energy_amplifier = energy_amplifier
        self.enabled = enabled
        self.use_baseline = use_baseline
        self.max_bin_decay_rate = max_bin_decay_rate
        self.debug = debug
        
        # Initialize baseline for low frequency band
        self.low_bin_baseline = deque(maxlen=baseline_window_size)
        
        # Current zoom factor value
        self.zoom_factor_value = 1.0
        
        # Store the highest low bin value seen for normalization
        self.max_low_bin = 0.1 * self.amplifying_factor # Starting with a small value to avoid division by zero

    def enable_debug(self, enabled=True):
        """Enable or disable debug printing"""
        self.debug = enabled
        
    def process_frequency_bins(self, binned_fft):
        """
        Process frequency bins to adjust zoom factor based on low frequency energy.
        
        Args:
            binned_fft: List containing the frequency bins (low, mid, high)
            
        Returns:
            float: Adjusted zoom factor value
        """
        # Validate input
        if not self.enabled:
            if self.debug:
                print("Controller is disabled, returning current zoom.")
            return self.zoom_factor_value

        if binned_fft is None or (isinstance(binned_fft, (list, tuple, np.ndarray)) and len(binned_fft) < 3):
            if self.debug:
                print(f"Invalid frequency bins, returning current zoom: {self.zoom_factor_value}")
            return self.zoom_factor_value
                    
        # Extract low frequency band from binned FFT multiplied by amplifying factor
        low_bin = max(binned_fft) * self.amplifying_factor

        self.low_bin_baseline.append(low_bin)

        if len(self.low_bin_baseline) > int(self.baseline_avg_pct * self.baseline_window_size):
            low_baseline_avg = sum(self.low_bin_baseline) / len(self.low_bin_baseline)
            pct_change = self.energy_amplifier * max(0, min(1, (low_bin - low_baseline_avg) / low_baseline_avg))
            target_zoom = 1 + pct_change

            print(f"Low bin: {low_bin:.2f}, Baseline: {low_baseline_avg:.2f}, Pct change: {pct_change:.2f}, Target zoom: {target_zoom:.2f}")
        else:
            target_zoom = self.min_zoom

        # Calculate smooth transition (approximately 3 iterations to reach target)
        transition_rate = 0.33  # 1/3 for ~3 frames to reach target
        new_zoom = self.zoom_factor_value + (target_zoom - self.zoom_factor_value) * transition_rate

        # Ensure zoom is within min/max bounds
        new_zoom = max(self.min_zoom, min(self.max_zoom, new_zoom))
        
        # Store the new zoom value for next iteration
        self.zoom_factor_value = new_zoom

        print(f"Current zoom: {self.zoom_factor_value:.2f}, Target zoom: {target_zoom:.2f}")

        return self.zoom_factor_value
    
    def get_zoom_factor(self):
        """
        Get the current zoom factor value
        
        Returns:
            float: Current zoom factor value
        """
        return self.zoom_factor_value
        
    def set_min_max_zoom(self, min_zoom=None, max_zoom=None):
        """
        Update the min and max zoom parameters
        
        Args:
            min_zoom (float, optional): New minimum zoom
            max_zoom (float, optional): New maximum zoom
        """
        if min_zoom is not None:
            self.min_zoom = min_zoom
        if max_zoom is not None:  
            self.max_zoom = max_zoom
            
    def set_smoothing_factor(self, smoothing_factor):
        """
        Update the smoothing factor
        
        Args:
            smoothing_factor (float): New smoothing factor (0.0-1.0)
        """
        self.smoothing_factor = max(0.0, min(1.0, smoothing_factor))
        
    def set_use_baseline(self, use_baseline):
        """
        Enable or disable the use of baseline for energy normalization
        
        Args:
            use_baseline (bool): Whether to use baseline subtraction in energy calculation
        """
        self.use_baseline = use_baseline
        if self.debug:
            print(f"Baseline usage set to: {use_baseline}")
            
    def set_max_bin_decay_rate(self, decay_rate):
        """
        Set the rate at which max_low_bin decays towards the baseline
        
        Args:
            decay_rate (float): Decay rate between 0.0 and 1.0
                               (higher values = slower decay)
        """
        self.max_bin_decay_rate = max(0.0, min(1.0, decay_rate))
        if self.debug:
            print(f"Max bin decay rate set to: {self.max_bin_decay_rate}")
