import numpy as np
from collections import deque

class FrequencyZoomController:
    """
    Processes frequency bins to adjust zoom factor based on changes in low and high frequency bands.
    With a balancing mechanism to return to neutral zoom (1.0) when there is no significant activity.
    """
    
    def __init__(self, 
                 baseline_window_size=30, 
                 low_bin_sensitivity=0.1, 
                 high_bin_sensitivity=0.1,
                 min_zoom=0.8,
                 max_zoom=1.5,
                 rebalance_rate=0.005,
                 activity_threshold=0.9,
                 amplifying_factor=1000,
                 enabled=True,
                 debug=False):
        """
        Initialize the frequency zoom controller.
        
        Args:
            baseline_window_size (int): Number of samples to keep for rolling baseline calculation
            low_bin_sensitivity (float): How much low frequencies affect zoom out (0.0-1.0)
            high_bin_sensitivity (float): How much high frequencies affect zoom in (0.0-1.0)
            min_zoom (float): Minimum allowed zoom factor
            max_zoom (float): Maximum allowed zoom factor
            rebalance_rate (float): Rate at which zoom returns to neutral when no activity
            activity_threshold (float): Threshold above which audio activity is considered significant
            enabled (bool): Whether the controller is active
        """
        # Store configuration parameters
        self.baseline_window_size = baseline_window_size
        self.low_bin_sensitivity = low_bin_sensitivity
        self.high_bin_sensitivity = high_bin_sensitivity
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom
        self.rebalance_rate = rebalance_rate
        self.activity_threshold = activity_threshold
        self.enabled = enabled

        self.amplifying_factor = amplifying_factor
        
        # Initialize baselines for frequency bands
        self.low_bin_baseline = deque(maxlen=baseline_window_size)
        self.high_bin_baseline = deque(maxlen=baseline_window_size)
        
        # Current zoom factor value
        self.zoom_factor_value = 1.0
        
        # Debug info
        self.debug = debug

    def enable_debug(self, enabled=True):
        """Enable or disable debug printing"""
        self.debug = enabled
        
    def process_frequency_bins(self, binned_fft):
        """
        Process frequency bins to adjust zoom factor based on changes in low and high frequency bands.
        
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

        # Fix: replace direct boolean evaluation of binned_fft with proper checks
        if binned_fft is None or (isinstance(binned_fft, (list, tuple, np.ndarray)) and len(binned_fft) < 3):
            if self.debug:
                print(f"Invalid frequency bins, returning current zoom: {self.zoom_factor_value}")
            return self.zoom_factor_value
            
        # Extract low and high frequency bands
        low_bin = binned_fft[0]
        high_bin = binned_fft[2]
        
        if self.debug:
            print(f"Processing frequency bins - Low: {low_bin}, High: {high_bin}")
        
        # Update the rolling baselines
        self.low_bin_baseline.append(low_bin)
        self.high_bin_baseline.append(high_bin)
        
        # Calculate baseline averages (if we have enough data)
        if len(self.low_bin_baseline) > 5 and len(self.high_bin_baseline) > 5:
            low_baseline_avg = sum(self.low_bin_baseline) / len(self.low_bin_baseline)
            high_baseline_avg = sum(self.high_bin_baseline) / len(self.high_bin_baseline)
            
            if self.debug:
                print(f"Baseline averages - Low: {low_baseline_avg}, High: {high_baseline_avg}")
            
            # Calculate delta from baseline as percentage changes
            low_delta_pct = max(0, (low_bin - low_baseline_avg) / low_baseline_avg if low_baseline_avg > 0 else 0)
            high_delta_pct = max(0, (high_bin - high_baseline_avg) / high_baseline_avg if high_baseline_avg > 0 else 0)
            
            if self.debug:
                print(f"Delta percentages - Low: {low_delta_pct}, High: {high_delta_pct}")
            
            # Cap percentage changes to avoid extreme reactions
            low_delta_pct = min(low_delta_pct, 1.0)   # Cap at 100% increase
            high_delta_pct = min(high_delta_pct, 1.0)  # Cap at 100% increase
            
            # Scale percentage changes to small increments appropriate for zoom
            zoom_out_factor = low_delta_pct * self.low_bin_sensitivity
            zoom_in_factor = high_delta_pct * self.high_bin_sensitivity
            
            if self.debug:
                print(f"Zoom factors - Out: {zoom_out_factor}, In: {zoom_in_factor}")
            
            # Calculate net zoom adjustment
            zoom_adjustment = zoom_in_factor - zoom_out_factor

            print(f"zoom_adjustment: {zoom_adjustment}")
            
            # Calculate activity level to determine if we should return to neutral
            max_activity = max(low_delta_pct, high_delta_pct)

            print(f"low_delta_pct: {low_delta_pct}")
            print(f"high_delta_pct: {high_delta_pct}")
            print(f"max_activity: {max_activity}")
            
            # If there's significant activity, apply the calculated adjustment
            # Otherwise gradually return to neutral (1.0)
            if max_activity > self.activity_threshold:
                # Apply normal adjustment based on frequency analysis
                new_zoom = self.zoom_factor_value + zoom_adjustment
            else:
                # Return to neutral (1.0) gradually when no significant activity
                if self.zoom_factor_value > 1.0:
                    new_zoom = self.zoom_factor_value - self.rebalance_rate
                elif self.zoom_factor_value < 1.0:
                    new_zoom = self.zoom_factor_value + self.rebalance_rate
                else:
                    new_zoom = 1.0
                    
                if self.debug:
                    print(f"No significant audio activity. Rebalancing zoom: {self.zoom_factor_value:.2f} -> {new_zoom:.2f}")
            
            # Keep within reasonable bounds
            new_zoom = max(self.min_zoom, min(self.max_zoom, new_zoom))
            
            if self.debug:
                print(f"Frequency bins - Low: {low_bin:.2f} (Δ%: {low_delta_pct:.2f}), High: {high_bin:.2f} (Δ%: {high_delta_pct:.2f})")
                print(f"Zoom adjustment: {zoom_adjustment:.4f}, New zoom: {new_zoom:.2f}")
            
            # self.zoom_factor_value = new_zoom
            return new_zoom
            
        return self.zoom_factor_value  # Return current value if not enough baseline data
        
    def get_zoom_factor(self):
        """
        Get the current zoom factor value
        
        Returns:
            float: Current zoom factor value
        """
        return self.zoom_factor_value
        
    def set_sensitivity(self, low_sensitivity=None, high_sensitivity=None):
        """
        Update the sensitivity parameters
        
        Args:
            low_sensitivity (float, optional): New low frequency sensitivity
            high_sensitivity (float, optional): New high frequency sensitivity
        """
        if low_sensitivity is not None:
            self.low_bin_sensitivity = low_sensitivity
        if high_sensitivity is not None:  
            self.high_bin_sensitivity = high_sensitivity