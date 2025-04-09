"""
Prompt Travel Scheduler for managed transitions between prompts.
This module provides scheduling functionality for the prompt travel feature,
automatically interpolating between source and target prompts.
"""

class PromptTravelScheduler:
    """
    A class that handles scheduled prompt travel transitions.
    This provides a deterministic schedule for moving between prompts.
    """
    def __init__(self, 
                min_factor=0.0,
                max_factor=1.0,
                factor_increment=0.025,
                stabilize_duration=3,
                oscillate=True,
                enabled=False,
                debug=False):
        """
        Initialize the prompt travel scheduler.
        
        Args:
            min_factor (float): Minimum prompt travel factor (default: 0.0)
            max_factor (float): Maximum prompt travel factor (default: 1.0)
            factor_increment (float): Amount to change factor per update (default: 0.025)
            stabilize_duration (int): Number of iterations to pause at min/max (default: 3)
            oscillate (bool): Whether to oscillate between min/max or one-way (default: True)
            enabled (bool): Whether the scheduler is active (default: False)
            debug (bool): Whether to print debug messages (default: False)
        """
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.factor_increment = factor_increment
        self.stabilize_duration = stabilize_duration
        self.oscillate = oscillate
        self.enabled = enabled
        self.debug = debug
        
        # Internal state
        self.factor_value = min_factor
        self.direction = 1  # 1 for increasing, -1 for decreasing
        self.stabilize_counter = 0  # Counter for stabilization pause
    
    def update(self):
        """
        Update the factor value for the next iteration.
        
        Returns:
            float: The current prompt travel factor value
        """
        if not self.enabled:
            return self.factor_value
            
        # Check if we're in stabilization pause at min or max
        at_boundary = (abs(self.factor_value - self.max_factor) < 0.001 or 
                       abs(self.factor_value - self.min_factor) < 0.001)
                       
        if at_boundary and self.stabilize_counter < self.stabilize_duration:
            # Hold at boundary for stabilize_duration iterations
            self.stabilize_counter += 1
            
            if self.debug and self.stabilize_counter == 1:
                boundary_type = "max" if abs(self.factor_value - self.max_factor) < 0.001 else "min"
                print(f"[PromptTravelScheduler] Stabilizing at {boundary_type} factor={self.factor_value:.2f} ({self.stabilize_counter}/{self.stabilize_duration})")
        else:
            if at_boundary and self.stabilize_counter >= self.stabilize_duration:
                # We've finished the stabilization period, continue with oscillation
                self.stabilize_counter = 0
                
                # If we're not oscillating and we've reached max, stop at max
                if not self.oscillate and abs(self.factor_value - self.max_factor) < 0.001:
                    return self.factor_value
                
            # Update factor for next iteration
            self.factor_value += self.direction * self.factor_increment
            
            # Ensure factor stays within bounds
            self.factor_value = max(self.min_factor, min(self.max_factor, self.factor_value))
            
            # Reverse direction if we hit limits
            if abs(self.factor_value - self.max_factor) < 0.001:
                self.direction = -1
            elif abs(self.factor_value - self.min_factor) < 0.001:
                self.direction = 1
        
        if self.debug:
            print(f"[PromptTravelScheduler] Factor value: {self.factor_value:.2f}")
            
        return self.factor_value
        
    def set_enabled(self, enabled):
        """Enable or disable the scheduler"""
        self.enabled = enabled
        if self.debug:
            print(f"[PromptTravelScheduler] Enabled set to: {enabled}")
        
    def set_debug(self, debug):
        """Enable or disable debug output"""
        self.debug = debug
        
    def set_oscillation(self, oscillate):
        """Set whether to oscillate between min/max or go one-way"""
        self.oscillate = oscillate
        if self.debug:
            print(f"[PromptTravelScheduler] Oscillation set to: {oscillate}")
            
    def set_factor_increment(self, increment):
        """Set the factor increment amount"""
        self.factor_increment = max(0.001, min(0.1, increment))
        if self.debug:
            print(f"[PromptTravelScheduler] Factor increment set to: {self.factor_increment}")
            
    def set_boundaries(self, min_factor=None, max_factor=None):
        """Set the min and max factor boundaries"""
        if min_factor is not None:
            self.min_factor = max(0.0, min(1.0, min_factor))
        if max_factor is not None:
            self.max_factor = max(0.0, min(1.0, max_factor))
            
        if self.debug:
            print(f"[PromptTravelScheduler] Boundaries set to: min={self.min_factor}, max={self.max_factor}")
            
    def reset(self):
        """Reset the scheduler to initial state"""
        self.factor_value = self.min_factor
        self.direction = 1
        self.stabilize_counter = 0
        if self.debug:
            print(f"[PromptTravelScheduler] Reset to initial state: factor={self.factor_value}") 