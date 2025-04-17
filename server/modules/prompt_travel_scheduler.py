"""
Prompt Travel Scheduler for managed transitions between prompts.
This module provides scheduling functionality for the prompt travel feature,
automatically interpolating between source and target prompts.
"""

import random  # Add import for random module

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
                debug=False,
                seed_enabled=True):
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
            seed_enabled (bool): Whether to generate random seeds (default: False)
        """
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.factor_increment = factor_increment
        self.stabilize_duration = stabilize_duration
        self.oscillate = oscillate
        self.enabled = enabled
        self.debug = debug
        self.seed_enabled = seed_enabled
        
        # Internal state
        self.factor_value = min_factor
        self.direction = 1  # 1 for increasing, -1 for decreasing
        self.stabilize_counter = 0  # Counter for stabilization pause
        
        # Initialize seeds if enabled
        if seed_enabled:
            self.current_seed = random.randint(0, 1000000)
            self.next_seed = random.randint(0, 1000000)
            if debug:
                print(f"[PromptTravelScheduler] Initialized with seeds: current={self.current_seed}, next={self.next_seed}")
        else:
            self.current_seed = None
            self.next_seed = None
    
    def update(self):
        """
        Update the factor value for the next iteration.
        
        Returns:
            tuple: A tuple containing (factor_value, seed_value) where seed_value is None if seed_enabled is False
        """
        if not self.enabled:
            return self.factor_value, self.current_seed if self.seed_enabled else None
            
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
                
                # Generate new seeds when we hit a boundary if enabled
                if self.seed_enabled:
                    # Determine which boundary we're at
                    at_max = abs(self.factor_value - self.max_factor) < 0.001
                    
                    if at_max:
                        # At max boundary (target), change the source seed (current_seed)
                        self.current_seed = random.randint(0, 1000000)
                        if self.debug:
                            print(f"[PromptTravelScheduler] At max boundary, changing source seed to: {self.current_seed}")
                    else:
                        # At min boundary (source), change the target seed (next_seed)
                        self.next_seed = random.randint(0, 1000000)
                        if self.debug:
                            print(f"[PromptTravelScheduler] At min boundary, changing target seed to: {self.next_seed}")
                
                # If we're not oscillating and we've reached max, stop at max
                if not self.oscillate and abs(self.factor_value - self.max_factor) < 0.001:
                    return self.factor_value, self.current_seed if self.seed_enabled else None
                
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
            
        return self.factor_value, self.current_seed if self.seed_enabled else None
    
    def get_seeds(self):
        """
        Get both the current seed and the next seed (for target generator).
        
        Returns:
            tuple: A tuple containing (current_seed, next_seed) where both can be None if seed_enabled is False
        """
        # if not self.seed_enabled:
        #     return None, None
            
        # If we don't have a current_seed yet, generate one
        if self.current_seed is None:
            self.current_seed = random.randint(0, 1000000)
            if self.debug:
                print(f"[PromptTravelScheduler] Generated current seed: {self.current_seed}")
                
        # If we don't have a next_seed yet, generate one
        if self.next_seed is None:
            self.next_seed = random.randint(0, 1000000)
            if self.debug:
                print(f"[PromptTravelScheduler] Generated next seed: {self.next_seed}")
                
        return self.current_seed, self.next_seed
        
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
            
    def set_seed_enabled(self, enabled):
        """Enable or disable seed generation"""
        self.seed_enabled = enabled
        if self.debug:
            print(f"[PromptTravelScheduler] Seed generation set to: {enabled}")
            
    def reset(self):
        """Reset the scheduler to initial state"""
        self.factor_value = self.min_factor
        self.direction = 1
        self.stabilize_counter = 0
        if self.seed_enabled:
            self.current_seed = random.randint(0, 1000000)  # Generate a new seed on reset
            self.next_seed = random.randint(0, 1000000)  # Generate a new seed on reset
            if self.debug:
                print(f"[PromptTravelScheduler] Reset with new seeds: current={self.current_seed}, next={self.next_seed}")
        if self.debug:
            print(f"[PromptTravelScheduler] Reset to initial state: factor={self.factor_value}") 