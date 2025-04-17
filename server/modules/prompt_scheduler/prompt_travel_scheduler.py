"""
Prompt Travel Scheduler for managed transitions between prompts.
This module provides scheduling functionality for the prompt travel feature,
automatically interpolating between source and target prompts.
"""

import random  # Add import for random module
import logging
import os
from datetime import datetime
from .prompt_scheduler import PromptScheduler

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
                seed_enabled=True,
                use_prompt_scheduler=False,
                prompts_dir="prompts",
                prompt_file_pattern="*.txt",
                loop_prompts=True,
                logging_enabled=False):
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
            use_prompt_scheduler (bool): Whether to use the prompt scheduler (default: False)
            prompts_dir (str): Directory containing prompt files (default: "prompts")
            prompt_file_pattern (str): Pattern to match prompt files (default: "*.txt")
            loop_prompts (bool): Whether to loop back to the beginning when reaching the end (default: True)
            logging_enabled (bool): Whether to enable logging (default: False)
        """
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.factor_increment = factor_increment
        self.stabilize_duration = stabilize_duration
        self.oscillate = oscillate
        self.enabled = enabled
        self.debug = debug
        self.seed_enabled = seed_enabled
        self.use_prompt_scheduler = use_prompt_scheduler
        self.logging_enabled = logging_enabled
        
        # Internal state
        self.factor_value = min_factor
        self.direction = 1  # 1 for increasing, -1 for decreasing
        self.stabilize_counter = 0  # Counter for stabilization pause
        
        # Setup logging if enabled
        if logging_enabled:
            self.setup_logging()
        else:
            # Create a dummy logger that does nothing
            self.logger = logging.getLogger("PromptTravelScheduler")
            self.logger.addHandler(logging.NullHandler())
        
        # Initialize seeds if enabled
        if seed_enabled:
            self.current_seed = random.randint(0, 1000000)
            self.next_seed = random.randint(0, 1000000)
            if self.logging_enabled:
                self.logger.info(f"Initialized with seeds: current={self.current_seed}, next={self.next_seed}")
        else:
            self.current_seed = None
            self.next_seed = None
            
        # Initialize prompt scheduler if enabled
        if use_prompt_scheduler:
            if self.logging_enabled:
                self.logger.info(f"Initializing prompt scheduler with prompts_dir={prompts_dir}, prompt_file_pattern={prompt_file_pattern}")
            self.prompt_scheduler = PromptScheduler(
                prompts_dir=prompts_dir,
                prompt_file_pattern=prompt_file_pattern,
                enabled=True,
                debug=debug,
                loop_prompts=loop_prompts,
                logging_enabled=logging_enabled
            )
            if self.logging_enabled:
                self.logger.info("Prompt scheduler initialized")
            # Check if prompts were loaded
            current_prompt, next_prompt = self.prompt_scheduler.get_current_prompts()
            if self.logging_enabled:
                self.logger.info(f"Initial prompts: source={current_prompt}, target={next_prompt}")
        else:
            self.prompt_scheduler = None
            if self.logging_enabled:
                self.logger.info("Prompt scheduler not enabled")
    
    def setup_logging(self):
        """
        Setup logging to a file.
        """
        # Create logs directory if it doesn't exist
        server_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        logs_dir = os.path.join(server_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        
        # Create a log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(logs_dir, f"prompt_travel_scheduler_{timestamp}.log")
        
        # Configure logging
        self.logger = logging.getLogger("PromptTravelScheduler")
        self.logger.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(file_handler)
        
        self.logger.info(f"Prompt Travel Scheduler logging initialized. Log file: {log_file}")
    
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
            
            if self.stabilize_counter == 1:
                boundary_type = "max" if abs(self.factor_value - self.max_factor) < 0.001 else "min"
                if self.logging_enabled:
                    self.logger.info(f"Stabilizing at {boundary_type} factor={self.factor_value:.2f} ({self.stabilize_counter}/{self.stabilize_duration})")
        else:
            if at_boundary and self.stabilize_counter >= self.stabilize_duration:
                # We've finished the stabilization period, continue with oscillation
                self.stabilize_counter = 0
                
                # Determine which boundary we're at
                at_max = abs(self.factor_value - self.max_factor) < 0.001
                at_min = abs(self.factor_value - self.min_factor) < 0.001
                
                # Update prompts if prompt scheduler is enabled
                if self.use_prompt_scheduler and self.prompt_scheduler is not None:
                    # Update the prompt scheduler with the correct boundary flags
                    self.prompt_scheduler.update(at_max_boundary=at_max, at_min_boundary=at_min)
                    current_prompt, next_prompt = self.prompt_scheduler.get_current_prompts()
                    if self.logging_enabled:
                        self.logger.info(f"Updated prompts: source={current_prompt}, target={next_prompt}")
                
                # Generate new seeds when we hit a boundary if enabled
                if self.seed_enabled:
                    if at_max:
                        # At max boundary (target), change the source seed (current_seed)
                        self.current_seed = random.randint(0, 1000000)
                        if self.logging_enabled:
                            self.logger.info(f"At max boundary, changing source seed to: {self.current_seed}")
                    else:
                        # At min boundary (source), change the target seed (next_seed)
                        self.next_seed = random.randint(0, 1000000)
                        if self.logging_enabled:
                            self.logger.info(f"At min boundary, changing target seed to: {self.next_seed}")
                
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
        
        if self.logging_enabled:
            self.logger.info(f"Factor value: {self.factor_value:.2f}")
            
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
            if self.logging_enabled:
                self.logger.info(f"Generated current seed: {self.current_seed}")
                
        # If we don't have a next_seed yet, generate one
        if self.next_seed is None:
            self.next_seed = random.randint(0, 1000000)
            if self.logging_enabled:
                self.logger.info(f"Generated next seed: {self.next_seed}")
                
        if self.logging_enabled:
            self.logger.info(f"Getting seeds: current={self.current_seed}, next={self.next_seed}")
        return self.current_seed, self.next_seed
        
    def get_prompts(self):
        """
        Get the current source and target prompts from the prompt scheduler.
        
        Returns:
            tuple: A tuple containing (current_prompt, next_prompt) or (None, None) if prompt scheduler is not enabled
        """
        if self.use_prompt_scheduler and self.prompt_scheduler is not None:
            current_prompt, next_prompt = self.prompt_scheduler.get_current_prompts()
            if self.logging_enabled:
                self.logger.info(f"Got prompts from scheduler: source={current_prompt}, target={next_prompt}")
            return current_prompt, next_prompt
        if self.logging_enabled:
            self.logger.info("Prompt scheduler not enabled, returning None, None")
        return None, None
        
    def set_enabled(self, enabled):
        """Enable or disable the scheduler"""
        self.enabled = enabled
        if self.logging_enabled:
            self.logger.info(f"Enabled set to: {enabled}")
        
    def set_debug(self, debug):
        """Enable or disable debug output"""
        self.debug = debug
        if self.prompt_scheduler is not None:
            self.prompt_scheduler.set_debug(debug)
        if self.logging_enabled:
            self.logger.info(f"Debug set to: {debug}")
        
    def set_oscillation(self, oscillate):
        """Set whether to oscillate between min/max or go one-way"""
        self.oscillate = oscillate
        if self.logging_enabled:
            self.logger.info(f"Oscillation set to: {oscillate}")
            
    def set_factor_increment(self, increment):
        """Set the factor increment amount"""
        self.factor_increment = max(0.001, min(0.1, increment))
        if self.logging_enabled:
            self.logger.info(f"Factor increment set to: {self.factor_increment}")
            
    def set_boundaries(self, min_factor=None, max_factor=None):
        """Set the min and max factor boundaries"""
        if min_factor is not None:
            self.min_factor = max(0.0, min(1.0, min_factor))
        if max_factor is not None:
            self.max_factor = max(0.0, min(1.0, max_factor))
            
        if self.logging_enabled:
            self.logger.info(f"Boundaries set to: min={self.min_factor}, max={self.max_factor}")
            
    def set_seed_enabled(self, enabled):
        """Enable or disable seed generation"""
        self.seed_enabled = enabled
        if self.logging_enabled:
            self.logger.info(f"Seed generation set to: {enabled}")
            
    def set_prompt_scheduler_enabled(self, enabled):
        """Enable or disable the prompt scheduler"""
        self.use_prompt_scheduler = enabled
        if self.prompt_scheduler is not None:
            self.prompt_scheduler.set_enabled(enabled)
        if self.logging_enabled:
            self.logger.info(f"Prompt scheduler enabled set to: {enabled}")
            
    def reload_prompts(self):
        """Reload prompts from the file"""
        if self.prompt_scheduler is not None:
            self.prompt_scheduler.reload_prompts()
        if self.logging_enabled:
            self.logger.info("Reloading prompts")
            
    def reset(self):
        """Reset the scheduler to initial state"""
        self.factor_value = self.min_factor
        self.direction = 1
        self.stabilize_counter = 0
        if self.seed_enabled:
            self.current_seed = random.randint(0, 1000000)  # Generate a new seed on reset
            self.next_seed = random.randint(0, 1000000)  # Generate a new seed on reset
            if self.logging_enabled:
                self.logger.info(f"Reset with new seeds: current={self.current_seed}, next={self.next_seed}")
        if self.prompt_scheduler is not None:
            self.prompt_scheduler.reset()
        if self.logging_enabled:
            self.logger.info(f"Reset to initial state: factor={self.factor_value}")
            
    def set_logging_enabled(self, enabled):
        """Enable or disable logging"""
        self.logging_enabled = enabled
        if enabled and not hasattr(self, 'logger') or self.logger.handlers == []:
            self.setup_logging()
        elif not enabled and hasattr(self, 'logger'):
            # Remove all handlers
            for handler in self.logger.handlers[:]:
                self.logger.removeHandler(handler)
            # Add a null handler
            self.logger.addHandler(logging.NullHandler()) 