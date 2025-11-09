"""
Seed Travel Scheduler for managed transitions between random seeds.
Provides smooth cycling through multiple seeds for continuous variation.
"""

import random
import logging

class SeedTravelScheduler:
    """
    A class that handles scheduled seed travel transitions.
    Cycles through multiple seeds continuously for smooth variation.
    """
    def __init__(self,
                 num_seeds=4,
                 factor_increment=0.025,
                 enabled=True,
                 match_prompt_travel=False,
                 seed_list=None,
                 logging_enabled=False):
        """
        Initialize the seed travel scheduler.

        Args:
            num_seeds (int): Number of seeds to cycle through (default: 4)
            factor_increment (float): Amount to change factor per update (default: 0.025)
            enabled (bool): Whether the scheduler is active (default: True)
            match_prompt_travel (bool): If True, sync with prompt travel factor (default: False)
            seed_list (list): Optional pre-defined list of seeds (default: None, generates random)
            logging_enabled (bool): Whether to enable logging (default: False)
        """
        self.num_seeds = num_seeds
        self.factor_increment = factor_increment
        self.enabled = enabled
        self.match_prompt_travel = match_prompt_travel
        self.logging_enabled = logging_enabled

        # Generate or use provided seed list
        if seed_list is not None and len(seed_list) > 0:
            self.seed_list = seed_list
            self.num_seeds = len(seed_list)
        else:
            self.seed_list = [random.randint(0, 1000000) for _ in range(num_seeds)]

        # Internal state
        self.factor_value = 0.0  # 0.0 to num_seeds (continuous)

        # Setup logging if enabled
        if logging_enabled:
            self.logger = logging.getLogger("SeedTravelScheduler")
            self.logger.setLevel(logging.INFO)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
                self.logger.addHandler(handler)
        else:
            self.logger = logging.getLogger("SeedTravelScheduler")
            self.logger.addHandler(logging.NullHandler())

        if self.logging_enabled:
            self.logger.info(f"Initialized with {self.num_seeds} seeds: {self.seed_list}")

    def update(self, external_factor=None):
        """
        Update the factor value for the next iteration.

        Args:
            external_factor (float): Optional external factor (e.g., from prompt travel)
                                    Only used if match_prompt_travel is True

        Returns:
            tuple: (current_seed, next_seed, interpolation_factor)
        """
        if not self.enabled:
            # Return first two seeds with no interpolation
            return self.seed_list[0], self.seed_list[1 % self.num_seeds], 0.0

        # Use external factor if matching prompt travel
        if self.match_prompt_travel and external_factor is not None:
            self.factor_value = external_factor
        else:
            # Increment factor
            self.factor_value += self.factor_increment

            # Wrap around at num_seeds (continuous cycling)
            if self.factor_value >= self.num_seeds:
                self.factor_value = self.factor_value - self.num_seeds

        # Calculate which two seeds we're interpolating between
        current_idx = int(self.factor_value) % self.num_seeds
        next_idx = (current_idx + 1) % self.num_seeds

        # Get interpolation weight (fractional part)
        interpolation_factor = self.factor_value - int(self.factor_value)

        current_seed = self.seed_list[current_idx]
        next_seed = self.seed_list[next_idx]

        if self.logging_enabled:
            self.logger.info(f"Factor: {self.factor_value:.3f}, Seeds: {current_seed} -> {next_seed}, Weight: {interpolation_factor:.3f}")

        return current_seed, next_seed, interpolation_factor

    def get_current_seeds(self):
        """
        Get the current and next seeds being interpolated.

        Returns:
            tuple: (current_seed, next_seed, interpolation_factor)
        """
        current_idx = int(self.factor_value) % self.num_seeds
        next_idx = (current_idx + 1) % self.num_seeds
        interpolation_factor = self.factor_value - int(self.factor_value)

        return self.seed_list[current_idx], self.seed_list[next_idx], interpolation_factor

    def set_seeds(self, seed_list):
        """
        Update the seed list.

        Args:
            seed_list (list): New list of seeds
        """
        self.seed_list = seed_list
        self.num_seeds = len(seed_list)
        if self.logging_enabled:
            self.logger.info(f"Updated to {self.num_seeds} seeds: {self.seed_list}")
