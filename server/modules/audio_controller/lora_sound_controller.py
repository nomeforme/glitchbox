import numpy as np
import random

class LoraSoundController:
    """
    Sound-reactive controller that synchronizes LoRA pipe selection 
    with audio frequency analysis.
    """
    
    def __init__(self, 
                 num_pipes=2,  # Number of available pipes in the pipeline
                 enabled=True,
                 debug=False,
                 freq_start_idx=0,  # Start index for frequency range
                 freq_end_idx=30,  # End index for frequency range (None means use all)
                 averaging_factor=6):  # Number of bins to average together
        """
        Initialize the sound-reactive LoRA controller.
        
        Args:
            num_pipes (int): Number of available pipes in the pipeline
            enabled (bool): Whether the controller is active
            debug (bool): Enable debug printing
            freq_start_idx (int): Start index for frequency range
            freq_end_idx (int): End index for frequency range (None means use all)
            averaging_factor (int): Number of bins to average together
        """
        # Store configuration parameters
        self.num_pipes = num_pipes
        self.enabled = enabled
        self.debug = debug
        self.freq_start_idx = freq_start_idx
        self.freq_end_idx = freq_end_idx
        self.averaging_factor = averaging_factor
        
        # Current pipe index value
        self.current_pipe_index = 0

    def enable_debug(self, enabled=True):
        """Enable or disable debug printing"""
        self.debug = enabled
        
    def _average_frequency_bins(self, frequency_data):
        """
        Average frequency bins into larger buckets.
        
        Args:
            frequency_data (np.ndarray or list): Array of frequency bin values
            
        Returns:
            np.ndarray: Averaged frequency bins
        """
        if self.averaging_factor <= 1:
            return frequency_data
            
        # Convert input to numpy array if it isn't already
        frequency_data = np.array(frequency_data)
        
        if self.debug:
            print(f"Input frequency data length: {len(frequency_data)}")
            print(f"Averaging factor: {self.averaging_factor}")
            
        # Calculate the number of complete buckets
        n_bins = len(frequency_data)
        n_buckets = n_bins // self.averaging_factor
        
        if self.debug:
            print(f"Number of complete buckets: {n_buckets}")
            print(f"Original bins: {frequency_data}")
        
        # If we don't have enough bins for even one complete bucket, return the original data
        if n_buckets == 0:
            if self.debug:
                print("Not enough bins for complete bucket, returning original data")
            return frequency_data
        
        # Reshape the array to group bins into buckets
        # Discard any remaining bins that don't fit into a complete bucket
        reshaped = frequency_data[:n_buckets * self.averaging_factor].reshape(n_buckets, self.averaging_factor)
        
        # Calculate mean for each bucket
        averaged = np.mean(reshaped, axis=1)
        
        if self.debug:
            print(f"Averaged {n_bins} bins into {n_buckets} buckets using factor {self.averaging_factor}")
            print(f"Averaged result: {averaged}")
            
        return averaged

    def process_frequency_bins(self, normalized_energies):
        """
        Process frequency bins to adjust pipe index based on audio input.
        For now, returns a random pipe index within allowed range.
        
        Args:
            normalized_energies: List containing the normalized energies
            
        Returns:
            int: Selected pipe index
        """
        # Validate input
        if not self.enabled:
            if self.debug:
                print("Controller is disabled, returning current pipe index.")
            return self.current_pipe_index

        if normalized_energies is None or (isinstance(normalized_energies, (list, tuple, np.ndarray)) and len(normalized_energies) < 3):
            if self.debug:
                print(f"Invalid frequency bins, returning current pipe index: {self.current_pipe_index}")
            return self.current_pipe_index
        
        # Slice the frequency range if specified
        if self.freq_end_idx is None:
            self.freq_end_idx = len(normalized_energies)
            
        normalized_energies_subset = normalized_energies[self.freq_start_idx:self.freq_end_idx]
        
        # Average the frequency bins
        averaged_energies = self._average_frequency_bins(normalized_energies_subset)
        
        # For now, just return the highest energy bin pipe index
        new_pipe_index = np.argmax(averaged_energies)

        if self.debug:
            print(f"Selected pipe index: {new_pipe_index}")
            
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