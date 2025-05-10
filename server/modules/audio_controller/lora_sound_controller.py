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
                 debug=False):
        """
        Initialize the sound-reactive LoRA controller.
        
        Args:
            num_pipes (int): Number of available pipes in the pipeline
            enabled (bool): Whether the controller is active
            debug (bool): Enable debug printing
        """
        # Store configuration parameters
        self.num_pipes = num_pipes
        self.enabled = enabled
        self.debug = debug
        
        # Current pipe index value
        self.current_pipe_index = 0

    def enable_debug(self, enabled=True):
        """Enable or disable debug printing"""
        self.debug = enabled
        
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
        
        # For now, just return a random pipe index
        new_pipe_index = np.argmax(normalized_energies)

        
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