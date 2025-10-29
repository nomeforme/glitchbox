"""
Prompt Scheduler for managing sequential prompt transitions.
This module provides functionality to read prompts from a file and manage
sequential transitions between prompts as source/target pairs.
"""

import os
import glob
import random
import logging
from datetime import datetime
from typing import List, Tuple, Optional, Dict

class PromptScheduler:
    """
    A class that handles scheduled prompt transitions from a file.
    This provides a deterministic schedule for moving between prompts.
    """
    def __init__(self,
                prompts_dir="prompts",
                prompt_file_pattern="*.txt",
                enabled=False,
                debug=False,
                loop_prompts=True,
                logging_enabled=False,
                prompts_file_names=None):
        """
        Initialize the prompt scheduler.

        Args:
            prompts_dir (str): Directory containing prompt files (default: "prompts")
            prompt_file_pattern (str): Pattern to match prompt files (default: "*.txt")
            enabled (bool): Whether the scheduler is active (default: False)
            debug (bool): Whether to print debug messages (default: False)
            loop_prompts (bool): Whether to loop back to the beginning when reaching the end (default: True)
            logging_enabled (bool): Whether to enable logging (default: False)
            prompts_file_names (list): List of prompts file names (default: None)
        """
        self.prompts_dir = prompts_dir
        self.prompt_file_pattern = prompt_file_pattern
        self.enabled = enabled
        self.debug = debug
        self.loop_prompts = loop_prompts
        self.logging_enabled = logging_enabled
        self.prompts_file_names = prompts_file_names if prompts_file_names else []

        # Internal state - multi-file support
        self.prompts_by_file = []  # List of lists: [[file1_prompts], [file2_prompts], ...]
        self.prompts = []  # Keep for backward compatibility (uses first file)
        self.current_index = 0
        self.current_prompt = None
        self.next_prompt = None

        # Progress ratio: tracks position in prompt cycle as 0.0-1.0
        # This preserves progress when switching between different prompt files
        self.progress_ratio = 0.0
        
        # Setup logging if enabled
        if logging_enabled:
            self.setup_logging()
        else:
            # Create a dummy logger that does nothing
            self.logger = logging.getLogger("PromptScheduler")
            self.logger.addHandler(logging.NullHandler())
        
        # Load prompts on initialization
        self.load_prompts()
    
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
        log_file = os.path.join(logs_dir, f"prompt_scheduler_{timestamp}.log")
        
        # Configure logging
        self.logger = logging.getLogger("PromptScheduler")
        self.logger.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(file_handler)
        
        self.logger.info(f"Prompt Scheduler logging initialized. Log file: {log_file}")
    
    def get_prompt_prefix(self):
        """
        Get the prompt prefix from the prompt_prefix_*.txt files.
        
        Returns:
            str: The prompt prefix
        """
        # Get the absolute path to the prompts directory
        server_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        prompts_path = os.path.join(server_dir, self.prompts_dir)
        
        # Find all prompt prefix files
        prefix_files = glob.glob(os.path.join(prompts_path, "prompt_prefix_*.txt"))
        
        if not prefix_files:
            if self.debug:
                print(f"[PromptScheduler] No prompt prefix files found in {prompts_path}")
            if self.logging_enabled:
                self.logger.warning(f"No prompt prefix files found in {prompts_path}")
            return ""
        
        # Read the first prefix file
        with open(prefix_files[0], 'r') as f:
            prefix = f.read().strip()
            
        if self.debug:
            print(f"[PromptScheduler] Loaded prompt prefix: {prefix}")
        if self.logging_enabled:
            self.logger.info(f"Loaded prompt prefix: {prefix}")
            
        return prefix
    
    def load_prompts(self):
        """
        Load prompts from multiple files specified in prompts_file_names.
        Each file should have the same number of lines.
        """
        # Get the prompt prefix
        prompt_prefix = self.get_prompt_prefix()

        # Get the absolute path to the prompts directory
        server_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        prompts_path = os.path.join(server_dir, self.prompts_dir)
        lora_prompts_path = os.path.join(prompts_path, "lora_prompts")

        if self.debug:
            print(f"[PromptScheduler] Looking for prompts in: {lora_prompts_path}")
        if self.logging_enabled:
            self.logger.info(f"Looking for prompts in: {lora_prompts_path}")

        # Clear existing prompts
        self.prompts_by_file = []
        self.prompts = []

        # If prompts_file_names is provided, load all specified files
        if self.prompts_file_names and len(self.prompts_file_names) > 0:
            print(f"[PromptScheduler] Loading {len(self.prompts_file_names)} prompt files: {self.prompts_file_names}")

            for file_name in self.prompts_file_names:
                prompt_file = os.path.join(lora_prompts_path, f"prompts_{file_name}.txt")

                if not os.path.exists(prompt_file):
                    print(f"[PromptScheduler] WARNING: File not found: {prompt_file}")
                    if self.logging_enabled:
                        self.logger.warning(f"File not found: {prompt_file}")
                    continue

                # Read prompts from file
                with open(prompt_file, 'r') as f:
                    raw_prompts = [line.strip() for line in f.readlines() if line.strip()]
                    # Apply the prompt prefix to each prompt
                    file_prompts = [prompt_prefix + " " + prompt for prompt in raw_prompts]
                    self.prompts_by_file.append(file_prompts)

                if self.debug:
                    print(f"[PromptScheduler] Loaded {len(file_prompts)} prompts from {file_name}")

            # Validate that all files have the same number of lines
            if self.prompts_by_file:
                line_counts = [len(prompts) for prompts in self.prompts_by_file]
                if len(set(line_counts)) > 1:
                    print(f"[PromptScheduler] ERROR: Prompt files have different line counts: {line_counts}")
                    print(f"[PromptScheduler] Files: {self.prompts_file_names}")
                    if self.logging_enabled:
                        self.logger.error(f"Prompt files have different line counts: {line_counts}")
                    # Keep going but warn user

                # For backward compatibility, set self.prompts to first file
                self.prompts = self.prompts_by_file[0] if self.prompts_by_file else []

                print(f"[PromptScheduler] Successfully loaded {len(self.prompts_by_file)} prompt files with {len(self.prompts)} prompts each")
                if self.logging_enabled:
                    self.logger.info(f"Loaded {len(self.prompts_by_file)} prompt files with {len(self.prompts)} prompts each")
        else:
            # Fallback to old behavior: find files matching pattern
            print(f"[PromptScheduler] No prompts_file_names provided, using fallback pattern matching")
            prompt_files = glob.glob(os.path.join(prompts_path, self.prompt_file_pattern))
            prompt_files = [f for f in prompt_files if not os.path.basename(f).startswith("prompt_prefix_")]

            if not prompt_files:
                print(f"[PromptScheduler] No prompt files found")
                return

            # Read prompts from the first file found
            with open(prompt_files[0], 'r') as f:
                raw_prompts = [line.strip() for line in f.readlines() if line.strip()]
                self.prompts = [prompt_prefix + " " + prompt for prompt in raw_prompts]
                self.prompts_by_file = [self.prompts]

        # Initialize current and next prompts (backward compatibility)
        if len(self.prompts) >= 2:
            self.current_index = int(self.progress_ratio * (len(self.prompts) - 1))
            self.current_index = max(0, min(len(self.prompts) - 1, self.current_index))
            self.current_prompt = self.prompts[self.current_index]
            next_index = (self.current_index + 1) % len(self.prompts)
            self.next_prompt = self.prompts[next_index]
        elif len(self.prompts) == 1:
            self.current_index = 0
            self.current_prompt = self.prompts[0]
            self.next_prompt = self.prompts[0]
    
    def update(self, at_max_boundary=False, at_min_boundary=False):
        """
        Update the current and next prompts based on boundary conditions.

        Args:
            at_max_boundary (bool): Whether we're at the maximum boundary (target prompt)
            at_min_boundary (bool): Whether we're at the minimum boundary (source prompt)

        Returns:
            tuple: A tuple containing (current_prompt, next_prompt)
        """
        print(f"[PromptScheduler.update] ENTER: at_max={at_max_boundary}, at_min={at_min_boundary}, current_index={self.current_index}")
        print(f"[PromptScheduler.update] BEFORE: current={self.current_prompt[:50] if self.current_prompt else 'None'}..., next={self.next_prompt[:50] if self.next_prompt else 'None'}...")

        if not self.enabled or len(self.prompts) < 2:
            return self.current_prompt, self.next_prompt

        # Update prompts based on boundary conditions
        if at_max_boundary:
            # At max boundary (target), advance to next prompt pair
            # current becomes what next was, and next advances to the following prompt
            old_index = self.current_index
            self.current_index = (self.current_index + 1) % len(self.prompts)
            self.current_prompt = self.prompts[self.current_index]

            # IMPORTANT: Also update next_prompt to avoid current==next
            next_index = (self.current_index + 1) % len(self.prompts)
            self.next_prompt = self.prompts[next_index]

            print(f"[PromptScheduler.update] MAX BOUNDARY: index {old_index} -> {self.current_index}")
            print(f"[PromptScheduler.update] MAX BOUNDARY: current_prompt updated to: {self.current_prompt[:50]}...")
            print(f"[PromptScheduler.update] MAX BOUNDARY: next_prompt updated to: {self.next_prompt[:50]}...")

            # Update progress ratio to preserve position across prompt file switches
            if len(self.prompts) > 1:
                self.progress_ratio = self.current_index / (len(self.prompts) - 1)

            if self.debug:
                print(f"[PromptScheduler] At max boundary, updated source prompt: {self.current_prompt} (ratio: {self.progress_ratio:.2f})")
            if self.logging_enabled:
                self.logger.info(f"At max boundary, updated source prompt: {self.current_prompt}")

        elif at_min_boundary:
            # At min boundary (source), change the target prompt (next_prompt)
            # Keep the source prompt (current_prompt) as is
            next_index = (self.current_index + 1) % len(self.prompts)
            self.next_prompt = self.prompts[next_index]

            print(f"[PromptScheduler.update] MIN BOUNDARY: current_index unchanged: {self.current_index}")
            print(f"[PromptScheduler.update] MIN BOUNDARY: current_prompt unchanged: {self.current_prompt[:50]}...")
            print(f"[PromptScheduler.update] MIN BOUNDARY: next_prompt updated to: {self.next_prompt[:50]}...")

            if self.debug:
                print(f"[PromptScheduler] At min boundary, updated target prompt: {self.next_prompt}")
            if self.logging_enabled:
                self.logger.info(f"At min boundary, updated target prompt: {self.next_prompt}")

        print(f"[PromptScheduler.update] EXIT: current={self.current_prompt[:50]}..., next={self.next_prompt[:50]}...")

        return self.current_prompt, self.next_prompt
    
    def get_current_prompts(self):
        """
        Get the current source and target prompts.

        Returns:
            tuple: A tuple containing (current_prompt, next_prompt)
        """
        if self.logging_enabled:
            self.logger.info(f"Getting current prompts: source={self.current_prompt}, target={self.next_prompt}")
        return self.current_prompt, self.next_prompt

    def get_prompts_from_factor(self, continuous_factor, weights=None):
        """
        Get source and target prompts based on a continuous factor value.
        Supports weighted blending across multiple prompt files.

        This method uses a simple linear progression through the prompt list:
        - Factor range: 0.0 to len(prompts)-1 (wraps around with modulo)
        - int(factor) gives the source prompt index
        - int(factor)+1 gives the target prompt index (with wrap)
        - frac(factor) gives the interpolation weight

        Multi-file blending:
        - If weights are provided and multiple files are loaded, returns lists of prompts
        - Pipeline should encode each prompt and blend embeddings using weights
        - weights should have same length as prompts_by_file

        Example with 4 prompts:
        - factor=0.3: source=prompts[0], target=prompts[1], weight=0.3
        - factor=1.7: source=prompts[1], target=prompts[2], weight=0.7
        - factor=3.2: source=prompts[3], target=prompts[0], weight=0.2

        Args:
            continuous_factor (float): Continuous factor from 0.0 to len(prompts)
            weights (list): Optional list of weights for multi-file blending

        Returns:
            tuple: (source_prompts, target_prompts, interpolation_weight, spatial_weights)
                   source_prompts and target_prompts can be strings or lists depending on weights
        """
        if not self.prompts or len(self.prompts) == 0:
            print("[PromptScheduler.get_prompts_from_factor] No prompts loaded")
            return "", "", 0.0, None

        if len(self.prompts) == 1:
            # Only one prompt, return it for both source and target
            if weights and len(self.prompts_by_file) > 1:
                # Multi-file mode with single prompt
                source_list = [file_prompts[0] for file_prompts in self.prompts_by_file]
                return source_list, source_list, 0.0, weights
            else:
                return self.prompts[0], self.prompts[0], 0.0, None

        # Wrap factor to valid range using modulo
        wrapped_factor = continuous_factor % len(self.prompts)

        # Get source and target indices
        source_index = int(wrapped_factor)
        target_index = (source_index + 1) % len(self.prompts)

        # Get interpolation weight (fractional part)
        interpolation_weight = wrapped_factor - source_index

        # Multi-file weighted blending
        if weights and len(self.prompts_by_file) > 1:
            # Validate weights
            if len(weights) != len(self.prompts_by_file):
                print(f"[PromptScheduler.get_prompts_from_factor] WARNING: weights length ({len(weights)}) != files count ({len(self.prompts_by_file)})")
                # Fall back to single file mode
                source_prompt = self.prompts[source_index]
                target_prompt = self.prompts[target_index]
                return source_prompt, target_prompt, interpolation_weight, None

            # Get prompts from all files at the same indices
            source_prompts = []
            target_prompts = []

            for file_prompts in self.prompts_by_file:
                if source_index < len(file_prompts):
                    source_prompts.append(file_prompts[source_index])
                else:
                    print(f"[PromptScheduler.get_prompts_from_factor] WARNING: source_index {source_index} out of range for file with {len(file_prompts)} prompts")
                    source_prompts.append(file_prompts[0])  # Fallback

                if target_index < len(file_prompts):
                    target_prompts.append(file_prompts[target_index])
                else:
                    print(f"[PromptScheduler.get_prompts_from_factor] WARNING: target_index {target_index} out of range for file with {len(file_prompts)} prompts")
                    target_prompts.append(file_prompts[0])  # Fallback

            if self.debug:
                print(f"[PromptScheduler.get_prompts_from_factor] MULTI-FILE MODE")
                print(f"[PromptScheduler.get_prompts_from_factor]   continuous_factor={continuous_factor:.3f}")
                print(f"[PromptScheduler.get_prompts_from_factor]   wrapped={wrapped_factor:.3f}, source_idx={source_index}, target_idx={target_index}")
                print(f"[PromptScheduler.get_prompts_from_factor]   interpolation_weight={interpolation_weight:.3f}")
                print(f"[PromptScheduler.get_prompts_from_factor]   spatial_weights={weights}")
                for i, (src, tgt) in enumerate(zip(source_prompts, target_prompts)):
                    print(f"[PromptScheduler.get_prompts_from_factor]   File {i} (weight={weights[i]:.2f}):")
                    print(f"[PromptScheduler.get_prompts_from_factor]     source: {src[:50]}...")
                    print(f"[PromptScheduler.get_prompts_from_factor]     target: {tgt[:50]}...")

            return source_prompts, target_prompts, interpolation_weight, weights

        # Single file mode (backward compatibility)
        source_prompt = self.prompts[source_index]
        target_prompt = self.prompts[target_index]

        if self.debug:
            print(f"[PromptScheduler.get_prompts_from_factor] SINGLE-FILE MODE")
            print(f"[PromptScheduler.get_prompts_from_factor]   continuous_factor={continuous_factor:.3f}")
            print(f"[PromptScheduler.get_prompts_from_factor]   wrapped={wrapped_factor:.3f}, source_idx={source_index}, target_idx={target_index}, weight={interpolation_weight:.3f}")
            print(f"[PromptScheduler.get_prompts_from_factor]   source={source_prompt[:50]}...")
            print(f"[PromptScheduler.get_prompts_from_factor]   target={target_prompt[:50]}...")

        return source_prompt, target_prompt, interpolation_weight, None
    
    def set_enabled(self, enabled):
        """Enable or disable the scheduler"""
        self.enabled = enabled
        if self.debug:
            print(f"[PromptScheduler] Enabled set to: {enabled}")
        if self.logging_enabled:
            self.logger.info(f"Enabled set to: {enabled}")
        
    def set_debug(self, debug):
        """Enable or disable debug output"""
        self.debug = debug
        if self.logging_enabled:
            self.logger.info(f"Debug set to: {debug}")
        
    def set_loop_prompts(self, loop):
        """Set whether to loop back to the beginning when reaching the end"""
        self.loop_prompts = loop
        if self.debug:
            print(f"[PromptScheduler] Loop prompts set to: {loop}")
        if self.logging_enabled:
            self.logger.info(f"Loop prompts set to: {loop}")
            
    def reload_prompts(self):
        """
        Reload prompts from the current prompts file.
        """
        if self.debug:
            print("[PromptScheduler] Reloading prompts")
        if self.logging_enabled:
            self.logger.info("Reloading prompts")
        self.load_prompts()
        
    def update_prompts_file_names(self, prompts_file_names):
        """
        Update the prompts file names and reload prompts.

        Args:
            prompts_file_names (list): New list of prompts file names
        """
        if self.debug:
            print(f"[PromptScheduler] Updating prompts file names from {self.prompts_file_names} to {prompts_file_names}")
        if self.logging_enabled:
            self.logger.info(f"Updating prompts file names from {self.prompts_file_names} to {prompts_file_names}")

        self.prompts_file_names = prompts_file_names if prompts_file_names else []
        self.reload_prompts()

    def reset(self):
        """Reset the scheduler to initial state"""
        if len(self.prompts) >= 2:
            self.current_prompt = self.prompts[0]
            self.next_prompt = self.prompts[1]
            self.current_index = 0
            self.progress_ratio = 0.0
        elif len(self.prompts) == 1:
            self.current_prompt = self.prompts[0]
            self.next_prompt = self.prompts[0]
            self.current_index = 0
            self.progress_ratio = 0.0

        if self.debug:
            print(f"[PromptScheduler] Reset to initial prompts: source={self.current_prompt}, target={self.next_prompt}")
        if self.logging_enabled:
            self.logger.info(f"Reset to initial prompts: source={self.current_prompt}, target={self.next_prompt}")
            
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