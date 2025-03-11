"""
Test oscillator classes for visual effects like zoom and shift.
These are primarily used for testing and development purposes.
"""

class ZoomOscillator:
    """
    A class that handles zoom oscillation effects for testing.
    """
    def __init__(self, 
                 min_zoom=0.5, 
                 max_zoom=1.5, 
                 zoom_increment=0.03,
                 stabilize_duration=3, 
                 enabled=True,
                 debug=False):
        """
        Initialize the zoom oscillator.
        
        Args:
            min_zoom (float): Minimum zoom value (default: 0.5)
            max_zoom (float): Maximum zoom value (default: 1.5)
            zoom_increment (float): Amount to change zoom per update (default: 0.03)
            stabilize_duration (int): Number of iterations to pause at zoom=1.0 (default: 3)
            enabled (bool): Whether oscillation is active (default: True)
            debug (bool): Whether to print debug messages (default: False)
        """
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom
        self.zoom_increment = zoom_increment
        self.stabilize_duration = stabilize_duration
        self.enabled = enabled
        self.debug = debug
        
        # Internal state
        self.zoom_value = 1.0
        self.zoom_direction = 1  # 1 for increasing, -1 for decreasing
        self.stabilize_counter = 0  # Counter for stabilization pause
    
    def update(self):
        """
        Update the zoom value for the next iteration.
        
        Returns:
            float: The current zoom value
        """
        if not self.enabled:
            return self.zoom_value
            
        # Check if we're in stabilization pause at zoom=1.0
        if abs(self.zoom_value - 1.0) < 0.01 and self.stabilize_counter < self.stabilize_duration:
            # Hold at zoom=1.0 for stabilize_duration iterations
            self.stabilize_counter += 1
            self.zoom_value = 1.0  # Ensure we're exactly at 1.0
            
            if self.debug and self.stabilize_counter == 1:
                print(f"[ZoomOscillator] Stabilizing at zoom=1.0 ({self.stabilize_counter}/{self.stabilize_duration})")
        else:
            if abs(self.zoom_value - 1.0) < 0.01 and self.stabilize_counter >= self.stabilize_duration:
                # We've finished the stabilization period, continue with oscillation
                self.stabilize_counter = 0
                
            # Update zoom for next iteration
            self.zoom_value += self.zoom_direction * self.zoom_increment
            
            # Reverse direction if we hit limits
            if self.zoom_value >= self.max_zoom:
                self.zoom_direction = -1
            elif self.zoom_value <= self.min_zoom:
                self.zoom_direction = 1
        
        if self.debug:
            print(f"[ZoomOscillator] Zoom value: {self.zoom_value:.2f}")
            
        return self.zoom_value
        
    def set_enabled(self, enabled):
        """Enable or disable the oscillator"""
        self.enabled = enabled
        
    def set_debug(self, debug):
        """Enable or disable debug output"""
        self.debug = debug


class ShiftOscillator:
    """
    A class that handles shift oscillation effects for testing.
    """
    def __init__(self, 
                 x_max=50,
                 y_max=50, 
                 x_increment=0, 
                 y_increment=0, 
                 enabled=True,
                 debug=False):
        """
        Initialize the shift oscillator.
        
        Args:
            x_max (int): Maximum horizontal shift in pixels (default: 50)
            y_max (int): Maximum vertical shift in pixels (default: 50)
            x_increment (int): Horizontal shift per update (default: 0)
            y_increment (int): Vertical shift per update (default: 0)
            enabled (bool): Whether oscillation is active (default: True)
            debug (bool): Whether to print debug messages (default: False)
        """
        self.x_max = x_max
        self.y_max = y_max
        self.x_increment = x_increment
        self.y_increment = y_increment
        self.enabled = enabled
        self.debug = debug
        
        # Internal state
        self.x_value = 0
        self.y_value = 0
        self.x_direction = 1  # 1 for right, -1 for left
        self.y_direction = 1  # 1 for down, -1 for up
    
    def update(self):
        """
        Update the shift values for the next iteration.
        
        Returns:
            tuple: (x_shift, y_shift) current shift values
        """
        if not self.enabled:
            return self.x_value, self.y_value

        # Update x_shift for next iteration
        self.x_value += self.x_direction * self.x_increment
        
        # Reverse x_shift direction if we hit limits
        if abs(self.x_value) >= self.x_max:
            self.x_direction *= -1
        
        # Update y_shift for next iteration
        self.y_value += self.y_direction * self.y_increment
        
        # Reverse y_shift direction if we hit limits
        if abs(self.y_value) >= self.y_max:
            self.y_direction *= -1
        
        if self.debug:
            print(f"[ShiftOscillator] Shift values: x={self.x_value}, y={self.y_value}")
            
        return self.x_value, self.y_value
        
    def set_enabled(self, enabled):
        """Enable or disable the oscillator"""
        self.enabled = enabled

    def set_debug(self, debug):
        """Enable or disable debug output"""
        self.debug = debug
        
    def set_increments(self, x_increment=None, y_increment=None):
        """Set the shift increments"""
        if x_increment is not None:
            self.x_increment = x_increment
        if y_increment is not None:
            self.y_increment = y_increment