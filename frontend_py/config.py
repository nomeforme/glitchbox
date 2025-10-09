"""
Configuration settings for the Glitch Machine Engine.
"""

# Display and camera dimensions
DISPLAY_WIDTH = 640#480
DISPLAY_HEIGHT = 360#480 
DISPLAY_SCALE = 4.0 #4.0
CAMERA_DEVICE_INDEX = 0 #0 #42
MIC_DEVICE_INDEX = 0 #16

# Audio settings
NUM_FFT_BINS = 50
FFT_WINDOW_SIZE_MS = 60
SMOOTHING_LENGTH_MS = 1000 #50
EQUALIZER_STRENGTH = 0.10 #0.20
ROLLING_STATS_WINDOW_S = 20

# FFT frequency range settings
FFT_FREQ_START_IDX = 0  # Starting index for frequency range
FFT_FREQ_END_IDX = None  # Ending index for frequency range (None means use all bins)

# UI behavior settings
AUTO_DISABLE_BLACK_FRAME_AFTER_CURATION_UPDATE = True  # Automatically disable black frame mode after successful curation index update
FORCE_MANUAL_RECONNECTION_AFTER_CURATION_UPDATE = True  # Force manual reconnection after curation index update
BLACK_FRAME_DISABLE_TIMEOUT = 0 #75

# Automatic curation index update settings
CURATION_INDEX_AUTO_UPDATE = False  # Enable automatic curation index updates
CURATION_INDEX_UPDATE_TIME = 120  # Update interval in seconds (1 hour = 3600 seconds)
CURATION_INDEX_MAX = 9#Maximum curation index value (0 to this value)
