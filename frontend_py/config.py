"""
Configuration settings for the Glitch Machine Engine.
"""

# Display and camera dimensions
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 360 
DISPLAY_SCALE = 1.0
CAMERA_DEVICE_INDEX = 0 #0 #42
MIC_DEVICE_INDEX = 4 #16

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
AUTO_DISABLE_BLACK_FRAME_AFTER_CURATION_UPDATE = False  # Automatically disable black frame mode after successful curation index update
