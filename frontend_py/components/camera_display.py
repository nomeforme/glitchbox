from PySide6.QtWidgets import QLabel
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QImage, QPixmap
import numpy as np

class CameraDisplay(QLabel):
    """Widget to display camera feed"""
    
    def __init__(self, min_size=(640, 480)):
        super().__init__()
        self.setMinimumSize(*min_size)
        self.setAlignment(Qt.AlignCenter)

    def update_frame(self, frame: np.ndarray):
        """Update the display with a new frame"""
        if frame is None:
            return
            
        height, width = frame.shape[:2]
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(q_image).scaled(
            640, 480, Qt.KeepAspectRatio))

    def clear_display(self):
        """Clear the display"""
        super().clear()