from PySide6.QtWidgets import QLabel, QWidget, QVBoxLayout
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
import numpy as np

class ProcessedDisplay(QWidget):
    """Widget to display processed image output"""
    
    def __init__(self, min_size=(640, 480)):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        # Image display
        self.image_label = QLabel()
        self.image_label.setMinimumSize(*min_size)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label)

    def update_frame(self, frame: np.ndarray):
        """Update the display with a new processed frame"""
        if frame is None:
            return
            
        height, width = frame.shape[:2]
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_image).scaled(
            640, 480, Qt.KeepAspectRatio))

    def clear_display(self):
        """Clear the display"""
        self.image_label.clear()