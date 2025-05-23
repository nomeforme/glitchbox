from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
import numpy as np

class FullscreenWindow(QMainWindow):
    """A detachable window that displays just the output image"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Output Display")
        # Remove WindowStaysOnTopHint and add normal window flags
        self.setWindowFlags(Qt.Window)
        
        # Set minimum size to prevent window from becoming too small
        self.setMinimumSize(480, 360)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: black;")
        layout.addWidget(self.image_label)
        
        # Set initial size
        self.resize(854, 480)
        
    def update_frame(self, frame: np.ndarray):
        """Update the display with a new frame"""
        if frame is None:
            return
            
        height, width = frame.shape[:2]
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # Scale the image to fit the window while maintaining aspect ratio
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        self.image_label.setPixmap(scaled_pixmap)
        
    def clear_display(self):
        """Clear the display"""
        self.image_label.clear()
        
    def keyPressEvent(self, event):
        """Handle key press events"""
        if event.key() == Qt.Key_Escape:
            self.close()
        elif event.key() == Qt.Key_F11 or event.key() == Qt.Key_F:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
        super().keyPressEvent(event)
        
    def resizeEvent(self, event):
        """Handle window resize events"""
        super().resizeEvent(event)
        # If we have a current image, rescale it to fit the new window size
        if self.image_label.pixmap():
            pixmap = self.image_label.pixmap()
            scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap) 