from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel
from PySide6.QtCore import Qt
import time

class StatusBar(QWidget):
    """Status bar showing connection and processing state"""
    
    def __init__(self):
        super().__init__()
        self.layout = QHBoxLayout()
        self.setLayout(self.layout)
        
        # Connection status
        self.conn_status = QLabel("Not Connected")
        self.conn_status.setStyleSheet("color: red;")
        self.layout.addWidget(self.conn_status)
        
        # FPS counter
        self.fps_label = QLabel("0 FPS")
        self.layout.addWidget(self.fps_label)
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.fps = 0
        
        # Processing status 
        self.proc_status = QLabel("")
        self.layout.addWidget(self.proc_status)
        
        # Add stretch to push labels to the left
        self.layout.addStretch()
        
        self.setMaximumHeight(30)

    def update_connection_status(self, connected: bool):
        """Update connection status display"""
        if connected:
            self.conn_status.setText("Connected")
            self.conn_status.setStyleSheet("color: green;")
        else:
            self.conn_status.setText("Not Connected")
            self.conn_status.setStyleSheet("color: red;")

    def update_fps(self):
        """Update FPS counter based on received frames"""
        current_time = time.time()
        self.frame_count += 1
        
        # Update FPS every second
        if current_time - self.last_frame_time >= 1.0:
            self.fps = self.frame_count
            self.fps_label.setText(f"{self.fps} FPS")
            self.frame_count = 0
            self.last_frame_time = current_time

    def _truncate_message(self, message: str, max_length: int = 50) -> str:
        """Truncate message if it's too long"""
        if len(message) > max_length:
            return message[:max_length-3] + "..."
        return message

    def update_processing_status(self, status: str):
        """Update processing status message"""
        truncated_status = self._truncate_message(status)
        self.proc_status.setText(truncated_status)