from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel
from PySide6.QtCore import Qt
import time

class StatusBar(QWidget):
    """Status bar showing connection and processing state"""
    
    def __init__(self):
        super().__init__()
        self.layout = QHBoxLayout()
        self.setLayout(self.layout)
        
        # Left side container
        left_container = QWidget()
        left_layout = QHBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # FPS counter
        self.fps_label = QLabel("0 FPS")
        left_layout.addWidget(self.fps_label)
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.fps = 0
        
        # Small spacing between FPS and connection status
        left_layout.addSpacing(5)
        
        # Connection status
        self.conn_status = QLabel("Not Connected")
        self.conn_status.setStyleSheet("color: red;")
        left_layout.addWidget(self.conn_status)
        
        # Add left container to main layout
        self.layout.addWidget(left_container)
        
        # Add stretch to push server info to the right
        self.layout.addStretch()
        
        # Server info on the right
        self.proc_status = QLabel("")
        self.proc_status.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.layout.addWidget(self.proc_status)
        
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