from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel
from PySide6.QtCore import Qt

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

    def update_processing_status(self, status: str):
        """Update processing status message"""
        self.proc_status.setText(status)