from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QLabel, QPushButton, QGroupBox, QSlider, QSpinBox, QGridLayout)
from PySide6.QtCore import Qt, Signal, QPoint, QRect
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QBrush, QMouseEvent
import numpy as np
import cv2
import json
import os
from .fullscreen_window import FullscreenWindow


class InteractiveImageLabel(QLabel):
    """Custom QLabel that allows dragging corner points for keystone correction"""

    corner_moved = Signal(int, float, float)  # corner_idx, x_percent, y_percent

    def __init__(self):
        super().__init__()
        self.corner_points = [[0, 0], [100, 0], [100, 100], [0, 100]]
        self.dragging_corner = None
        self.handle_radius = 10
        self.setMouseTracking(True)

    def set_corners(self, corners):
        """Update corner points"""
        self.corner_points = corners
        self.update()

    def paintEvent(self, event):
        """Override paint to draw corner handles"""
        super().paintEvent(event)

        if not self.pixmap():
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Get the actual displayed image rect (accounting for scaling)
        pixmap_rect = self._get_scaled_pixmap_rect()
        if pixmap_rect.isEmpty():
            return

        # Draw corner handles
        for i, corner in enumerate(self.corner_points):
            # Convert percentage to pixel position on the displayed image
            x = pixmap_rect.x() + (corner[0] / 100.0) * pixmap_rect.width()
            y = pixmap_rect.y() + (corner[1] / 100.0) * pixmap_rect.height()

            # Draw handle
            if i == self.dragging_corner:
                painter.setPen(QPen(QColor(255, 255, 0), 3))
                painter.setBrush(QBrush(QColor(255, 255, 0, 180)))
            else:
                painter.setPen(QPen(QColor(0, 255, 0), 2))
                painter.setBrush(QBrush(QColor(0, 255, 0, 150)))

            painter.drawEllipse(QPoint(int(x), int(y)), self.handle_radius, self.handle_radius)

            # Draw label
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            labels = ["TL", "TR", "BR", "BL"]
            painter.drawText(int(x) - 10, int(y) - 15, labels[i])

        # Draw lines connecting corners
        painter.setPen(QPen(QColor(0, 255, 0, 100), 1, Qt.DashLine))
        for i in range(4):
            next_i = (i + 1) % 4
            x1 = pixmap_rect.x() + (self.corner_points[i][0] / 100.0) * pixmap_rect.width()
            y1 = pixmap_rect.y() + (self.corner_points[i][1] / 100.0) * pixmap_rect.height()
            x2 = pixmap_rect.x() + (self.corner_points[next_i][0] / 100.0) * pixmap_rect.width()
            y2 = pixmap_rect.y() + (self.corner_points[next_i][1] / 100.0) * pixmap_rect.height()
            painter.drawLine(int(x1), int(y1), int(x2), int(y2))

    def _get_scaled_pixmap_rect(self):
        """Get the rectangle where the scaled pixmap is actually drawn"""
        if not self.pixmap():
            return QRect()

        pixmap = self.pixmap()
        label_size = self.size()

        # Calculate scaled size maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # Calculate position (centered)
        x = (label_size.width() - scaled_pixmap.width()) // 2
        y = (label_size.height() - scaled_pixmap.height()) // 2

        return QRect(x, y, scaled_pixmap.width(), scaled_pixmap.height())

    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press to start dragging a corner"""
        if event.button() != Qt.LeftButton:
            return

        pixmap_rect = self._get_scaled_pixmap_rect()
        if pixmap_rect.isEmpty():
            return

        # Check if click is near any corner
        for i, corner in enumerate(self.corner_points):
            x = pixmap_rect.x() + (corner[0] / 100.0) * pixmap_rect.width()
            y = pixmap_rect.y() + (corner[1] / 100.0) * pixmap_rect.height()

            distance = ((event.position().x() - x) ** 2 + (event.position().y() - y) ** 2) ** 0.5
            if distance <= self.handle_radius + 5:
                self.dragging_corner = i
                self.update()
                return

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move to drag corner"""
        if self.dragging_corner is None:
            # Update cursor if hovering over handle
            pixmap_rect = self._get_scaled_pixmap_rect()
            if not pixmap_rect.isEmpty():
                for corner in self.corner_points:
                    x = pixmap_rect.x() + (corner[0] / 100.0) * pixmap_rect.width()
                    y = pixmap_rect.y() + (corner[1] / 100.0) * pixmap_rect.height()

                    distance = ((event.position().x() - x) ** 2 + (event.position().y() - y) ** 2) ** 0.5
                    if distance <= self.handle_radius + 5:
                        self.setCursor(Qt.PointingHandCursor)
                        return
            self.setCursor(Qt.ArrowCursor)
            return

        pixmap_rect = self._get_scaled_pixmap_rect()
        if pixmap_rect.isEmpty():
            return

        # Convert mouse position to percentage
        x_percent = ((event.position().x() - pixmap_rect.x()) / pixmap_rect.width()) * 100
        y_percent = ((event.position().y() - pixmap_rect.y()) / pixmap_rect.height()) * 100

        # Clamp to reasonable bounds
        x_percent = max(-50, min(150, x_percent))
        y_percent = max(-50, min(150, y_percent))

        # Update corner
        self.corner_points[self.dragging_corner] = [x_percent, y_percent]
        self.update()

        # Emit signal
        self.corner_moved.emit(self.dragging_corner, x_percent, y_percent)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release to stop dragging"""
        if event.button() == Qt.LeftButton:
            self.dragging_corner = None
            self.update()


class ProjectionMapperWindow(QMainWindow):
    """A window for projection mapping with trapezoidal/keystone correction"""

    window_closed = Signal()  # Signal emitted when window is closed

    def __init__(self, config_path=None):
        super().__init__()
        self.setWindowTitle("Projection Mapper")

        # Configuration
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "..", "projection_config.json")
        self.config_path = config_path

        # Store the current frame
        self.current_frame = None
        self.transformed_frame = None

        # Initialize corner points (as percentages of image size: 0-100)
        # Format: [top-left, top-right, bottom-right, bottom-left]
        self.corner_points = [
            [0, 0],      # Top-left
            [100, 0],    # Top-right
            [100, 100],  # Bottom-right
            [0, 100]     # Bottom-left
        ]

        # Load saved configuration if it exists
        self.load_config()

        # Set window size
        self.setMinimumSize(800, 600)
        self.resize(1024, 768)

        # Create UI
        self.setup_ui()

        # Fullscreen window
        self.fullscreen_window = None
        self.is_fullscreen = False

    def setup_ui(self):
        """Setup the user interface"""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left side - Image display
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.image_label = InteractiveImageLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: black;")
        self.image_label.setMinimumSize(640, 480)
        self.image_label.set_corners(self.corner_points)
        self.image_label.corner_moved.connect(self.on_corner_dragged)
        left_layout.addWidget(self.image_label)

        # Fullscreen button
        self.fullscreen_button = QPushButton("Go Fullscreen")
        self.fullscreen_button.clicked.connect(self.toggle_fullscreen)
        self.fullscreen_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        left_layout.addWidget(self.fullscreen_button)

        main_layout.addWidget(left_widget, stretch=3)

        # Right side - Controls
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # Corner controls
        corner_group = QGroupBox("Corner Adjustment")
        corner_layout = QGridLayout()

        self.corner_spinboxes = []
        corner_labels = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]

        for i, label in enumerate(corner_labels):
            # Label
            corner_layout.addWidget(QLabel(f"{label}:"), i, 0)

            # X coordinate
            x_label = QLabel("X:")
            corner_layout.addWidget(x_label, i, 1)

            x_spinbox = QSpinBox()
            x_spinbox.setRange(-50, 150)
            x_spinbox.setValue(int(self.corner_points[i][0]))
            x_spinbox.setSuffix("%")
            x_spinbox.valueChanged.connect(lambda val, idx=i, coord=0: self.update_corner(idx, coord, val))
            corner_layout.addWidget(x_spinbox, i, 2)

            # Y coordinate
            y_label = QLabel("Y:")
            corner_layout.addWidget(y_label, i, 3)

            y_spinbox = QSpinBox()
            y_spinbox.setRange(-50, 150)
            y_spinbox.setValue(int(self.corner_points[i][1]))
            y_spinbox.setSuffix("%")
            y_spinbox.valueChanged.connect(lambda val, idx=i, coord=1: self.update_corner(idx, coord, val))
            corner_layout.addWidget(y_spinbox, i, 4)

            self.corner_spinboxes.append((x_spinbox, y_spinbox))

        corner_group.setLayout(corner_layout)
        right_layout.addWidget(corner_group)

        # Quick adjust buttons
        quick_group = QGroupBox("Quick Adjustments")
        quick_layout = QVBoxLayout()

        reset_button = QPushButton("Reset to Default")
        reset_button.clicked.connect(self.reset_corners)
        quick_layout.addWidget(reset_button)

        save_button = QPushButton("Save Configuration")
        save_button.clicked.connect(self.save_config)
        quick_layout.addWidget(save_button)

        load_button = QPushButton("Load Configuration")
        load_button.clicked.connect(self.load_config)
        quick_layout.addWidget(load_button)

        quick_group.setLayout(quick_layout)
        right_layout.addWidget(quick_group)

        right_layout.addStretch()

        main_layout.addWidget(right_widget, stretch=1)

    def update_corner(self, corner_idx, coord_idx, value):
        """Update a corner point coordinate from spinbox"""
        self.corner_points[corner_idx][coord_idx] = value
        self.image_label.set_corners(self.corner_points)
        self.apply_transform()

    def on_corner_dragged(self, corner_idx, x_percent, y_percent):
        """Handle corner being dragged on the image"""
        # Update the corner points
        self.corner_points[corner_idx] = [x_percent, y_percent]

        # Update spinboxes without triggering their signals
        x_spinbox, y_spinbox = self.corner_spinboxes[corner_idx]
        x_spinbox.blockSignals(True)
        y_spinbox.blockSignals(True)
        x_spinbox.setValue(int(x_percent))
        y_spinbox.setValue(int(y_percent))
        x_spinbox.blockSignals(False)
        y_spinbox.blockSignals(False)

        # Apply transform
        self.apply_transform()

    def reset_corners(self):
        """Reset corners to default positions"""
        self.corner_points = [
            [0, 0],      # Top-left
            [100, 0],    # Top-right
            [100, 100],  # Bottom-right
            [0, 100]     # Bottom-left
        ]

        # Update UI
        for i, (x_spinbox, y_spinbox) in enumerate(self.corner_spinboxes):
            x_spinbox.setValue(int(self.corner_points[i][0]))
            y_spinbox.setValue(int(self.corner_points[i][1]))

        # Update interactive label
        self.image_label.set_corners(self.corner_points)
        self.apply_transform()

    def update_frame(self, frame: np.ndarray):
        """Update the display with a new frame"""
        if frame is None:
            return

        self.current_frame = frame.copy()
        self.apply_transform()

    def apply_transform(self):
        """Apply perspective transform to the current frame"""
        if self.current_frame is None:
            return

        frame = self.current_frame
        height, width = frame.shape[:2]

        # Convert corner percentages to pixel coordinates
        src_points = np.float32([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ])

        dst_points = np.float32([
            [self.corner_points[0][0] * width / 100, self.corner_points[0][1] * height / 100],
            [self.corner_points[1][0] * width / 100, self.corner_points[1][1] * height / 100],
            [self.corner_points[2][0] * width / 100, self.corner_points[2][1] * height / 100],
            [self.corner_points[3][0] * width / 100, self.corner_points[3][1] * height / 100]
        ])

        # Calculate perspective transform matrix
        try:
            matrix = cv2.getPerspectiveTransform(src_points, dst_points)

            # Apply transform
            self.transformed_frame = cv2.warpPerspective(frame, matrix, (width, height))

            # Display the transformed frame in mapper window
            self.display_frame(self.transformed_frame)

            # Also update fullscreen window if active (for multi-screen setups)
            if self.fullscreen_window and self.is_fullscreen:
                self.fullscreen_window.update_frame(self.transformed_frame)
        except Exception as e:
            print(f"[ProjectionMapper] Error applying transform: {e}")
            # If transform fails, display original frame
            self.display_frame(frame)

            # Also update fullscreen window with original frame if active
            if self.fullscreen_window and self.is_fullscreen:
                self.fullscreen_window.update_frame(frame)

    def display_frame(self, frame: np.ndarray):
        """Display a frame in the image label"""
        if frame is None:
            return

        height, width = frame.shape[:2]
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)

        # Scale to fit window
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

        self.image_label.setPixmap(scaled_pixmap)

    def toggle_fullscreen(self):
        """Toggle fullscreen output window"""
        if not self.is_fullscreen:
            # Create and show fullscreen window
            if not self.fullscreen_window:
                self.fullscreen_window = FullscreenWindow()
                # Connect close signal to update button state
                self.fullscreen_window.window_closed.connect(self.on_fullscreen_closed)

            self.fullscreen_window.show()
            self.fullscreen_window.showFullScreen()

            # Update the fullscreen window with current transformed frame if available
            if self.transformed_frame is not None:
                self.fullscreen_window.update_frame(self.transformed_frame)

            self.is_fullscreen = True
            self.fullscreen_button.setText("Exit Fullscreen")
        else:
            # Close fullscreen window
            if self.fullscreen_window:
                self.fullscreen_window.close()
                self.fullscreen_window = None
            self.is_fullscreen = False
            self.fullscreen_button.setText("Go Fullscreen")

    def on_fullscreen_closed(self):
        """Handle fullscreen window being closed via X button"""
        self.fullscreen_window = None
        self.is_fullscreen = False
        self.fullscreen_button.setText("Go Fullscreen")

    def save_config(self):
        """Save corner configuration to JSON file"""
        try:
            config = {
                "corner_points": self.corner_points
            }
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"[ProjectionMapper] Configuration saved to {self.config_path}")
        except Exception as e:
            print(f"[ProjectionMapper] Error saving configuration: {e}")

    def load_config(self):
        """Load corner configuration from JSON file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                self.corner_points = config.get("corner_points", self.corner_points)
                print(f"[ProjectionMapper] Configuration loaded from {self.config_path}")

                # Update UI if spinboxes exist
                if hasattr(self, 'corner_spinboxes'):
                    for i, (x_spinbox, y_spinbox) in enumerate(self.corner_spinboxes):
                        x_spinbox.setValue(int(self.corner_points[i][0]))
                        y_spinbox.setValue(int(self.corner_points[i][1]))

                # Update interactive label if it exists
                if hasattr(self, 'image_label'):
                    self.image_label.set_corners(self.corner_points)
        except Exception as e:
            print(f"[ProjectionMapper] Error loading configuration: {e}")

    def clear_display(self):
        """Clear the display"""
        self.image_label.clear()
        self.current_frame = None
        self.transformed_frame = None

        # Clear fullscreen window if active
        if self.fullscreen_window:
            self.fullscreen_window.clear_display()

    def keyPressEvent(self, event):
        """Handle key press events"""
        if event.key() == Qt.Key_Escape:
            self.close()
        elif event.key() == Qt.Key_F11 or event.key() == Qt.Key_F:
            self.toggle_fullscreen()
        super().keyPressEvent(event)

    def closeEvent(self, event):
        """Handle window close event"""
        # Close fullscreen window if open
        if self.fullscreen_window:
            self.fullscreen_window.close()
            self.fullscreen_window = None

        # Emit signal to notify parent
        self.window_closed.emit()
        super().closeEvent(event)

    def resizeEvent(self, event):
        """Handle window resize events"""
        super().resizeEvent(event)
        # Redisplay the current frame when window is resized
        if self.transformed_frame is not None:
            self.display_frame(self.transformed_frame)
