from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QLabel, QPushButton, QGroupBox, QSlider, QSpinBox, QGridLayout)
from PySide6.QtCore import Qt, Signal, QPoint, QRect
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QBrush, QMouseEvent
import numpy as np
import cv2
import json
import os
from .fullscreen_window import FullscreenWindow
from .mesh_warp import MeshWarp


class InteractiveImageLabel(QLabel):
    """Custom QLabel that allows dragging mesh grid vertices"""

    vertex_moved = Signal(int, int, float, float)  # row, col, x_percent, y_percent

    def __init__(self, mesh_warp: MeshWarp):
        super().__init__()
        self.mesh_warp = mesh_warp
        self.dragging_vertex = None  # (row, col) of vertex being dragged
        self.handle_radius = 8
        self.setMouseTracking(True)

    def set_mesh(self, mesh_warp: MeshWarp):
        """Update mesh reference"""
        self.mesh_warp = mesh_warp
        self.update()

    def paintEvent(self, event):
        """Override paint to draw mesh grid and vertex handles"""
        super().paintEvent(event)

        if not self.pixmap() or not self.mesh_warp:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Get the actual displayed image rect (accounting for scaling)
        pixmap_rect = self._get_scaled_pixmap_rect()
        if pixmap_rect.isEmpty():
            return

        grid_size = self.mesh_warp.grid_size

        # Draw horizontal grid lines
        painter.setPen(QPen(QColor(0, 255, 0, 80), 1, Qt.SolidLine))
        for row in range(grid_size + 1):
            for col in range(grid_size):
                p1 = self.mesh_warp.get_point(row, col)
                p2 = self.mesh_warp.get_point(row, col + 1)

                x1 = pixmap_rect.x() + (p1[0] / 100.0) * pixmap_rect.width()
                y1 = pixmap_rect.y() + (p1[1] / 100.0) * pixmap_rect.height()
                x2 = pixmap_rect.x() + (p2[0] / 100.0) * pixmap_rect.width()
                y2 = pixmap_rect.y() + (p2[1] / 100.0) * pixmap_rect.height()

                painter.drawLine(int(x1), int(y1), int(x2), int(y2))

        # Draw vertical grid lines
        for col in range(grid_size + 1):
            for row in range(grid_size):
                p1 = self.mesh_warp.get_point(row, col)
                p2 = self.mesh_warp.get_point(row + 1, col)

                x1 = pixmap_rect.x() + (p1[0] / 100.0) * pixmap_rect.width()
                y1 = pixmap_rect.y() + (p1[1] / 100.0) * pixmap_rect.height()
                x2 = pixmap_rect.x() + (p2[0] / 100.0) * pixmap_rect.width()
                y2 = pixmap_rect.y() + (p2[1] / 100.0) * pixmap_rect.height()

                painter.drawLine(int(x1), int(y1), int(x2), int(y2))

        # Draw vertex handles
        for row in range(grid_size + 1):
            for col in range(grid_size + 1):
                point = self.mesh_warp.get_point(row, col)
                x = pixmap_rect.x() + (point[0] / 100.0) * pixmap_rect.width()
                y = pixmap_rect.y() + (point[1] / 100.0) * pixmap_rect.height()

                # Highlight if being dragged
                if self.dragging_vertex == (row, col):
                    painter.setPen(QPen(QColor(255, 255, 0), 3))
                    painter.setBrush(QBrush(QColor(255, 255, 0, 200)))
                # Highlight corner vertices differently
                elif (row == 0 or row == grid_size) and (col == 0 or col == grid_size):
                    painter.setPen(QPen(QColor(0, 200, 255), 2))
                    painter.setBrush(QBrush(QColor(0, 200, 255, 150)))
                else:
                    painter.setPen(QPen(QColor(0, 255, 0), 2))
                    painter.setBrush(QBrush(QColor(0, 255, 0, 120)))

                painter.drawEllipse(QPoint(int(x), int(y)), self.handle_radius, self.handle_radius)

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
        """Handle mouse press to start dragging a vertex"""
        if event.button() != Qt.LeftButton or not self.mesh_warp:
            return

        pixmap_rect = self._get_scaled_pixmap_rect()
        if pixmap_rect.isEmpty():
            return

        grid_size = self.mesh_warp.grid_size

        # Check if click is near any vertex
        for row in range(grid_size + 1):
            for col in range(grid_size + 1):
                point = self.mesh_warp.get_point(row, col)
                x = pixmap_rect.x() + (point[0] / 100.0) * pixmap_rect.width()
                y = pixmap_rect.y() + (point[1] / 100.0) * pixmap_rect.height()

                distance = ((event.position().x() - x) ** 2 + (event.position().y() - y) ** 2) ** 0.5
                if distance <= self.handle_radius + 5:
                    self.dragging_vertex = (row, col)
                    self.update()
                    return

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move to drag vertex"""
        if not self.mesh_warp:
            return

        if self.dragging_vertex is None:
            # Update cursor if hovering over handle
            pixmap_rect = self._get_scaled_pixmap_rect()
            if not pixmap_rect.isEmpty():
                grid_size = self.mesh_warp.grid_size
                for row in range(grid_size + 1):
                    for col in range(grid_size + 1):
                        point = self.mesh_warp.get_point(row, col)
                        x = pixmap_rect.x() + (point[0] / 100.0) * pixmap_rect.width()
                        y = pixmap_rect.y() + (point[1] / 100.0) * pixmap_rect.height()

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

        # Update vertex in mesh
        row, col = self.dragging_vertex
        self.mesh_warp.set_point(row, col, x_percent, y_percent)
        self.update()

        # Emit signal
        self.vertex_moved.emit(row, col, x_percent, y_percent)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release to stop dragging"""
        if event.button() == Qt.LeftButton:
            self.dragging_vertex = None
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

        # Initialize mesh warp system (4x4 grid by default)
        self.mesh_warp = MeshWarp(grid_size=4)

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

        # Debounce timer for mesh updates
        from PySide6.QtCore import QTimer
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.apply_transform)
        self.is_dragging = False

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

        self.image_label = InteractiveImageLabel(self.mesh_warp)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: black;")
        self.image_label.setMinimumSize(640, 480)
        self.image_label.vertex_moved.connect(self.on_vertex_dragged)
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

        # Grid resolution controls
        grid_group = QGroupBox("Mesh Grid Settings")
        grid_layout = QGridLayout()

        grid_layout.addWidget(QLabel("Grid Resolution:"), 0, 0)

        self.grid_size_spinbox = QSpinBox()
        self.grid_size_spinbox.setRange(2, 10)
        self.grid_size_spinbox.setValue(self.mesh_warp.grid_size)
        self.grid_size_spinbox.setSuffix(" x " + str(self.mesh_warp.grid_size))
        self.grid_size_spinbox.valueChanged.connect(self.on_grid_size_changed)
        grid_layout.addWidget(self.grid_size_spinbox, 0, 1)

        grid_layout.addWidget(QLabel("(Drag vertices to warp)"), 1, 0, 1, 2)

        grid_group.setLayout(grid_layout)
        right_layout.addWidget(grid_group)

        # Quick adjust buttons
        quick_group = QGroupBox("Quick Adjustments")
        quick_layout = QVBoxLayout()

        reset_button = QPushButton("Reset Mesh to Default")
        reset_button.clicked.connect(self.reset_mesh)
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

    def on_vertex_dragged(self, row, col, x_percent, y_percent):
        """Handle vertex being dragged on the image"""
        # Vertex is already updated in mesh by InteractiveImageLabel
        # Debounce: only update after dragging stops for 500ms (building is slow)
        self.update_timer.stop()
        self.update_timer.start(500)

    def on_grid_size_changed(self, new_size):
        """Handle grid resolution change"""
        self.mesh_warp.set_grid_size(new_size)
        self.grid_size_spinbox.setSuffix(" x " + str(new_size))
        self.image_label.update()
        self.apply_transform()

    def reset_mesh(self):
        """Reset mesh to default grid"""
        self.mesh_warp.reset_to_default()
        self.image_label.update()
        self.apply_transform()

    def update_frame(self, frame: np.ndarray):
        """Update the display with a new frame"""
        if frame is None:
            return

        self.current_frame = frame.copy()
        self.apply_transform()

    def apply_transform(self):
        """Apply mesh-based warp to the current frame"""
        if self.current_frame is None:
            return

        frame = self.current_frame

        # Apply mesh warp
        try:
            self.transformed_frame = self.mesh_warp.apply_warp(frame)

            # Display the transformed frame in mapper window
            self.display_frame(self.transformed_frame)

            # Also update fullscreen window if active (for multi-screen setups)
            if self.fullscreen_window and self.is_fullscreen:
                self.fullscreen_window.update_frame(self.transformed_frame)
        except Exception as e:
            print(f"[ProjectionMapper] Error applying mesh warp: {e}")
            import traceback
            traceback.print_exc()
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
        """Save mesh configuration to JSON file"""
        try:
            config = self.mesh_warp.to_dict()
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"[ProjectionMapper] Configuration saved to {self.config_path}")
        except Exception as e:
            print(f"[ProjectionMapper] Error saving configuration: {e}")

    def load_config(self):
        """Load mesh configuration from JSON file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                self.mesh_warp.from_dict(config)
                print(f"[ProjectionMapper] Configuration loaded from {self.config_path}")

                # Update UI if spinbox exists
                if hasattr(self, 'grid_size_spinbox'):
                    self.grid_size_spinbox.setValue(self.mesh_warp.grid_size)
                    self.grid_size_spinbox.setSuffix(" x " + str(self.mesh_warp.grid_size))

                # Update interactive label if it exists
                if hasattr(self, 'image_label'):
                    self.image_label.set_mesh(self.mesh_warp)
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
