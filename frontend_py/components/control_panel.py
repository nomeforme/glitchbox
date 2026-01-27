from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                             QSlider, QCheckBox, QLabel, QLineEdit,
                             QScrollArea, QFrame)
from PySide6.QtCore import Qt, Signal

# Minimum heights for controls
SLIDER_MIN_HEIGHT = 40
CHECKBOX_MIN_HEIGHT = 30
TEXT_INPUT_MIN_HEIGHT = 40

class ControlPanel(QWidget):
    """Widget containing pipeline controls and settings"""

    parameter_changed = Signal(str, object)  # (param_id, value)

    def __init__(self):
        super().__init__()
        self.controls = {}

        # Create outer layout
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        # Create scroll area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setMinimumHeight(300)
        self.scroll_area.setMaximumHeight(500)

        # Create content widget for scroll area
        self.content_widget = QWidget()
        self.main_layout = QVBoxLayout(self.content_widget)
        self.main_layout.setSpacing(8)

        self.scroll_area.setWidget(self.content_widget)
        outer_layout.addWidget(self.scroll_area)

    def setup_pipeline_options(self, settings):
        """Setup UI controls based on received pipeline settings"""
        # Clear existing controls
        self.clear_layout(self.main_layout)
        self.controls = {}
        
        # Extract parameters from settings
        params = settings.get('input_params', {}).get('properties', {})
        default_values = {}
        
        # Get default values from server settings
        for param_id, param in params.items():
            default_values[param_id] = param.get('default', 0)
        
        # Add controls based on parameters
        for param_id, param in params.items():
            field_type = param.get('field')
            default_value = default_values.get(param_id)

            if field_type == 'range':
                self.add_slider(param_id, param, default_value)
            elif field_type == 'checkbox':
                self.add_checkbox(param_id, param, default_value)
            elif field_type == 'textarea':
                self.add_text_input(param_id, param, default_value)

        # Add stretch at the end so controls stay at top
        self.main_layout.addStretch()

    def add_slider(self, param_id, param, default_value=None):
        """Add a slider control"""
        # Create container widget with minimum height
        container = QWidget()
        container.setMinimumHeight(SLIDER_MIN_HEIGHT)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 4, 0, 4)

        # Label with fixed width for alignment
        label = QLabel(param.get('title', param_id))
        label.setMinimumWidth(150)
        layout.addWidget(label)
        
        # Slider
        slider = QSlider(Qt.Horizontal)
        slider.setRange(int(param.get('min', 0) * 100), int(param.get('max', 1) * 100))
        if default_value is not None:
            default_val = int(default_value * 100)
        else:
            default_val = int(param.get('default', 0) * 100)
        slider.setValue(default_val)
        
        # Value display
        value_edit = QLineEdit()
        value_edit.setFixedWidth(50)
        value_edit.setText(str(default_val/100))
        
        # Connect signals
        def on_slider_change(value):
            value_edit.setText(f"{value/100:.2f}")
            self.parameter_changed.emit(param_id, value/100)
            
        def on_text_change():
            try:
                value = float(value_edit.text()) * 100
                slider.setValue(int(value))
                self.parameter_changed.emit(param_id, float(value_edit.text()))
            except ValueError:
                pass
                
        slider.valueChanged.connect(on_slider_change)
        value_edit.editingFinished.connect(on_text_change)

        layout.addWidget(slider)
        layout.addWidget(value_edit)

        self.main_layout.addWidget(container)
        self.controls[param_id] = (slider, value_edit)

    def add_checkbox(self, param_id, param, default_value=None):
        """Add a checkbox control"""
        # Create container widget with minimum height
        container = QWidget()
        container.setMinimumHeight(CHECKBOX_MIN_HEIGHT)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 4, 0, 4)

        checkbox = QCheckBox(param.get('title', param_id))
        if default_value is not None:
            checkbox.setChecked(default_value)
        else:
            checkbox.setChecked(param.get('default', False))

        checkbox.stateChanged.connect(
            lambda v, pid=param_id: self.parameter_changed.emit(pid, bool(v)))
        layout.addWidget(checkbox)
        layout.addStretch()

        self.main_layout.addWidget(container)
        self.controls[param_id] = checkbox

    def add_text_input(self, param_id, param, default_value=None):
        """Add a text input control"""
        # Create container widget with minimum height
        container = QWidget()
        container.setMinimumHeight(TEXT_INPUT_MIN_HEIGHT)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 4, 0, 4)

        # Label with fixed width for alignment
        label = QLabel(param.get('title', param_id))
        label.setMinimumWidth(150)
        layout.addWidget(label)

        text_input = QLineEdit()
        if default_value is not None:
            text_input.setText(str(default_value))
        else:
            text_input.setText(str(param.get('default', '')))

        text_input.textChanged.connect(
            lambda v, pid=param_id: self.parameter_changed.emit(pid, v))
        layout.addWidget(text_input)

        self.main_layout.addWidget(container)
        self.controls[param_id] = text_input

    def clear_layout(self, layout):
        """Recursively clear a layout and its widgets"""
        if layout is None:
            return
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
            elif item.layout():
                self.clear_layout(item.layout())

    def update_control(self, param_id, value):
        """Update a control's value programmatically"""
        if param_id not in self.controls:
            return
            
        control = self.controls[param_id]
        if isinstance(control, tuple):  # Slider
            slider, value_edit = control
            slider.setValue(int(value * 100))
            value_edit.setText(f"{value:.2f}")
        elif isinstance(control, QCheckBox):
            control.setChecked(value)
        elif isinstance(control, QLineEdit):
            control.setText(str(value))