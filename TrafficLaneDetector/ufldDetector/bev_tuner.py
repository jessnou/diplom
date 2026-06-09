from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .perspectiveTransformation import BevConfig
from .utils import CurvatureType, OffsetType


def _bgr_to_qimage(frame_bgr: np.ndarray) -> QImage:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = frame_rgb.shape
    bytes_per_line = ch * w
    return QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)


class BevTunerWidget(QWidget):
    config_changed = Signal()

    def __init__(self, bev_config: BevConfig):
        super().__init__()
        self._cfg = bev_config
        self._suppress_signals = False

        self.setWindowTitle("BEV Tuning")
        self.setMinimumWidth(500)

        layout = QVBoxLayout(self)

        self.image_label = QLabel("(no frame)")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumHeight(200)
        self.image_label.setStyleSheet("QLabel { background: #111; border: 1px solid #444; }")
        layout.addWidget(self.image_label, 1)

        src_group = QGroupBox("Source points (fraction of W,H)")
        src_layout = QVBoxLayout(src_group)
        self._sliders = {}
        src_layout.addLayout(self._make_slider("top_y", self._cfg.src_top_y))
        src_layout.addLayout(self._make_slider("top_L_x", self._cfg.src_top_left_x))
        src_layout.addLayout(self._make_slider("bot_L_x", self._cfg.src_bot_left_x))
        src_layout.addLayout(self._make_slider("bot_R_x", self._cfg.src_bot_right_x))
        src_layout.addLayout(self._make_slider("top_R_x", self._cfg.src_top_right_x))
        layout.addWidget(src_group)

        dst_group = QGroupBox("Destination")
        dst_layout = QVBoxLayout(dst_group)
        dst_layout.addLayout(self._make_slider("dst_offs", self._cfg.dst_offset_x, max_val=50))
        layout.addWidget(dst_group)

        crop_layout = QHBoxLayout()
        crop_layout.addLayout(self._make_slider("crop", self._cfg.hood_crop_ratio, max_val=30))
        layout.addLayout(crop_layout)

        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("BEV mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["dynamic", "static_bottom", "static_default"])
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(self.mode_combo, 1)
        layout.addLayout(mode_layout)

        self.info_label = QLabel("Offset: --  |  Curve: --")
        self.info_label.setStyleSheet("QLabel { color: #0f0; font-size: 13px; }")
        layout.addWidget(self.info_label)

    def _make_slider(self, name: str, default: float, max_val: int = 100) -> QHBoxLayout:
        row = QHBoxLayout()
        label = QLabel(f"{name}:")
        label.setFixedWidth(70)
        row.addWidget(label)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, max_val)
        slider.setValue(int(default * max_val))
        slider.valueChanged.connect(self._on_slider_changed)
        row.addWidget(slider, 1)

        val_label = QLabel(f"{default:.2f}")
        val_label.setFixedWidth(45)
        val_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        row.addWidget(val_label)

        self._sliders[name] = (slider, val_label, max_val)
        return row

    def _on_slider_changed(self):
        if self._suppress_signals:
            return
        self._apply_all()
        self.config_changed.emit()

    def _on_mode_changed(self, idx: int):
        if self._suppress_signals:
            return
        self.config_changed.emit()

    def _apply_all(self):
        for name, (slider, val_label, max_val) in self._sliders.items():
            frac = slider.value() / max_val
            val_label.setText(f"{frac:.2f}")
            setattr(self._cfg, self._field_name(name), frac)

    def _field_name(self, slider_name: str) -> str:
        mapping = {
            "top_y": "src_top_y",
            "top_L_x": "src_top_left_x",
            "bot_L_x": "src_bot_left_x",
            "bot_R_x": "src_bot_right_x",
            "top_R_x": "src_top_right_x",
            "dst_offs": "dst_offset_x",
            "crop": "hood_crop_ratio",
        }
        return mapping[slider_name]

    def _set_all_sliders(self):
        self._suppress_signals = True
        for name, (slider, val_label, max_val) in self._sliders.items():
            val = getattr(self._cfg, self._field_name(name))
            slider.setValue(int(val * max_val))
            val_label.setText(f"{val:.2f}")
        self._suppress_signals = False

    def update_bev_image(self, bev_bgr: np.ndarray):
        h, w = bev_bgr.shape[:2]
        disp_w = min(w, 500)
        disp_h = int(h * disp_w / w)
        resized = cv2.resize(bev_bgr, (disp_w, disp_h))
        qimg = _bgr_to_qimage(resized)
        self.image_label.setPixmap(QPixmap.fromImage(qimg))

    def update_info(self, offset: Optional[OffsetType] = None,
                    curvature: Optional[CurvatureType] = None):
        parts = []
        if offset is not None:
            parts.append(f"Offset: {offset.value}")
        else:
            parts.append("Offset: --")
        if curvature is not None:
            parts.append(f"Curve: {curvature.value}")
        else:
            parts.append("Curve: --")
        self.info_label.setText("  |  ".join(parts))

    def apply_to_config(self) -> None:
        self._apply_all()

    def refresh_from_config(self) -> None:
        self._set_all_sliders()

    def closeEvent(self, event):
        event.accept()
