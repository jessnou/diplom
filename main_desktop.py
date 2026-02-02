import os
import sys
from dataclasses import dataclass
from typing import Optional

import cv2

from adas_pipeline import ADASProcessor
from ObjectDetector.utils import ObjectModelType
from TrafficLaneDetector.ufldDetector.utils import LaneModelType


def _require_pyside6() -> None:
    try:
        import PySide6  # noqa: F401
    except Exception as e:
        raise SystemExit(
            "PySide6 не установлен.\n"
            "Установи зависимости и запусти снова:\n"
            "  pip install -r requirements.txt\n\n"
            f"Причина: {e}\n"
        )


@dataclass
class AppConfig:
    lane_config: dict
    object_config: dict
    save_output: bool = True


def default_config() -> AppConfig:
    lane_config = {
        "model_path": os.path.join(
            os.path.dirname(__file__),
            "TrafficLaneDetector/ufldDetector/exportLib/ultrafastLaneV2/tusimple_res18.onnx",
        ),
        "model_type": LaneModelType.UFLD_TUSIMPLE,
    }
    object_config = {
        "model_path": os.path.join(os.path.dirname(__file__), "yolov8l.onnx"),
        "model_type": ObjectModelType.YOLOV8,
        "classes_path": "./ObjectDetector/models/coco_label.txt",
        "box_score": 0.4,
        "box_nms_iou": 0.5,
    }
    return AppConfig(lane_config=lane_config, object_config=object_config, save_output=True)


def _bgr_to_qimage(frame_bgr):
    from PySide6.QtGui import QImage

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = frame_rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    return qimg


class VideoDropLabel:
    def __init__(self, label):
        from PySide6.QtCore import QObject

        class _Filter(QObject):
            def __init__(self, outer):
                super().__init__()
                self.outer = outer

            def eventFilter(self, obj, event):
                from PySide6.QtCore import QEvent

                if event.type() == QEvent.Type.DragEnter:
                    if event.mimeData().hasUrls():
                        event.acceptProposedAction()
                        return True
                if event.type() == QEvent.Type.Drop:
                    urls = event.mimeData().urls()
                    if urls:
                        path = urls[0].toLocalFile()
                        if self.outer.on_path:
                            self.outer.on_path(path)
                    event.acceptProposedAction()
                    return True
                return False

        self.label = label
        self.on_path = None
        self.label.setAcceptDrops(True)
        self.filter = _Filter(self)


class ADASWindow:
    def __init__(self, cfg: AppConfig):
        from PySide6.QtCore import QTimer, Qt
        from PySide6.QtGui import QPixmap
        from PySide6.QtWidgets import (
            QCheckBox,
            QFileDialog,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QPushButton,
            QVBoxLayout,
            QWidget,
        )

        self.cfg = cfg
        self.processor = ADASProcessor(cfg.lane_config, cfg.object_config, allowed_labels={"person", "car"})

        self.cap: Optional[cv2.VideoCapture] = None
        self.vout: Optional[cv2.VideoWriter] = None
        self.video_path: Optional[str] = None
        self.initialized = False
        self.last_frame_size = None

        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)

        self.root = QWidget()
        self.root.setWindowTitle("ADAS Desktop")
        self.root.resize(1280, 800)

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Перетащи видео сюда или выбери через 'Открыть…'")

        self.open_btn = QPushButton("Открыть…")
        self.open_btn.clicked.connect(self._open_dialog)

        self.start_btn = QPushButton("Старт")
        self.start_btn.clicked.connect(self.start)
        self.stop_btn = QPushButton("Стоп")
        self.stop_btn.clicked.connect(self.stop)
        self.stop_btn.setEnabled(False)

        self.save_cb = QCheckBox("Сохранять *_out.mp4")
        self.save_cb.setChecked(bool(cfg.save_output))

        top = QHBoxLayout()
        top.addWidget(self.path_edit, 1)
        top.addWidget(self.open_btn)
        top.addWidget(self.start_btn)
        top.addWidget(self.stop_btn)
        top.addWidget(self.save_cb)

        self.video_label = QLabel("Drop video here")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumHeight(600)
        self.video_label.setStyleSheet("QLabel { background: #111; color: #bbb; border: 1px dashed #444; }")
        self.video_label.setScaledContents(True)

        self.status = QLabel("Готово")

        layout = QVBoxLayout()
        layout.addLayout(top)
        layout.addWidget(self.video_label, 1)
        layout.addWidget(self.status)
        self.root.setLayout(layout)

        self._pixmap = QPixmap()

        self.drop = VideoDropLabel(self.video_label)
        self.drop.on_path = self.set_video_path
        self.video_label.installEventFilter(self.drop.filter)

    def show(self):
        self.root.show()

    def set_video_path(self, path: str) -> None:
        if not path:
            return
        self.video_path = path
        self.path_edit.setText(path)
        self.status.setText(f"Выбрано: {path}")

    def _open_dialog(self):
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getOpenFileName(
            self.root,
            "Выбери видео",
            os.path.dirname(self.video_path) if self.video_path else os.getcwd(),
            "Video (*.mp4 *.avi *.mkv *.mov *.m4v);;All files (*)",
        )
        if path:
            self.set_video_path(path)

    def start(self):
        if not self.video_path:
            self.status.setText("Сначала выбери видео")
            return

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            self.status.setText("Не удалось открыть видео")
            self.cap = None
            return

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 1e-3:
            fps = 30.0

        frame_size = (width, height)
        if (not self.initialized) or (self.last_frame_size != frame_size):
            self.processor.initialize(frame_size)
            self.initialized = True
            self.last_frame_size = frame_size

        if self.save_cb.isChecked():
            out_path = os.path.splitext(self.video_path)[0] + "_out.mp4"
            fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
            self.vout = cv2.VideoWriter(out_path, fourcc, float(fps), (width, height))
        else:
            self.vout = None

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        interval_ms = max(1, int(1000.0 / float(fps)))
        self.timer.start(interval_ms)
        self.status.setText("Воспроизведение… (q в этом окне не работает, жми 'Стоп')")

    def stop(self):
        self.timer.stop()
        if self.vout is not None:
            self.vout.release()
            self.vout = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status.setText("Остановлено")

    def _tick(self):
        from PySide6.QtGui import QPixmap

        if self.cap is None:
            self.stop()
            return

        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.stop()
            return

        frame_show, metrics = self.processor.process_frame(frame)

        if self.vout is not None:
            self.vout.write(frame_show)

        qimg = _bgr_to_qimage(frame_show)
        pixmap = QPixmap.fromImage(qimg)
        self.video_label.setPixmap(pixmap)

        self.status.setText(
            f"FCWS={metrics.collision.name} | LDWS={metrics.offset.name} | LKAS={metrics.curvature.name} "
            f"| obj={metrics.object_infer_s:.2f}s lane={metrics.lane_infer_s:.4f}s"
        )


def main() -> None:
    _require_pyside6()
    from PySide6.QtWidgets import QApplication

    cfg = default_config()
    app = QApplication(sys.argv)
    window = ADASWindow(cfg)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
