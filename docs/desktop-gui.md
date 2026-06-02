# Десктопное приложение (main_desktop.py)

## Назначение

Модуль `main_desktop.py` реализует графический интерфейс пользователя (GUI) на базе **PySide6** (Qt для Python). Приложение позволяет:

- Выбирать видеофайл через диалог или drag-and-drop.
- Запускать/останавливать обработку видео.
- Наблюдать результаты ADAS в реальном времени (BGR-кадр с оверлеями).
- Сохранять выходное видео с наложенными предупреждениями.
- Настраивать размер модели YOLO и параметры инференса через CLI.

---

## Классы

```
AppConfig (dataclass)
    │   lane_config: dict
    │   object_config: dict
    │   save_output: bool = True
    │
VideoDropLabel
    │   Обработка drag-and-drop видео
    │
ADASWindow
    │   Главное окно приложения
    │   - ADASProcessor (пайплайн)
    │   - QTimer (цикл обработки)
    │   - cv2.VideoCapture (ввод)
    │   - cv2.VideoWriter (вывод, опционально)
```

---

## Конфигурация по умолчанию

```python
# main_desktop.py:32-57

YOLO_MODEL_FILES = {
    "n": "yolov8n.onnx",     # Nano — 13 MB
    "s": "yolov8s.onnx",     # Small
    "m": "yolov8m.onnx",     # Medium
    "l": "yolov8l.onnx",     # Large — 167 MB
}

def default_config(yolo_size="n") -> AppConfig:
    lane_config = {
        "model_path": "TrafficLaneDetector/models/ufldv2_tusimple_res18_320x800.onnx",
        "model_type": LaneModelType.UFLDV2_TUSIMPLE,
    }
    object_config = {
        "model_path": "ObjectDetector/models/yolov8n.onnx",
        "model_type": ObjectModelType.YOLOV8,
        "classes_path": "ObjectDetector/models/coco_label.txt",
        "box_score": 0.4,
        "box_nms_iou": 0.5,
    }
    return AppConfig(lane_config=lane_config, object_config=object_config, save_output=True)
```

**Параметры:**
- Детектор полос: UFLDv2 (TuSimple), модель 320×800.
- Детектор объектов: YOLOv8n (Nano), порог скоринга 0.4, NMS IoU 0.5.

---

## CLI-параметры

```bash
python main_desktop.py [OPTIONS]

Options:
  --yolo-size {n,s,m,l}    Размер модели YOLO (default: n)
  --yolo-path PATH          Путь к ONNX модели (переопределяет --yolo-size)
  --no-parallel             Отключить параллельный инференс
  --lane-skip N             Пропускать N кадров между детекциями полос (0=каждый)
  --downscale FLOAT         Коэффициент уменьшения кадра (напр. 0.5)
```

---

## ADASWindow — главное окно

### Инициализация

```python
# main_desktop.py:102-177

class ADASWindow:
    def __init__(self, cfg: AppConfig, parallel=True, lane_skip_frames=0, downscale=1.0):
        self.processor = ADASProcessor(
            cfg.lane_config, cfg.object_config,
            allowed_labels={"person", "car", "truck", "bus", "motorbike"},
            parallel=parallel, lane_skip_frames=lane_skip_frames, downscale=downscale,
        )
        self.cap: Optional[cv2.VideoCapture] = None
        self.vout: Optional[cv2.VideoWriter] = None
        # ... Qt-виджеты ...
```

### Цикл обработки — метод `_tick()`

Вызывается по `QTimer` с интервалом `1000 / fps` мс:

```python
# main_desktop.py:250-275

def _tick(self):
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
        f"FCWS={metrics.collision.name} | LDWS={metrics.offset.name} "
        f"| LKAS={metrics.curvature.name} "
        f"| obj={metrics.object_infer_s:.2f}s lane={metrics.lane_infer_s:.4f}s"
    )
```

**Поток данных в GUI:**

```
QTimer.timeout
    │
    ▼
cv2.VideoCapture.read()  →  BGR-кадр
    │
    ▼
ADASProcessor.process_frame()  →  (frame_show, ADASMetrics)
    │
    ├──► cv2.VideoWriter.write()  (если включено сохранение)
    │
    ├──► _bgr_to_qimage()  →  QPixmap  →  QLabel.setPixmap()
    │
    └──► StatusBar: FCWS/LDWS/LKAS/время инференса
```

---

## Конвертация BGR → QImage

```python
# main_desktop.py:60-67

def _bgr_to_qimage(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = frame_rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    return qimg
```

---

## Drag-and-Drop — VideoDropLabel

```python
# main_desktop.py:70-100

class VideoDropLabel:
    def __init__(self, label):
        class _Filter(QObject):
            def eventFilter(self, obj, event):
                if event.type() == QEvent.Type.DragEnter:
                    if event.mimeData().hasUrls():
                        event.acceptProposedAction()
                        return True
                if event.type() == QEvent.Type.Drop:
                    path = event.mimeData().urls()[0].toLocalFile()
                    if self.outer.on_path:
                        self.outer.on_path(path)
                    event.acceptProposedAction()
                    return True
                return False

        self.label = label
        self.on_path = None
        self.label.setAcceptDrops(True)
        self.filter = _Filter(self)
```

Пользователь может перетащить видеофайл на область отображения — путь будет установлен автоматически.

---

## Управление воспроизведением

```python
# main_desktop.py:201-249

def start(self):
    # Инициализация ADASProcessor при первом запуске или смене размера
    if not self.initialized or self.last_frame_size != frame_size:
        self.processor.initialize(frame_size)
        self.initialized = True

    # Опциональное сохранение выходного видео
    if self.save_cb.isChecked():
        out_path = os.path.splitext(self.video_path)[0] + "_out.mp4"
        self.vout = cv2.VideoWriter(out_path, fourcc, float(fps), (width, height))

    # Запуск таймера
    interval_ms = max(1, int(1000.0 / float(fps)))
    self.timer.start(interval_ms)

def stop(self):
    self.timer.stop()
    if self.vout is not None:
        self.vout.release()
    if self.cap is not None:
        self.cap.release()
```

**Ключевая особенность:** `ADASProcessor.initialize()` вызывается при изменении размера кадра, что позволяет обрабатывать видео разных разрешений без перезапуска приложения.

---

## Точка входа — main_new.py

```python
# main_new.py:7-15

def main(argv=None):
    parser = argparse.ArgumentParser(description="ADAS entrypoint (desktop GUI).")
    parser.add_argument("--desktop", action="store_true", help="Launch desktop GUI (PySide6).")
    args, remaining = parser.parse_known_args(argv)

    root = os.path.dirname(os.path.realpath(__file__))
    sys.argv = [sys.argv[0]] + remaining
    runpy.run_path(os.path.join(root, "main_desktop.py"), run_name="__main__")
    return 0
```

Запуск:
```bash
python main_new.py
# или напрямую:
python main_desktop.py --yolo-size n
```

---

## Схема главного окна

```
┌──────────────────────────────────────────────────────────────────────┐
│ ┌─────────────────────────────────┐ ┌──────┐ ┌───────┐ ┌──────┐ ☑  │
│ │  Путь к видео / drag-and-drop   │ │Открыть│ │ Старт │ │ Стоп │ ☑  │
│ └─────────────────────────────────┘ └──────┘ └───────┘ └──────┘ ☑  │
│ ┌──────────────────────────────────────────────────────────────────┐ │
│ │                                                                  │ │
│ │            Видео с оверлеями (QLabel + QPixmap)                  │ │
│ │                                                                  │ │
│ │   ┌──────────────┐                      ┌────────────────────┐   │ │
│ │   │ Иконки       │                      │  BEV мини-окно     │   │ │
│ │   │ поворотов/   │                      │  (правый верхний    │   │ │
│ │   │ смещений     │                      │   угол)             │   │ │
│ │   └──────────────┘                      └────────────────────┘   │ │
│ │                                                                  │ │
│ └──────────────────────────────────────────────────────────────────┘ │
│ FCWS=WARNING | LDWS=LEFT | LKAS=HARD_RIGHT | obj=0.12s lane=0.0045s│
└──────────────────────────────────────────────────────────────────────┘
```

---

## Связанные разделы

- [Пайплайн обработки](adas-pipeline.md)
- [Архитектура решения](architecture.md)