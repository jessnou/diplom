# Детекция объектов (ObjectDetector/)

## Назначение

Модуль `ObjectDetector` отвечает за обнаружение объектов на видеопотоке (автомобили, пешеходы, мотоциклы и др.) с помощью нейросетевых моделей YOLO или EfficientDet, работающих через ONNX Runtime. Результатом является список объектов с координатами bounding box, классом и уверенностью.

---

## Структура модуля

```
ObjectDetector/
├── __init__.py                  # Экспорт YoloDetector, EfficientdetDetector
├── core.py                     # Базовый класс ObjectDetectBase, RectInfo
├── yoloDetector.py             # YOLO-детектор (v5–v10)
├── efficientdetDetector.py     # EfficientDet-детектор (альтернатива)
├── distanceMeasure.py          # Оценка дистанции (см. отдельный раздел)
└── utils.py                    # NMS, Scaler, перечисления
```

---

## Иерархия классов

```
ObjectDetectBase (abc.ABC)
    │   DetectFrame()           — абстрактный метод
    │   DrawDetectedOnFrame()   — абстрактный метод
    │   object_info → list[RectInfo]
    │
    ├── YoloDetector (+ YoloLiteParameters)
    │       _object_info: list[RectInfo]
    │       engine: OnnxEngine
    │
    └── EfficientdetDetector
            _object_info: list[RectInfo]
            engine: OnnxEngine
```

---

## Ключевые структуры данных

### RectInfo — информация об обнаруженном объекте

```python
# ObjectDetector/core.py:9-16

@dataclass
class RectInfo:
    x: float           # координата X левого верхнего угла
    y: float           # координата Y левого верхнего угла
    width: float        # ширина bounding box
    height: float       # высота bounding box
    conf: float         # уверенность (confidence score)
    label: str          # название класса («car», «person» и т.д.)
    kpss: List[Tuple[int, int]] = field(default_factory=list)  # ключевые точки

    def tolist(self, dtype=int, format_type="xyxy"):
        if format_type == "xyxy":
            return [self.x, self.y, self.x + self.width, self.y + self.height]
        else:
            return [self.x, self.y, self.width, self.height]
```

### ObjectModelType — поддерживаемые модели

```python
# ObjectDetector/utils.py:15-24

class ObjectModelType(Enum):
    YOLOV5 = 0
    YOLOV5_LITE = 1
    YOLOV6 = 2
    YOLOV7 = 3
    YOLOV8 = 4
    YOLOV9 = 5
    YOLOV10 = 6
    EfficientDet = 7
```

---

## YoloDetector — основной детектор

### Инициализация и конфигурация

```python
# ObjectDetector/yoloDetector.py:45-61

class YoloDetector(ObjectDetectBase, YoloLiteParameters):
    _defaults = {
        "model_path": './models/yolov5n-coco.onnx',
        "model_type": ObjectModelType.YOLOV8,
        "classes_path": './models/coco_label.txt',
        "box_score": 0.4,
        "box_nms_iou": 0.45
    }

    def __init__(self, logger=None, num_threads=None, **kwargs):
        ObjectDetectBase.__init__(self, logger)
        self.__dict__.update(kwargs)
        self._num_threads = num_threads
        self._initialize_class(self.classes_path)
        self._initialize_model(self.model_path)
        YoloLiteParameters.__init__(self, self.model_type, self.input_shapes, len(self.class_names))
```

- `_defaults` — словарь конфигурации, переопределяется через `set_defaults()`.
- `box_score` (0.4) — порог уверенности для фильтрации детекций.
- `box_nms_iou` (0.45) — порог IoU для non-maximum suppression.

### Препроцессинг входного изображения

```python
# ObjectDetector/yoloDetector.py:87-93

def __prepare_input(self, srcimg):
    scaler = Scaler(self.input_shapes[-2:], True)
    image = scaler.process_image(srcimg)
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (image.shape[1], image.shape[0]),
                                  swapRB=True, crop=False).astype(self.input_types)
    return blob, scaler
```

1. `Scaler` приводит изображение к входному размеру модели с сохранением пропорций (letterbox + padding).
2. `cv2.dnn.blobFromImage` выполняет нормализацию `[0, 255] → [0, 1]`, меняет порядок каналов BGR→RGB и трансформирует в формат NCHW.

### Обработка выхода сети

```python
# ObjectDetector/yoloDetector.py:95-126

def __process_output(self, output):
    # YOLOv8/9/10: транспонирование выхода
    if self.model_type in [ObjectModelType.YOLOV8, ObjectModelType.YOLOV9, ObjectModelType.YOLOV10]:
        output = output.T

    output = self.lite_postprocess(output)

    # Извлечение вероятностей классов
    if self.model_type in [ObjectModelType.YOLOV8, ObjectModelType.YOLOV9, ObjectModelType.YOLOV10]:
        obj_cls_probs = output[:, 4:]       # YOLOv8: box(xywh) + cls_probs
    else:
        obj_cls_probs = output[:, 5:] * output[:, 4:5]  # YOLOv5: obj_conf * cls_conf

    class_ids = np.argmax(obj_cls_probs, axis=1)
    class_confs = np.take_along_axis(obj_cls_probs, class_ids[:, np.newaxis], axis=1).squeeze()
    mask = class_confs > self.box_score
    # ... фильтрация и преобразование координат
```

**Различия формата выхода:**
- **YOLOv8/9/10**: выход формы `(N, 4+num_classes)` — координаты + вероятности классов напрямую.
- **YOLOv5/6/7**: выход формы `(N, 5+num_classes)` — `obj_conf * cls_conf` для итогового скора.

### NMS — Non-Maximum Suppression

Используется **Soft-NMS** (метод `linear` по умолчанию):

```python
# ObjectDetector/utils.py:161-191

class NMS:
    @staticmethod
    def fast_soft_nms(dets, scores, iou_thr=0.3, sigma=0.5,
                      score_thr=0.001, dets_type="xyxy", method='linear'):
```

Soft-NMS в отличие от жёсткого NMS не отбрасывает перекрывающиеся боксы, а снижает их скор в зависимости от IoU, что повышает полноту детекции для близко расположенных объектов.

Детектор вызывает:
```python
nms_results = NMS.fast_soft_nms(boxes, class_confs, self.box_nms_iou, dets_type="xywh")
```

---

## Scaler — преобразование координат

`Scaler` решает задачу приведения координат детекций из пространства входа модели в пространство исходного изображения.

```python
# ObjectDetector/utils.py:31-99

@dataclass
class Scaler(object):
    target_size: Tuple[int, int]   # (H, W) — размер входа модели
    keep_ratio: bool = True         # сохранять ли пропорции (letterbox)

    def process_image(self, srcimg):
        # Если keep_ratio=True: letterbox + padding до target_size
        # Иначе: прямой resize

    def convert_boxes_coordinate(self, boxes, in_format="xyxy", out_format="xywh"):
        # Обратное преобразование: (x,y) из модельных → оригинальные
        # Учитывает padding и масштабирование
        boxes[..., [0, 2]] = (boxes[..., [0, 2]] - padw) * ratiow
        boxes[..., [1, 3]] = (boxes[..., [1, 3]] - padh) * ratioh
```

---

## Полный цикл DetectFrame

```python
# ObjectDetector/yoloDetector.py:152-161

def DetectFrame(self, srcimg):
    input_tensor, scaler = self.__prepare_input(srcimg)
    output_from_network = self.engine.engine_inference(input_tensor)[0].squeeze(axis=0)
    _raw_boxes, _raw_class_ids, _raw_class_confs, _raw_kpss = self.__process_output(output_from_network)
    transform_boxes = scaler.convert_boxes_coordinate(_raw_boxes)
    transform_kpss = scaler.convert_kpss_coordinate(_raw_kpss)
    self._object_info = self.get_nms_results(transform_boxes, _raw_class_confs, _raw_class_ids, transform_kpss)
```

```
BGR-кадр → Scaler.process_image (letterbox+padding)
         → blobFromImage (нормализация, NCHW)
         → OnnxEngine.engine_inference
         → __process_output (фильтрация по box_score)
         → Scaler.convert_boxes_coordinate (обратное преобразование координат)
         → NMS.fast_soft_nms
         → self._object_info: list[RectInfo]
```

---

## Фильтрация по классам

В пайплайне (`adas_pipeline.py:353-356`) результаты детекции фильтруются по разрешённым классам:

```python
if self.allowed_labels is not None:
    self.object_detector._object_info = [
        obj for obj in self.object_detector.object_info
        if str(obj.label).lower() in self.allowed_labels
    ]
```

По умолчанию в ADASProcessor разрешены классы: `{"person", "car", "truck", "bus", "motorbike"}`.

---

## Визуализация результатов

```python
# ObjectDetector/yoloDetector.py:163-184

def DrawDetectedOnFrame(self, frame_show):
    for _info in self._object_info:
        xmin, ymin, xmax, ymax = _info.tolist()
        label = _info.label
        # Отрисовка bounding box с характерными «уголками»
        self.cornerRect(frame_show, _info.tolist(), colorR=self.colors_dict[label], ...)
        # Подпись класса
        cv2.putText(frame_show, label, (xmin + 2, ymin - 7), ...)
```

---

## Связанные разделы

- [Движок инференса ONNX](inference-engine.md)
- [Трекинг объектов](object-tracking.md)
- [Оценка дистанции](distance-estimation.md)
- [Пайплайн обработки](adas-pipeline.md)