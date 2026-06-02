# Объединённый пайплайн ADAS (adas_pipeline.py)

## Назначение

Модуль `adas_pipeline.py` — центральный оркестратор системы. Класс `ADASProcessor` объединяет все компоненты (детекция объектов, трекинг, оценка дистанции, детекция полос, BEV-преобразование, логика предупреждений, визуализация) в единый конвейер обработки кадра.

---

## Класс ADASProcessor

### Инициализация

```python
# adas_pipeline.py:256-286

class ADASProcessor:
    def __init__(self, lane_config, object_config, logger=None,
                 allowed_labels=None, parallel=True,
                 lane_skip_frames=0, downscale=1.0):
        self.lane_config = dict(lane_config)
        self.object_config = dict(object_config)
        self.logger = logger or Logger(None, logging.INFO, logging.INFO)
        self.allowed_labels = {s.lower() for s in allowed_labels} if allowed_labels else None
        self.parallel = parallel
        self.lane_skip_frames = lane_skip_frames
        self.downscale = downscale

        self._executor = ThreadPoolExecutor(max_workers=2) if parallel else None
        self._frame_idx = 0
        self._cached_lane_result = None
```

| Параметр | Описание |
|----------|----------|
| `lane_config` | Конфигурация детектора полос (путь к модели, тип) |
| `object_config` | Конфигурация детектора объектов (путь к модели, тип, пороги) |
| `allowed_labels` | Множество разрешённых классов (`{"person", "car", ...}`) |
| `parallel` | Параллельный инференс через `ThreadPoolExecutor` |
| `lane_skip_frames` | Пропуск N кадров между детекциями полос (0 = каждый кадр) |
| `downscale` | Коэффициент масштабирования кадра для инференса |

### initialize — создание всех компонентов

```python
# adas_pipeline.py:288-313

def initialize(self, frame_size: Tuple[int, int]) -> None:
    width, height = frame_size
    num_threads = max(1, (os.cpu_count() or 4) // 2) if self.parallel else None

    # Детектор полос (UFLD или UFLDv2)
    if "UFLDV2" in self.lane_config["model_type"].name:
        UltrafastLaneDetectorV2.set_defaults(self.lane_config)
        self.lane_detector = UltrafastLaneDetectorV2(logger=self.logger, num_threads=num_threads)
    else:
        UltrafastLaneDetector.set_defaults(self.lane_config)
        self.lane_detector = UltrafastLaneDetector(logger=self.logger, num_threads=num_threads)

    self.transform_view = PerspectiveTransformation((width, height), logger=self.logger)

    # Детектор объектов (YOLO или EfficientDet)
    if ObjectModelType.EfficientDet == self.object_config["model_type"]:
        EfficientdetDetector.set_defaults(self.object_config)
        self.object_detector = EfficientdetDetector(logger=self.logger, num_threads=num_threads)
    else:
        YoloDetector.set_defaults(self.object_config)
        self.object_detector = YoloDetector(logger=self.logger, num_threads=num_threads)

    self.distance_detector = SingleCamDistanceMeasure()
    self.object_tracker = BYTETracker(names=self.object_detector.colors_dict)
    self.display_panel = ControlPanel()
    self.analyze_msg = TaskConditions()
```

---

## process_frame — обработка одного кадра

Это центральный метод, вызываемый на каждом кадре.

```python
# adas_pipeline.py:338-426

def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, ADASMetrics]:
```

### Шаг 1: Препроцессинг

```python
infer_frame = self._prepare_frame(frame)
# Если downscale < 1.0, кадр уменьшается для ускорения инференса
```

### Шаг 2: Параллельный или последовательный инференс

```python
run_lane = (self.lane_skip_frames == 0) or (self._frame_idx % (self.lane_skip_frames + 1) == 0)

if self.parallel and self._executor is not None:
    obj_future = self._executor.submit(self.object_detector.DetectFrame, infer_frame)
    lane_future = self._executor.submit(self.lane_detector.DetectFrame, infer_frame) if run_lane else None

    obj_future.result()
    object_infer_time = round(time.time() - object_time, 2)

    # Фильтрация по разрешённым классам
    if self.allowed_labels is not None:
        self.object_detector._object_info = [
            obj for obj in self.object_detector.object_info
            if str(obj.label).lower() in self.allowed_labels
        ]

    if lane_future is not None:
        lane_future.result()
        lane_infer_time = round(time.time() - object_time, 4)
```

**Оптимизация:** детекция полос может запускаться не на каждом кадре (`lane_skip_frames > 0`), что значительно ускоряет процесс, так как полосы меняются медленно.

### Шаг 3: Трекинг объектов

```python
boxes = [obj.tolist(format_type="xyxy") for obj in self.object_detector.object_info]
scores = [obj.conf for obj in self.object_detector.object_info]
class_ids = [obj.label for obj in self.object_detector.object_info]

self.object_tracker.update(boxes, scores, class_ids, frame)
```

### Шаг 4: Оценка дистанции и зоны столкновения

```python
self.distance_detector.updateDistance(self.object_detector.object_info)
vehicle_distance = self.distance_detector.calcCollisionPoint(
    self.lane_detector.lane_info.area_points
)
```

### Шаг 5: BEV-преобразование и расчёт кривизны/смещения

```python
if self.analyze_msg.CheckStatus() and self.lane_detector.lane_info.area_status:
    self.transform_view.updateTransformParams(
        *self.lane_detector.lane_info.lanes_points[1:3],
        self.analyze_msg.transform_status
    )

frame_show = frame.copy()
birdview_show = self.transform_view.transformToBirdView(frame_show)
birdview_lanes_points = [
    self.transform_view.transformToBirdViewPoints(lp)
    for lp in self.lane_detector.lane_info.lanes_points
]
(vehicle_direction, vehicle_curvature), vehicle_offset = \
    self.transform_view.calcCurveAndOffset(birdview_show, *birdview_lanes_points[1:3])
```

### Шаг 6: Обновление предупреждений

```python
self.analyze_msg.UpdateCollisionStatus(vehicle_distance, self.lane_detector.lane_info.area_status)
self.analyze_msg.UpdateOffsetStatus(vehicle_offset)
self.analyze_msg.UpdateRouteStatus(vehicle_direction, vehicle_curvature)
```

### Шаг 7: Визуализация

```python
self.transform_view.DrawDetectedOnBirdView(birdview_show, birdview_lanes_points, self.analyze_msg.offset_msg)
self.lane_detector.DrawDetectedOnFrame(frame_show, self.analyze_msg.offset_msg)
self.lane_detector.DrawAreaOnFrame(frame_show, self.display_panel.CollisionDict[self.analyze_msg.collision_msg])
self.object_detector.DrawDetectedOnFrame(frame_show)
self.object_tracker.DrawTrackedOnFrame(frame_show, False)
self.distance_detector.DrawDetectedOnFrame(frame_show)

self.display_panel.DisplayBirdViewPanel(frame_show, birdview_show)
self.display_panel.DisplaySignsPanel(frame_show, self.analyze_msg.offset_msg,
                                      self.analyze_msg.curvature_msg, self.analyze_msg.collision_msg)
```

### Шаг 8: Формирование метрик

```python
metrics = ADASMetrics(
    object_infer_s=object_infer_time,
    lane_infer_s=lane_infer_time,
    collision=self.analyze_msg.collision_msg,
    offset=self.analyze_msg.offset_msg,
    curvature=self.analyze_msg.curvature_msg,
)
return frame_show, metrics
```

---

## ADASMetrics — структура результата

```python
# adas_pipeline.py:247-253

@dataclass
class ADASMetrics:
    object_infer_s: float      # время инференса детектора объектов (секунды)
    lane_infer_s: float        # время инференса детектора полос (секунды)
    collision: CollisionType    # статус предупреждения столкновения
    offset: OffsetType          # статус смещения от центра полосы
    curvature: CurvatureType    # статус кривизны дороги
```

---

## ControlPanel — визуализация предупреждений

### Цветовая схема

```python
# adas_pipeline.py:39-61

class ControlPanel(object):
    CollisionDict = {
        CollisionType.UNKNOWN:  (0, 255, 255),    # жёлтый
        CollisionType.NORMAL:   (0, 255, 0),       # зелёный
        CollisionType.PROMPT:   (0, 102, 255),    # оранжевый
        CollisionType.WARNING:  (0, 0, 255),       # красный
    }

    OffsetDict = {
        OffsetType.UNKNOWN:  (0, 255, 255),         # жёлтый
        OffsetType.RIGHT:    (0, 0, 255),           # красный
        OffsetType.LEFT:     (0, 0, 255),           # красный
        OffsetType.CENTER:  (0, 255, 0),            # зелёный
    }

    CurvatureDict = {
        CurvatureType.UNKNOWN:   (0, 255, 255),    # жёлтый
        CurvatureType.STRAIGHT:  (0, 255, 0),      # зелёный
        CurvatureType.EASY_LEFT: (0, 102, 255),    # оранжевый
        CurvatureType.EASY_RIGHT:(0, 102, 255),    # оранжевый
        CurvatureType.HARD_LEFT: (0, 0, 255),      # красный
        CurvatureType.HARD_RIGHT:(0, 0, 255),      # красный
    }
```

### Отображение BEV-панели

```python
# adas_pipeline.py:111-118

def DisplayBirdViewPanel(self, main_show, min_show, show_ratio=0.25):
    W = int(main_show.shape[1] * show_ratio)
    H = int(main_show.shape[0] * show_ratio)
    min_birdview_show = cv2.resize(min_show, (W, H))
    min_birdview_show = cv2.copyMakeBorder(min_birdview_show, 10, 10, 10, 10, ...)
    main_show[0:min_birdview_show.shape[0], -min_birdview_show.shape[1]:] = min_birdview_show
```

Мини-окно BEV отображается в правом верхнем углу кадра (25% от размера).

### Отображение иконок и текстовых предупреждений

```python
# adas_pipeline.py:120-191 DisplaySignsPanel

# Верхний левый угол: иконки поворотов/смещения + текст LDWS/FCWS/FPS
# Иконки с alpha-каналом накладываются через предвычисленные маски (_alpha_masks)
```

---

## Полная схема потока данных

```
                          Входной кадр (BGR)
                                │
                                ▼
                         _prepare_frame()
                         (downscale, опционально)
                                │
                    ┌───────────┴───────────┐
                    │  ThreadPoolExecutor    │ (если parallel=True)
                    │                       │
          ┌─────────▼─────────┐  ┌─────────▼──────────┐
          │  ObjectDetector   │  │  LaneDetector       │
          │  .DetectFrame()   │  │  .DetectFrame()     │
          │  (YOLO/EffDet)    │  │  (UFLD/UFLDv2)      │
          └─────────┬─────────┘  └─────────┬──────────┘
                    │                       │
                    │   Фильтрация по       │
                    │   allowed_labels       │
                    │                       │
          ┌─────────▼─────────┐             │
          │   BYTETracker     │             │
          │   .update()       │             │
          └─────────┬─────────┘             │
                    │                       │
          ┌─────────▼─────────┐             │
          │  DistanceMeasure  │◄────────────┘
          │  .updateDistance()│  (area_points)
          │  .calcCollisionPt│
          └─────────┬─────────┘
                    │
                    │           CheckStatus()
                    │           updateTransformParams()
                    ▼
          ┌─────────────────────────────────┐
          │  PerspectiveTransformation       │
          │  .transformToBirdView()         │
          │  .transformToBirdViewPoints()    │
          │  .calcCurveAndOffset()           │
          └──────┬──────────┬───────────────┘
                 │          │
                 ▼          ▼
      vehicle_offset    (direction, curvature)
                 │          │
    ┌────────────▼──────────▼─────────────┐
    │         TaskConditions               │
    │  .UpdateCollisionStatus()            │
    │  .UpdateOffsetStatus()               │
    │  .UpdateRouteStatus()                 │
    └────────────┬─────────────────────────┘
                 │
                 ▼
    ┌─────────────────────────────────────┐
    │     ControlPanel (визуализация)      │
    │  + lane/object/track/distance draw   │
    └────────────┬────────────────────────┘
                 │
                 ▼
     frame_show (BGR) + ADASMetrics
```

---

## Связанные разделы

- [Архитектура решения](architecture.md)
- [Детекция объектов](object-detection.md)
- [Трекинг объектов](object-tracking.md)
- [Детекция разметки](lane-detection.md)
- [Оценка дистанции](distance-estimation.md)
- [BEV-преобразование](perspective-transformation.md)
- [Логика предупреждений](task-conditions.md)
- [Десктопное приложение](desktop-gui.md)