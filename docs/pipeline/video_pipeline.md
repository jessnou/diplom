# Пайплайн обработки видео (End-to-End)

Ключевая реализация пайплайна находится в `adas_pipeline.py` (класс `ADASProcessor`).

## Краткое описание

На каждом кадре система выполняет:

1. Получение кадра BGR из `cv2.VideoCapture`
2. Предобработка под входные размеры моделей (детектор объектов / детектор полос)
3. Детекция объектов (YOLO/EfficientDet)
4. Трекинг объектов (ByteTrack)
5. Расчёт расстояния (по bbox) + выбор ближайшего объекта в зоне своей полосы
6. Детекция полос (UFLD/UFLDv2)
7. Преобразование в Bird Eye View (BEV)
8. Расчёт смещения и кривизны
9. Генерация предупреждений (TaskConditions)
10. Визуализация и/или сохранение результата

## Поток данных (сигналы и структуры)

```text
frame (np.ndarray, BGR uint8)
  ├─ objectDetector.DetectFrame(frame)
  │    └─ objectDetector.object_info: List[RectInfo]
  ├─ objectTracker.update(boxes, scores, class_ids, frame)
  ├─ distanceDetector.updateDistance(object_info)
  ├─ laneDetector.DetectFrame(frame)
  │    └─ lane_info.area_points (polygon), lane_info.area_status
  ├─ vehicle_distance = calcCollisionPoint(area_points)
  ├─ bird = transformToBirdView(frame_show)
  ├─ bird_lanes = transformToBirdViewPoints(lanes_points)
  ├─ (direction, curvature), offset = calcCurveAndOffset(bird, left_ego, right_ego)
  └─ TaskConditions.Update*Status(...) -> (FCWS/LDWS/LKAS)
```

## Псевдокод (уровень `ADASProcessor`)

```pseudo
initialize(frame_size):
  laneDetector = UFLDv2 or UFLD
  objectDetector = YOLO or EfficientDet
  transformView = PerspectiveTransformation(frame_size)
  objectTracker = BYTETracker(names=objectDetector.colors_dict)
  distanceDetector = SingleCamDistanceMeasure()
  analyzeMsg = TaskConditions()

process_frame(frame):
  objectDetector.DetectFrame(frame)
  objectTracker.update(...)
  laneDetector.DetectFrame(frame)

  distanceDetector.updateDistance(object_info)
  vehicle_distance = distanceDetector.calcCollisionPoint(lane_info.area_points)

  if analyzeMsg.CheckStatus() and lane_info.area_status:
      transformView.updateTransformParams(left_ego, right_ego, analyzeMsg.transform_status)

  bird = transformView.transformToBirdView(frame)
  bird_lanes = transformView.transformToBirdViewPoints(lanes_points)
  (dir, R), offset = transformView.calcCurveAndOffset(bird, left_ego_bird, right_ego_bird)

  analyzeMsg.UpdateCollisionStatus(vehicle_distance, lane_info.area_status)
  analyzeMsg.UpdateOffsetStatus(offset)
  analyzeMsg.UpdateRouteStatus(dir, R)

  draw overlays
  return frame_show, metrics
```

## Параметры, влияющие на поведение (по умолчанию)

| Блок | Параметр | Где | Значение |
|---|---|---|---:|
| Детектор | `box_score` | `object_config` | 0.4 |
| Детектор | `box_nms_iou` | `object_config` | 0.45–0.5 |
| Трекинг | `track_thresh` | `BYTETracker` | 0.5 |
| Трекинг | `track_buffer` | `BYTETracker` | 30 |
| Дистанция | `f` | `SingleCamDistanceMeasure` | 200 |
| FCWS | `distance_thres` | `UpdateCollisionStatus` | 1.5 м |
| LDWS | `offset_thres` | `UpdateOffsetStatus` | 0.65 м |
| LKAS | `curvae_thres` | `UpdateRouteStatus` | 500 |

