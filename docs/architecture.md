# Архитектура решения ADAS (Vision-Guard)

## Назначение системы

Система помощи водителю (ADAS — Advanced Driver Assistance System) предназначена для анализа видеопотока с фронтальной камеры автомобиля и выдачи предупреждений:

- **LDWS** (Lane Departure Warning System) — контроль смещения автомобиля относительно центра полосы движения.
- **LKAS** (Lane Keeping Assist System) — информирование о кривизне дороги (повороты, прямые участки).
- **FCWS** (Forward Collision Warning System) — контроль дистанции до объектов впереди и предупреждение о риске столкновения.

---

## Общая схема архитектуры

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Входной видеопоток                           │
│                      (BGR-кадр из cv2.VideoCapture)                │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       ADASProcessor                                 │
│                    (adas_pipeline.py:256)                           │
│                                                                     │
│  ┌──────────────────────┐    ┌─────────────────────────────────┐   │
│  │  ObjectDetector      │    │  LaneDetector                    │   │
│  │  (YOLO / EfficientDet)│    │  (UFLD / UFLDv2)                │   │
│  │   DetectFrame()      │    │   DetectFrame()                  │   │
│  └─────────┬────────────┘    └────────────┬────────────────────┘   │
│            │                              │                         │
│            ▼                              ▼                         │
│  ┌──────────────────────┐    ┌─────────────────────────────────┐   │
│  │  BYTETracker          │    │  PerspectiveTransformation      │   │
│  │   update()            │    │   transformToBirdView()          │   │
│  │                       │    │   calcCurveAndOffset()           │   │
│  └─────────┬────────────┘    └────────────┬────────────────────┘   │
│            │                              │                         │
│            ▼                              ▼                         │
│  ┌──────────────────────┐    ┌─────────────────────────────────┐   │
│  │  DistanceMeasure     │    │  Lane polygon (area_points)      │   │
│  │   updateDistance()   │◄───┤                                  │   │
│  │   calcCollisionPoint()│    └─────────────────────────────────┘   │
│  └─────────┬────────────┘                                            │
│            │                                                         │
│            ▼                                                         │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    TaskConditions                             │   │
│  │   UpdateCollisionStatus()  → CollisionType                   │   │
│  │   UpdateOffsetStatus()     → OffsetType                      │   │
│  │   UpdateRouteStatus()      → CurvatureType                   │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│            ▼                                                         │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                  ControlPanel (визуализация)                   │   │
│  │   DisplayBirdViewPanel()  DisplaySignsPanel()                 │   │
│  │   DisplayCollisionPanel()                                    │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
               ┌──────────────────────────┐
               │  Выходной кадр (оверлеи)  │
               │  + ADASMetrics            │
               └──────────────────────────┘
```

---

## Модульная структура проекта

```
diplom/
├── main_new.py                  # Точка входа (CLI)
├── main_desktop.py               # Точка входа (PySide6 GUI)
├── adas_pipeline.py              # Оркестратор пайплайна
├── taskConditions.py             # Логика предупреждений
├── coreEngine.py                 # Базовый движок инференса ONNX
│
├── ObjectDetector/               # Модуль детекции объектов
│   ├── core.py                   #   Базовый класс ObjectDetectBase, RectInfo
│   ├── yoloDetector.py           #   YOLO (v5–v10) детектор
│   ├── efficientdetDetector.py   #   EfficientDet детектор
│   ├── distanceMeasure.py        #   Оценка дистанции по одной камере
│   └── utils.py                  #   NMS, Scaler, перечисления
│
├── ObjectTracker/                # Модуль трекинга объектов
│   ├── core.py                   #   Базовый класс ObjectTrackBase, отрисовка
│   └── byteTrack/
│       ├── byteTracker.py        #   BYTETracker
│       ├── matching.py           #   IoU + fuse_score + LAP assignment
│       └── dtypes/               #   STrack, KalmanFilter, BaseTrack
│
├── TrafficLaneDetector/          # Модуль детекции дорожной разметки
│   ├── imageDetection.py
│   ├── videoDetection.py
│   └── ufldDetector/
│       ├── core.py               #   Базовый класс LaneDetectBase, LaneInfo
│       ├── ultrafastLaneDetector.py    #   UFLD v1
│       ├── ultrafastLaneDetectorV2.py #   UFLD v2
│       ├── perspectiveTransformation.py # BEV-преобразование
│       └── utils.py              #   LaneModelType, OffsetType, CurvatureType
│
├── assets/                       # Иконки предупреждений
└── models/                       # ONNX/Wts модели
```

---

## Поток данных на один кадр

1. **Захват кадра** — `cv2.VideoCapture` или PySide6-таймер читает BGR-кадр.
2. **Препроцессинг** — кадр масштабируется (опционально `downscale`).
3. **Параллельный инференс** — `ThreadPoolExecutor` запускает детекцию объектов и детекцию полос одновременно.
4. **Трекинг** — обнаруженные bounding boxes передаются в `BYTETracker.update()`.
5. **Оценка дистанции** — `SingleCamDistanceMeasure` вычисляет расстояние до каждого объекта через формулу тонкой линзы; выбирается ближайший объект внутри полигона текущей полосы.
6. **BEV-преобразование** — точки полос проецируются в Bird's Eye View; `calcCurveAndOffset()` вычисляет смещение и кривизну дороги.
7. **Логика предупреждений** — `TaskConditions` агрегирует измерения во времени (медианная фильтрация) и выдаёт финальные статусы `CollisionType`, `OffsetType`, `CurvatureType`.
8. **Визуализация** — все детекции, треки, дистанции, зоны полос, BEV-панель, иконки предупреждений накладываются на кадр.

---

## Ключевые технологии

| Компонент | Технология | Формат модели |
|-----------|-----------|---------------|
| Детекция объектов | YOLOv5–v10 / EfficientDet | ONNX Runtime |
| Детекция разметки | UFLD / UFLDv2 | ONNX Runtime |
| Трекинг | ByteTrack (Kalman Filter + IoU) | — |
| Инференс | ONNX Runtime (CPU/CUDA) | `.onnx` |
| Визуализация | OpenCV | — |
| GUI | PySide6 (Qt) | — |
| Язык | Python 3.12 | — |

---

## Связь с другими разделами

- [Движок инференса ONNX](inference-engine.md)
- [Детекция объектов](object-detection.md)
- [Трекинг объектов](object-tracking.md)
- [Детекция разметки](lane-detection.md)
- [Оценка дистанции](distance-estimation.md)
- [BEV-преобразование](perspective-transformation.md)
- [Логика предупреждений](task-conditions.md)
- [Пайплайн обработки](adas-pipeline.md)
- [Десктопное приложение](desktop-gui.md)