# Модель: YOLO (v5–v10) для детекции объектов

## Название модели

YOLO family (v5…v10).

## Источник

YOLO (You Only Look Once) — семейство одностадийных детекторов объектов. В репозитории обучение не ведётся; предполагается использование готовых весов, экспортированных в ONNX/TRT.

Ссылки:
- Ultralytics YOLO (современные версии, экспорт в ONNX): https://github.com/ultralytics/ultralytics
- YOLOv1 (оригинальная статья): https://arxiv.org/abs/1506.02640

## Архитектура (обобщённо)

Backbone + Neck (FPN/PAN) + Detection Head (bbox + objectness + class scores). Конкретная архитектура зависит от выбранных весов (например, `yolov8l`).

## Входные данные

- исходный кадр: BGR `np.ndarray uint8`,
- preprocess (см. `ObjectDetector/yoloDetector.py`):
  - resize/letterbox под вход модели,
  - нормализация `1/255`,
  - NCHW,
  - dtype `float16/float32`.

## Выходные данные

После постпроцессинга:
- список `RectInfo` (bbox + `conf` + `label`),
- подавление дублей: Soft-NMS (`NMS.fast_soft_nms`).

## Обучение / дообучение

В текущем репозитории не реализовано.

## Используемые веса

В коде фигурирует пример:
- `yolov8l.onnx` (см. `main_desktop.py:default_config()`).

## Производительность и latency

Снимается в пайплайне через `ADASMetrics.object_infer_s` (см. `adas_pipeline.py`).

## Ограничения и риски

- Нужно корректно выбрать `ObjectModelType` под формат выхода экспортированного ONNX.
- Для `.trt` требуется совместимость с версией TensorRT/CUDA, на которой создан engine.
