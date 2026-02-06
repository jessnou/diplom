# Модель: EfficientDet (ONNX) для детекции объектов

## Название модели

EfficientDet (D0/D1/…).

## Источник

Семейство EfficientDet (EfficientNet + BiFPN). В проекте ожидается ONNX-модель, которая возвращает boxes/class_ids/class_confs (см. `ObjectDetector/efficientdetDetector.py`).

Ссылки:
- EfficientDet (статья): https://arxiv.org/abs/1911.09070
- EfficientNet (статья): https://arxiv.org/abs/1905.11946

## Архитектура (обобщённо)

- Backbone: EfficientNet.
- Neck: BiFPN.
- Heads: bbox regression + classification.

## Входные данные

Preprocess (см. `ObjectDetector/efficientdetDetector.py`):
- resize/letterbox (`Scaler`),
- нормализация mean/std,
- NCHW.

## Выходные данные

Список `RectInfo` после порога `box_score`.

## Используемые веса

Пример в `__main__`:
- `models/efficientdet-d0-coco_fp32.onnx`.

## Производительность и latency

Измерение аналогично YOLO (время инференса на кадр).

## Ограничения

- В текущей реализации нет NMS внутри детектора (зависит от конкретного экспорта).
