# Модель: UFLD / UFLDv2 для детекции полос

## Название модели

- Ultrafast Lane Detection (UFLD)
- Ultrafast Lane Detection v2 (UFLDv2)

## Источник

В репозитории присутствует код экспортируемых PyTorch моделей и скрипт конвертации в ONNX:

- `TrafficLaneDetector/convertPytorchToONNX.py`
- `TrafficLaneDetector/ufldDetector/exportLib/`

Ссылки:
- Ultra Fast Structure-aware Deep Lane Detection (UFLD, статья): https://arxiv.org/abs/2004.11757
- Репозиторий UFLD (референс-реализация): https://github.com/cfzd/Ultra-Fast-Lane-Detection

## Архитектура (обобщённо)

Модель предсказывает:
- дискретные положения линии по griding (row/col),
- флаги существования линий (exist),
- (опционально, для v2) confidence-каналы.

## Входные данные

UFLD:
- RGB, normalize mean/std, resize до входа модели, NCHW.

UFLDv2:
- RGB, resize с учётом `crop_ratio` + crop, normalize mean/std, NCHW.

См. `TrafficLaneDetector/ufldDetector/ultrafastLaneDetector*.py`.

## Выходные данные

`LaneInfo`:
- `lanes_points` (4 линии, точки `(x,y)`),
- `lanes_status` (bool по каждой линии),
- `area_points` / `area_status` для области своей полосы.

## Способ обучения / дообучения

В репозитории обучение не реализовано. Предполагается работа с готовыми весами (например, TuSimple/CULane) и экспорт в ONNX.

## Используемые веса

В репозитории присутствует PyTorch вес:
- `configs/tusimple_18.pth`

ONNX веса должны быть предоставлены пользователем и указаны в `lane_config["model_path"]`.

## Производительность и latency

Снимается через `ADASMetrics.lane_infer_s` (см. `adas_pipeline.py`).

## Ограничения

- Требуется корректный `LaneModelType` под конкретный вес и датасет.
- Качество сильно зависит от условий съемки.
