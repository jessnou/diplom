# Алгоритм: UFLDv2 lane detection (детекция дорожной разметки)

Документ описывает UFLDv2‑подсистему: теорию “grid + anchors” и конкретную реализацию восстановления точек линий в проекте.

Ссылки:
- UFLD (статья, базовая идея): https://arxiv.org/abs/2004.11757
- Reference repo: https://github.com/cfzd/Ultra-Fast-Lane-Detection

---

## 1. Назначение алгоритма

**Задача.** По кадру получить:

- точки линий разметки (обычно 4 линии: `left-side`, `left-ego`, `right-ego`, `right-side`),
- флаги наличия линий `lanes_status` (bool на каждую линию).

Эти выходы используются дальше:

- для построения collision zone (полигон своей полосы),
- для BEV‑преобразования и расчёта offset/curvature (LDWS/LKAS).

**Почему выбран UFLDv2.**

- Высокая скорость (ultrafast).
- Структурированный выход (точки линий), удобный для геометрии и downstream‑алгоритмов.

**Альтернативы.**

- Lane segmentation (SCNN, ENet‑Seg и др.): проще postprocess, но часто тяжелее.
- Polyline regression (LaneATT, CondLaneNet): лучше на сложных сценах, но сложнее интеграция.
- Классические методы (Canny+Hough): нестабильны в реальных условиях.

---

## 2. Теоретическая основа

### 2.1. Представление “grid + anchors”

UFLD‑семейство формулирует lane detection как задачу классификации по дискретной сетке (griding):

- задаётся набор якорей по оси y (row anchors) и/или по оси x (col anchors),
- для каждой линии и каждого якоря модель выдаёт распределение по `G` grid‑позициям,
- позиция линии восстанавливается как индекс grid (и затем переводится в пиксели).

Для row‑анкеров:

$$
g^\*_{k,l}=\arg\max_g P(g\mid k,l),
\quad
x_{k,l}=\frac{g^\*_{k,l}}{G-1}\cdot W
$$

Для col‑анкеров:

$$
g^\*_{k,l}=\arg\max_g P(g\mid k,l),
\quad
y_{k,l}=\frac{g^\*_{k,l}}{G-1}\cdot H
$$

где `W/H` — ширина/высота исходного кадра.

### 2.2. Локальное уточнение (soft‑argmax в окне)

В проекте используется refinement около argmax:

1) взять окно индексов $g \in [g^\* - w, g^\* + w]$,
2) применить softmax по логитам в этом окне,
3) вычислить взвешенное среднее индекса:

$$
\hat{g}=\sum_{g \in window} \mathrm{softmax}(s_g)\cdot g + 0.5
$$

Это уменьшает квантование по grid и делает точки более гладкими.

### 2.3. Exist‑ветка (есть ли линия)

Модель также предсказывает `exist_row/exist_col`. Для каждой линии считается “сколько якорей валидны”; если валидных якорей достаточно, линия считается найденной.

В проекте пороги заданы как доли от числа якорей:

- для row‑линий: `valid_row.sum() > num_cls_row / 2`
- для col‑линий: `valid_col.sum() > num_cls_col / 4`

---

## 3. Архитектура модели (нейросеть)

UFLDv2 — CNN‑модель со структурированным выходом. В проекте поддерживаются варианты ONNX‑экспорта с 4 или 6 выходными тензорами:

- 4 канала: `loc_row`, `loc_col`, `exist_row`, `exist_col`
- 6 каналов: + `conf_row`, `conf_col` (доп. confidence)

### Входные данные (как подаются в проекте)

См. `TrafficLaneDetector/ufldDetector/ultrafastLaneDetectorV2.py::__prepare_input`:

- `BGR → RGB`,
- resize до `(input_width, input_height / crop_ratio)`,
- crop нижней части до `input_height`,
- normalize mean/std (ImageNet):

$$
I'=\frac{I/255-\mu}{\sigma}
$$

- `HWC → NCHW`, batch‑dim.

### Выходные данные (как интерпретируются)

См. `UltrafastLaneDetectorV2.__process_output`:

- `loc_row`: shape `(B, G_row, K_row, L_row)`
- `exist_row`: shape `(B, 2, K_row, L_row)` (как правило; в коде берётся `argmax` по axis=1)
- аналогично для col‑ветки.

### Loss‑функции (типовые)

В репозитории обучение не реализовано, но для понимания:

- `loc_*`: cross‑entropy по grid‑классам,
- `exist_*`: cross‑entropy/BCE,
- возможны aux losses (например, confidence или segmentation‑ветки).

---

## 4. Pipeline обработки данных

### Входные данные

- `image: np.ndarray (H,W,3) BGR uint8`.

### Предобработка

1. RGB conversion.
2. Resize + crop (учёт `crop_ratio`).
3. Normalize mean/std.
4. NCHW + batch.

### Основной алгоритм

1. Инференс ONNX/TRT.
2. Восстановление точек:
   - row‑ветка формирует центральные `left-ego/right-ego`,
   - col‑ветка формирует `left-side/right-side`.
3. Формирование `lanes_points` (массив из 4 списков точек).
4. Формирование `lanes_status` (bool по линиям).

### Постобработка и интеграция

`LaneDetectBase` (см. `TrafficLaneDetector/ufldDetector/core.py`) автоматически:

- выставляет `area_status=True`, если центральная пара линий найдена,
- строит `area_points` (полигон своей полосы),
- при `adjust_lanes=True` может “дотянуть” линии полиномом (2‑й степени), чтобы сгладить контур.

---

## 5. Реализация в проекте

### Основной класс

- Файл: `TrafficLaneDetector/ufldDetector/ultrafastLaneDetectorV2.py`
- Класс: `UltrafastLaneDetectorV2(LaneDetectBase)`
- Методы:
  - `DetectFrame(image, adjust_lanes=True)`
  - `DrawDetectedOnFrame(image, type=OffsetType, alpha=0.3)`
  - `DrawAreaOnFrame(image, color, alpha)`

### Конфигурация anchors/griding (по коду)

`ModelConfig` выбирается по `LaneModelType`:

- `UFLDV2_TUSIMPLE`: `img_w=800`, `img_h=320`, `griding_num=100`, `crop_ratio=0.8`, `row_anchor=56`, `col_anchor=41`.
- `UFLDV2_CULANE`: `img_w=1600`, `img_h=320`, `griding_num=200`, `crop_ratio=0.6`, `row_anchor=72`, `col_anchor=81`.
- `UFLDV2_CURVELANES`: `img_w=1600`, `img_h=800`, `griding_num=200`, `crop_ratio=0.8`, `row_anchor=72`, `col_anchor=81`.

### Восстановление точек (конкретные формулы из реализации)

Row‑ветка:

- проверка валидности линии: `valid_row.sum() > num_cls_row/2`,
- для каждого якоря `k`:
  - `g* = argmax(loc_row[:, k, lane])`,
  - `window = [g*-w, g*+w]`,
  - $\hat{g}$ через локальный softmax,
  - `x = hat_g/(G-1) * W_orig`,
  - `y = row_anchor[k] * H_orig`.

Col‑ветка:

- проверка валидности линии: `valid_col.sum() > num_cls_col/4`,
- `y = hat_g/(G-1) * H_orig`,
- `x = col_anchor[k] * W_orig`.

---

## 6. Псевдокод алгоритма

```pseudo
ufldv2_detect(frame_bgr):
  x = preprocess(frame_bgr)  # RGB + resize/crop + normalize + NCHW
  out = engine.infer(x)

  # unpack 4 or 6 outputs
  loc_row, loc_col, exist_row, exist_col = out[0:4]

  lanes_points = {left_side:[], left_ego:[], right_ego:[], right_side:[]}
  lanes_status = {.. all False ..}

  # row lanes: left_ego/right_ego
  for lane in [left_ego, right_ego]:
    if sum(exist_row[lane]) > K_row/2:
      for k in anchors_row:
        if exist_row[k,lane]:
          g0 = argmax(loc_row[:,k,lane])
          g  = local_softargmax(loc_row[g0-w:g0+w, k, lane])
          x = g/(G_row-1) * W_orig
          y = row_anchor[k] * H_orig
          lanes_points[lane].append((x,y))
      lanes_status[lane] = (len(lanes_points[lane]) > 2)

  # col lanes: left_side/right_side (аналогично)
  ...

  return lanes_points, lanes_status
```

---

## 7. Параметры и конфигурация

| Параметр | Где | Значение по умолчанию | Влияние |
|---|---|---:|---|
| `lane_config.model_type` | конфиг | `UFLDV2_TUSIMPLE` | anchors/griding/crop |
| `lane_config.model_path` | конфиг | зависит от пользователя | веса ONNX/TRT |
| `crop_ratio` | `ModelConfig` | 0.6–0.8 | сколько отрезать сверху |
| `griding_num` | `ModelConfig` | 100/200 | дискретизация |
| `local_width` | `__process_output` | 1 | сглаживание “grid→pixel” |

Инженерные рекомендации:

- на старте логировать `lanes_status` и долю `area_status=True`,
- валидировать размеры входа/выхода ONNX и число выходов (4/6),
- при дребезге точек добавлять temporal smoothing (EMA по коэффициентам полинома или по точкам).

