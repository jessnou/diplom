# Детекция дорожной разметки (TrafficLaneDetector/)

## Назначение

Модуль `TrafficLaneDetector` реализует обнаружение линий дорожной разметки с помощью нейросетевых моделей **UFLD** (Ultra-Fast Lane Detection) и **UFLDv2**. Результат — набор точек каждой полосы (до 4 линий), статус их обнаружения и полигон «своей» полосы для дальнейшего анализа дистанции.

---

## Структура модуля

```
TrafficLaneDetector/
├── __init__.py                      # Экспорт UltrafastLaneDetector, UltrafastLaneDetectorV2
├── imageDetection.py                # CLI для одиночных изображений
├── videoDetection.py                # CLI для видео
├── convertPytorchToONNX.py          # Конвертация моделей PyTorch → ONNX
├── models/                          # ONNX-модели
└── ufldDetector/
    ├── core.py                      # LaneDetectBase, LaneInfo
    ├── ultrafastLaneDetector.py     # UFLD v1
    ├── ultrafastLaneDetectorV2.py   # UFLD v2
    ├── perspectiveTransformation.py # BEV-преобразование (отдельный раздел)
    └── utils.py                     # LaneModelType, OffsetType, CurvatureType
```

---

## Иерархия классов

```
LaneDetectBase (abc.ABC)
    │   lane_info: LaneInfo
    │   set_input_details(engine)    → input_shape, input_types
    │   set_output_details(engine)   → output_shape, output_names
    │   DetectFrame()                — абстрактный
    │   DrawDetectedOnFrame()        — абстрактный
    │   DrawAreaOnFrame()            — абстрактный
    │
    ├── UltrafastLaneDetector      (UFLD v1)
    └── UltrafastLaneDetectorV2    (UFLD v2)
```

---

## LaneInfo — структура результата детекции

```python
# TrafficLaneDetector/ufldDetector/core.py:8-11

@dataclass
class LaneInfo:
    _lanes_points: np.ndarray   # Массив 4 массивов точек [(x,y), ...] для каждой полосы
    _lanes_status: np.ndarray   # Массив bool: обнаружена ли каждая полоса
    _area_points: np.ndarray    # Полигон «своей» полосы (для FCWS)
    _area_status: bool          # Есть ли валидная зона полосы
```

### Формирование area_points

Полигон формируется из двух центральных полос (`lanes_points[1]` — левая эго-полоса, `lanes_points[2]` — правая эго-полоса):

```python
# TrafficLaneDetector/ufldDetector/core.py:150-158

def __update_lanes_area(self, lanes_points, img_height):
    if self.lane_info._area_status:
        index = len(lanes_points) // 2
        left_lanes_points = lanes_points[index - 1]
        right_lanes_points = lanes_points[index]
        self.lane_info._area_points = np.vstack((
            left_lanes_points,          # левая граница: сверху вниз
            np.flipud(right_lanes_points)  # правая граница: снизу вверх
        ))
```

**Важно:** `area_status` становится `True` только когда обнаружены **обе** центральные полосы (`lanes_status[index-1] == True and lanes_status[index] == True`).

---

## UltrafastLaneDetectorV2 — основной детектор

### Конфигурация по типу модели

```python
# TrafficLaneDetector/ufldDetector/ultrafastLaneDetectorV2.py:15-26

class ModelConfig:
    def __init__(self, model_type):
        if model_type == LaneModelType.UFLDV2_TUSIMPLE:
            self.init_tusimple_config()    # img_w=800, img_h=320, griding_num=100
        elif model_type == LaneModelType.UFLDV2_CURVELANES:
            self.init_curvelanes_config()  # img_w=1600, img_h=800, griding_num=200
        else:
            self.init_culane_config()      # img_w=1600, img_h=320, griding_num=200
        self.num_lanes = 4
```

### Препроцессинг входного изображения

```python
# TrafficLaneDetector/ufldDetector/ultrafastLaneDetectorV2.py:92-107

def __prepare_input(self, image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    new_size = (self.input_width, int(self.input_height / self.cfg.crop_ratio))
    img_input = cv2.resize(img, new_size).astype(np.float32)
    img_input = img_input[-self.input_height:, :, :]  # кроп снизу

    mean = [0.485, 0.456, 0.406]  # ImageNet mean
    std = [0.229, 0.224, 0.225]   # ImageNet std
    img_input = ((img_input / 255.0 - mean) / std)
    img_input = img_input.transpose(2, 0, 1)           # HWC → CHW
    img_input = img_input[np.newaxis, :, :, :]            # добавляем batch
    return img_input.astype(self.input_types)
```

Особенности:
- Изображение сначала масштабируется до `(input_width, input_height / crop_ratio)`, затем берётся нижняя часть (кроп). Это связано с тем, что полосы обычно находятся в нижней половине кадра.
- Нормализация — стандартная ImageNet.

### Обработка выхода сети

UFLDv2 может выдавать 4 или 6 выходных тензоров (в зависимости от модели):

```python
# TrafficLaneDetector/ufldDetector/ultrafastLaneDetectorV2.py:109-121

def __process_output(self, output, cfg, local_width=1):
    if len(output) == 6:
        output = {
            "loc_row": output[0], 'loc_col': output[1],
            "exist_row": output[2], "exist_col": output[3],
            'conf_row': output[4], 'conf_col': output[5],
        }
    else:
        output = {
            "loc_row": output[0], 'loc_col': output[1],
            "exist_row": output[2], "exist_col": output[3],
        }
```

**Ключевые выходы UFLDv2:**
- `loc_row` — локации точек вдоль горизонтальной оси (для горизонтальных полос).
- `loc_col` — локации точек вдоль вертикальной оси (для вертикальных полос).
- `exist_row/col` — бинарные флаги существования каждой полосы на данной строке.

### Декодирование точек полос

```python
# Ключевой фрагмент декодирования (ultrafastLaneDetectorV2.py:135-152)

for i in row_lane_idx:  # [1, 2] → left-ego и right-ego
    if valid_row[0, :, i].sum() > num_cls_row / 2:
        for k in range(valid_row.shape[1]):
            if valid_row[0, k, i]:
                all_ind = list(range(max(0, max_indices_row[0, k, i] - local_width),
                                      min(num_grid_row - 1, max_indices_row[0, k, i] + local_width) + 1))
                out_tmp = (_softmax(output['loc_row'][0, all_ind, k, i]) * list(map(float, all_ind))).sum() + 0.5
                out_tmp = out_tmp / (num_grid_row - 1) * original_image_width
                tmp.append((int(out_tmp), int(cfg.row_anchor[k] * original_image_height)))
```

**Алгоритм декодирования:**
1. Для каждой `row_anchor`-строки проверяем `exist_row` — обнаружена ли точка.
2. Берём `argmax` по `loc_row` и окрестность `±local_width` для сабпиксельной точности.
3. Применяем **softmax** для получения распределения вероятностей в окрестности.
4. Взвешенная сумма даёт субпиксельную x-координату.
5. y-координата определяется из `row_anchor` (предопределённые строки).

---

## Полный цикл DetectFrame

```python
# TrafficLaneDetector/ufldDetector/ultrafastLaneDetectorV2.py:175-182

def DetectFrame(self, image, adjust_lanes=True):
    input_tensor = self.__prepare_input(image)
    output = self.engine.engine_inference(input_tensor)
    self.lane_info.lanes_points, self.lane_info.lanes_status = self.__process_output(output, self.cfg)

    self.adjust_lanes = adjust_lanes
    self._LaneDetectBase__update_lanes_status(self.lane_info.lanes_status)
    self._LaneDetectBase__update_lanes_area(self.lane_info.lanes_points, self.img_height)
```

```
BGR-кадр → cvtColor_BGR2RGB → resize + crop снизу
          → ImageNet-нормализация → HWC→NCHW
          → OnnxEngine.engine_inference
          → __process_output → lanes_points[4], lanes_status[4]
          → __update_lanes_status → area_status: bool
          → __update_lanes_area → area_points: np.ndarray (полигон)
```

---

## Визуализация результатов

### DrawDetectedOnFrame — точки полос

```python
# UltrafastLaneDetectorV2.DrawDetectedOnFrame

for lane_num, lane_points in enumerate(self.lane_info.lanes_points):
    if lane_num == 1 and type == OffsetType.RIGHT:
        color = (0, 0, 255)      # красный — предупреждение
    elif lane_num == 2 and type == OffsetType.LEFT:
        color = (0, 0, 255)      # красный — предупреждение
    else:
        color = lane_colors[lane_num]  # стандартные цвета

    for lane_point in lane_points:
        cv2.circle(image, (lane_point[0], lane_point[1]), 3, color, thickness=-1)
```

### DrawAreaOnFrame — полигон зоны полосы

```python
# UltrafastLaneDetectorV2.DrawAreaOnFrame

if self.lane_info.area_status:
    overlay = np.zeros_like(image)
    cv2.fillPoly(overlay, pts=[self.lane_info.area_points], color=(255, 191, 0))
    cv2.addWeighted(overlay, 1 - alpha, image, alpha, 0, dst=image)
```

Полигон заливается полупрозрачным цветом `(255, 191, 0)` — голубым в BGR.

---

## Полиномиальная корректировка полос (adjust_lanes)

При `adjust_lanes=True` базовый класс выполняет полиномиальную аппроксимацию точек каждой полосы 2-й степени:

```python
# TrafficLaneDetector/ufldDetector/core.py:103-141

def __adjust_lanes_points(left_lanes_points, right_lanes_points, image_height):
    if len(left_lanes_points[1]) != 0:
        leftx, lefty = list(zip(*left_lanes_points))
        if len(lefty) > 10:
            left_fit = np.polyfit(lefty, leftx, 2)  # полином 2-й степени

    # Аналогично для правой полосы
    # ...
    left_fitx = left_fit[0]*both_fity**2 + left_fit[1]*both_fity + left_fit[2]
    right_fitx = right_fit[0]*both_fity**2 + right_fit[1]*both_fity + right_fit[2]
```

Это сглаживает шумные результаты детекции и обеспечивает непрерывность полос.

---

## LaneModelType — поддерживаемые модели

```python
# TrafficLaneDetector/ufldDetector/utils.py:3-8

class LaneModelType(Enum):
    UFLD_TUSIMPLE = 0
    UFLD_CULANE = 1
    UFLDV2_TUSIMPLE = 2
    UFLDV2_CULANE = 3
    UFLDV2_CURVELANES = 4
```

---

## Связанные разделы

- [BEV-преобразование](perspective-transformation.md)
- [Логика предупреждений](task-conditions.md)
- [Оценка дистанции](distance-estimation.md)
- [Пайплайн обработки](adas-pipeline.md)