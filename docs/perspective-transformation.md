# BEV-преобразование и расчёт кривизны/смещения (perspectiveTransformation.py)

## Назначение

Модуль `PerspectiveTransformation` выполняет два ключевых преобразования:

1. **Перспективное преобразование** (Bird's Eye View — BEV) — проецирует изображение и точки полос из фронтального вида в вид сверху для корректного измерения геометрических параметров дороги.
2. **Расчёт смещения и кривизны** — по BEV-точкам левой и правой полос вычисляется радиус кривизны дороги и смещение автомобиля относительно центра полосы.

---

## Класс PerspectiveTransformation

### Инициализация — исходные и целевые точки

```python
# TrafficLaneDetector/ufldDetector/perspectiveTransformation.py:10-37

class PerspectiveTransformation(object):
    def __init__(self, img_size=(1280, 720), logger=None):
        self.img_size = img_size
        self.logger = logger

        self.src = np.float32([
            (self.img_size[0]*0.3,  self.img_size[1]*0.7),   # top-left
            (self.img_size[0]*0.2,  self.img_size[1]),         # bottom-left
            (self.img_size[0]*0.95, self.img_size[1]),         # bottom-right
            (self.img_size[0]*0.8,  self.img_size[1]*0.7)      # top-right
        ])

        offset_x = self.img_size[0] / 4
        offset_y = 0
        self.dst = np.float32([
            (offset_x,            offset_y),
            (offset_x,            img_size[1] - offset_y),
            (img_size[0] - offset_x, img_size[1] - offset_y),
            (img_size[0] - offset_x, offset_y),
        ])

        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.M_inv = cv2.getPerspectiveTransform(self.dst, self.src)
```

**Схема исходной зоны (src):**

```
  ┌────────────────────────────────────────┐
  │          ●top-left    top-right●        │  y = 0.7 * H
  │           (0.3W, 0.7H)  (0.8W, 0.7H)   │
  │             ╲              ╱             │
  │              ╲            ╱              │
  │               ╲          ╱               │
  │                ╲        ╱                │
  │  bottom-left ●────────● bottom-right    │  y = H
  │  (0.2W, H)                (0.95W, H)    │
  └────────────────────────────────────────┘
```

Исходная трапеция вырезает нижнюю часть кадра, где видны полосы, и проецирует её в прямоугольник (BEV).

**Матрица `M`** — матрица перспективного преобразования src → dst, `M_inv` — обратное преобразование.

---

## updateTransformParams — динамическая коррекция зоны

Зона перспективного преобразования адаптируется в зависимости от положения полос на изображении. Тип коррекции определяется в `TaskConditions`:

```python
# perspectiveTransformation.py:39-86

def updateTransformParams(self, left_lanes, right_lanes, type="Bottom"):
    if len(left_lanes) and len(right_lanes):
        left_lanes = np.squeeze(left_lanes)
        right_lanes = np.squeeze(right_lanes)

        if type == "Top":
            # Верхняя граница подстраивается под верх полос
            top_y = min(min(left_lanes[:, 1]), min(right_lanes[:, 1]))
            top_left     = (max(left_lanes[:, 0]) - 20, top_y)
            bottom_left  = (self.src[1][0] - 10, self.src[1][1])
            bottom_right = (self.src[2][0] + 10, self.src[2][1])
            top_right    = (min(right_lanes[:, 0]) + 20, top_y)

        elif type == "Bottom":
            # Нижняя граница подстраивается под низ полос
            top_left     = (self.src[0][0], self.src[0][1])
            bottom_left  = (min(left_lanes[:, 0]) - 20, self.src[1][1])
            bottom_right = (max(right_lanes[:, 0]) + 20, self.src[2][1])
            top_right    = (self.src[3][0], self.src[3][1])

        elif type == "Default":
            # Полная подстройка по обеим границам
            top_y = min(min(left_lanes[:, 1]), min(right_lanes[:, 1]))
            top_left     = (max(left_lanes[:, 0]) - 20, top_y)
            bottom_left  = (min(left_lanes[:, 0]) - 5, self.src[1][1])
            bottom_right = (max(right_lanes[:, 0]) + 5, self.src[2][1])
            top_right    = (min(right_lanes[:, 0]) + 20, top_y)

        self.src = np.float32([top_left, bottom_left, bottom_right, top_right])
        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.M_inv = cv2.getPerspectiveTransform(self.dst, self.src)
```

**Типы коррекции:**
| Тип | Описание |
|-----|----------|
| `Bottom` | Сужает нижнюю часть зоны по обнаруженным полосам |
| `Top` | Расширяет верхнюю часть зоны до начала полос |
| `Default` | Полная подстройка по обоим краям |

---

## transformToBirdView — проекция изображения

```python
# perspectiveTransformation.py:89-103

def transformToBirdView(self, img, flags=cv2.INTER_LINEAR):
    return cv2.warpPerspective(img, self.M, self.img_size, flags=flags)
```

Применяет матрицу `M` ко всему изображению через `cv2.warpPerspective`, создавая вид сверху.

---

## transformToBirdViewPoints — проекция точек

```python
# perspectiveTransformation.py:120-142

def transformToBirdViewPoints(self, points: list) -> Union[list, np.ndarray]:
    points_array = []
    if len(points):
        for x, y in points:
            points_array.append([x, y])
        if len(points_array):
            points_array = np.array(points_array)
            # Однородные координаты + матричное умножение
            new_points = np.einsum('kl, ...l->...k', self.M,
                np.concatenate([points_array,
                                np.broadcast_to(1, (*points_array.shape[:-1], 1))], axis=-1))
            return np.asarray(new_points[..., :2] / new_points[..., 2][..., None], dtype='int')
    return []
```

**Алгоритм:**
1. Точки `(x, y)` переводятся в однородные координаты `(x, y, 1)`.
2. Умножаются на матрицу `M` через `np.einsum`.
3. Результат нормализуется делением на третью координату (`w` в однородных координатах).

---

## calcCurveAndOffset — расчёт кривизны и смещения

Это центральный метод для LDWS/LKAS — вычисляет направление и радиус кривизны дороги, а также смещение автомобиля относительно центра полосы.

```python
# perspectiveTransformation.py:145-213

def calcCurveAndOffset(self, img, left_lanes, right_lanes):
    if len(left_lanes) and len(right_lanes):
        left_lanes = np.squeeze(left_lanes)
        right_lanes = np.squeeze(right_lanes)

        # Полиномиальная аппроксимация 2-й степени: x = Ay² + By + C
        left_fit = np.polyfit(left_lanes[:, 1], left_lanes[:, 0], 2)
        right_fit = np.polyfit(right_lanes[:, 1], right_lanes[:, 0], 2)

        # Определение направления кривизны
        if abs(left_fit[0]) > abs(right_fit[0]):
            side_cr = left_fit[0]
        else:
            side_cr = right_fit[0]

        if side_cr < -0.00015 and (left_lanes[0, 0] <= left_lanes[len(left_lanes)//2, 0]):
            curvature_direction = "L"
        elif side_cr > 0.00015 and (right_lanes[0, 0] >= right_lanes[len(right_lanes)//2, 0]):
            curvature_direction = "R"
        else:
            curvature_direction = "F"

        # Пересчёт в метры (U.S. стандарты: 30 м/720px по Y, 3.7 м/700px по X)
        ym_per_pix = 30 / 720
        xm_per_pix = 3.7 / 700
        y_eval = np.max(ploty)

        left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)

        # Радиус кривизны: R = (1 + (2Ay+B)²)^(3/2) / |2A|
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) \
                        / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) \
                         / np.absolute(2*right_fit_cr[0])

        curvature = (left_curverad + right_curverad) / 2

        # Смещение от центра
        lane_width = np.absolute(leftx[719] - rightx[719])
        lane_xm_per_pix = 3.7 / lane_width
        veh_pos = (leftx[719] + rightx[719]) / 2
        cen_pos = (img.shape[1] / 2)
        distance_from_center = (veh_pos - cen_pos) * lane_xm_per_pix

        return (curvature_direction, curvature), distance_from_center
```

---

## Формулы расчёта

### Радиус кривизны

Полиномиальная аппроксимация BEV-точек полосы 2-й степени:

$$x = A \cdot y^2 + B \cdot y + C$$

Радиус кривизны в точке $y$:

$$R = \frac{(1 + (2Ay + B)^2)^{3/2}}{|2A|}$$

Пересчёт в метры через масштабные коэффициенты:
- $y_{m/pixel}$ = 30/720 м/px (длина дороги ~30 м на 720 пикселей)
- $x_{m/pixel}$ = 3.7/700 м/px (ширина полосы ~3.7 м на 700 пикселей)

### Смещение от центра

$$offset = (veh\_pos - cen\_pos) \times \frac{3.7}{lane\_width\_px}$$

- `veh_pos` — центр между левой и правой полосой внизу BEV-изображения.
- `cen_pos` — геометрический центр изображения.
- Ширина полосы принимается равной **3.7 м** (стандартная ширина полосы в США).

---

## Визуализация на BEV-изображении

```python
# perspectiveTransformation.py:216-225

def DrawDetectedOnBirdView(self, image, lanes_points, type=OffsetType.UNKNOWN):
    for lane_num, lane_points in enumerate(lanes_points):
        if lane_num == 1 and type == OffsetType.RIGHT:
            color = (0, 0, 255)        # красный — предупреждение о смещении вправо
        elif lane_num == 2 and type == OffsetType.LEFT:
            color = (0, 0, 255)        # красный — предупреждение о смещении влево
        else:
            color = lane_colors[lane_num]
        for x, y in lane_points:
            cv2.circle(image, (int(x), int(y)), 10, color, -1)
```

Также на BEV-изображении рисуются стрелки:
- Центр полосы (белая): `cv2.arrowedLine` от `(veh_pos, y_eval)` вверх.
- Центр изображения (серая): `cv2.arrowedLine` от `(cen_pos, y_eval)` вниз.

---

## Интеграция с пайплайном

```python
# adas_pipeline.py:392-407

birdview_show = self.transform_view.transformToBirdView(frame_show)
birdview_lanes_points = [
    self.transform_view.transformToBirdViewPoints(lanes_point)
    for lanes_point in self.lane_detector.lane_info.lanes_points
]
(vehicle_direction, vehicle_curvature), vehicle_offset = \
    self.transform_view.calcCurveAndOffset(birdview_show, *birdview_lanes_points[1:3])
```

Результаты `(curvature_direction, curvature)` и `offset` передаются в `TaskConditions` для принятия решения о предупреждении.

---

## Связанные разделы

- [Детекция разметки](lane-detection.md)
- [Логика предупреждений](task-conditions.md)
- [Пайплайн обработки](adas-pipeline.md)