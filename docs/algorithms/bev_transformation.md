# Алгоритм: Bird’s Eye View (BEV) преобразование (гомография)

## 1. Назначение алгоритма

**Задача.** Преобразовать изображение из фронтальной перспективы камеры в “вид сверху” (BEV), чтобы:

- упростить геометрию дорожной сцены (линии полос становятся ближе к параллельным),
- стабилизировать расчёт смещения и кривизны по разметке,
- унифицировать координаты для downstream‑логики (offset/curvature).

**Почему выбран.** BEV через гомографию — быстрый, детерминированный и легко калибруемый метод, не требующий глубинной модели.

**Альтернативы.**

- Полная калибровка камеры + обратная проекция на плоскость дороги (intrinsics/extrinsics).
- Монокулярная глубина (MiDaS, DPT) + plane fitting (дорого по compute, сложнее валидация).
- Learning‑based BEV (Lift‑Splat‑Shoot, BEVFormer) — существенно сложнее и тяжелее.

---

## 2. Теоретическая основа

### 2.1. Гомография (проективное преобразование)

Если сцена аппроксимируется плоскостью (дорога), то точки на изображении связаны преобразованием:

$$
\mathbf{p'} \sim \mathbf{H}\mathbf{p},
\quad
\mathbf{p} = (x, y, 1)^T,\;
\mathbf{p'} = (x', y', w')^T
$$

Нормализация:

$$
x_{bev} = \frac{x'}{w'},\quad y_{bev} = \frac{y'}{w'}.
$$

Матрица гомографии $\mathbf{H}$ вычисляется по четырём соответствиям (src↔dst):

$$
\mathbf{H} = \text{getPerspectiveTransform}(\text{src}, \text{dst})
$$

### 2.2. Предположения и применимость

- Дорога локально плоская (plane assumption).
- Камера неподвижна относительно кузова (или изменение мало).
- Выбранная область (src‑четырёхугольник) действительно соответствует дорожной плоскости.

При нарушениях (уклон, кочки, сильный pitch/roll) BEV начинает искажать метрические оценки.

---

## 3. Архитектура модели

Алгоритм не является нейросетью (классическая геометрия, OpenCV).

---

## 4. Pipeline обработки данных

### Входные данные

- `img`: кадр `np.ndarray` (BGR, `uint8`) размера `(H, W, 3)`.
- `lanes_points`: точки линий разметки в координатах исходного кадра (из UFLDv2).

### Предобработка

- Задание `src` и `dst` (4 точки в пикселях) на основе `img_size`.

### Основной алгоритм

1. Вычислить $\mathbf{H}$ и $\mathbf{H}^{-1}$:
   - `M = cv2.getPerspectiveTransform(src, dst)`
   - `M_inv = cv2.getPerspectiveTransform(dst, src)`
2. Преобразовать изображение:
   - `bird = cv2.warpPerspective(img, M, img_size)`
3. Преобразовать точки линий:
   - для каждой точки $(x, y)$ посчитать $(x_{bev}, y_{bev})$ по $\mathbf{H}$.

### Постобработка

- Использовать точки в BEV для оценки offset/curvature.
- (Опционально) отрисовать область преобразования на фронтальном кадре (`DrawTransformFrontalViewArea`).

### Интеграция с другими модулями

- Входы:
  - `TrafficLaneDetector` → `lane_info.lanes_points`
  - `TaskConditions` → `transform_status` (режим адаптации зоны BEV)
- Выходы:
  - `calcCurveAndOffset` (см. `lane_offset_estimation.md`)

---

## 5. Реализация в проекте

### Классы/функции

- Файл: `TrafficLaneDetector/ufldDetector/perspectiveTransformation.py`
- Класс: `PerspectiveTransformation`
- Ключевые методы:
  - `transformToBirdView(img)`
  - `transformToBirdViewPoints(points)`
  - `updateTransformParams(left_lanes, right_lanes, type)`

### Как именно реализовано преобразование точек

В `transformToBirdViewPoints` применяется умножение на матрицу `M` в однородных координатах:

- формируется массив точек `[[x, y], ...]`,
- дописывается компонент `1`,
- считается произведение через `np.einsum`,
- нормализация делением на третий компонент.

### Адаптация области `src` (динамический BEV)

Метод `updateTransformParams(...)` изменяет 4 точки `src` в зависимости от текущих линий:

- `type="Top"`: подстраивает верхние вершины по минимальному `y` линий.
- `type="Bottom"`: подстраивает нижние вершины по экстремумам `x` на нижней части.
- `type="Default"`: комбинированная подстройка.

Режим `type` управляется `TaskConditions` (см. `taskConditions.py`) — это попытка стабилизировать BEV при смене условий/детекта.

---

## 6. Псевдокод алгоритма

```pseudo
init(img_size):
  src = 4 точки на дороге в кадре
  dst = прямоугольник BEV
  M = perspective(src -> dst)
  M_inv = perspective(dst -> src)

update_transform(left_lane_pts, right_lane_pts, mode):
  if mode == "Top":
      src.top_y = min(y в lane pts)
      src.top_left.x  = max(x в left)  - margin
      src.top_right.x = min(x в right) + margin
  if mode == "Bottom":
      src.bottom_left.x  = min(x в left)  - margin
      src.bottom_right.x = max(x в right) + margin
  recompute M, M_inv

to_bev_image(img):
  return warpPerspective(img, M, img_size)

to_bev_points(points):
  for (x,y) in points:
     (x',y',w') = M * (x,y,1)
     append (x'/w', y'/w')
  return points_bev
```

---

## 7. Параметры и конфигурация (из реализации)

### Параметры зоны преобразования по умолчанию

В `__init__` (для `img_size=(W,H)`):

- `src`:
  - top-left: `(0.3W, 0.7H)`
  - bottom-left: `(0.2W, 1.0H)`
  - bottom-right: `(0.95W, 1.0H)`
  - top-right: `(0.8W, 0.7H)`
- `dst` (прямоугольник):
  - `offset_x = W/4`, `offset_y = 0`

### Влияние параметров

- Чем “уже” `dst` (больше `offset_x`), тем сильнее сжатие по x и выше чувствительность offset.
- Положение `src` определяет, какая часть дорожной плоскости считается валидной; неправильный выбор приводит к систематическому смещению.

### Рекомендации по инженерной настройке

- Зафиксировать реальную калибровку камеры и “привязать” `src` к реальным точкам на дороге.
- Логировать `src`/`dst` и метрики offset/curvature, чтобы оценить стабильность при разных условиях.

