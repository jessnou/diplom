# Алгоритм: расчёт смещения относительно центра полосы и кривизны

## 1. Назначение алгоритма

**Задача.**

- Оценить **смещение автомобиля** относительно центра своей полосы (LDWS‑сигнал).
- Оценить **кривизну траектории/дороги** и направление поворота (LKAS‑сигнал) для информирования/стабилизации.

**Почему выбран.** Полиномиальная аппроксимация линий в BEV:

- проста и быстра,
- хорошо работает при наличии качественных точек разметки,
- даёт интерпретируемые метрики (offset в метрах, радиус кривизны).

**Альтернативы.**

- Модели lane centerline + vehicle pose estimation (с IMU/odometry).
- Fit сплайнами (cubic spline, clothoids) — точнее на сложных кривых, но сложнее.
- Оптимизационные методы на основе карты/HD‑map (если доступно).

---

## 2. Теоретическая основа

### 2.1. Аппроксимация линии полиномом

В BEV принято аппроксимировать линию как функцию $x(y)$:

$$
x(y) = ay^2 + by + c
$$

Коэффициенты оцениваются методом наименьших квадратов (в коде: `np.polyfit(y, x, 2)`).

### 2.2. Радиус кривизны

Если линия задана как $x(y)$ в **метрах**, радиус кривизны:

$$
R(y) = \frac{(1 + (2ay + b)^2)^{3/2}}{|2a|}
$$

В реализации:

- сначала строится полином в пикселях,
- затем выполняется перевод `pixel → meter` через коэффициенты:
  - `ym_per_pix`, `xm_per_pix`,
- затем пересчитывается полином в “мировых” координатах и считается $R$.

### 2.3. Смещение относительно центра полосы

На выбранной линии “опоры” (в коде — нижняя часть кадра) вычисляются:

- `leftx(y_eval)` и `rightx(y_eval)`,
- ширина полосы `lane_width_px = |leftx - rightx|`,
- центр полосы `veh_pos_px = (leftx + rightx)/2`,
- центр изображения `cen_pos_px = W/2`.

Смещение (в метрах):

$$
offset = (veh\_pos\_{px} - cen\_pos\_{px}) \cdot \frac{3.7}{lane\_width\_{px}}
$$

Где `3.7 m` — типовая ширина полосы (эвристика).

### 2.4. Предположения

- Центральные линии (“ego”) корректно определены и соответствуют своей полосе.
- Ширина полосы примерно постоянна и близка к 3.7 м.
- BEV достаточно точен, чтобы использовать линейную метрику по пикселям.

---

## 3. Архитектура модели

Алгоритм не является нейросетью (классическая математика на точках разметки).

---

## 4. Pipeline обработки данных

### Входные данные

- `left_lanes`, `right_lanes`: точки центральных линий в **BEV‑координатах** (обычно `left-ego` и `right-ego`).
- `bird_img`: изображение BEV (используется для отрисовки).

### Основной алгоритм

1. Fit полиномов `left_fit`, `right_fit` по точкам.
2. Сгенерировать `ploty` по высоте BEV и вычислить `leftx/rightx`.
3. Оценить направление поворота по знаку/величине коэффициента `a` (квадратичный член) с порогом.
4. Перевести пиксели в метры (`ym_per_pix`, `xm_per_pix`) и вычислить радиус кривизны `R`.
5. Вычислить `offset` по центру полосы внизу кадра.

### Выходные данные

`(direction, curvature_radius_m), offset_m`

### Интеграция

- `TaskConditions.UpdateOffsetStatus(offset)` → LDWS статус (LEFT/RIGHT/CENTER).
- `TaskConditions.UpdateRouteStatus(direction, curvature)` → LKAS статус (STRAIGHT/EASY/HARD).

---

## 5. Реализация в проекте

- Файл: `TrafficLaneDetector/ufldDetector/perspectiveTransformation.py`
- Метод: `PerspectiveTransformation.calcCurveAndOffset(img, left_lanes, right_lanes)`

Особенности реализации:

- Если точки отсутствуют, возвращает `(None, None), None`.
- Направление кривизны определяется эвристикой:
  - используется `side_cr = max(|a_left|, |a_right|)` и сравнение с порогом `0.00015`.
- Радиус кривизны считается как среднее между левым и правым радиусом.
- Смещение считается на фиксированном индексе `y=719` (это предполагает высоту BEV ~720).

Инженерный риск: жёсткая привязка к `719` может быть некорректна при другом размере BEV‑кадра.

---

## 6. Псевдокод алгоритма

```pseudo
calc_curve_and_offset(left_pts, right_pts, W, H):
  if not left_pts or not right_pts:
     return (None, None), None

  # Fit x(y) = a*y^2 + b*y + c in pixels
  left_fit  = polyfit(y_left,  x_left,  deg=2)
  right_fit = polyfit(y_right, x_right, deg=2)

  ploty = linspace(0, H-1, H)
  leftx  = eval_poly(left_fit, ploty)
  rightx = eval_poly(right_fit, ploty)

  # Direction heuristic
  a = argmax(|left_fit.a|, |right_fit.a|)
  if a < -a_thr: dir = "L"
  elif a > a_thr: dir = "R"
  else: dir = "F"

  # Pixel->meter scaling
  ym_per_pix = 30/720
  xm_per_pix = 3.7/700

  left_fit_m  = polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
  right_fit_m = polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
  y_eval = max(ploty)*ym_per_pix
  R_left  = curvature_radius(left_fit_m, y_eval)
  R_right = curvature_radius(right_fit_m, y_eval)
  R = (R_left + R_right)/2

  y0 = H-1  # bottom row
  lane_width_px = abs(leftx[y0] - rightx[y0])
  lane_xm_per_pix = 3.7 / lane_width_px
  lane_center_px = (leftx[y0] + rightx[y0]) / 2
  image_center_px = W / 2
  offset_m = (lane_center_px - image_center_px) * lane_xm_per_pix

  return (dir, R), offset_m
```

---

## 7. Параметры и конфигурация

Параметры зашиты в коде (требуют калибровки под камеру/разрешение):

| Параметр | Где | Значение | Что делает |
|---|---|---:|---|
| `a_thr` | `calcCurveAndOffset` | `0.00015` | порог определения направления L/R/F |
| `ym_per_pix` | `calcCurveAndOffset` | `30/720` | масштаб по оси y (м/пикс) |
| `xm_per_pix` | `calcCurveAndOffset` | `3.7/700` | масштаб по оси x (м/пикс) |
| `lane_width_m` | offset | `3.7` | принятая ширина полосы (м) |

Инженерные рекомендации:

- заменить жёсткий индекс `719` на `H-1`,
- вынести коэффициенты масштаба в конфиг (и калибровать по реальным данным),
- логировать offset до/после сглаживания `TaskConditions`.

