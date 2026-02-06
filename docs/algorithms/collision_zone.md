# Алгоритм: определение collision zone (зона “своей полосы”) и проверка объекта

## 1. Назначение алгоритма

**Задача.** Выделить область “своей полосы” на кадре и определить, находится ли обнаруженный объект внутри этой области. Это требуется для:

- уменьшения ложных FCWS‑срабатываний на объекты в соседних полосах,
- выбора “релевантного” ближайшего объекта при многократных детекциях.

**Почему выбран.** Простая геометрическая модель:

- детектор полос даёт две центральные линии,
- из них строится полигон,
- проверка попадания точки объекта в полигон выполняется быстро и надёжно.

**Альтернативы.**

- Semantic segmentation drivable area + lane association.
- 3D‑подход (depth + 3D bbox) с реальным измерением lateral offset.
- Multi‑lane tracking (lane graph) и назначение объекта на lane‑ID.

---

## 2. Теоретическая основа

### 2.1. Полигон полосы

Пусть $L=\{(x_i^L, y_i^L)\}$ — точки левой границы своей полосы, а $R=\{(x_i^R, y_i^R)\}$ — правой.

Полигон области полосы строится как конкатенация:

$$
P = [L,\; \text{reverse}(R)]
$$

То есть замкнутый контур между линиями.

### 2.2. Проверка попадания точки в полигон

Для репрезентативной точки объекта выбирается “контактная” точка bbox:

$$
p_{obj} = (x_{center}, y_{bottom})
$$

Далее выполняется point‑in‑polygon тест. В реализации используется `cv2.pointPolygonTest`, который возвращает:

- `> 0` — точка строго внутри,
- `= 0` — на границе,
- `< 0` — вне полигона.

---

## 3. Архитектура модели

Алгоритм не является нейросетью. Он использует выходы нейросетевых модулей:

- UFLD/UFLDv2 → точки центральных линий,
- YOLO → bbox объекта.

---

## 4. Pipeline обработки данных

### Входные данные

- `lanes_points`: точки линий разметки (как минимум центральная пара).
- `boxes`: детекции объектов (`RectInfo`) с bbox.

### Основной алгоритм

1. Сформировать `area_status`:
   - валиден только если обе центральные линии найдены.
2. Построить `area_points`:
   - полигон между `left-ego` и `right-ego`.
3. Для каждого объекта:
   - вычислить `(x_center, y_bottom)` по bbox,
   - проверить попадание в полигон.
4. Выбрать ближайший объект в зоне (в проекте — по минимальному estimated distance).

### Выходные данные

- `area_status` (bool),
- `area_points` (полигон),
- релевантный объект впереди (или `None`).

---

## 5. Реализация в проекте

### Построение зоны своей полосы

- Файл: `TrafficLaneDetector/ufldDetector/core.py`
- Класс: `LaneDetectBase`
- Метод (внутренний): `__update_lanes_area(lanes_points, img_height)`

Логика:

- `__update_lanes_status(lanes_status)` выставляет `area_status=True`, если найдена центральная пара линий.
- `__update_lanes_area(...)` формирует `lane_info._area_points` как:
  - `vstack(left_ego_points, flipud(right_ego_points))`
  - (опционально) предварительно “дотягивает” точки полиномом при `adjust_lanes=True`.

### Проверка объекта на принадлежность зоне

- Файл: `ObjectDetector/distanceMeasure.py`
- Класс: `SingleCamDistanceMeasure`
- Метод: `calcCollisionPoint(poly)`

Внутри:
- объекты сортируются по расстоянию (которое уже посчитано в `updateDistance`),
- выполняется `cv2.pointPolygonTest(poly, (x,y), False) >= 0`,
- возвращается первый объект, попавший внутрь.

### Интеграция в пайплайн

- Файл: `adas_pipeline.py`
- В `ADASProcessor.process_frame`:
  - после `laneDetector.DetectFrame(frame)` полигон доступен как `laneDetector.lane_info.area_points`,
  - после `distanceDetector.updateDistance(object_info)` вызывается `calcCollisionPoint(area_points)`.

---

## 6. Псевдокод алгоритма

```pseudo
build_lane_polygon(left_ego_pts, right_ego_pts):
  if not left_ego_pts or not right_ego_pts:
     return None
  return concat(left_ego_pts, reverse(right_ego_pts))

is_object_in_zone(bbox_xyxy, polygon):
  x1,y1,x2,y2 = bbox
  p = ((x1+x2)/2, y2)  # bottom-center
  return point_in_polygon(polygon, p)

select_target(objects, polygon):
  candidates = []
  for obj in objects:
     if is_object_in_zone(obj.bbox, polygon):
         candidates.append(obj)
  return argmin(candidates, key=obj.estimated_distance)
```

---

## 7. Параметры и конфигурация

Критичные параметры/допущения:

| Параметр | Где | Значение | Влияние |
|---|---|---:|---|
| “центральная пара линий” | `LaneDetectBase` | индексы `index-1` и `index` | определяет, какие линии считаются своей полосой |
| точка объекта | `SingleCamDistanceMeasure` | `(x_center, y_bottom)` | приближение зоны контакта |
| `adjust_lanes` | `LaneDetectBase` | `True/False` | полиномиальная “дотяжка” точек, влияет на форму полигона |

Инженерные рекомендации:

- если камера/модель часто ошибается на центральных линиях, добавлять фильтр “lane sanity checks” (минимальная ширина, монотонность y, отсутствие самопересечений).
- расширять/сужать полигон с учётом скорости и required time-to-collision (TTC), если цель — именно предупреждение столкновения, а не “объект в полосе”.

