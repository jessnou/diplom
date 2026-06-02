# Оценка дистанции по одной камере (distanceMeasure.py)

## Назначение

Модуль `SingleCamDistanceMeasure` реализует оценку расстояния до объектов на изображении с одной фронтальной камеры. Используется приближение тонкой линзы для вычисления расстояния по высоте bounding box объекта. Дополнительно проверяется, находится ли объект внутри зоны текущей полосы (`area_points`), для определения риска столкновения (FCWS).

---

## Принцип работы: формула тонкой линзы

Расстояние до объекта вычисляется по формуле:

```
D = (H_real × f) / H_px
```

где:
- `H_real` — реальная высота объекта (из справочной таблицы `RefSizeDict`), в дюймах.
- `f` — фокусное расстояние камеры (константа `f = 200`), в пикселях.
- `H_px` — высота bounding box в пикселях (`ymax - ymin`).
- `D` — результат в футах, который конвертируется в метры: `D_meters = D_ft × 0.3048`.

---

## Код: справочная таблица размеров

```python
# ObjectDetector/distanceMeasure.py:7-16

class SingleCamDistanceMeasure(object):
    INCH = 0.39  # 1 см = 0.39 дюйма
    RefSizeDict = {
        "person":    (160 * INCH, 50 * INCH),    # высота × ширина (в дюймах)
        "bicycle":   (98 * INCH, 65 * INCH),
        "motorbike": (100 * INCH, 100 * INCH),
        "car":       (150 * INCH, 180 * INCH),
        "bus":       (319 * INCH, 250 * INCH),
        "truck":     (346 * INCH, 250 * INCH),
    }

    def __init__(self, object_list=["person", "bicycle", "car", "motorbike", "bus", "truck"]):
        self.object_list = object_list
        self.f = 200  # фокусное расстояние в пикселях
        self.distance_points = []
```

> **Примечание:** Фокусное расстояние `f=200` — калибровочная константа, которая подбирается для конкретной камеры. В идеале она должна быть получена из параметров калибрации камеры.

---

## updateDistance — обновление расстояний до всех объектов

```python
# ObjectDetector/distanceMeasure.py:49-73

def updateDistance(self, boxes: typing.List[RectInfo]) -> None:
    self.distance_points = []
    if len(boxes) != 0:
        for box in boxes:
            xmin, ymin, xmax, ymax = box.tolist()
            label = box.label

            if label in self.object_list and ymax <= 650:
                point_x = (xmax + xmin) // 2   # центр объекта по X
                point_y = ymax                    # нижняя граница объекта

                try:
                    distance = (self.RefSizeDict[label][0] * self.f) / (ymax - ymin)
                    distance = distance / 12 * 0.3048  # дюймы → футы → метры
                    self.distance_points.append([point_x, point_y, distance])
                except (KeyError, ZeroDivisionError):
                    pass
```

**Ключевые моменты:**
- Учитываются только объекты из `object_list` (person, car, truck, bus, motorbike, bicycle).
- Фильтрация `ymax <= 650` — объекты в верхней части кадра (далёкие) не учитываются для уменьшения шума. Порог 650px связан с разрешением 720p.
- Точка привязки для каждого объекта: `(center_x, bottom_y)` — точка под центром объекта на уровне дороги.
- При `ZeroDivisionError` (нулевой размер bbox) или `KeyError` (неизвестный класс) объект пропускается.

---

## calcCollisionPoint — ближайший объект в зоне полосы

```python
# ObjectDetector/distanceMeasure.py:75-92

def calcCollisionPoint(self, poly: np.ndarray) -> typing.Union[list, None]:
    if len(self.distance_points) != 0 and len(poly):
        sorted_distance_points = sorted(self.distance_points, key=lambda arr: arr[2])
        for x, y, d in sorted_distance_points:
            # Проверка: находится ли точка объекта внутри полигона полосы
            status = True if cv2.pointPolygonTest(poly, ((x, y)), False) >= 0 else False
            if status:
                return [x, y, d]
    return None
```

**Алгоритм:**
1. Все обнаруженные объекты сортируются по расстоянию (от ближайшего к дальнему).
2. Для каждого объекта проверяется: попадает ли его нижняя точка `(x, y)` внутрь полигона `area_points` текущей полосы.
3. Первый попавший внутрь объект становится «точкой столкновения».
4. Если ни один объект не находится в полосе — возвращается `None`.

```python
# В пайплайне (adas_pipeline.py:389)
vehicle_distance = self.distance_detector.calcCollisionPoint(
    self.lane_detector.lane_info.area_points
)
```

---

## DrawDetectedOnFrame — визуализация

```python
# ObjectDetector/distanceMeasure.py:94-114

def DrawDetectedOnFrame(self, frame_show):
    if len(self.distance_points) != 0:
        for x, y, d in self.distance_points:
            cv2.circle(frame_show, (x, y), 4, (255, 255, 255), thickness=-1)

            unit = 'm'
            if d < 0:
                text = ' {} {}'.format("unknown", unit)
            else:
                text = ' {:.2f} {}'.format(d, unit)

            fontScale = max(0.4, min(1, 1/d))
            textsize = cv2.getTextSize(text, 0, fontScale=fontScale, thickness=3)[0]
            textX = int((x - textsize[0]/2))
            textY = int((y + textsize[1]))
            putText_shadow(frame_show, text, (textX + 1, textY + 5), ...)
```

- `fontScale` масштабируется обратно пропорционально расстоянию — чем ближе объект, тем крупнее текст.
- Используется `putText_shadow` для отрисовки с тенью, обеспечивающей читаемость на любом фоне.

---

## Схема работы

```
  ┌──────────────────────────────────────────────────────────────┐
  │                     Видеокадр (BGR)                           │
  │                                                              │
  │    ┌─────────────┐          ┌─────┐                          │
  │    │  ObjectDet  │          │ Car │  ← bounding box          │
  │    │  (RectInfo) │          └─────┘                          │
  │    └──────┬──────┘                                            │
  │           │                                                    │
  │           ▼                                                    │
  │  ┌─────────────────────┐                                     │
  │  │ updateDistance()     │                                     │
  │  │  D = H_real·f/H_px  │  → distance_points: [[x, y, d]]    │
  │  └──────────┬──────────┘                                     │
  │             │                                                 │
  │             │     ┌──────────────────────────────────┐        │
  │             │     │ LaneDetector.lane_info.area_points│        │
  │             │     │  (полигон текущей полосы)         │        │
  │             │     └──────────────┬───────────────────┘        │
  │             │                    │                             │
  │             ▼                    ▼                             │
  │  ┌─────────────────────────────────────────────┐              │
  │  │ calcCollisionPoint(area_points)              │              │
  │  │  → сортировка по дистанции                  │              │
  │  │  → cv2.pointPolygonTest для каждого объекта  │              │
  │  │  → возврат ближайшего [x, y, d] в полосе    │              │
  │  └─────────────────────────────────────────────┘              │
  └──────────────────────────────────────────────────────────────┘
```

---

## Ограничения

1. **Калибровочная константа `f=200`** — не адаптируется к конкретной камере. В идеале нужна калибрация через шахматную доску или известные маркеры.
2. **Фиксированные размеры объектов** — реальная высота может значительно отличаться (например, малолитражка vs внедорожник).
3. **Порог `ymax <= 650`** — завязан на разрешение 720p; при других разрешениях требует корректировки.
4. **Не учитывается наклон камеры** — не компенсируется pitch/roll.

---

## Связанные разделы

- [Детекция объектов](object-detection.md)
- [Детекция разметки](lane-detection.md)
- [Логика предупреждений FCWS](task-conditions.md)
- [Пайплайн обработки](adas-pipeline.md)