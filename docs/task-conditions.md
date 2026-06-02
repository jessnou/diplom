# Логика предупреждений — TaskConditions (taskConditions.py)

## Назначение

Модуль `TaskConditions` реализует стабилизацию и агрегацию результатов детекции во времени, а также пороговую логику для выдачи предупреждений:

- **FCWS** (Forward Collision Warning System) — предупреждение о столкновении на основе дистанции до ближайшего объекта в полосе.
- **LDWS** (Lane Departure Warning System) — предупреждение о смещении от центра полосы.
- **LKAS** (Lane Keeping Assist System) — информирование о кривизне дороги (повороты, прямые участки).

Стабилизация необходима, потому что результаты детекции на отдельных кадрах могут быть шумными: объект может временно исчезнуть, детекция полос — дать ложный результат. Медианная фильтрация и пороги по нескольким кадрам устраняют дребезг.

---

## Типы предупреждений

```python
# ObjectDetector/utils.py:8-12

class CollisionType(Enum):
    UNKNOWN = "Determined ..."       # Недостаточно данных
    NORMAL = "Normal Risk"           # Безопасная дистанция
    PROMPT = "Prompt Risk"          # Повышенный риск
    WARNING = "Warning Risk"        # Опасная дистанция

# TrafficLaneDetector/ufldDetector/utils.py:10-22

class OffsetType(Enum):
    UNKNOWN = "To Be Determined ..."    # Недостаточно данных
    RIGHT = "Please Keep Right"         # Смещение вправо
    LEFT = "Please Keep Left"            # Смещение влево
    CENTER = "Good Lane Keeping"         # Движение по центру

class CurvatureType(Enum):
    UNKNOWN = "To Be Determined ..."
    STRAIGHT = "Keep Straight Ahead"
    EASY_LEFT = "Gentle Left Curve Ahead"
    HARD_LEFT = "Hard Left Curve Ahead"
    EASY_RIGHT = "Gentle Right Curve Ahead"
    HARD_RIGHT = "Hard Right Curve Ahead"
```

---

## Класс LimitedList — кольцевой буфер

```python
# taskConditions.py:12-35

class LimitedList(list):
    def __init__(self, maxlen):
        super().__init__()
        self._maxlen = maxlen
        self._is_full = False

    def append(self, element):
        self.__delitem__(slice(0, len(self) == self._maxlen))
        super(LimitedList, self).append(element)
        self._is_full = len(self) >= self._maxlen

    def full(self):
        return self._is_full
```

| Буфер | Размер | Назначение |
|-------|--------|-----------|
| `vehicle_collision_record` | 5 | Агрегация дистанций для FCWS |
| `vehicle_offset_record` | 5 | Агрегация смещений для LDWS |
| `vehicle_curvature_record` | 10 | Агрегация кривизны для LKAS |

Метод `full()` возвращает `True`, когда буфер заполнен — только тогда принимается решение.

---

## Класс TaskConditions — основная логика

### Инициализация

```python
# taskConditions.py:84-98

class TaskConditions(object):
    def __init__(self):
        self.collision_msg = CollisionType.UNKNOWN
        self.offset_msg = OffsetType.UNKNOWN
        self.curvature_msg = CurvatureType.UNKNOWN
        self.vehicle_collision_record = LimitedList(5)
        self.vehicle_offset_record = LimitedList(5)
        self.vehicle_curvature_record = LimitedList(10)
        self.transform_status = None

        self.toggle_status = "Bottom"
        self.toggle_oscillator_status = [False, False]
        self.toggle_status_counter = {"Offset": 0, "Curvae": 0, "BirdViewAngle": 0}
```

`toggle_status` управляет динамической коррекцией зоны BEV-преобразования, `toggle_oscillator_status` — детекция смены полосы.

---

## FCWS: UpdateCollisionStatus

```python
# taskConditions.py:278-307

def UpdateCollisionStatus(self, vehicle_distance, lane_area, distance_thres=1.5):
    if vehicle_distance is not None:
        x, y, d = vehicle_distance
        self.vehicle_collision_record.append(d)
        if self.vehicle_collision_record.full():
            avg_vehicle_collision = np.median(self.vehicle_collision_record)
            if avg_vehicle_collision <= distance_thres:           # ≤ 1.5 м
                self.collision_msg = CollisionType.WARNING
            elif distance_thres < avg_vehicle_collision <= 2 * distance_thres:  # 1.5–3.0 м
                self.collision_msg = CollisionType.PROMPT
            else:                                                   # > 3.0 м
                self.collision_msg = CollisionType.NORMAL
    else:
        if lane_area:
            self.collision_msg = CollisionType.NORMAL
        else:
            self.collision_msg = CollisionType.UNKNOWN
        self.vehicle_collision_record.clear()
```

**Пороги дистанции (по умолчанию `distance_thres=1.5` м):**

```
┌──────────────────────────────────────────────────────────┐
│                                                         │
│  <── 1.5 м ──>  WARNING    (красный)                    │
│  <── 3.0 м ──>  PROMPT     (оранжевый)                  │
│  >── 3.0 м ──>  NORMAL     (зелёный)                    │
│                                                         │
│  Нет данных    UNKNOWN     (жёлтый)                      │
└──────────────────────────────────────────────────────────┘
```

- Используется **медиана** (не среднее), что устойчиво к выбросам.
- Если объектов в полосе нет, но полоса обнаружена — `NORMAL`.
- Если полоса не обнаружена — `UNKNOWN` + очистка буфера.

---

## LDWS: UpdateOffsetStatus

```python
# taskConditions.py:195-234

def UpdateOffsetStatus(self, vehicle_offset, offset_thres=0.65):
    if vehicle_offset is not None:
        self.vehicle_offset_record.append(vehicle_offset)
        if self.vehicle_offset_record.full():
            avg_vehicle_offset = np.median(self.vehicle_offset_record)
            self.offset_msg = self._calc_deviation(avg_vehicle_offset, offset_thres)
            # ... логика детекции смены полосы (toggle_oscillator_status)
    else:
        self.offset_msg = OffsetType.UNKNOWN
        self.vehicle_offset_record.clear()
```

```python
# taskConditions.py:121-143

def _calc_deviation(self, offset, offset_thres):
    if abs(offset) > offset_thres:            # |смещение| > 0.65 м
        if offset > 0 and self.curvature_msg not in {HARD_LEFT, EASY_LEFT}:
            return OffsetType.RIGHT
        elif offset < 0 and self.curvature_msg not in {HARD_RIGHT, EASY_RIGHT}:
            return OffsetType.LEFT
        else:
            return OffsetType.UNKNOWN          # во время поворота смещение игнорируется
    else:
        return OffsetType.CENTER               # |смещение| ≤ 0.65 м
```

**Логика подавления ложных срабатываний:** если автомобиль смещён влево при правом повороте (или наоборот), статус устанавливается в `UNKNOWN`, а не `LEFT`/`RIGHT`.

---

## LKAS: UpdateRouteStatus

```python
# taskConditions.py:236-276

def UpdateRouteStatus(self, vehicle_direction, vehicle_curvature, curvae_thres=500):
    if vehicle_curvature is not None:
        if vehicle_direction is not None and self.offset_msg == OffsetType.CENTER:
            self.vehicle_curvature_record.append([vehicle_direction, vehicle_curvature])

            if self.vehicle_curvature_record.full():
                avg_direction = max(set(...), key=...)  # мода направлений
                avg_curvature = np.median([...])         # медиана кривизны
                self.curvature_msg = self._calc_direction(avg_curvature, avg_direction, curvae_thres)
    else:
        self.vehicle_curvature_record.clear()
        self.curvature_msg = CurvatureType.UNKNOWN
```

```python
# taskConditions.py:145-172

def _calc_direction(self, curvature, curvae_dir, curvae_thres):
    if curvature <= curvae_thres:     # радиус ≤ 500 (резкий поворот)
        if curvae_dir == "L":
            return CurvatureType.HARD_LEFT
        elif curvae_dir == "R":
            return CurvatureType.HARD_RIGHT
        else:
            return CurvatureType.UNKNOWN
    else:                               # радиус > 500 (пологий поворот или прямая)
        if curvae_dir == "L":
            return CurvatureType.EASY_LEFT
        elif curvae_dir == "R":
            return CurvatureType.EASY_RIGHT
        else:
            return CurvatureType.STRAIGHT
```

**Пороги кривизны:**
- `curvature ≤ 500` → «резкий» поворот (HARD_LEFT / HARD_RIGHT).
- `curvature > 500` → «пологий» поворот или прямая.

> **Примечание:** порог 500 — это радиус кривизны в условных единицах. Меньшее значение = более резкий поворот, так как радиус определяется как $R = \frac{(1+(2Ay+B)^2)^{3/2}}{|2A|}$, и при маленьком радиусе значение `curvature` тоже мало.

**Условие оценки:** кривизна оценивается **только когда автомобиль находится по центру полосы** (`offset_msg == OffsetType.CENTER`), чтобы избежать ложных срабатываний при смене полосы.

---

## Динамическая коррекция BEV: CheckStatus и toggle-механизм

```python
# taskConditions.py:174-193

def CheckStatus(self):
    if self.curvature_msg == CurvatureType.UNKNOWN and self.offset_msg == OffsetType.UNKNOWN:
        self.toggle_oscillator_status = [False, False]

    if self.toggle_status != self.transform_status:
        self.transform_status = self.toggle_status
        self.toggle_status = None
        return True     # → обновить зону BEV
    else:
        return False    # → оставить текущую зону
```

`toggle_status` обновляется в `UpdateOffsetStatus` и `UpdateRouteStatus` при осцилляции смещения (детекция смены полосы):

```python
# В UpdateOffsetStatus (фрагмент):
if np.array(self.toggle_oscillator_status).all():
    self.toggle_status = "Top"       # обе фазы осцилляции → расширить зону сверху
    self.toggle_oscillator_status = [False, False]

# В UpdateRouteStatus (фрагмент):
if self.curvature_msg != STRAIGHT and abs(offset) < 0.2 and not any(oscillator):
    self.toggle_status = "Bottom"     # кривая + малое смещение → сузить зону снизу
```

---

## _calibration_curve — калибровка по прямой дороге

```python
# taskConditions.py:99-119

def _calibration_curve(self, vehicle_curvature, frequency=3, curvae_thres=15000):
    if self.toggle_status_counter["BirdViewAngle"] <= frequency:
        if vehicle_curvature >= curvae_thres:
            self.toggle_status_counter["BirdViewAngle"] += 1
        else:
            self.toggle_status_counter["BirdViewAngle"] = 0
    else:
        self.toggle_status_counter["BirdViewAngle"] = 0
        self.toggle_status = "Default"   # перейти к стандартной зоне BEV
```

Если автомобиль едет по прямой (`curvature ≥ 15000`) на протяжении `frequency` кадров, зона BEV сбрасывается к `Default`.

---

## Схема взаимодействия

```
                     ┌────────────────────────────────────────┐
                     │         ADASProcessor.process_frame     │
                     └──────┬─────────┬─────────┬──────────────┘
                            │         │         │
          vehicle_distance  │         offset   │ (direction, curvature)
                            │         │         │
                            ▼         ▼         ▼
               ┌──────────────────────────────────────────┐
               │           TaskConditions                  │
               │                                          │
               │  ┌─────────────────┐  ┌────────────────┐  │
               │  │ collision_record│  │ offset_record  │  │
               │  │   LimitedList(5)│  │ LimitedList(5) │  │
               │  └───────┬─────────┘  └───────┬────────┘  │
               │          │                     │          │
               │          ▼                     ▼          │
               │  ┌─────────────────┐  ┌────────────────┐  │
               │  │ CollisionType   │  │  OffsetType    │  │
               │  │ WARNING/PROMPT/ │  │  LEFT/RIGHT/   │  │
               │  │ NORMAL/UNKNOWN  │  │  CENTER/UNKNOWN│  │
               │  └─────────────────┘  └────────────────┘  │
               │                                          │
               │  ┌─────────────────┐  ┌────────────────┐  │
               │  │curvature_record │  │CurvatureType   │  │
               │  │  LimitedList(10)│  │STRAIGHT/EASY_* │  │
               │  └───────┬─────────┘  │/HARD_*/UNKNOWN │  │
               │          │             └────────────────┘  │
               │          │  toggle_status ─────────┐      │
               │          │                         │      │
               │          ▼                         ▼      │
               │  CheckStatus() ──> PerspectiveTransformation│
               └──────────────────────────────────────────┘
```

---

## Связанные разделы

- [BEV-преобразование](perspective-transformation.md)
- [Оценка дистанции](distance-estimation.md)
- [Пайплайн обработки](adas-pipeline.md)