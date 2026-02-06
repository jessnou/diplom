# Алгоритм: предупреждения и сглаживание статусов (TaskConditions)

Этот алгоритм не является отдельной “детекцией”, но критичен для инженерного поведения ADAS: он превращает непрерывные метрики (distance/offset/curvature) в устойчивые дискретные статусы FCWS/LDWS/LKAS.

---

## 1. Назначение алгоритма

**Задача.**

- Сгладить шумные оценки по времени.
- Выдать дискретные статусы:
  - `FCWS`: `UNKNOWN/NORMAL/PROMPT/WARNING`
  - `LDWS`: `UNKNOWN/LEFT/RIGHT/CENTER`
  - `LKAS`: `UNKNOWN/STRAIGHT/EASY/HARD`
- Управлять режимом адаптации BEV (`transform_status`), чтобы уменьшить дрожание геометрии.

**Почему выбран.**

- Простые правила и медиана по окну дают устойчивость без сложных фильтров.
- Лёгкость подстройки порогов под конкретную камеру/датасет.

**Альтернативы.**

- HMM/CRF для дискретных статусов.
- Калман/EMA для непрерывных метрик (offset/curvature) + hysteresis для статусов.
- TTC‑логика для FCWS вместо порогов по расстоянию.

---

## 2. Теоретическая основа

### 2.1. Temporal aggregation

Пусть $x_t$ — измерение (например, offset) на кадре $t$. Для устойчивости берётся агрегатор по окну:

$$
\tilde{x}_t = \mathrm{median}(x_{t-n+1:t})
$$

Медиана устойчива к выбросам и “скачкам” из‑за ошибок детекции.

### 2.2. Пороговая классификация

Пример для FCWS:

$$
status=
\begin{cases}
WARNING, & d \le \tau \\
PROMPT, & \tau < d \le 2\tau \\
NORMAL, & d > 2\tau
\end{cases}
$$

---

## 3. Архитектура модели

Алгоритм не является нейросетью. Он использует выходы:

- `SingleCamDistanceMeasure` → расстояние до объекта в полосе,
- `PerspectiveTransformation` → offset и curvature,
- `LaneDetectBase` → `area_status`.

---

## 4. Pipeline обработки данных

### Входные данные (на кадр)

- `vehicle_distance`: `[x, y, d_m]` или `None`,
- `lane_area`: `bool` (`lane_info.area_status`),
- `vehicle_offset`: `float` или `None`,
- `vehicle_direction`: `L/R/F` или `None`,
- `vehicle_curvature`: `float` или `None`.

### Основной алгоритм

1. Добавить значения в буферы фиксированной длины.
2. Если буфер заполнен — взять медиану и применить пороговые правила.
3. При отсутствии измерения — сбросить буфер и выставить статус `UNKNOWN` (или `NORMAL` для FCWS при валидной полосе).
4. По паттернам поведения переключить `toggle_status`, который далее влияет на BEV‑адаптацию.

### Выходные данные

- `collision_msg`, `offset_msg`, `curvature_msg` (enum),
- `transform_status` (строка режима `Top/Bottom/Default`).

---

## 5. Реализация в проекте

- Файл: `taskConditions.py`
- Класс: `TaskConditions`

Ключевые методы:

- `UpdateCollisionStatus(vehicle_distance, lane_area, distance_thres=1.5)`
- `UpdateOffsetStatus(vehicle_offset, offset_thres=0.65)`
- `UpdateRouteStatus(vehicle_direction, vehicle_curvature, curvae_thres=500)`
- `CheckStatus()` — сигнал “обновить режим BEV”.

Буферы:

- `vehicle_collision_record = LimitedList(5)`
- `vehicle_offset_record = LimitedList(5)`
- `vehicle_curvature_record = LimitedList(10)`

---

## 6. Псевдокод алгоритма

```pseudo
update_fcws(distance, lane_area):
  if distance exists:
     push(d)
     if window_full:
        d_med = median(window)
        if d_med <= th: status=WARNING
        elif d_med <= 2*th: status=PROMPT
        else: status=NORMAL
  else:
     clear(window)
     status = NORMAL if lane_area else UNKNOWN

update_ldws(offset):
  if offset exists:
     push(offset)
     if window_full:
        o = median(window)
        status = LEFT/RIGHT/CENTER by thresholds
  else:
     clear(window); status=UNKNOWN
```

---

## 7. Параметры и конфигурация

| Параметр | Метод | Значение по умолчанию | Влияние |
|---|---|---:|---|
| `distance_thres` | `UpdateCollisionStatus` | 1.5 м | WARNING/PROMPT пороги |
| `offset_thres` | `UpdateOffsetStatus` | 0.65 м | LEFT/RIGHT пороги |
| `curvae_thres` | `UpdateRouteStatus` | 500 | hard/easy граница |
| окна медианы | `LimitedList` | 5/5/10 | задержка/стабильность |

Инженерные рекомендации:

- делать пороги зависимыми от скорости (например, через TTC),
- добавить hysteresis (разные пороги на вход/выход статуса), чтобы уменьшить дребезг,
- логировать “сырые” и “сглаженные” метрики для отладки.

