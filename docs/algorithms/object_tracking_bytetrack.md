# Алгоритм: трекинг объектов (ByteTrack / ByteSort)

Ссылка на первоисточник:
- ByteTrack (статья): https://arxiv.org/abs/2110.06864

---

## 1. Назначение алгоритма

**Задача.** Связать детекции объектов между кадрами видеопотока и поддерживать:

- стабильные ID (track_id),
- историю траектории (для визуализации/оценки движения),
- устойчивость к кратким пропускам детекций (окклюзии/blur).

В ADAS‑контуре трекинг нужен, чтобы:

- стабилизировать “впереди идущий объект” (FCWS),
- уменьшить мерцание bbox и улучшить UI/интерпретируемость,
- подготовить базу для более продвинутых метрик (скорость сближения, TTC).

**Почему выбран ByteTrack.**

- Хорошо работает без ReID (только motion + IoU).
- Быстрый и широко применяемый baseline MOT.
- Устойчив к шуму за счёт “двухуровневой” ассоциации high/low score.

**Альтернативы.**

- SORT (проще, но менее устойчив).
- DeepSORT/ByteSort (добавляют ReID, меньше ID‑switch, но дороже по compute).
- OC‑SORT, BoT‑SORT, StrongSORT (улучшенные ассоциации/модели движения).

---

## 2. Теоретическая основа

### 2.1. Модель трека и фильтр Калмана

ByteTrack использует фильтр Калмана, который поддерживает гауссово распределение состояния:

$$
\mathbf{x}_t \sim \mathcal{N}(\mu_t, \Sigma_t)
$$

В стандартной параметризации ByteTrack/SORT состояние часто хранится как `xyah` (center x, center y, aspect ratio, height) + скорости (и/или дополнительные компоненты). В реализации проекта трек предоставляет представления:

- `tlwh`: (x, y, w, h)
- `tlbr`: (x1, y1, x2, y2)
- `xyah`: (cx, cy, a, h), где $a=w/h$

(см. `ObjectTracker/byteTrack/dtypes/strack.py`).

### 2.2. Стоимость ассоциации: IoU distance

Сходство трека и детекции оценивается по IoU:

$$
\mathrm{IoU}(A,B)=\frac{|A\cap B|}{|A\cup B|}
$$

В матрице стоимостей используется:

$$
cost = 1 - \mathrm{IoU}
$$

В проекте: `ObjectTracker/byteTrack/matching.py:iou_distance`.

### 2.3. Assignment (LAPJV / Hungarian)

Задача сопоставления “треки ↔ детекции” решается как задача линейного назначения:

$$
\min \sum_{i,j} C_{ij} X_{ij}
$$

при ограничениях “каждый трек максимум с одной детекцией и наоборот”.

В коде используется `lap.lapjv` (Jonker‑Volgenant) с порогом `cost_limit`:

- `linear_assignment(cost_matrix, thresh)`
- детекции с `cost > thresh` считаются несопоставленными.

### 2.4. Fuse score (как в реализации)

В проекте после IoU‑стоимости применяется корректировка по score детекций:

$$
sim = (1 - cost) \cdot score
$$

$$
fused\_cost = 1 - sim
$$

Это реализовано в `matching.py:fuse_score`.

### 2.5. Двухшаговая ассоциация (ключевой “трюк” ByteTrack)

Детекции делятся на:

- **high score**: `score > track_thresh`,
- **low score**: `0.1 < score < track_thresh`.

Алгоритм:
1) сопоставляет треки с high‑детекциями,
2) затем пытается сопоставить оставшиеся треки с low‑детекциями,
3) затем создаёт новые треки из оставшихся high‑детекций.

Это повышает устойчивость при “просадках” confidence.

---

## 3. Архитектура модели

Трекинг не является нейросетью в данной конфигурации (ByteTrack без ReID).

### ByteSort (объяснение)

ByteSort обычно понимают как “ByteTrack + ReID”: добавляется вектор признаков (эмбеддинг) для ассоциации, чтобы:

- уменьшить ID‑switch при пересечениях,
- лучше переживать окклюзии.

В текущем репозитории ReID‑ветка не реализована, поэтому фактически используется ByteTrack (motion + IoU + score).

---

## 4. Pipeline обработки данных

### Входные данные (на кадр)

- `bboxes`: список bbox в формате `xyxy` (из `RectInfo.tolist(format_type="xyxy")`).
- `scores`: confidence детекций.
- `class_ids`: идентификаторы классов.
- `frame`: кадр BGR (для кропов/визуализации).

### Предобработка

- Разделение детекций на high/low по порогам.
- Преобразования bbox между `tlbr` и `tlwh` (см. `STrack.tlbr_to_tlwh`).

### Основной алгоритм

1. **Predict**: для всех активных и потерянных треков прогнозируем bbox фильтром Калмана.
2. **First association**: IoU distance + fuse_score → assignment (порог `match_thresh`).
3. **Second association**: оставшиеся tracked‑треки сопоставляются с low‑детекциями (порог 0.5 в коде).
4. **Unconfirmed tracks**: отдельная обработка треков, которые были активны только 1 кадр.
5. **Init new tracks**: из оставшихся детекций создаются новые треки, если score выше `det_thresh`.
6. **Lifecycle**: если трек lost слишком долго — перевод в removed.

### Выходные данные

- обновлённые списки треков,
- сообщения активных треков (`get_track_message`),
- опциональная визуализация (`DrawTrackedOnFrame`).

---

## 5. Реализация в проекте

### Основные файлы

- `ObjectTracker/byteTrack/byteTracker.py`:
  - `BYTETracker.update(...)` — основной цикл.
- `ObjectTracker/byteTrack/matching.py`:
  - `iou_distance`, `fuse_score`, `linear_assignment`.
- `ObjectTracker/byteTrack/dtypes/strack.py`:
  - `STrack` + bbox конвертеры + история траекторий.

### Особенности реализации, которые важно знать инженеру

1) **Assignment через LAPJV.** Это быстро, но чувствительно к масштабу стоимостей и порогам.

2) **Fuse score** в текущем коде фактически делает ассоциацию более строгой к детекциям с низким confidence.

3) **class_id**: `STrack` хранит `class_id` и историю `class_id_history`. Однако в пайплайне `adas_pipeline.py` в `class_ids` передаётся `obj.label` (строка). Если `class_id` не является `int`, то:

- типовая логика “один цвет на класс” может работать иначе,
- `class_id_history` станет словарём по строкам (что не обязательно плохо, но важно осознавать).

Рекомендация для промышленного контура: передавать `int class_id` (индекс из классов), а строковый label хранить отдельно.

---

## 6. Псевдокод алгоритма

```pseudo
update_tracks(detections):
  # split detections
  high = det where score > track_thresh
  low  = det where 0.1 < score < track_thresh

  # predict existing tracks
  predict(tracked + lost)

  # first association (high)
  C = 1 - IoU(tracks, high)
  C = fuse_score(C, high.scores)
  matches, u_tracks, u_high = linear_assignment(C, thresh=match_thresh)
  update matched tracks

  # second association (low)
  C2 = 1 - IoU(unmatched_tracked, low)
  matches2 = linear_assignment(C2, thresh=0.5)
  update matched tracks
  mark remaining unmatched as lost

  # init new
  for det in remaining high:
     if det.score >= det_thresh:
        create new track

  # cleanup removed
  if lost too long -> removed
```

---

## 7. Параметры и конфигурация

Параметры конструктора `BYTETracker` (см. `ObjectTracker/byteTrack/byteTracker.py`):

| Параметр | Значение по умолчанию | Влияние |
|---|---:|---|
| `track_thresh` | 0.5 | порог high/low и активации |
| `match_thresh` | 0.8 | строгость ассоциации на high |
| `track_buffer` | 30 | сколько кадров держать lost |
| `frame_rate` | 30 | масштабирует buffer |
| `min_box_area` | 10 | фильтр для отрисовки/траекторий |

Ключевые “внутренние” пороги по коду:

- low‑детекции: `score > 0.1`
- second association: `thresh=0.5`
- `det_thresh = track_thresh + 0.1`

Инженерная настройка:

- при “дрожании” ID: повышать `match_thresh` и/или добавлять ReID (ByteSort‑путь),
- при частых пропусках: увеличить `track_buffer`,
- при шумных детекциях: поднять `track_thresh` и `box_score` в детекторе.
