# Трекинг объектов — ByteTrack (ObjectTracker/)

## Назначение

Модуль `ObjectTracker` реализует алгоритм **ByteTrack** для устойчивого сопровождения объектов между кадрами. ByteTrack присваивает каждому обнаруженному объекту уникальный `track_id`, который сохраняется при кратковременных окклюзиях, и формирует траекторию движения.

Это критически важно для системы FCWS: трекинг позволяет не только обнаруживать объект, но и отслеживать его приближение/удаление, вычисляя стабильную оценку дистанции.

---

## Структура модуля

```
ObjectTracker/
├── __init__.py                  # Экспорт BYTETracker, ObjectTrackBase
├── core.py                     # Базовый класс ObjectTrackBase, отрисовка
└── byteTrack/
    ├── byteTracker.py           # Основной класс BYTETracker
    ├── matching.py              # IoU-distance, fuse_score, LAP assignment
    └── dtypes/
        ├── base_track.py       # Счётчик ID, TrackState
        ├── kalman_filter.py    # Фильтр Калмана для предсказания позиции
        └── strack.py           # STrack — отдельный трек
```

---

## Иерархия классов

```
ObjectTrackBase (ABC)
    │   names: dict            — имена классов/цвета
    │   class_colors: list     — цвета для визуализации
    │   update()               — абстрактный, возвращает List[Dict]
    │
    └── BYTETracker
            tracked_stracks: List[STrack]
            lost_stracks: List[STrack]
            removed_stracks: List[STrack]
            kalman_filter: KalmanFilter
            track_thresh: float = 0.5
            match_thresh: float = 0.8
            det_thresh: float = 0.6  (track_thresh + 0.1)
```

---

## Алгоритм ByteTrack: ключевая идея

Классический трекинг (SORT, DeepSORT) отбрасывает детекции с низким скором. **ByteTrack** использует двухэтапную ассоциацию:

1. **Первый этап** — высокоуверенные детекции (`score > track_thresh`) ассоциируются с существующими треками через IoU + fuse_score.
2. **Второй этап** — низкоуверенные детекции (`0.1 < score < track_thresh`) ассоциируются с оставшимися (unmatched) треками второй раз. Это позволяет «подхватывать» объекты при частичной окклюзии, когда детектор выдаёт низкий скор.

```
                ┌────────────────────────────────────────────────┐
                │           Все детекции (N штук)                │
                └───────────┬────────────────────┬───────────────┘
                            │                    │
                 score > track_thresh    0.1 < score < track_thresh
                            │                    │
                    ┌───────▼──────┐     ┌───────▼──────┐
                    │  Высокоур.   │     │ Низкоур.     │
                    │  детекции    │     │  детекции    │
                    └───────┬──────┘     └───────┬──────┘
                            │                    │
                 ┌──────────▼──────────┐  ┌──────▼──────────┐
                 │ Ассоциация 1:       │  │ Ассоциация 2:    │
                 │ IoU + fuse_score    │  │ IoU (thresh=0.5) │
                 │ с tracked+lost      │  │ с unmatched      │
                 └────────────────────┘  └──────────────────┘
```

---

## Код: BYTETracker.update()

Основной метод вызывается на каждом кадре с новыми детекциями.

```python
# ObjectTracker/byteTrack/byteTracker.py:62-185

def update(self, bboxes, scores, class_ids, frame: np.ndarray):
    self.frame_id += 1
    activated_stracks = []
    refind_stracks = []
    lost_stracks = []
    removed_stracks = []

    # Разделение детекций на высоко- и низкоуверенные
    remain_inds = scores > self.track_thresh
    inds_low = scores > 0.1
    inds_high = scores < self.track_thresh
    inds_second = np.logical_and(inds_low, inds_high)

    dets = bboxes[remain_inds]          # высокоуверенные
    dets_second = bboxes[inds_second]   # низкоуверенные

    # Шаг 1: Создание STrack из высокоуверенных детекций
    detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, class_id)
                  for (tlbr, s, class_id) in zip(dets, scores_keep, class_ids_keep)]

    # Шаг 2: Первая ассоциация (KF-предсказание + IoU + fuse_score)
    strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
    STrack.multi_predict(strack_pool)
    dists = matching.iou_distance(strack_pool, detections)
    dists = matching.fuse_score(dists, detections)
    matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.match_thresh)

    # ... обновление matched треков ...

    # Шаг 3: Вторая ассоциация (низкоуверенные с unmatched-треками)
    r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
    dists = matching.iou_distance(r_tracked_stracks, detections_second)
    matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)

    # ... обновление и mark_lost для unmatched ...

    # Шаг 4: Ассоциация неподтверждённых треков
    detections = [detections[i] for i in u_detection]
    dists = matching.iou_distance(unconfirmed, detections)
    dists = matching.fuse_score(dists, detections)
    matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)

    # Шаг 5: Инициализация новых треков
    for inew in u_detection:
        track = detections[inew]
        if track.score < self.det_thresh:
            continue
        track.activate(self.kalman_filter, self.frame_id)

    # Шаг 5: Удаление старых lost-треков
    for track in self.lost_stracks:
        if self.frame_id - track.end_frame > self.max_time_lost:
            track.mark_removed()

    return self._get_tracker_messages()
```

---

## Ассоциация: fuse_score + линейное назначение

```python
# ObjectTracker/byteTrack/matching.py:108-116

def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost
```

**Суть:** вместо чистого IoU-distance (расстояние = 1 - IoU), `fuse_score` умножает IoU-похожесть на скоре детекции. Это наказывает пары с высоким IoU, но низким скором детекции.

Для решения задачи назначения используется библиотека `lap` (Linear Assignment Problem solver):

```python
# ObjectTracker/byteTrack/matching.py:20-31

def linear_assignment(cost_matrix, thresh):
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    # x[i] = j → трек i назначен детекции j (или -1 если не назначен)
```

---

## Kalman Filter в STrack

Каждый `STrack` ведёт фильтр Калмана для предсказания позиции на текущем кадре, даже когда детекция отсутствует.

Состояние KF: `[x_center, y_center, aspect_ratio, height, vx, vy, va, vh]` (8 параметров).

```python
# ObjectTracker/byteTrack/dtypes/strack.py (ключевые методы)

class STrack(BaseTrack):
    def predict(self):
        mean = np.copy(self.mean)
        if self.state != TrackState.Tracked:
            mean[6] = 0  # обнуление скорости при потере
        self.mean, self.covariance = self.kalman_filter.predict(mean, self.covariance)

    def update(self, new_track, frame_id):
        self.frame_id = frame_id
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_track.to_xyah()
        )
```

---

## Интеграция с пайплайном ADAS

```python
# adas_pipeline.py:382-386

boxes = [obj.tolist(format_type="xyxy") for obj in self.object_detector.object_info]
scores = [obj.conf for obj in self.object_detector.object_info]
class_ids = [obj.label for obj in self.object_detector.object_info]

self.object_tracker.update(boxes, scores, class_ids, frame)
```

Вызов `DrawTrackedOnFrame` формирует визуализацию: bounding box + ID + траектория + направление движения.

```python
# adas_pipeline.py:413

self.object_tracker.DrawTrackedOnFrame(frame_show, False)  # show_box=False, show_traject=True
```

---

## Визуализация треков

`ObjectTrackBase` предоставляет методы отрисовки:

| Метод | Назначение |
|-------|-----------|
| `plot_bbox()` | Bounding Box + ID + класс |
| `plot_trajectories()` | Точки траектории с нарастающей толщиной |
| `plot_directions()` | Стрелка направления движения (медиана векторов) |

Особенности:
- `plot_directions()` показывает стрелку только при `len(directions) >= lock_count` (5 наблюдений).
- Текст отрисовывается с тенью (`putText_shadow`) для лучшей читаемости.

---

## Связанные разделы

- [Детекция объектов](object-detection.md)
- [Оценка дистанции](distance-estimation.md)
- [Пайплайн обработки](adas-pipeline.md)