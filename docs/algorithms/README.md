# Алгоритмы (инженерная документация)

Этот раздел описывает ключевые алгоритмы ADAS‑пайплайна и их реализацию в коде проекта.

## Контроль дистанции (FCWS)

1. `object_detection.md` — YOLO‑детекция объектов (inference + postprocess).
2. `object_tracking_bytetrack.md` — ByteTrack / ByteSort (теория и текущая реализация ByteTrack).
3. `distance_estimation_single_cam.md` — Single‑camera distance estimation по размеру bbox.
4. `collision_zone.md` — определение “collision zone” (полигон своей полосы + попадание объекта в зону).

## Контроль полос (LDWS/LKAS)

5. `lane_detection_ufld_ufldv2.md` — UFLDv2 lane detection (восстановление точек линий).
6. `bev_transformation.md` — преобразование в Bird’s Eye View (гомография) и адаптация области.
7. `lane_offset_estimation.md` — расчёт смещения относительно центра полосы и радиуса кривизны.

## Дополнительно (для интеграции)

- `warnings_and_smoothing.md` — пороги и сглаживание статусов (TaskConditions).
