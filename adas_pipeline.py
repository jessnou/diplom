import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from ObjectTracker import BYTETracker
from taskConditions import Logger, TaskConditions
from ObjectDetector import EfficientdetDetector, YoloDetector
from ObjectDetector.distanceMeasure import SingleCamDistanceMeasure
from ObjectDetector.utils import CollisionType, ObjectModelType
from TrafficLaneDetector import UltrafastLaneDetector, UltrafastLaneDetectorV2
from TrafficLaneDetector.ufldDetector.perspectiveTransformation import BevConfig, PerspectiveTransformation
from TrafficLaneDetector.ufldDetector.utils import CurvatureType, LaneModelType, OffsetType


def load_image_safe(path: str, size: Tuple[int, int]) -> np.ndarray:
    if os.path.exists(path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if img is not None:
            img = cv2.resize(img, size)

            # --- гарантируем 4 канала ---
            if img.shape[2] == 3:
                alpha = np.full((img.shape[0], img.shape[1], 1), 255, dtype=np.uint8)
                img = np.concatenate([img, alpha], axis=2)

            return img

    h, w = size[1], size[0]
    return np.zeros((h, w, 4), dtype=np.uint8)


class ControlPanel(object):
    CollisionDict = {
        CollisionType.UNKNOWN: (0, 255, 255),
        CollisionType.NORMAL: (0, 255, 0),
        CollisionType.PROMPT: (0, 102, 255),
        CollisionType.WARNING: (0, 0, 255),
    }

    OffsetDict = {
        OffsetType.UNKNOWN: (0, 255, 255),
        OffsetType.RIGHT: (0, 0, 255),
        OffsetType.LEFT: (0, 0, 255),
        OffsetType.CENTER: (0, 255, 0),
    }

    CurvatureDict = {
        CurvatureType.UNKNOWN: (0, 255, 255),
        CurvatureType.STRAIGHT: (0, 255, 0),
        CurvatureType.EASY_LEFT: (0, 102, 255),
        CurvatureType.EASY_RIGHT: (0, 102, 255),
        CurvatureType.HARD_LEFT: (0, 0, 255),
        CurvatureType.HARD_RIGHT: (0, 0, 255),
    }

    def __init__(self, assets_dir: Optional[str] = None):
        script_dir = os.path.dirname(os.path.realpath(__file__))
        assets_dir = assets_dir or os.path.join(script_dir, "assets")

        self.collision_warning_img = load_image_safe(os.path.join(assets_dir, "FCWS-warning.png"), (100, 100))
        self.collision_prompt_img = load_image_safe(os.path.join(assets_dir, "FCWS-prompt.png"), (100, 100))
        self.collision_normal_img = load_image_safe(os.path.join(assets_dir, "FCWS-normal.jpg"), (100, 100))

        self.left_curve_img = load_image_safe(os.path.join(assets_dir, "left_turn.png"), (200, 200))
        self.right_curve_img = load_image_safe(os.path.join(assets_dir, "right_turn.jpg"), (200, 200))
        self.keep_straight_img = load_image_safe(os.path.join(assets_dir, "straight.png"), (200, 200))
        self.determined_img = load_image_safe(os.path.join(assets_dir, "warn.png"), (200, 200))

        self.left_lanes_img = load_image_safe(os.path.join(assets_dir, "LTA-left_lanes.png"), (300, 200))
        self.right_lanes_img = load_image_safe(os.path.join(assets_dir, "LTA-right_lanes.png"), (300, 200))

        self._alpha_masks = {}
        self._precompute_alpha_masks()

        self.fps = 0.0
        self.frame_count = 0
        self.start = time.time()
        self.curve_status = None

    def _precompute_alpha_masks(self):
        for name in [
            "determined_img", "left_curve_img", "right_curve_img",
            "keep_straight_img", "collision_warning_img",
            "collision_prompt_img", "collision_normal_img",
        ]:
            img = getattr(self, name, None)
            if img is not None and img.shape[2] == 4:
                mask = img[:, :, 3]
                self._alpha_masks[name] = mask.nonzero()
        for name in ["left_lanes_img", "right_lanes_img"]:
            img = getattr(self, name, None)
            if img is not None and img.shape[2] >= 3:
                mask = img[:, :, 2]
                self._alpha_masks[name] = mask.nonzero()

    def updateFPS(self):
        self.frame_count += 1
        if self.frame_count >= 30:
            end = time.time()
            self.fps = self.frame_count / max(1e-6, (end - self.start))
            self.frame_count = 0
            self.start = time.time()

    def DisplayBirdViewPanel(self, main_show, min_show, show_ratio=0.25):
        W = int(main_show.shape[1] * show_ratio)
        H = int(main_show.shape[0] * show_ratio)
        min_birdview_show = cv2.resize(min_show, (W, H))
        min_birdview_show = cv2.copyMakeBorder(
            min_birdview_show, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        main_show[0 : min_birdview_show.shape[0], -min_birdview_show.shape[1] :] = min_birdview_show

    def DisplaySignsPanel(self, main_show, offset_type, curvature_type, collision_type):
        W = 400
        H = 365
        main_show[:H, :W] //= 2
        main_show[0:3, :W] = [0, 0, 255]
        main_show[H - 3:H - 1, :W] = [0, 0, 255]
        main_show[:H, 0:3] = [0, 0, 255]
        main_show[:H, W - 3:W - 1] = [0, 0, 255]

        if curvature_type == CurvatureType.UNKNOWN and offset_type in {OffsetType.UNKNOWN, OffsetType.CENTER}:
            y, x = self._alpha_masks.get("determined_img", ([], []))
            if len(y) > 0:
                main_show[y + 10, x - 100 + W // 2] = self.determined_img[y, x, :3]
            self.curve_status = None
        elif (curvature_type == CurvatureType.HARD_LEFT or self.curve_status == "Left") and (
            curvature_type not in {CurvatureType.EASY_RIGHT, CurvatureType.HARD_RIGHT}
        ):
            y, x = self._alpha_masks.get("left_curve_img", ([], []))
            if len(y) > 0:
                main_show[y + 10, x - 100 + W // 2] = self.left_curve_img[y, x, :3]
            self.curve_status = "Left"
        elif (curvature_type == CurvatureType.HARD_RIGHT or self.curve_status == "Right") and (
            curvature_type not in {CurvatureType.EASY_LEFT, CurvatureType.HARD_LEFT}
        ):
            y, x = self._alpha_masks.get("right_curve_img", ([], []))
            if len(y) > 0:
                main_show[y + 10, x - 100 + W // 2] = self.right_curve_img[y, x, :3]
            self.curve_status = "Right"

        if offset_type == OffsetType.RIGHT:
            y, x = self._alpha_masks.get("left_lanes_img", ([], []))
            if len(y) > 0:
                main_show[y + 10, x - 150 + W // 2] = self.left_lanes_img[y, x, :3]
        elif offset_type == OffsetType.LEFT:
            y, x = self._alpha_masks.get("right_lanes_img", ([], []))
            if len(y) > 0:
                main_show[y + 10, x - 150 + W // 2] = self.right_lanes_img[y, x, :3]
        elif curvature_type == CurvatureType.STRAIGHT or self.curve_status == "Straight":
            y, x = self._alpha_masks.get("keep_straight_img", ([], []))
            if len(y) > 0:
                main_show[y + 10, x - 100 + W // 2] = self.keep_straight_img[y, x, :3]
            self.curve_status = "Straight"

        self.updateFPS()
        cv2.putText(
            main_show,
            "LDWS : " + offset_type.value,
            (10, 240),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.7,
            color=self.OffsetDict[offset_type],
            thickness=2,
        )
        cv2.putText(
            main_show,
            "FCWS : " + collision_type.value,
            (10, 280),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.6,
            color=self.CollisionDict[collision_type],
            thickness=2,
        )
        cv2.putText(
            main_show,
            "FPS  : %.2f" % self.fps,
            (10, H - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def DisplayCollisionPanel(self, main_show, collision_type, object_infer_time, lane_infer_time, show_ratio=0.25):
        W = int(main_show.shape[1] * show_ratio)
        H = int(main_show.shape[0] * show_ratio)

        main_show[H + 20 : 2 * H, -W - 20 :] //= 2
        main_show[H + 20 : H + 23, -W - 20 :] = [0, 0, 255]
        main_show[2 * H - 2 : 2 * H, -W - 20 :] = [0, 0, 255]
        main_show[H + 20 : 2 * H, -W - 20 : -W - 17] = [0, 0, 255]
        main_show[H + 20 : 2 * H, -3:] = [0, 0, 255]

        if collision_type == CollisionType.WARNING:
            y, x = self._alpha_masks.get("collision_warning_img", ([], []))
            if len(y) > 0:
                main_show[H + y + 50, (x - W - 5)] = self.collision_warning_img[y, x, :3]
        elif collision_type == CollisionType.PROMPT:
            y, x = self._alpha_masks.get("collision_prompt_img", ([], []))
            if len(y) > 0:
                main_show[H + y + 50, (x - W - 5)] = self.collision_prompt_img[y, x, :3]
        elif collision_type == CollisionType.NORMAL:
            y, x = self._alpha_masks.get("collision_normal_img", ([], []))
            if len(y) > 0:
                main_show[H + y + 50, (x - W - 5)] = self.collision_normal_img[y, x, :3]

        cv2.putText(
            main_show,
            "FCWS : " + collision_type.value,
            (main_show.shape[1] - int(W) + 100, 240),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.6,
            color=self.CollisionDict[collision_type],
            thickness=2,
        )
        cv2.putText(
            main_show,
            "object-infer : %.2f s" % object_infer_time,
            (main_show.shape[1] - int(W) + 100, 300),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            main_show,
            "lane-infer : %.2f s" % lane_infer_time,
            (main_show.shape[1] - int(W) + 100, 320),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )


@dataclass
class ADASMetrics:
    object_infer_s: float
    lane_infer_s: float
    collision: CollisionType
    offset: OffsetType
    curvature: CurvatureType
    birdview: Optional[np.ndarray] = None


class ADASProcessor:
    def __init__(
        self,
        lane_config: dict,
        object_config: dict,
        logger: Optional[Logger] = None,
        allowed_labels: Optional[set[str]] = None,
        parallel: bool = True,
        lane_skip_frames: int = 0,
        downscale: float = 1.0,
        bev_config: Optional[BevConfig] = None,
        bev_debug: bool = False,
    ):
        self.lane_config = dict(lane_config)
        self.object_config = dict(object_config)
        self.logger = logger or Logger(None, logging.INFO, logging.INFO)
        self.allowed_labels = {s.lower() for s in allowed_labels} if allowed_labels else None
        self.parallel = parallel
        self.lane_skip_frames = lane_skip_frames
        self.downscale = downscale
        self.bev_config = bev_config or BevConfig()
        self.bev_debug = bev_debug

        self.lane_detector = None
        self.object_detector = None
        self.transform_view = None
        self.distance_detector = None
        self.object_tracker = None
        self.display_panel = None
        self.analyze_msg = None

        self._executor = ThreadPoolExecutor(max_workers=2) if parallel else None
        self._frame_idx = 0
        self._cached_lane_result = None
        self._small_frame = None

    def initialize(self, frame_size: Tuple[int, int]) -> None:
        width, height = frame_size

        if self.parallel and self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=2)

        num_threads = max(1, (os.cpu_count() or 4) // 2) if self.parallel else None

        if "UFLDV2" in self.lane_config["model_type"].name:
            UltrafastLaneDetectorV2.set_defaults(self.lane_config)
            self.lane_detector = UltrafastLaneDetectorV2(logger=self.logger, num_threads=num_threads)
        else:
            UltrafastLaneDetector.set_defaults(self.lane_config)
            self.lane_detector = UltrafastLaneDetector(logger=self.logger, num_threads=num_threads)

        self.transform_view = PerspectiveTransformation((width, height), logger=self.logger, bev_config=self.bev_config)

        if ObjectModelType.EfficientDet == self.object_config["model_type"]:
            EfficientdetDetector.set_defaults(self.object_config)
            self.object_detector = EfficientdetDetector(logger=self.logger, num_threads=num_threads)
        else:
            YoloDetector.set_defaults(self.object_config)
            self.object_detector = YoloDetector(logger=self.logger, num_threads=num_threads)

        self.distance_detector = SingleCamDistanceMeasure()
        self.object_tracker = BYTETracker(names=self.object_detector.colors_dict)

        self.display_panel = ControlPanel()
        self.analyze_msg = TaskConditions()

    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.downscale >= 1.0:
            return frame
        small = cv2.resize(frame, None, fx=self.downscale, fy=self.downscale, interpolation=cv2.INTER_LINEAR)
        return small

    def _run_detection(self, frame: np.ndarray, run_lane: bool):
        if self.parallel and self._executor is not None:
            obj_future = self._executor.submit(self.object_detector.DetectFrame, frame)
            lane_future = self._executor.submit(self.lane_detector.DetectFrame, frame) if run_lane else None

            obj_exception = obj_future.exception()
            if obj_exception:
                raise obj_exception
            if lane_future is not None:
                lane_exception = lane_future.exception()
                if lane_exception:
                    raise lane_exception
        else:
            self.object_detector.DetectFrame(frame)
            if run_lane:
                self.lane_detector.DetectFrame(frame)

    def process_frame(self, frame: np.ndarray, no_hud: bool = False) -> Tuple[np.ndarray, ADASMetrics]:
        if self.lane_detector is None or self.object_detector is None:
            raise RuntimeError("ADASProcessor is not initialized. Call initialize(frame_size) first.")

        infer_frame = self._prepare_frame(frame)
        run_lane = (self.lane_skip_frames == 0) or (self._frame_idx % (self.lane_skip_frames + 1) == 0)

        object_time = time.time()
        if self.parallel and self._executor is not None:
            obj_future = self._executor.submit(self.object_detector.DetectFrame, infer_frame)
            lane_future = self._executor.submit(self.lane_detector.DetectFrame, infer_frame) if run_lane else None

            obj_future.result()
            object_infer_time = round(time.time() - object_time, 2)

            if self.allowed_labels is not None:
                self.object_detector._object_info = [
                    obj for obj in self.object_detector.object_info if str(obj.label).lower() in self.allowed_labels
                ]

            if lane_future is not None:
                lane_future.result()
                lane_infer_time = round(time.time() - object_time, 4)
                self._cached_lane_result = None
            else:
                lane_infer_time = 0.0
        else:
            self.object_detector.DetectFrame(infer_frame)
            if self.allowed_labels is not None:
                self.object_detector._object_info = [
                    obj for obj in self.object_detector.object_info if str(obj.label).lower() in self.allowed_labels
                ]
            object_infer_time = round(time.time() - object_time, 2)

            if run_lane:
                lane_time = time.time()
                self.lane_detector.DetectFrame(infer_frame)
                lane_infer_time = round(time.time() - lane_time, 4)
                self._cached_lane_result = None
            else:
                lane_infer_time = 0.0

        self._frame_idx += 1

        boxes = [obj.tolist(format_type="xyxy") for obj in self.object_detector.object_info]
        scores = [obj.conf for obj in self.object_detector.object_info]
        class_ids = [obj.label for obj in self.object_detector.object_info]

        self.object_tracker.update(boxes, scores, class_ids, frame)

        self.distance_detector.updateDistance(self.object_detector.object_info)
        vehicle_distance = self.distance_detector.calcCollisionPoint(self.lane_detector.lane_info.area_points)

        if (not self.bev_config.static_mode and not self.bev_debug
                and self.analyze_msg.CheckStatus() and self.lane_detector.lane_info.area_status):
            self.transform_view.updateTransformParams(
                *self.lane_detector.lane_info.lanes_points[1:3], self.analyze_msg.transform_status
            )

        if self.bev_debug:
            self.transform_view.rebuildFromConfig(self.bev_config)

        frame_show = frame.copy()
        birdview_show = self.transform_view.transformToBirdView(frame_show)
        birdview_lanes_points = [
            self.transform_view.transformToBirdViewPoints(lanes_point) for lanes_point in self.lane_detector.lane_info.lanes_points
        ]
        (vehicle_direction, vehicle_curvature), vehicle_offset = self.transform_view.calcCurveAndOffset(
            birdview_show, *birdview_lanes_points[1:3]
        )

        self.analyze_msg.UpdateCollisionStatus(vehicle_distance, self.lane_detector.lane_info.area_status)
        self.analyze_msg.UpdateOffsetStatus(vehicle_offset)
        self.analyze_msg.UpdateRouteStatus(vehicle_direction, vehicle_curvature)

        self.transform_view.DrawDetectedOnBirdView(birdview_show, birdview_lanes_points, self.analyze_msg.offset_msg)
        self.lane_detector.DrawDetectedOnFrame(frame_show, self.analyze_msg.offset_msg)
        self.lane_detector.DrawAreaOnFrame(frame_show, self.display_panel.CollisionDict[self.analyze_msg.collision_msg])
        self.object_detector.DrawDetectedOnFrame(frame_show)
        self.object_tracker.DrawTrackedOnFrame(frame_show, False)
        self.distance_detector.DrawDetectedOnFrame(frame_show)

        if not no_hud:
            self.display_panel.DisplayBirdViewPanel(frame_show, birdview_show)
            self.display_panel.DisplaySignsPanel(frame_show, self.analyze_msg.offset_msg, self.analyze_msg.curvature_msg, self.analyze_msg.collision_msg)

        metrics = ADASMetrics(
            object_infer_s=object_infer_time,
            lane_infer_s=lane_infer_time,
            collision=self.analyze_msg.collision_msg,
            offset=self.analyze_msg.offset_msg,
            curvature=self.analyze_msg.curvature_msg,
            birdview=birdview_show,
        )
        return frame_show, metrics

    def cleanup(self):
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
