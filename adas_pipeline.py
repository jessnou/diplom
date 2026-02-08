import logging
import os
import time
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
from TrafficLaneDetector.ufldDetector.perspectiveTransformation import PerspectiveTransformation
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
        self.collision_prompt_img = load_image_safe(os.path.join(assets_dir, "FCWS-normal.jpg"), (100, 100))
        self.collision_normal_img = load_image_safe(os.path.join(assets_dir, "FCWS-normal.jpg"), (100, 100))

        self.left_curve_img = load_image_safe(os.path.join(assets_dir, ""), (200, 200))
        self.right_curve_img = load_image_safe(os.path.join(assets_dir, ""), (200, 200))
        self.keep_straight_img = load_image_safe(os.path.join(assets_dir, ""), (200, 200))
        self.determined_img = load_image_safe(os.path.join(assets_dir, ""), (200, 200))

        self.left_lanes_img = load_image_safe(os.path.join(assets_dir, "right_turn.jpg"), (300, 200))
        self.right_lanes_img = load_image_safe(os.path.join(assets_dir, "left_turn.png"), (300, 200))

        self.fps = 0.0
        self.frame_count = 0
        self.start = time.time()
        self.curve_status = None

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
        widget = np.copy(main_show[:H, :W])
        widget //= 2
        widget[0:3, :] = [0, 0, 255]
        widget[-3:-1, :] = [0, 0, 255]
        widget[:, 0:3] = [0, 0, 255]
        widget[:, -3:-1] = [0, 0, 255]
        main_show[:H, :W] = widget

        if curvature_type == CurvatureType.UNKNOWN and offset_type in {OffsetType.UNKNOWN, OffsetType.CENTER}:
            y, x = self.determined_img[:, :, 3].nonzero()
            main_show[y + 10, x - 100 + W // 2] = self.determined_img[y, x, :3]
            self.curve_status = None
        elif (curvature_type == CurvatureType.HARD_LEFT or self.curve_status == "Left") and (
            curvature_type not in {CurvatureType.EASY_RIGHT, CurvatureType.HARD_RIGHT}
        ):
            y, x = self.left_curve_img[:, :, 3].nonzero()
            main_show[y + 10, x - 100 + W // 2] = self.left_curve_img[y, x, :3]
            self.curve_status = "Left"
        elif (curvature_type == CurvatureType.HARD_RIGHT or self.curve_status == "Right") and (
            curvature_type not in {CurvatureType.EASY_LEFT, CurvatureType.HARD_LEFT}
        ):
            y, x = self.right_curve_img[:, :, 3].nonzero()
            main_show[y + 10, x - 100 + W // 2] = self.right_curve_img[y, x, :3]
            self.curve_status = "Right"

        if offset_type == OffsetType.RIGHT:
            y, x = self.left_lanes_img[:, :, 2].nonzero()
            main_show[y + 10, x - 150 + W // 2] = self.left_lanes_img[y, x, :3]
        elif offset_type == OffsetType.LEFT:
            y, x = self.right_lanes_img[:, :, 2].nonzero()
            main_show[y + 10, x - 150 + W // 2] = self.right_lanes_img[y, x, :3]
        elif curvature_type == CurvatureType.STRAIGHT or self.curve_status == "Straight":
            y, x = self.keep_straight_img[:, :, 3].nonzero()
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
            (10, widget.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def DisplayCollisionPanel(self, main_show, collision_type, object_infer_time, lane_infer_time, show_ratio=0.25):
        W = int(main_show.shape[1] * show_ratio)
        H = int(main_show.shape[0] * show_ratio)

        widget = np.copy(main_show[H + 20 : 2 * H, -W - 20 :])
        widget //= 2
        widget[0:3, :] = [0, 0, 255]
        widget[-3:-1, :] = [0, 0, 255]
        widget[:, -3:-1] = [0, 0, 255]
        widget[:, 0:3] = [0, 0, 255]
        main_show[H + 20 : 2 * H, -W - 20 :] = widget

        if collision_type == CollisionType.WARNING:
            y, x = self.collision_warning_img[:, :, 3].nonzero()
            main_show[H + y + 50, (x - W - 5)] = self.collision_warning_img[y, x, :3]
        elif collision_type == CollisionType.PROMPT:
            y, x = self.collision_prompt_img[:, :, 3].nonzero()
            main_show[H + y + 50, (x - W - 5)] = self.collision_prompt_img[y, x, :3]
        elif collision_type == CollisionType.NORMAL:
            y, x = self.collision_normal_img[:, :, 3].nonzero()
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


class ADASProcessor:
    def __init__(
        self,
        lane_config: dict,
        object_config: dict,
        logger: Optional[Logger] = None,
        allowed_labels: Optional[set[str]] = None,
    ):
        self.lane_config = dict(lane_config)
        self.object_config = dict(object_config)
        self.logger = logger or Logger(None, logging.INFO, logging.INFO)
        self.allowed_labels = {s.lower() for s in allowed_labels} if allowed_labels else None

        self.laneDetector = None
        self.objectDetector = None
        self.transformView = None
        self.distanceDetector = None
        self.objectTracker = None
        self.displayPanel = None
        self.analyzeMsg = None

    def initialize(self, frame_size: Tuple[int, int]) -> None:
        width, height = frame_size

        if "UFLDV2" in self.lane_config["model_type"].name:
            UltrafastLaneDetectorV2.set_defaults(self.lane_config)
            self.laneDetector = UltrafastLaneDetectorV2(logger=self.logger)
        else:
            UltrafastLaneDetector.set_defaults(self.lane_config)
            self.laneDetector = UltrafastLaneDetector(logger=self.logger)

        self.transformView = PerspectiveTransformation((width, height), logger=self.logger)

        if ObjectModelType.EfficientDet == self.object_config["model_type"]:
            EfficientdetDetector.set_defaults(self.object_config)
            self.objectDetector = EfficientdetDetector(logger=self.logger)
        else:
            YoloDetector.set_defaults(self.object_config)
            self.objectDetector = YoloDetector(logger=self.logger)

        self.distanceDetector = SingleCamDistanceMeasure()
        self.objectTracker = BYTETracker(names=self.objectDetector.colors_dict)

        self.displayPanel = ControlPanel()
        self.analyzeMsg = TaskConditions()

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, ADASMetrics]:
        if self.laneDetector is None or self.objectDetector is None:
            raise RuntimeError("ADASProcessor is not initialized. Call initialize(frame_size) first.")

        frame_show = frame.copy()

        object_time = time.time()
        self.objectDetector.DetectFrame(frame)
        if self.allowed_labels is not None:
            self.objectDetector._object_info = [
                obj for obj in self.objectDetector.object_info if str(obj.label).lower() in self.allowed_labels
            ]
        object_infer_time = round(time.time() - object_time, 2)

        boxes = [obj.tolist(format_type="xyxy") for obj in self.objectDetector.object_info]
        scores = [obj.conf for obj in self.objectDetector.object_info]
        class_ids = [obj.label for obj in self.objectDetector.object_info]

        self.objectTracker.update(boxes, scores, class_ids, frame)

        lane_time = time.time()
        self.laneDetector.DetectFrame(frame)
        lane_infer_time = round(time.time() - lane_time, 4)

        self.distanceDetector.updateDistance(self.objectDetector.object_info)
        vehicle_distance = self.distanceDetector.calcCollisionPoint(self.laneDetector.lane_info.area_points)

        if self.analyzeMsg.CheckStatus() and self.laneDetector.lane_info.area_status:
            self.transformView.updateTransformParams(
                *self.laneDetector.lane_info.lanes_points[1:3], self.analyzeMsg.transform_status
            )

        birdview_show = self.transformView.transformToBirdView(frame_show)
        birdview_lanes_points = [
            self.transformView.transformToBirdViewPoints(lanes_point) for lanes_point in self.laneDetector.lane_info.lanes_points
        ]
        (vehicle_direction, vehicle_curvature), vehicle_offset = self.transformView.calcCurveAndOffset(
            birdview_show, *birdview_lanes_points[1:3]
        )

        self.analyzeMsg.UpdateCollisionStatus(vehicle_distance, self.laneDetector.lane_info.area_status)
        self.analyzeMsg.UpdateOffsetStatus(vehicle_offset)
        self.analyzeMsg.UpdateRouteStatus(vehicle_direction, vehicle_curvature)

        self.transformView.DrawDetectedOnBirdView(birdview_show, birdview_lanes_points, self.analyzeMsg.offset_msg)
        self.laneDetector.DrawDetectedOnFrame(frame_show, self.analyzeMsg.offset_msg)
        self.laneDetector.DrawAreaOnFrame(frame_show, self.displayPanel.CollisionDict[self.analyzeMsg.collision_msg])
        self.objectDetector.DrawDetectedOnFrame(frame_show)
        self.objectTracker.DrawTrackedOnFrame(frame_show, False)
        self.distanceDetector.DrawDetectedOnFrame(frame_show)

        self.displayPanel.DisplayBirdViewPanel(frame_show, birdview_show)
        self.displayPanel.DisplaySignsPanel(frame_show, self.analyzeMsg.offset_msg, self.analyzeMsg.curvature_msg, self.analyzeMsg.collision_msg)

        metrics = ADASMetrics(
            object_infer_s=object_infer_time,
            lane_infer_s=lane_infer_time,
            collision=self.analyzeMsg.collision_msg,
            offset=self.analyzeMsg.offset_msg,
            curvature=self.analyzeMsg.curvature_msg,
        )
        return frame_show, metrics
