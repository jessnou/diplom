import os
import cv2
import logging
import numpy as np
import pycuda.driver as drv
import time
import sys

# Add the project's root directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ObjectTracker import BYTETracker
from taskConditions import TaskConditions, Logger
from ObjectDetector import YoloDetector, EfficientdetDetector
from ObjectDetector.utils import ObjectModelType, CollisionType
from ObjectDetector.distanceMeasure import SingleCamDistanceMeasure

from TrafficLaneDetector import UltrafastLaneDetector, UltrafastLaneDetectorV2
from TrafficLaneDetector.ufldDetector.perspectiveTransformation import PerspectiveTransformation
from TrafficLaneDetector.ufldDetector.utils import LaneModelType, OffsetType, CurvatureType
import os
print("Current working directory:", os.getcwd())
print("Classes path exists:", os.path.isfile("./ObjectDetector/models/coco_label.txt"))


def load_image_safe(path, size):
    """Загружает картинку и ресайзит. Если нет файла — возвращает пустой квадрат."""
    if os.path.exists(path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            return cv2.resize(img, size)
    # Заглушка: прозрачная картинка
    h, w = size[1], size[0]
    return np.zeros((h, w, 4), dtype=np.uint8)
LOGGER = Logger(None, logging.INFO, logging.INFO)

# ----------------------------------------------------
# Project and Model Configuration
# ----------------------------------------------------
video_path = r"/home/jessnou/TSU/diplom/data/archive/drivingDataset/normalDay/nD_16.mp4" #Paste video path here
lane_config = {
    "model_path": r"/home/jessnou/TSU/diplom/Vision-Guard-ADAS/TrafficLaneDetector/ufldDetector/exportLib/ultrafastLaneV2/tusimple_res18.onnx", #Paste Lane detection model path here
    "model_type": LaneModelType.UFLD_TUSIMPLE, # Change to UFLDV2_CURVELANE when curved Lane
}

object_config = {
    "model_path": r'/home/jessnou/TSU/diplom/Vision-Guard-ADAS/yolov8l.onnx', #Paste Object Detection model path here
    "model_type": ObjectModelType.YOLOV8,
    "classes_path": './ObjectDetector/models/coco_label.txt',
    "box_score": 0.4,
    "box_nms_iou": 0.5,
}


# ----------------------------------------------------
# Control Panel Class
# ----------------------------------------------------
class ControlPanel(object):
    CollisionDict = {
        CollisionType.UNKNOWN: (0, 255, 255),
        CollisionType.NORMAL: (0, 255, 0),
        CollisionType.PROMPT: (0, 102, 255),
        CollisionType.WARNING: (0, 0, 255)
    }

    OffsetDict = {
        OffsetType.UNKNOWN: (0, 255, 255),
        OffsetType.RIGHT: (0, 0, 255),
        OffsetType.LEFT: (0, 0, 255),
        OffsetType.CENTER: (0, 255, 0)
    }

    CurvatureDict = {
        CurvatureType.UNKNOWN: (0, 255, 255),
        CurvatureType.STRAIGHT: (0, 255, 0),
        CurvatureType.EASY_LEFT: (0, 102, 255),
        CurvatureType.EASY_RIGHT: (0, 102, 255),
        CurvatureType.HARD_LEFT: (0, 0, 255),
        CurvatureType.HARD_RIGHT: (0, 0, 255)
    }

    def __init__(self):
        # Load panel images
        script_dir = os.path.dirname(os.path.realpath(__file__))
        assets_dir = os.path.join(script_dir, 'assets')

        self.collision_warning_img = load_image_safe(os.path.join(assets_dir, 'FCWS-warning.png'), (100, 100))
        self.collision_prompt_img  = load_image_safe(os.path.join(assets_dir, 'FCWS-prompt.png'), (100, 100))
        self.collision_normal_img  = load_image_safe(os.path.join(assets_dir, 'FCWS-normal.png'), (100, 100))

        self.left_curve_img        = load_image_safe(os.path.join(assets_dir, 'left_turn.png'), (200, 200))
        self.right_curve_img       = load_image_safe(os.path.join(assets_dir, 'right_turn.png'), (200, 200))
        self.keep_straight_img     = load_image_safe(os.path.join(assets_dir, 'straight.png'), (200, 200))
        self.determined_img        = load_image_safe(os.path.join(assets_dir, 'warn.png'), (200, 200))

        self.left_lanes_img        = load_image_safe(os.path.join(assets_dir, 'LTA-left_lanes.png'), (300, 200))
        self.right_lanes_img       = load_image_safe(os.path.join(assets_dir, 'LTA-right_lanes.png'), (300, 200))

        # FPS
        self.fps = 0
        self.frame_count = 0
        self.start = time.time()
        self.curve_status = None

    def updateFPS(self):
        self.frame_count += 1
        if self.frame_count >= 30:
            end = time.time()
            self.fps = self.frame_count / (end - self.start)
            self.frame_count = 0
            self.start = time.time()

    def DisplayBirdViewPanel(self, main_show, min_show, show_ratio=0.25):
        W = int(main_show.shape[1] * show_ratio)
        H = int(main_show.shape[0] * show_ratio)
        min_birdview_show = cv2.resize(min_show, (W, H))
        min_birdview_show = cv2.copyMakeBorder(min_birdview_show, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        main_show[0:min_birdview_show.shape[0], -min_birdview_show.shape[1]:] = min_birdview_show

    def DisplaySignsPanel(self, main_show, offset_type, curvature_type):
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
        elif (curvature_type == CurvatureType.HARD_LEFT or self.curve_status == "Left") and \
                (curvature_type not in {CurvatureType.EASY_RIGHT, CurvatureType.HARD_RIGHT}):
            y, x = self.left_curve_img[:, :, 3].nonzero()
            main_show[y + 10, x - 100 + W // 2] = self.left_curve_img[y, x, :3]
            self.curve_status = "Left"
        elif (curvature_type == CurvatureType.HARD_RIGHT or self.curve_status == "Right") and \
                (curvature_type not in {CurvatureType.EASY_LEFT, CurvatureType.HARD_LEFT}):
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
        cv2.putText(main_show, "LDWS : " + offset_type.value, (10, 240), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.7, color=self.OffsetDict[offset_type], thickness=2)
        cv2.putText(main_show, "LKAS : " + curvature_type.value, (10, 280), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.7, color=self.CurvatureDict[curvature_type], thickness=2)
        cv2.putText(main_show, "FPS  : %.2f" % self.fps, (10, widget.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    def DisplayCollisionPanel(self, main_show, collision_type, object_infer_time, lane_infer_time, show_ratio=0.25):
        W = int(main_show.shape[1] * show_ratio)
        H = int(main_show.shape[0] * show_ratio)

        widget = np.copy(main_show[H + 20:2 * H, -W - 20:])
        widget //= 2
        widget[0:3, :] = [0, 0, 255]
        widget[-3:-1, :] = [0, 0, 255]
        widget[:, -3:-1] = [0, 0, 255]
        widget[:, 0:3] = [0, 0, 255]
        main_show[H + 20:2 * H, -W - 20:] = widget

        if collision_type == CollisionType.WARNING:
            y, x = self.collision_warning_img[:, :, 3].nonzero()
            main_show[H + y + 50, (x - W - 5)] = self.collision_warning_img[y, x, :3]
        elif collision_type == CollisionType.PROMPT:
            y, x = self.collision_prompt_img[:, :, 3].nonzero()
            main_show[H + y + 50, (x - W - 5)] = self.collision_prompt_img[y, x, :3]
        elif collision_type == CollisionType.NORMAL:
            y, x = self.collision_normal_img[:, :, 3].nonzero()
            main_show[H + y + 50, (x - W - 5)] = self.collision_normal_img[y, x, :3]

        cv2.putText(main_show, "FCWS : " + collision_type.value, (main_show.shape[1] - int(W) + 100, 240),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=self.CollisionDict[collision_type], thickness=2)
        cv2.putText(main_show, "object-infer : %.2f s" % object_infer_time, (main_show.shape[1] - int(W) + 100, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(main_show, "lane-infer : %.2f s" % lane_infer_time, (main_show.shape[1] - int(W) + 100, 320),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)


# ----------------------------------------------------
# Main
# ----------------------------------------------------
if __name__ == "__main__":
    # Initialize read and save video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Video path error. Please check it.")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    vout = cv2.VideoWriter(video_path[:-4] + '_out.mp4', fourcc, 30.0, (width, height))
    cv2.namedWindow("ADAS Simulation", cv2.WINDOW_NORMAL)

    # ==========================================================
    # Initialize Classes
    # ==========================================================
    LOGGER.info("[Pycuda] Cuda Version: {}".format(drv.get_version()))
    LOGGER.info("[Driver] Cuda Version: {}".format(drv.get_driver_version()))
    LOGGER.info("-" * 40)

    # Lane detection model
    LOGGER.info("Detector Model Type : {}".format(lane_config["model_type"].name))
    if "UFLDV2" in lane_config["model_type"].name:
        UltrafastLaneDetectorV2.set_defaults(lane_config)
        laneDetector = UltrafastLaneDetectorV2(logger=LOGGER)
    else:
        UltrafastLaneDetector.set_defaults(lane_config)
        laneDetector = UltrafastLaneDetector(logger=LOGGER)
    transformView = PerspectiveTransformation((width, height), logger=LOGGER)

    # Object detection model
    LOGGER.info("ObjectDetector Model Type : {}".format(object_config["model_type"].name))
    if ObjectModelType.EfficientDet == object_config["model_type"]:
        EfficientdetDetector.set_defaults(object_config)
        objectDetector = EfficientdetDetector(logger=LOGGER)
    else:
        YoloDetector.set_defaults(object_config)
        objectDetector = YoloDetector(logger=LOGGER)

    distanceDetector = SingleCamDistanceMeasure()
    objectTracker = BYTETracker(names=objectDetector.colors_dict)

    # Display panel
    displayPanel = ControlPanel()
    analyzeMsg = TaskConditions()
    frame_no = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_show = frame.copy()

        # ========================== Detect Model =========================
        object_time = time.time()
        objectDetector.DetectFrame(frame)
        object_infer_time = round(time.time() - object_time, 2)

        box = [obj.tolist(format_type="xyxy") for obj in objectDetector.object_info]
        score = [obj.conf for obj in objectDetector.object_info]
        id = [obj.label for obj in objectDetector.object_info]

        objectTracker.update(box, score, id, frame)

        lane_time = time.time()
        laneDetector.DetectFrame(frame)
        lane_infer_time = round(time.time() - lane_time, 4)

        # ========================= Analyze Status ========================
        distanceDetector.updateDistance(objectDetector.object_info)
        vehicle_distance = distanceDetector.calcCollisionPoint(laneDetector.lane_info.area_points)

        if analyzeMsg.CheckStatus() and laneDetector.lane_info.area_status:
            transformView.updateTransformParams(*laneDetector.lane_info.lanes_points[1:3],
                                                analyzeMsg.transform_status)
        birdview_show = transformView.transformToBirdView(frame_show)

        birdview_lanes_points = [transformView.transformToBirdViewPoints(lanes_point)
                                 for lanes_point in laneDetector.lane_info.lanes_points]
        (vehicle_direction, vehicle_curvature), vehicle_offset = transformView.calcCurveAndOffset(
            birdview_show, *birdview_lanes_points[1:3])

        analyzeMsg.UpdateCollisionStatus(vehicle_distance, laneDetector.lane_info.area_status)
        analyzeMsg.UpdateOffsetStatus(vehicle_offset)
        analyzeMsg.UpdateRouteStatus(vehicle_direction, vehicle_curvature)

        # ========================== Draw Results =========================
        transformView.DrawDetectedOnBirdView(birdview_show, birdview_lanes_points, analyzeMsg.offset_msg)
        if LOGGER.clevel == logging.DEBUG:
            transformView.DrawTransformFrontalViewArea(frame_show)

        laneDetector.DrawDetectedOnFrame(frame_show, analyzeMsg.offset_msg)
        laneDetector.DrawAreaOnFrame(frame_show, displayPanel.CollisionDict[analyzeMsg.collision_msg])
        objectDetector.DrawDetectedOnFrame(frame_show)
        objectTracker.DrawTrackedOnFrame(frame_show, False)
        distanceDetector.DrawDetectedOnFrame(frame_show)

        displayPanel.DisplayBirdViewPanel(frame_show, birdview_show)
        displayPanel.DisplaySignsPanel(frame_show, analyzeMsg.offset_msg, analyzeMsg.curvature_msg)
        displayPanel.DisplayCollisionPanel(frame_show, analyzeMsg.collision_msg,
                                           object_infer_time, lane_infer_time)


        frame_no += 1
        cv2.imshow("ADAS Simulation", frame_show)
        vout.write(frame_show)
        if cv2.waitKey(1) == ord('q'):
            break

    vout.release()
    cap.release()
    cv2.destroyAllWindows()
