import numpy as np
import cv2


class LDWDetector:
    def __init__(self, img_width, img_height, threshold_px=60):
        self.w = img_width
        self.h = img_height
        self.threshold = threshold_px

        # --- ego vehicle footprint (MVP) ---
        self.left_wheel = (
            int(self.w * 0),
            int(self.h * 0.92)
        )
        self.right_wheel = (
            int(self.w * 1),
            int(self.h * 0.92)
        )


    # ---------- geometry ----------

    @staticmethod
    def point_to_segment_distance(p, a, b):
        p = np.array(p, dtype=float)
        a = np.array(a, dtype=float)
        b = np.array(b, dtype=float)

        ab = b - a
        if np.allclose(ab, 0):
            return np.linalg.norm(p - a), tuple(a.astype(int))

        t = np.dot(p - a, ab) / np.dot(ab, ab)
        t = np.clip(t, 0.0, 1.0)

        closest = a + t * ab
        return np.linalg.norm(p - closest), tuple(closest.astype(int))
    
    @staticmethod
    def x_on_segment_at_y(p1, p2, y):
        y1, y2 = p1[1], p2[1]
        if abs(y2 - y1) < 1e-3:
            return None

        t = (y - y1) / (y2 - y1)
        if 0.0 <= t <= 1.0:
            return p1[0] + t * (p2[0] - p1[0])

        return None

    def lane_side(self, points):
        avg_x = sum(p[0] for p in points) / len(points)
        return "left" if avg_x < self.w / 2 else "right"

    # ---------- main logic ----------

    def process_lanes(self, lane_polylines):
        left_dist = float("inf")
        right_dist = float("inf")
        left_crossed = False
        right_crossed = False

        lw_x, lw_y = self.left_wheel
        rw_x, rw_y = self.right_wheel

        for pts in lane_polylines:
            if len(pts) < 2:
                continue

            avg_x = sum(p[0] for p in pts) / len(pts)
            side = "left" if avg_x < self.w / 2 else "right"

            for i in range(len(pts) - 1):
                p1, p2 = pts[i], pts[i + 1]

                if side == "left":
                    x_line = self.x_on_segment_at_y(p1, p2, lw_y)
                    if x_line is None:
                        continue

                    dist = lw_x - x_line   # signed
                    left_dist = min(left_dist, abs(dist))

                    if dist < 0:
                        left_crossed = True

                else:  # right
                    x_line = self.x_on_segment_at_y(p1, p2, rw_y)
                    if x_line is None:
                        continue

                    dist = x_line - rw_x   # signed
                    right_dist = min(right_dist, abs(dist))

                    if dist < 0:
                        right_crossed = True
            state = "OK"

            if left_crossed or right_crossed:
                state = "OUT_OF_LANE"
            elif left_dist < self.threshold or right_dist < self.threshold:
                state = "WARNING"

            return {
                "state": state,
                "left_dist": left_dist,
                "right_dist": right_dist,
                "left_crossed": left_crossed,
                "right_crossed": right_crossed,
                "left_wheel": self.left_wheel,
                "right_wheel": self.right_wheel
            }



    # ---------- visualization ----------

    def draw(self, img, result):
        vis = img.copy()

        lw = result["left_wheel"]
        rw = result["right_wheel"]

        # --- wheels ---
        cv2.circle(vis, lw, 7, (255, 0, 0), -1)
        cv2.circle(vis, rw, 7, (255, 0, 0), -1)

        # --- wheel baseline ---
        cv2.line(
            vis,
            (0, lw[1]),
            (self.w, lw[1]),
            (100, 100, 100),
            1
        )

        # --- left lane distance ---
        if "left_x_line" in result and result["left_x_line"] is not None:
            x = int(result["left_x_line"])
            y = lw[1]

            color = (0, 0, 255) if result["left_crossed"] else (0, 255, 255)

            cv2.circle(vis, (x, y), 6, color, -1)
            cv2.line(vis, lw, (x, y), color, 2)

            cv2.putText(
                vis,
                f"{int(result['left_dist'])} px",
                (x - 60, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        # --- right lane distance ---
        if "right_x_line" in result and result["right_x_line"] is not None:
            x = int(result["right_x_line"])
            y = rw[1]

            color = (0, 0, 255) if result["right_crossed"] else (0, 255, 255)

            cv2.circle(vis, (x, y), 6, color, -1)
            cv2.line(vis, rw, (x, y), color, 2)

            cv2.putText(
                vis,
                f"{int(result['right_dist'])} px",
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        # --- LDW state ---
        if result["state"] == "OK":
            color = (0, 255, 0)
        elif result["state"] == "WARNING":
            color = (0, 255, 255)
        else:
            color = (0, 0, 255)

        cv2.putText(
            vis,
            f"LDW: {result['state']}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            color,
            3
        )

        return vis

