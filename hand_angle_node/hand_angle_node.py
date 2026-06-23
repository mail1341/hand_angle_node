#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
from datetime import datetime
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32


# ============================================================
# 調整しやすい設定
# ============================================================

CAMERA_ID = 0

# 画質を優先して1280x720で取得します。
# FPSが落ちる場合は640x480に戻してください。
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

TIMER_SEC = 1.0 / 30.0

# MediaPipe検出条件
MAX_NUM_HANDS = 1
MODEL_COMPLEXITY = 1
MIN_DETECTION_CONFIDENCE = 0.60
MIN_TRACKING_CONFIDENCE = 0.50

# 移動平均
SMOOTH_WINDOW = 7

# ランドマーク座標の指数移動平均
# 大きいほど滑らか、ただし遅れる
LANDMARK_ALPHA = 0.65

# 1フレームで許容する角度変化量 [deg]
MAX_OUTPUT_JUMP_PER_FRAME = 18.0
MAX_INDEX_JUMP_PER_FRAME = 18.0

# 側面判定
# side_score = abs(lm[5].x - lm[17].x) / palm_size
# 小さいほど横向きに近い
SIDE_ENTER_THRESHOLD = 0.35
SIDE_EXIT_THRESHOLD = 0.55

# 側面/正面モードが頻繁に切り替わらないようにする連続確認フレーム数
MODE_SWITCH_CONFIRM_FRAMES = 4

# 検出喪失時
# HOLD中は最後の値をpublishし続ける
HOLD_FRAMES = 10

# これ以上見失ったら内部状態をリセット
MAX_MISSED_FRAMES = 20

# CSV
CSV_FLUSH_INTERVAL = 10

# 表示画面の拡大倍率
# 1280x720で取得するため、追加拡大はしない。
DISPLAY_SCALE = 1.0
WINDOW_NAME = "Hand"


class HandAngleNode(Node):
    def __init__(self):
        super().__init__('hand_angle_node')

        # ========= ROS2 publish =========
        self.pub_norm = self.create_publisher(Float32, '/hand_norm', 10)
        self.pub_plane_angle = self.create_publisher(Float32, '/hand_plane_angle', 10)
        self.pub_index_angle = self.create_publisher(Float32, '/hand_index_angle', 10)

        # ========= Camera =========
        self.cap = cv2.VideoCapture(CAMERA_ID)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

        # 遅延を減らす
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        if not self.cap.isOpened():
            self.get_logger().warn("Camera could not be opened. Check CAMERA_ID.")

        # ========= display window =========
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            WINDOW_NAME,
            int(CAMERA_WIDTH * DISPLAY_SCALE),
            int(CAMERA_HEIGHT * DISPLAY_SCALE)
        )

        # ========= MediaPipe =========
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=MAX_NUM_HANDS,
            model_complexity=MODEL_COMPLEXITY,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        )

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_draw_styles = mp.solutions.drawing_styles

        # ========= smoothing =========
        self.norm_history = deque(maxlen=SMOOTH_WINDOW)
        self.output_angle_history = deque(maxlen=SMOOTH_WINDOW)
        self.index_angle_history = deque(maxlen=SMOOTH_WINDOW)

        self.prev_landmarks = None
        self.prev_output_angle = None
        self.prev_index_angle = None

        # ========= side mode =========
        self.is_side_mode = False
        self.mode_switch_count = 0

        # ========= lost tracking =========
        self.missed_frames = 0
        self.last_hand_norm = None
        self.last_output_angle = None
        self.last_index_angle = None
        self.last_mode = "NONE"

        # ========= CSV =========
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = f"hand_data_{now_str}.csv"
        self.csv_file = open(self.csv_path, mode="w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "time",
            "tracking_state",
            "mode",
            "hand_norm",
            "output_angle",
            "index_output_angle",
            "side_score",
            "raw_plane_angle",
            "raw_2d_bend_angle",
            "index_plane_angle",
            "middle_plane_angle",
            "ring_plane_angle",
            "pinky_plane_angle",
            "index_2d_bend",
            "middle_2d_bend",
            "ring_2d_bend",
            "pinky_2d_bend"
        ])
        self.frame_count = 0
        self.get_logger().info(f"CSV保存開始: {self.csv_path}")
        self.get_logger().info("Publishing: /hand_norm, /hand_plane_angle, /hand_index_angle")

        self.timer = self.create_timer(TIMER_SEC, self.timer_callback)

    # ------------------------------------------------------------
    # math utilities
    # ------------------------------------------------------------

    def calc_distance(self, a, b):
        return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))

    def calc_angle(self, p1, p2, p3):
        v1 = np.asarray(p1) - np.asarray(p2)
        v2 = np.asarray(p3) - np.asarray(p2)

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 <= 1.0e-12 or norm2 <= 1.0e-12:
            return 0.0

        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        return float(np.degrees(np.arccos(cos_angle)))

    def calc_2d_bend_angle(self, p1, p2, p3):
        """
        MediaPipeの関節角度は、指がまっすぐなとき約180度。
        ここでは 180 - angle とし、曲がるほど大きい値にする。
        """
        joint_angle = self.calc_angle(p1, p2, p3)
        return float(np.clip(180.0 - joint_angle, 0.0, 180.0))

    def calc_angle_between_vector_and_plane(self, vec, plane_normal):
        vec_norm = np.linalg.norm(vec)
        normal_norm = np.linalg.norm(plane_normal)

        if vec_norm <= 1.0e-12 or normal_norm <= 1.0e-12:
            return 0.0

        vec_unit = vec / vec_norm
        normal_unit = plane_normal / normal_norm

        cos_to_normal = np.clip(abs(np.dot(vec_unit, normal_unit)), -1.0, 1.0)
        angle_to_normal = np.degrees(np.arccos(cos_to_normal))

        return float(90.0 - angle_to_normal)

    def moving_average(self, history, value):
        history.append(float(value))
        return float(sum(history) / len(history))

    def smooth_landmarks(self, landmark_list):
        current = np.array([[p.x, p.y, p.z] for p in landmark_list], dtype=np.float64)

        if self.prev_landmarks is None:
            smoothed = current
        else:
            smoothed = LANDMARK_ALPHA * self.prev_landmarks + (1.0 - LANDMARK_ALPHA) * current

        self.prev_landmarks = smoothed
        return smoothed

    def limit_jump(self, value, prev_value, max_jump):
        value = float(value)

        if prev_value is None:
            return value

        diff = value - prev_value
        if abs(diff) > max_jump:
            value = prev_value + np.sign(diff) * max_jump

        return float(value)

    def update_side_mode(self, side_score):
        desired_mode = self.is_side_mode

        if self.is_side_mode:
            if side_score > SIDE_EXIT_THRESHOLD:
                desired_mode = False
        else:
            if side_score < SIDE_ENTER_THRESHOLD:
                desired_mode = True

        if desired_mode != self.is_side_mode:
            self.mode_switch_count += 1
            if self.mode_switch_count >= MODE_SWITCH_CONFIRM_FRAMES:
                self.is_side_mode = desired_mode
                self.mode_switch_count = 0
        else:
            self.mode_switch_count = 0

        return self.is_side_mode

    def reset_tracking_state(self):
        self.prev_landmarks = None
        self.prev_output_angle = None
        self.prev_index_angle = None
        self.norm_history.clear()
        self.output_angle_history.clear()
        self.index_angle_history.clear()
        self.is_side_mode = False
        self.mode_switch_count = 0
        self.last_hand_norm = None
        self.last_output_angle = None
        self.last_index_angle = None
        self.last_mode = "NONE"

    # ------------------------------------------------------------
    # ROS publish / drawing
    # ------------------------------------------------------------

    def publish_values(self, hand_norm, output_angle, index_angle):
        msg = Float32()
        msg.data = float(hand_norm)
        self.pub_norm.publish(msg)

        msg_angle = Float32()
        msg_angle.data = float(output_angle)
        self.pub_plane_angle.publish(msg_angle)

        msg_index = Float32()
        msg_index.data = float(index_angle)
        self.pub_index_angle.publish(msg_index)

    def draw_text(
        self,
        frame,
        tracking_state,
        mode,
        hand_norm,
        output_angle,
        index_angle,
        side_score,
        raw_plane_angle,
        raw_2d_bend_angle,
        index_plane_angle,
        index_2d_bend
    ):
        """
        画面左上の結果欄を見やすく表示する。
        カメラ範囲・処理解像度は変更しない。
        """
        x = 22
        y = 34
        line_h = 25
        font_scale = 0.58
        thickness = 2

        lines = [
            f"tracking : {tracking_state}",
            f"mode     : {mode}",
            f"index    : {index_angle:.1f} deg",
            f"output   : {output_angle:.1f} deg",
            f"norm     : {hand_norm:.2f}",
            f"side     : {side_score:.3f}",
        ]

        panel_x1 = 10
        panel_y1 = 10
        panel_w = 380
        panel_h = 28 + line_h * len(lines)
        panel_x2 = panel_x1 + panel_w
        panel_y2 = panel_y1 + panel_h

        # 半透明の黒背景
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (panel_x1, panel_y1),
            (panel_x2, panel_y2),
            (0, 0, 0),
            -1
        )
        alpha = 0.68
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

        # 外枠を追加して見やすくする
        cv2.rectangle(
            frame,
            (panel_x1, panel_y1),
            (panel_x2, panel_y2),
            (80, 80, 80),
            1
        )

        for i, text in enumerate(lines):
            color = (0, 255, 0)
            if i in (0, 1, 5):
                color = (0, 255, 255)

            cv2.putText(
                frame,
                text,
                (x, y + i * line_h),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                thickness
            )

    def write_csv_row(
        self,
        tracking_state,
        mode,
        hand_norm,
        output_angle,
        index_output_angle,
        side_score,
        raw_plane_angle,
        raw_2d_bend_angle,
        index_plane_angle,
        middle_plane_angle,
        ring_plane_angle,
        pinky_plane_angle,
        index_2d_bend,
        middle_2d_bend,
        ring_2d_bend,
        pinky_2d_bend
    ):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        self.csv_writer.writerow([
            now,
            tracking_state,
            mode,
            round(float(hand_norm), 3),
            round(float(output_angle), 1),
            round(float(index_output_angle), 1),
            round(float(side_score), 5),
            round(float(raw_plane_angle), 1),
            round(float(raw_2d_bend_angle), 1),
            round(float(index_plane_angle), 1),
            round(float(middle_plane_angle), 1),
            round(float(ring_plane_angle), 1),
            round(float(pinky_plane_angle), 1),
            round(float(index_2d_bend), 1),
            round(float(middle_2d_bend), 1),
            round(float(ring_2d_bend), 1),
            round(float(pinky_2d_bend), 1),
        ])

        self.frame_count += 1
        if self.frame_count % CSV_FLUSH_INTERVAL == 0:
            self.csv_file.flush()

    def show_frame(self, frame):
        """
        処理解像度はそのままに、表示だけ拡大する。
        これにより映像は大きく見えるが、MediaPipeの処理負荷は増えにくい。
        """
        if DISPLAY_SCALE != 1.0:
            display_frame = cv2.resize(
                frame,
                None,
                fx=DISPLAY_SCALE,
                fy=DISPLAY_SCALE,
                interpolation=cv2.INTER_LINEAR
            )
        else:
            display_frame = frame

        cv2.imshow(WINDOW_NAME, display_frame)

    # ------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self.hands.process(rgb)
        rgb.flags.writeable = True

        if result.multi_hand_landmarks:
            self.missed_frames = 0
            tracking_state = "DETECTED"

            hand = result.multi_hand_landmarks[0]

            self.mp_draw.draw_landmarks(
                frame,
                hand,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_draw_styles.get_default_hand_landmarks_style(),
                self.mp_draw_styles.get_default_hand_connections_style()
            )

            lm = self.smooth_landmarks(hand.landmark)

            palm_size = self.calc_distance(lm[0][:2], lm[9][:2])
            if palm_size <= 1.0e-6:
                self.show_frame(frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    rclpy.shutdown()
                return

            # hand_norm
            index_len = self.calc_distance(lm[8][:2], lm[5][:2]) / palm_size
            middle_len = self.calc_distance(lm[12][:2], lm[9][:2]) / palm_size
            ring_len = self.calc_distance(lm[16][:2], lm[13][:2]) / palm_size
            pinky_len = self.calc_distance(lm[20][:2], lm[17][:2]) / palm_size
            hand_norm_raw = index_len + middle_len + ring_len + pinky_len

            # 2D bend
            index_2d_bend = self.calc_2d_bend_angle(lm[5][:2], lm[6][:2], lm[7][:2])
            middle_2d_bend = self.calc_2d_bend_angle(lm[9][:2], lm[10][:2], lm[11][:2])
            ring_2d_bend = self.calc_2d_bend_angle(lm[13][:2], lm[14][:2], lm[15][:2])
            pinky_2d_bend = self.calc_2d_bend_angle(lm[17][:2], lm[18][:2], lm[19][:2])

            hand_2d_bend_angle_raw = (
                index_2d_bend + middle_2d_bend + ring_2d_bend + pinky_2d_bend
            ) / 4.0

            # hand plane
            wrist = lm[0]
            index_mcp = lm[5]
            middle_mcp = lm[9]
            ring_mcp = lm[13]
            pinky_mcp = lm[17]

            palm_v1 = index_mcp - wrist
            palm_v2 = pinky_mcp - wrist
            palm_normal = np.cross(palm_v1, palm_v2)
            palm_normal_norm = np.linalg.norm(palm_normal)

            index_plane_angle = self.calc_angle_between_vector_and_plane(
                lm[6] - lm[5],
                palm_normal
            )
            middle_plane_angle = self.calc_angle_between_vector_and_plane(
                lm[10] - lm[9],
                palm_normal
            )
            ring_plane_angle = self.calc_angle_between_vector_and_plane(
                lm[14] - lm[13],
                palm_normal
            )
            pinky_plane_angle = self.calc_angle_between_vector_and_plane(
                lm[18] - lm[17],
                palm_normal
            )

            hand_plane_angle_raw = (
                index_plane_angle + middle_plane_angle + ring_plane_angle + pinky_plane_angle
            ) / 4.0

            # side mode
            side_score = abs(float(lm[5][0] - lm[17][0])) / palm_size
            side_mode = self.update_side_mode(side_score)
            plane_valid = palm_normal_norm > 1.0e-5

            if side_mode or not plane_valid:
                output_angle_raw = hand_2d_bend_angle_raw
                index_output_angle_raw = index_2d_bend
                mode = "SIDE_2D"
            else:
                output_angle_raw = hand_plane_angle_raw
                index_output_angle_raw = index_plane_angle
                mode = "PLANE_3D"

            # jump limit
            output_angle_limited = self.limit_jump(
                output_angle_raw,
                self.prev_output_angle,
                MAX_OUTPUT_JUMP_PER_FRAME
            )
            self.prev_output_angle = output_angle_limited

            index_angle_limited = self.limit_jump(
                index_output_angle_raw,
                self.prev_index_angle,
                MAX_INDEX_JUMP_PER_FRAME
            )
            self.prev_index_angle = index_angle_limited

            # moving average
            hand_norm = self.moving_average(self.norm_history, hand_norm_raw)
            output_angle = self.moving_average(self.output_angle_history, output_angle_limited)
            index_angle = self.moving_average(self.index_angle_history, index_angle_limited)

            output_angle_rounded = round(float(output_angle), 1)
            index_angle_rounded = round(float(index_angle), 1)

            # save last values
            self.last_hand_norm = float(hand_norm)
            self.last_output_angle = float(output_angle_rounded)
            self.last_index_angle = float(index_angle_rounded)
            self.last_mode = mode

            self.publish_values(hand_norm, output_angle_rounded, index_angle_rounded)

            self.write_csv_row(
                tracking_state,
                mode,
                hand_norm,
                output_angle_rounded,
                index_angle_rounded,
                side_score,
                hand_plane_angle_raw,
                hand_2d_bend_angle_raw,
                index_plane_angle,
                middle_plane_angle,
                ring_plane_angle,
                pinky_plane_angle,
                index_2d_bend,
                middle_2d_bend,
                ring_2d_bend,
                pinky_2d_bend
            )

            self.draw_text(
                frame,
                tracking_state,
                mode,
                hand_norm,
                output_angle_rounded,
                index_angle_rounded,
                side_score,
                hand_plane_angle_raw,
                hand_2d_bend_angle_raw,
                index_plane_angle,
                index_2d_bend
            )

        else:
            self.missed_frames += 1

            if (
                self.missed_frames <= HOLD_FRAMES
                and self.last_hand_norm is not None
                and self.last_output_angle is not None
                and self.last_index_angle is not None
            ):
                tracking_state = "HOLD"
                self.publish_values(
                    self.last_hand_norm,
                    self.last_output_angle,
                    self.last_index_angle
                )

                overlay = frame.copy()
                cv2.rectangle(overlay, (10, 10), (390, 115), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)
                cv2.rectangle(frame, (10, 10), (390, 115), (80, 80, 80), 1)

                cv2.putText(
                    frame,
                    f"tracking : {tracking_state}",
                    (22, 38),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (0, 255, 255),
                    2
                )
                cv2.putText(
                    frame,
                    f"hold idx : {self.last_index_angle:.1f} deg",
                    (22, 68),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (0, 255, 0),
                    2
                )
                cv2.putText(
                    frame,
                    f"mode     : {self.last_mode}",
                    (22, 98),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (0, 255, 0),
                    2
                )

            if self.missed_frames >= MAX_MISSED_FRAMES:
                self.reset_tracking_state()

        self.show_frame(frame)

        if cv2.waitKey(1) & 0xFF == 27:
            rclpy.shutdown()

    def destroy_node(self):
        self.get_logger().info("CSV保存終了")

        try:
            self.csv_file.flush()
            self.csv_file.close()
        except Exception:
            pass

        try:
            self.cap.release()
            cv2.destroyAllWindows()
        except Exception:
            pass

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = HandAngleNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
