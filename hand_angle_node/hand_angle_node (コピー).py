#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

import cv2
import mediapipe as mp
import numpy as np

import csv
from datetime import datetime


class HandAngleNode(Node):
    def __init__(self):
        super().__init__('hand_angle_node')

        self.pub = self.create_publisher(Float32, '/hand_norm', 10)
        self.pub_plane_angle = self.create_publisher(Float32, '/hand_plane_angle', 10)

        self.cap = cv2.VideoCapture(0)

        # 解像度を上げる
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_draw_styles = mp.solutions.drawing_styles

        # 移動平均用
        self.angle_history = []
        self.norm_history = []
        self.smooth_window = 5

        # CSV保存設定
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = f"hand_data_{now_str}.csv"
        self.csv_file = open(self.csv_path, mode="w", newline="")
        self.csv_writer = csv.writer(self.csv_file)

        self.frame_count = 0
        self.flush_interval = 10

        # CSVには時間、平均plane_angle、各指のplane_angleを記録する
        self.csv_writer.writerow([
            "time",
            "plane_angle",
            "index_plane_angle",
            "middle_plane_angle",
            "ring_plane_angle",
            "pinky_plane_angle"
        ])

        self.get_logger().info(f"CSV保存開始: {self.csv_path}")

        self.timer = self.create_timer(0.033, self.timer_callback)

    def calc_distance(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    def calc_angle(self, p1, p2, p3):
        v1 = np.array(p1) - np.array(p2)
        v2 = np.array(p3) - np.array(p2)

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        return float(np.degrees(np.arccos(cos_angle)))

    def landmark_to_3d(self, lm_point):
        return np.array([lm_point.x, lm_point.y, lm_point.z], dtype=np.float64)

    def calc_angle_between_vector_and_plane(self, vec, plane_normal):
        vec_norm = np.linalg.norm(vec)
        normal_norm = np.linalg.norm(plane_normal)

        if vec_norm == 0 or normal_norm == 0:
            return 0.0

        vec_unit = vec / vec_norm
        normal_unit = plane_normal / normal_norm

        cos_to_normal = np.clip(abs(np.dot(vec_unit, normal_unit)), -1.0, 1.0)
        angle_to_normal = np.degrees(np.arccos(cos_to_normal))

        angle_to_plane = 90.0 - angle_to_normal
        return float(angle_to_plane)

    def moving_average(self, history, value):
        history.append(value)

        if len(history) > self.smooth_window:
            history.pop(0)

        return sum(history) / len(history)

    def draw_plane_angle_text(
        self,
        frame,
        hand_plane_angle,
        index_plane_angle,
        middle_plane_angle,
        ring_plane_angle,
        pinky_plane_angle
    ):
        # ========= 映像上にplane_angleを表示 =========
        x = 20
        y = 40
        line_h = 32

        # 見やすくするため背景を付ける
        cv2.rectangle(frame, (10, 10), (430, 205), (0, 0, 0), -1)

        cv2.putText(
            frame,
            f"plane_angle avg : {hand_plane_angle:.1f} deg",
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2
        )

        y += line_h
        cv2.putText(
            frame,
            f"index  plane : {index_plane_angle:.1f} deg",
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        y += line_h
        cv2.putText(
            frame,
            f"middle plane : {middle_plane_angle:.1f} deg",
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        y += line_h
        cv2.putText(
            frame,
            f"ring   plane : {ring_plane_angle:.1f} deg",
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        y += line_h
        cv2.putText(
            frame,
            f"pinky  plane : {pinky_plane_angle:.1f} deg",
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = self.hands.process(rgb)

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]

            self.mp_draw.draw_landmarks(
                frame,
                hand,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_draw_styles.get_default_hand_landmarks_style(),
                self.mp_draw_styles.get_default_hand_connections_style()
            )

            lm = hand.landmark

            # ========= 手の大きさで正規化するための基準 =========
            palm_size = self.calc_distance(
                [lm[0].x, lm[0].y],
                [lm[9].x, lm[9].y]
            )

            if palm_size <= 0:
                return

            # ========= 各指の距離（親指なし） =========
            index = self.calc_distance([lm[8].x, lm[8].y], [lm[5].x, lm[5].y]) / palm_size
            middle = self.calc_distance([lm[12].x, lm[12].y], [lm[9].x, lm[9].y]) / palm_size
            ring = self.calc_distance([lm[16].x, lm[16].y], [lm[13].x, lm[13].y]) / palm_size
            pinky = self.calc_distance([lm[20].x, lm[20].y], [lm[17].x, lm[17].y]) / palm_size

            # ========= 各指の2D推定角度 =========
            index_angle = self.calc_angle(
                [lm[5].x, lm[5].y],
                [lm[6].x, lm[6].y],
                [lm[7].x, lm[7].y]
            )

            middle_angle = self.calc_angle(
                [lm[9].x, lm[9].y],
                [lm[10].x, lm[10].y],
                [lm[11].x, lm[11].y]
            )

            ring_angle = self.calc_angle(
                [lm[13].x, lm[13].y],
                [lm[14].x, lm[14].y],
                [lm[15].x, lm[15].y]
            )

            pinky_angle = self.calc_angle(
                [lm[17].x, lm[17].y],
                [lm[18].x, lm[18].y],
                [lm[19].x, lm[19].y]
            )

            # ========= 手の甲平面との角度 =========
            wrist = self.landmark_to_3d(lm[0])
            index_mcp = self.landmark_to_3d(lm[5])
            middle_mcp = self.landmark_to_3d(lm[9])
            ring_mcp = self.landmark_to_3d(lm[13])
            pinky_mcp = self.landmark_to_3d(lm[17])

            palm_v1 = index_mcp - wrist
            palm_v2 = pinky_mcp - wrist
            palm_normal = np.cross(palm_v1, palm_v2)

            index_pip = self.landmark_to_3d(lm[6])
            middle_pip = self.landmark_to_3d(lm[10])
            ring_pip = self.landmark_to_3d(lm[14])
            pinky_pip = self.landmark_to_3d(lm[18])

            index_vec = index_pip - index_mcp
            middle_vec = middle_pip - middle_mcp
            ring_vec = ring_pip - ring_mcp
            pinky_vec = pinky_pip - pinky_mcp

            index_plane_angle = self.calc_angle_between_vector_and_plane(index_vec, palm_normal)
            middle_plane_angle = self.calc_angle_between_vector_and_plane(middle_vec, palm_normal)
            ring_plane_angle = self.calc_angle_between_vector_and_plane(ring_vec, palm_normal)
            pinky_plane_angle = self.calc_angle_between_vector_and_plane(pinky_vec, palm_normal)

            # 親指を除いた4指の合計
            hand_norm_raw = index + middle + ring + pinky

            # 親指を除いた4指の平均plane_angle
            hand_plane_angle_raw = (
                index_plane_angle +
                middle_plane_angle +
                ring_plane_angle +
                pinky_plane_angle
            ) / 4.0

            # ========= 移動平均で平滑化 =========
            hand_norm = self.moving_average(self.norm_history, hand_norm_raw)
            hand_plane_angle = self.moving_average(self.angle_history, hand_plane_angle_raw)

            # ========= 小数第1位で丸める =========
            hand_plane_angle_rounded = round(float(hand_plane_angle), 1)

            index_plane_angle_rounded = round(float(index_plane_angle), 1)
            middle_plane_angle_rounded = round(float(middle_plane_angle), 1)
            ring_plane_angle_rounded = round(float(ring_plane_angle), 1)
            pinky_plane_angle_rounded = round(float(pinky_plane_angle), 1)

            # ========= ROS2トピックにpublish =========
            msg = Float32()
            msg.data = float(hand_norm)
            self.pub.publish(msg)

            msg_angle = Float32()
            msg_angle.data = float(hand_plane_angle_rounded)
            self.pub_plane_angle.publish(msg_angle)

            # ========= CSV記録 =========
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

            self.csv_writer.writerow([
                now,
                hand_plane_angle_rounded,
                index_plane_angle_rounded,
                middle_plane_angle_rounded,
                ring_plane_angle_rounded,
                pinky_plane_angle_rounded
            ])

            self.frame_count += 1
            if self.frame_count % self.flush_interval == 0:
                self.csv_file.flush()

            # ========= 映像上にplane_angleを表示 =========
            self.draw_plane_angle_text(
                frame,
                hand_plane_angle_rounded,
                index_plane_angle_rounded,
                middle_plane_angle_rounded,
                ring_plane_angle_rounded,
                pinky_plane_angle_rounded
            )

        cv2.imshow("Hand", frame)

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

    node.destroy_node()

    try:
        if rclpy.ok():
            rclpy.shutdown()
    except Exception:
        pass


if __name__ == '__main__':
    main()