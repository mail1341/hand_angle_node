#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

# ====== 設定値 ===========================
SMOOTH_WIN = 5        # 角度の移動平均窓サイズ
REFRESH_HZ = 30.0     # メインループ周期 [Hz]
CALIB_SEC  = 5.0      # open/closed それぞれのキャリブ時間 [秒]

# 指ごとのランドマークID (MediaPipe Hands)
FINGERS = {
    "index":  (5, 6, 7),
    "middle": (9, 10, 11),
    "ring":   (13, 14, 15),
    "pinky":  (17, 18, 19),
}

# キャリブ結果を入れる（open / closed それぞれの平均角度）
# 例: {"index": {"open": 120.0, "closed": 50.0}, ...}
calibration = {}


def calculate_angle(a, b, c):
    """3点(a,b,c)で∠ABC [deg] を計算"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    na = np.linalg.norm(ba)
    nc = np.linalg.norm(bc)
    if na == 0 or nc == 0:
        return None
    cos_angle = np.dot(ba, bc) / (na * nc)
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def normalize_angle(finger, angle):
    """キャリブ結果に基づいて 0〜1 に正規化（0:閉じ, 1:開き）"""
    if (
        finger in calibration
        and "open" in calibration[finger]
        and "closed" in calibration[finger]
    ):
        open_a   = calibration[finger]["open"]
        closed_a = calibration[finger]["closed"]
        if open_a is None or closed_a is None or abs(open_a - closed_a) < 1e-6:
            return 0.0
        # closed → open が 0→1 になるように
        norm = (angle - closed_a) / (open_a - closed_a)
        return float(np.clip(norm, 0.0, 1.0))
    return 0.0


class HandAngleNode(Node):
    def __init__(self):
        super().__init__('hand_angle_node')

        # Publisher: 指の正規化角度 [index, middle, ring, pinky]
        self.pub_norm = self.create_publisher(
            Float32MultiArray, 'hand_norm', 10
        )

        # MediaPipe Hands 初期化
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # カメラ初期化（とりあえず /dev/video0）
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("カメラが開けませんでした。 (/dev/video0)")

        # 各指ごとの移動平均バッファ
        self.buf = {
            finger: deque(maxlen=SMOOTH_WIN) for finger in FINGERS.keys()
        }

        # 起動時に open / closed キャリブを実行
        self.perform_calibration("open")
        self.perform_calibration("closed")

        self.get_logger().info("Calibration done. Start main loop.")

        # タイマーで周期実行
        self.timer = self.create_timer(1.0 / REFRESH_HZ, self.timer_callback)

    # === キャリブレーション ===
    def perform_calibration(self, state: str):
        """open / closed それぞれの状態で CALIB_SEC 秒キャリブ"""
        self.get_logger().info(
            f"Put your hand in {state.upper()} position for {CALIB_SEC:.0f} seconds."
        )
        angles_list = []
        start_time = time.time()
        last_log_t = start_time

        while time.time() - start_time < CALIB_SEC:
            ret, frame = self.cap.read()
            if not ret:
                continue

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image)

            if results.multi_hand_landmarks:
                lm = results.multi_hand_landmarks[0].landmark
                for finger, (a_id, b_id, c_id) in FINGERS.items():
                    a = [lm[a_id].x, lm[a_id].y, lm[a_id].z]
                    b = [lm[b_id].x, lm[b_id].y, lm[b_id].z]
                    c = [lm[c_id].x, lm[c_id].y, lm[c_id].z]
                    angle = calculate_angle(a, b, c)
                    if angle is not None:
                        angles_list.append((finger, angle))

            # 1秒ごとに進捗をログに出す
            now = time.time()
            if now - last_log_t > 1.0:
                elapsed = int(now - start_time)
                self.get_logger().info(
                    f"  {state} calibration... {elapsed}/{int(CALIB_SEC)} s"
                )
                last_log_t = now

        # 指ごとに平均角度を保存
        for finger in FINGERS.keys():
            vals = [a for f, a in angles_list if f == finger]
            calibration.setdefault(finger, {})[state] = (
                float(np.mean(vals)) if vals else None
            )
            self.get_logger().info(
                f"Calib[{finger}][{state}] = {calibration[finger][state]}"
            )

    # === メインループ ===
    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Failed to read frame from camera.")
            return

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image)

        norms_per_finger  = {f: None for f in FINGERS.keys()}
        angles_per_finger = {f: None for f in FINGERS.keys()}

        if results and results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0].landmark  # 最初の手だけ

            for finger, (a_id, b_id, c_id) in FINGERS.items():
                a = [lm[a_id].x, lm[a_id].y, lm[a_id].z]
                b = [lm[b_id].x, lm[b_id].y, lm[b_id].z]
                c = [lm[c_id].x, lm[c_id].y, lm[c_id].z]
                ang = calculate_angle(a, b, c)
                if ang is None:
                    continue

                # 移動平均
                self.buf[finger].append(ang)
                smoothed = float(np.mean(self.buf[finger]))
                angles_per_finger[finger] = smoothed

                # 0〜1 に正規化
                norm = normalize_angle(finger, smoothed)
                norms_per_finger[finger] = norm

        # パブリッシュ用に配列を作成（index, middle, ring, pinky の順）
        msg = Float32MultiArray()
        msg.data = [
            norms_per_finger["index"]  if norms_per_finger["index"]  is not None else 0.0,
            norms_per_finger["middle"] if norms_per_finger["middle"] is not None else 0.0,
            norms_per_finger["ring"]   if norms_per_finger["ring"]   is not None else 0.0,
            norms_per_finger["pinky"]  if norms_per_finger["pinky"]  is not None else 0.0,
        ]
        self.pub_norm.publish(msg)

        # ===== 映像に角度を左上に1行ずつ描画 =====
        y0 = 30        # 開始Y
        dy = 25        # 行間
        for idx, finger in enumerate(["index", "middle", "ring", "pinky"]):
            ang = angles_per_finger[finger]
            if ang is None:
                text = f"{finger}: ---"
            else:
                text = f"{finger}: {ang:6.1f} deg"
            y = y0 + idx * dy
            cv2.putText(
                frame, text, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2, cv2.LINE_AA
            )

        # 画面表示
        cv2.imshow("HandAngleNode", frame)
        cv2.waitKey(1)

    def destroy_node(self):
        # 終了処理
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.hands.close()
        cv2.destroyAllWindows()
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
        rclpy.shutdown()


if __name__ == '__main__':
    main()
