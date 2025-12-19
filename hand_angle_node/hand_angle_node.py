#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import deque
import time

import cv2
import mediapipe as mp
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


# 指ごとのランドマークID (MediaPipe Hands)
FINGERS = {
    "index":  (5, 6, 7),
    "middle": (9, 10, 11),
    "ring":   (13, 14, 15),
    "pinky":  (17, 18, 19),
}
FINGER_ORDER = ["index", "middle", "ring", "pinky"]


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


class HandAngleNode(Node):
    """
    /hand_norm に [index, middle, ring, pinky] の順で 0〜1 を publish
    0: closed, 1: open
    """

    def __init__(self):
        super().__init__("hand_angle_node")

        # ===== パラメータ =====
        self.topic_name = self.declare_parameter("topic_name", "/hand_norm").value
        self.camera_index = int(self.declare_parameter("camera_index", 0).value)
        self.refresh_hz = float(self.declare_parameter("refresh_hz", 30.0).value)
        self.smooth_win = int(self.declare_parameter("smooth_win", 5).value)
        self.calib_sec = float(self.declare_parameter("calib_sec", 5.0).value)

        self.max_num_hands = int(self.declare_parameter("max_num_hands", 1).value)
        self.min_det_conf = float(self.declare_parameter("min_detection_confidence", 0.5).value)
        self.min_trk_conf = float(self.declare_parameter("min_tracking_confidence", 0.5).value)

        self.display = bool(self.declare_parameter("display", True).value)
        self.window_name = self.declare_parameter("window_name", "HandAngleNode").value

        # 手を見失ったら何フレームでゼロ＆バッファクリアするか
        self.miss_reset_frames = int(self.declare_parameter("miss_reset_frames", 10).value)

        # ===== publisher =====
        self.pub_norm = self.create_publisher(Float32MultiArray, self.topic_name, 10)

        # ===== MediaPipe =====
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.max_num_hands,
            min_detection_confidence=self.min_det_conf,
            min_tracking_confidence=self.min_trk_conf,
        )

        # ===== Camera =====
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"カメラが開けませんでした。(index={self.camera_index})")

        # ===== 状態 =====
        self.buf = {finger: deque(maxlen=self.smooth_win) for finger in FINGERS.keys()}
        self.calibration = {finger: {"open": None, "closed": None} for finger in FINGERS.keys()}

        # キャリブ用データ
        self.mode = "CALIB_OPEN"  # CALIB_OPEN -> CALIB_CLOSED -> RUN
        self.calib_start_t = time.monotonic()
        self.last_progress_log_t = self.calib_start_t
        self.calib_samples = {finger: [] for finger in FINGERS.keys()}

        self.miss_count = 0

        self.get_logger().info(f"Calibration start: OPEN for {self.calib_sec:.1f} sec")

        # ===== timer =====
        self.timer = self.create_timer(1.0 / max(self.refresh_hz, 1.0), self._tick)

    def _normalize_angle(self, finger: str, angle: float) -> float:
        """キャリブ結果に基づいて 0〜1 に正規化（0:閉じ, 1:開き）"""
        open_a = self.calibration[finger]["open"]
        closed_a = self.calibration[finger]["closed"]
        if open_a is None or closed_a is None or abs(open_a - closed_a) < 1e-6:
            return 0.0
        # closed → open が 0→1 になるように
        norm = (angle - closed_a) / (open_a - closed_a)
        return float(np.clip(norm, 0.0, 1.0))

    def _extract_angles(self, lm):
        """ランドマークから指角度を計算して dictで返す（見えない指は None）"""
        angles = {f: None for f in FINGERS.keys()}
        for finger, (a_id, b_id, c_id) in FINGERS.items():
            a = (lm[a_id].x, lm[a_id].y, lm[a_id].z)
            b = (lm[b_id].x, lm[b_id].y, lm[b_id].z)
            c = (lm[c_id].x, lm[c_id].y, lm[c_id].z)
            ang = calculate_angle(a, b, c)
            angles[finger] = ang
        return angles

    def _publish_norms(self, norms_dict):
        msg = Float32MultiArray()
        msg.data = [float(norms_dict.get(f, 0.0) or 0.0) for f in FINGER_ORDER]
        self.pub_norm.publish(msg)

    def _reset_buffers(self):
        for f in self.buf:
            self.buf[f].clear()

    def _tick(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warning("Failed to read frame from camera.")
            self._publish_norms({f: 0.0 for f in FINGERS.keys()})
            return

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        has_hand = bool(results and results.multi_hand_landmarks)

        # ---- 見失い処理 ----
        if not has_hand:
            self.miss_count += 1
            if self.miss_count >= self.miss_reset_frames:
                self._reset_buffers()
                self._publish_norms({f: 0.0 for f in FINGERS.keys()})
            self._draw_status(frame, extra="NO HAND")
            self._maybe_show(frame)
            return
        else:
            self.miss_count = 0

        lm = results.multi_hand_landmarks[0].landmark
        angles = self._extract_angles(lm)

        # ---- キャリブ状態機械 ----
        now = time.monotonic()
        if self.mode in ("CALIB_OPEN", "CALIB_CLOSED"):
            state = "open" if self.mode == "CALIB_OPEN" else "closed"

            # サンプル追加
            for finger, ang in angles.items():
                if ang is not None:
                    self.calib_samples[finger].append(ang)

            # 1秒ごとにログ
            if now - self.last_progress_log_t >= 1.0:
                elapsed = now - self.calib_start_t
                self.get_logger().info(
                    f"{state} calibration... {elapsed:.1f}/{self.calib_sec:.1f} sec"
                )
                self.last_progress_log_t = now

            # 終了判定
            if now - self.calib_start_t >= self.calib_sec:
                # 平均を保存
                for finger in FINGERS.keys():
                    vals = self.calib_samples[finger]
                    self.calibration[finger][state] = float(np.mean(vals)) if vals else None
                    self.get_logger().info(
                        f"Calib[{finger}][{state}] = {self.calibration[finger][state]}"
                    )

                # 次状態へ
                self.calib_samples = {finger: [] for finger in FINGERS.keys()}
                self.calib_start_t = now
                self.last_progress_log_t = now
                self._reset_buffers()

                if self.mode == "CALIB_OPEN":
                    self.mode = "CALIB_CLOSED"
                    self.get_logger().info(f"Calibration: CLOSED for {self.calib_sec:.1f} sec")
                else:
                    self.mode = "RUN"
                    self.get_logger().info("Calibration done. Start publishing /hand_norm.")

            self._draw_status(frame, extra=f"CALIB: {state.upper()}")
            self._maybe_show(frame)
            return

        # ---- RUN: 平滑化 + 正規化 + publish ----
        norms = {f: 0.0 for f in FINGERS.keys()}
        debug_angles = {f: None for f in FINGERS.keys()}

        for finger, ang in angles.items():
            if ang is None:
                continue
            self.buf[finger].append(ang)
            smoothed = float(np.mean(self.buf[finger])) if len(self.buf[finger]) > 0 else ang
            debug_angles[finger] = smoothed
            norms[finger] = self._normalize_angle(finger, smoothed)

        self._publish_norms(norms)

        # 表示（任意）
        self._draw_angles(frame, debug_angles)
        self._draw_status(frame, extra="RUN")
        self._maybe_show(frame)

    def _draw_angles(self, frame, angles_per_finger):
        y0, dy = 30, 25
        for idx, finger in enumerate(FINGER_ORDER):
            ang = angles_per_finger.get(finger)
            text = f"{finger}: ---" if ang is None else f"{finger}: {ang:6.1f} deg"
            cv2.putText(frame, text, (10, y0 + idx * dy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    def _draw_status(self, frame, extra=""):
        cv2.putText(frame, f"mode={self.mode} {extra}", (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA)
        if self.display:
            cv2.putText(frame, "Press q or ESC to quit", (10, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA)

    def _maybe_show(self, frame):
        if not self.display:
            return
        cv2.imshow(self.window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):  # ESC or q
            rclpy.shutdown()

    def destroy_node(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        try:
            self.hands.close()
        except Exception:
            pass
        if self.display:
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
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
