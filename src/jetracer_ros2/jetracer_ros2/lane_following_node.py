"""Camera lane follower for the 2026 autorace mission stack.

This node never publishes ``/cmd_vel``.  It publishes a proposed command on
``inference/cmd_vel`` and the mission manager is the sole vehicle commander.
"""

import os
import sys
import time
from typing import Optional

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist, Vector3Stamped
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import Bool, String

from jetracer_ros2.lane_pipeline import LanePipeline, LanePipelineConfig
from jetracer_ros2.lane_attributes import scale_steering
from jetracer_ros2.lane_steering import (
    reference_side, tracking_error, curve_release_target, limited_steering,
)
from jetracer_ros2.utils.ackermann_utils import inner_angle_to_omega
from jetracer_ros2.live_debug_view import LiveViewer
from jetracer_ros2.lane_ackermann import target_to_inner_angle
from jetracer_ros2.driving_profile import (
    ZigzagTracker, select_speed, yellow_zigzag_evidence,
)


VALID_MODES = {'center', 'left', 'right', 'heading_hold'}


class LaneFollowingNode(Node):
    def __init__(self):
        super().__init__('lane_following_node')
        self._declare_parameters()
        self.pipeline = LanePipeline(self._pipeline_config())
        self._camera_matrix: Optional[np.ndarray] = None
        self._distortion: Optional[np.ndarray] = None
        self._last_image_ns = 0
        self._last_warning_ns = 0
        self._last_status_ns = 0
        self._zigzag_tracker = ZigzagTracker()
        self._zigzag_active = False
        self._mode = 'center'
        self._held_steering = 0.0
        self._applied_steering = 0.0
        self._last_steering_ns = 0
        self._curve_active = False
        self._curve_direction = 0.0
        self._curve_candidate_direction = 0.0
        self._curve_hits = 0
        self._geometry_hits = 0
        self._curve_weight = 0.0
        self._invalid_frames = 0
        self._last_valid_cmd: Optional[Twist] = None
        self._last_valid_ns = 0
        self._last_valid_curve = False

        self.commands_enabled = self.get_parameter('publish_commands').value
        self.cmd_pub = (self.create_publisher(Twist, self.get_parameter('output_topic').value, 10)
                        if self.commands_enabled else None)
        self.viewer = None
        self._actual_cmd = None
        self._actual_cmd_time = 0.0
        if self.get_parameter('live_debug_enabled').value:
            try:
                self.viewer = LiveViewer(float(self.get_parameter('live_debug_fps').value))
                self.actual_cmd_sub = self.create_subscription(
                    Twist, '/cmd_vel', self._actual_cmd_callback, qos_profile_sensor_data)
            except (OSError, ValueError, RuntimeError) as exc:
                self.get_logger().warn(f'Live debugger unavailable: {exc}')
        self.model_pub = self.create_publisher(
            Vector3Stamped, '/lane/model', 10)
        self.debug_pub = self.create_publisher(
            CompressedImage, '/lane/debug/compressed', 2)
        self.zigzag_pub = self.create_publisher(Bool, '/lane/zigzag', 10)
        self.mode_state_pub = self.create_publisher(String, '/lane/mode_state', 10)
        self.mode_sub = self.create_subscription(
            String, self.get_parameter('mode_topic').value,
            self._mode_callback, 10)
        self.image_sub = self.create_subscription(
            CompressedImage, self.get_parameter('image_topic').value,
            self._image_callback, qos_profile_sensor_data)
        if self.get_parameter('use_camera_info').value:
            self.camera_info_sub = self.create_subscription(
                CameraInfo, self.get_parameter('camera_info_topic').value,
                self._camera_info_callback, qos_profile_sensor_data)
        self.watchdog = self.create_timer(0.05, self._watchdog_callback)
        self.get_logger().info(
            f"lane follower ready: {self.get_parameter('image_topic').value} -> "
            f"{self.get_parameter('output_topic').value}; command output={self.commands_enabled}; "
            f"metric Ackermann={self.get_parameter('metric_lookahead_enabled').value}")

    def _declare_parameters(self):
        values = {
            'image_topic': '/camera/color/image_raw/compressed',
            'camera_info_topic': '/camera/color/camera_info',
            'output_topic': 'inference/cmd_vel',
            'mode_topic': 'mission/lane_mode',
            'use_camera_info': True,
            'publish_debug': True,
            'publish_commands': True,  # Startup-only. False creates no command publisher.
            'live_debug_enabled': False,
            'live_debug_fps': 5.0,
            'command_timeout_sec': 0.30,
            'minimum_confidence': 0.48,
            'normal_speed': 0.70,
            'curve_speed': 0.30,
            'curve_steering_speed_threshold': 0.35,
            'zigzag_speed': 0.18,
            'lateral_gain': 0.95,
            'heading_gain': 0.70,
            'steering_gain': 1.20,
            'left_reference_gain': 1.35,
            'preview_heading_gain': 0.25,
            'right_reference_gain': 1.0,
            'status_log_interval_sec': 1.0,
            'max_steer_angle': 0.42,
            'ackermann_wheelbase_m': 0.20,
            'ackermann_track_m': 0.172,
            'curve_enter_error': 0.55,
            'curve_exit_error': 0.25,
            'curve_enter_frames': 2,
            'curve_shape_min_span': 0.20,
            'curve_shape_full_span': 0.40,
            'curve_shape_confirm_frames': 2,
            'curve_max_steering_ratio': 1.0,
            'steering_attack_rate': 4.0,
            'steering_release_rate': 1.2,
            'steering_reversal_rate': 4.0,
            'heading_hold_decay': 0.98,
            'invalid_hold_sec': 0.25,
            'curve_recovery_sec': 0.50,
            'curve_recovery_speed': 0.12,
            'reset_after_invalid_frames': 5,
            'zigzag_confirm_sec': 2.0,
            'zigzag_gap_sec': 0.35,
            'zigzag_hold_sec': 1.5,
            'image_width': 640,
            'image_height': 480,
            'bev_src': [0.3605, 0.46, 0.6395, 0.46, 1.00, 0.82, 0.00, 0.82],
            'bev_dst': [0.22, 0.00, 0.78, 0.00, 0.78, 1.00, 0.22, 1.00],
            'detect_white': True,
            'white_lightness_min': 170,
            'white_saturation_max': 100,
            'detect_yellow': False,
            'yellow_hue_min': 15,
            'yellow_hue_max': 35,
            'yellow_saturation_min': 80,
            'yellow_lightness_min': 60,
            'gradient_min': 35,
            'gradient_require_color_support': True,
            'gradient_support_dilation': 9,
            'morphology_kernel': 3,
            'windows': 10,
            'window_margin': 55,
            'opposite_window_margin': 30,
            'search_opposite_in_single_mode': False,
            'histogram_min_peak': 12,
            'bottom_tracking_windows': 3,
            'recenter_min_pixels': 35,
            'fit_min_points': 120,
            'polynomial_degree': 3,
            'ransac_iterations': 60,
            'ransac_residual_px': 7.0,
            'lane_width_px': 370.0,
            'lane_width_tolerance': 0.35,
            'max_lane_width_change_px': 70.0,
            'max_fit_jump_px': 90.0,
            'strong_single_inlier_ratio': 0.65,
            'strong_single_point_multiplier': 1.5,
            'lookahead_y_ratio': 0.92,
            'smoothing_alpha': 0.35,
            'zigzag_min_step_px': 8.0,
            'zigzag_min_span_px': 55.0,
            'zigzag_min_direction_changes': 1,
            'duplicate_overlap_ratio': 0.30,
            'duplicate_gap_ratio': 0.50,
            'low_light_enabled': True,
            'low_light_max_gain': 3.0,
            'preview_y_ratio': 0.76,
            'metric_lookahead_enabled': False,
            'lookahead_distance_m': 0.40,
            'camera_forward_offset_m': 0.184,
            'metric_lateral_scale_m': 0.30,
            'bev_ground_points_m': [0.0] * 8,
            'metric_max_near_clamp_m': 0.25,
            'metric_edge_margin_m': 0.02,
            'target_roi_margin_px': 20.0,
        }
        for name, default in values.items():
            self.declare_parameter(name, default)

    def _pipeline_config(self):
        p = lambda name: self.get_parameter(name).value
        return LanePipelineConfig(
            low_light_enabled=p('low_light_enabled'), low_light_max_gain=p('low_light_max_gain'),
            preview_y_ratio=p('preview_y_ratio'),
            metric_lookahead_enabled=p('metric_lookahead_enabled'),
            lookahead_distance_m=p('lookahead_distance_m'),
            camera_forward_offset_m=p('camera_forward_offset_m'),
            metric_lateral_scale_m=p('metric_lateral_scale_m'),
            bev_ground_points_m=tuple(p('bev_ground_points_m')),
            metric_max_near_clamp_m=p('metric_max_near_clamp_m'),
            metric_edge_margin_m=p('metric_edge_margin_m'),
            target_roi_margin_px=p('target_roi_margin_px'),
            width=p('image_width'), height=p('image_height'),
            src_points=tuple(p('bev_src')), dst_points=tuple(p('bev_dst')),
            detect_white=p('detect_white'),
            white_lightness_min=p('white_lightness_min'),
            white_saturation_max=p('white_saturation_max'),
            detect_yellow=p('detect_yellow'),
            yellow_hue_min=p('yellow_hue_min'), yellow_hue_max=p('yellow_hue_max'),
            yellow_saturation_min=p('yellow_saturation_min'),
            yellow_lightness_min=p('yellow_lightness_min'),
            gradient_min=p('gradient_min'),
            gradient_require_color_support=p('gradient_require_color_support'),
            gradient_support_dilation=p('gradient_support_dilation'),
            morphology_kernel=p('morphology_kernel'),
            windows=p('windows'), window_margin=p('window_margin'),
            opposite_window_margin=p('opposite_window_margin'),
            search_opposite_in_single_mode=p('search_opposite_in_single_mode'),
            histogram_min_peak=p('histogram_min_peak'),
            bottom_tracking_windows=p('bottom_tracking_windows'),
            recenter_min_pixels=p('recenter_min_pixels'), fit_min_points=p('fit_min_points'),
            polynomial_degree=p('polynomial_degree'),
            ransac_iterations=p('ransac_iterations'), ransac_residual_px=p('ransac_residual_px'),
            lane_width_px=p('lane_width_px'), lane_width_tolerance=p('lane_width_tolerance'),
            max_lane_width_change_px=p('max_lane_width_change_px'),
            max_fit_jump_px=p('max_fit_jump_px'),
            strong_single_inlier_ratio=p('strong_single_inlier_ratio'),
            strong_single_point_multiplier=p('strong_single_point_multiplier'),
            lookahead_y_ratio=p('lookahead_y_ratio'),
            smoothing_alpha=p('smoothing_alpha'), zigzag_min_step_px=p('zigzag_min_step_px'),
            zigzag_min_span_px=p('zigzag_min_span_px'),
            zigzag_min_direction_changes=p('zigzag_min_direction_changes'),
            duplicate_overlap_ratio=p('duplicate_overlap_ratio'),
            duplicate_gap_ratio=p('duplicate_gap_ratio'))

    def _camera_info_callback(self, msg: CameraInfo):
        if len(msg.k) == 9 and msg.k[0] > 0.0:
            self._camera_matrix = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
            self._distortion = np.asarray(msg.d, dtype=np.float64)

    def _mode_callback(self, msg: String):
        mode = msg.data.strip().lower()
        if mode not in VALID_MODES:
            self.get_logger().warn(f'ignored invalid lane mode: {mode}')
            return
        if mode != self._mode:
            self.pipeline.reset()
            self._reset_steering()
            self._mode = mode
            self.get_logger().info(f'lane mode -> {mode}')
        self.mode_state_pub.publish(String(data=self._mode))

    def _image_callback(self, msg: CompressedImage):
        self._last_image_ns = self.get_clock().now().nanoseconds
        image = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            self._safe_stop('image decode failed')
            return
        raw_image = image
        if self._camera_matrix is not None and self._distortion is not None:
            image = cv2.undistort(image, self._camera_matrix, self._distortion)
        try:
            tracking_mode = self._mode if self._mode in ('left', 'right') else 'auto'
            result = self.pipeline.process(image, tracking_mode=tracking_mode)
        except (cv2.error, ValueError, TypeError, np.linalg.LinAlgError) as error:
            self._safe_stop(f'lane processing failed: {error}')
            return

        now_ns = self.get_clock().now().nanoseconds
        self._update_zigzag(yellow_zigzag_evidence(result), now_ns)
        usable = result.valid and result.confidence >= self.get_parameter('minimum_confidence').value
        if usable:
            self._invalid_frames = 0
            self._update_curve_geometry(result)
        else:
            self._invalid_frames += 1
            self._geometry_hits = 0
            self._curve_weight = 0.0
            self._curve_hits = 0
            self._curve_candidate_direction = 0.0
            reset_after = int(self.get_parameter('reset_after_invalid_frames').value)
            if reset_after > 0 and self._invalid_frames % reset_after == 0:
                # 이전 차선 기준이 오래되면 급변 검사(max_fit_jump_px)에 계속 걸려
                # 영영 복구하지 못하므로, 연속 실패 시 기준을 초기화한다.
                self.pipeline.reset()
                self._warn(f'lane lost for {self._invalid_frames} frames, pipeline reset')

        cmd = Twist()
        if usable:
            error = self._mode_error(result)
            max_angle = float(self.get_parameter('max_steer_angle').value)
            if result.target_lateral_m is not None:
                # Ground calibration enabled: steer from the rear axle to a
                # measured metric target, without adding image-space gains.
                target_steering = target_to_inner_angle(
                    result.target_lateral_m, result.target_forward_m,
                    self.get_parameter('ackermann_wheelbase_m').value,
                    self.get_parameter('ackermann_track_m').value,
                    max_angle) / max_angle
            else:
                raw_steering = float(np.clip(-error, -1.0, 1.0))
                target_steering = self._curve_steering_target(raw_steering)
                target_steering = scale_steering(
                    target_steering, 1.0 + self._curve_weight * (
                        self.get_parameter('steering_gain').value - 1.0))
            steering = self._rate_limit_steering(target_steering, now_ns)
            self._held_steering = steering
            curve_detected = (
                self._curve_active
                or result.curve_heading_span >= self.get_parameter('curve_shape_min_span').value
                # Continue at curve speed while steering is still being released.
                or abs(steering) >= self.get_parameter('curve_steering_speed_threshold').value)
            speed = select_speed(
                self._zigzag_active, curve_detected,
                self.get_parameter('normal_speed').value,
                self.get_parameter('curve_speed').value,
                self.get_parameter('zigzag_speed').value)
            cmd.linear.x = speed
            cmd.angular.z = self._omega(steering, speed)
            self._last_valid_cmd = cmd
            self._last_valid_ns = now_ns
            self._last_valid_curve = curve_detected and result.confidence >= 0.70
        elif (self._last_valid_cmd is not None and self._last_valid_curve
              and 0 <= (now_ns - self._last_valid_ns) / 1e9 <= min(
                  0.8, max(0.0, self.get_parameter('curve_recovery_sec').value))):
            # Short detection gap only; camera watchdog still stops stale input.
            speed = max(0.0, min(0.12, self._last_valid_cmd.linear.x,
                                self.get_parameter('curve_recovery_speed').value))
            cmd.linear.x = speed
            cmd.angular.z = self._omega(self._held_steering, speed)
        elif self._mode == 'heading_hold' and abs(self._held_steering) > 1e-3:
            self._held_steering *= self.get_parameter('heading_hold_decay').value
            self._applied_steering = self._held_steering
            self._last_steering_ns = now_ns
            cmd.linear.x = self.get_parameter('zigzag_speed').value
            cmd.angular.z = self._omega(self._held_steering, cmd.linear.x)
        elif (self._last_valid_cmd is not None
              and (now_ns - self._last_valid_ns) / 1e9 <= self.get_parameter('invalid_hold_sec').value):
            # 반사·그림자 등으로 잠깐 놓친 프레임은 마지막 명령을 유지해 덜컥거림을 막는다.
            # Even a brief blind interval must not retain the 0.70 m/s cruise.
            cmd.linear.x = min(0.12, self._last_valid_cmd.linear.x)
            cmd.angular.z = self._omega(self._held_steering, cmd.linear.x)
        else:
            self._reset_steering()
            self._warn(f'lane invalid: {result.reason}, confidence={result.confidence:.2f}')
        self._publish_command(cmd)
        interval = float(self.get_parameter('status_log_interval_sec').value)
        if interval > 0 and (now_ns - self._last_status_ns) / 1e9 >= interval:
            self._last_status_ns = now_ns
            steering_status = (
                f'target_norm={target_steering:+.2f} applied_norm={steering:+.2f} '
                f'angle_cmd_deg={np.degrees(steering * self.get_parameter("max_steer_angle").value):+.1f} '
                if usable else 'steering=held_or_stopped '
            )
            self.get_logger().info(
                f'lane type(est): L={result.left_type} R={result.right_type} '
                f'confidence={result.confidence:.2f} usable={usable} '
                f'lateral={result.lateral_error:+.3f} heading={result.heading_error:+.3f} '
                f'shape_span={result.curve_heading_span:.3f} curve_weight={self._curve_weight:.2f} '
                f'reference={reference_side(result)} '
                f'lookahead={result.lookahead_description} light_gain={result.illumination_gain:.2f} '
                f'curve_control={self._curve_active} zigzag_slow={self._zigzag_active} '
                f'{steering_status}'
                f'v={cmd.linear.x:.3f} omega={cmd.angular.z:.3f} '
                f'reason={result.reason}')
        if self.viewer and self.viewer.wants_frame():
            state = dict(
                received_mono=time.monotonic(), source_stamp_ns=msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec,
                commands_enabled=self.commands_enabled, valid=bool(result.valid), usable=bool(usable),
                confidence=float(result.confidence), reason=result.reason, mode=self._mode,
                reference=reference_side(result), left_type=result.left_type, right_type=result.right_type,
                lateral=float(result.lateral_error), heading_rad=float(result.heading_error),
                shape_span=float(result.curve_heading_span), curve_weight=float(self._curve_weight),
                curve_control=bool(self._curve_active), lookahead=result.lookahead_description,
                angle_cmd_deg=float(np.degrees(self._applied_steering * self.get_parameter('max_steer_angle').value)),
                lane_v=float(cmd.linear.x), lane_omega=float(cmd.angular.z), actual_cmd=self._actual_cmd,
                actual_cmd_age_sec=round(time.monotonic() - self._actual_cmd_time, 2) if self._actual_cmd_time else None)
            self.viewer.submit(dict(
                state=state, raw=raw_image,
                camera=cv2.resize(image, (self.pipeline.config.width, self.pipeline.config.height)),
                bev=result.bev_image, windows=result.debug_image, binary=result.binary_image,
                color=result.color_binary_image, gradient=result.gradient_binary_image,
                left_fit=result.left_fit, right_fit=result.right_fit, target_bev_xy=result.target_bev_xy,
                inverse_matrix=self.pipeline._inverse_matrix,
                roi=self.pipeline._normalized_points(self.pipeline.config.src_points),
                bev_roi=self.pipeline._normalized_points(self.pipeline.config.dst_points)))
        self._publish_result(msg, result)

    def _publish_command(self, command):
        if self.cmd_pub is not None:
            self.cmd_pub.publish(command)

    def _omega(self, normalized_steering, speed):
        return inner_angle_to_omega(
            normalized_steering * self.get_parameter('max_steer_angle').value,
            speed, self.get_parameter('ackermann_wheelbase_m').value,
            self.get_parameter('ackermann_track_m').value)

    def _actual_cmd_callback(self, msg):
        self._actual_cmd = dict(v=round(float(msg.linear.x), 3), omega=round(float(msg.angular.z), 3))
        self._actual_cmd_time = time.monotonic()

    def close_viewer(self):
        if self.viewer:
            self.viewer.close()

    def _update_curve_geometry(self, result):
        """Require measured bending on consecutive valid images before boosting."""
        minimum = max(0.0, float(self.get_parameter('curve_shape_min_span').value))
        full = max(minimum + 1e-6, float(self.get_parameter('curve_shape_full_span').value))
        span = result.curve_heading_span
        if not np.isfinite(span) or span <= minimum:
            self._geometry_hits = 0
            self._curve_weight = 0.0
            return
        self._geometry_hits += 1
        frames = max(1, int(self.get_parameter('curve_shape_confirm_frames').value))
        if self._geometry_hits < frames:
            self._curve_weight = 0.0
            return
        t = float(np.clip((span - minimum) / (full - minimum), 0.0, 1.0))
        self._curve_weight = t * t * (3.0 - 2.0 * t)

    def _curve_steering_target(self, raw_steering):
        """곡선 진입은 크게 조향하고 오차 감소에 따라 목표 조향을 푼다.

        한 프레임의 노이즈로 최대 조향이 걸리지 않도록 연속 프레임을 확인하고,
        진입/탈출 임계값을 다르게 둬 곡선 경계에서 상태가 흔들리지 않게 한다.
        """
        if self._curve_weight <= 0.0:
            self._curve_active = False
            self._curve_hits = 0
            self._curve_direction = self._curve_candidate_direction = 0.0
            return raw_steering
        magnitude = abs(raw_steering)
        direction = float(np.sign(raw_steering))
        enter = self.get_parameter('curve_enter_error').value
        exit_error = self.get_parameter('curve_exit_error').value

        if self._curve_active:
            if direction == 0.0 or direction != self._curve_direction or magnitude <= exit_error:
                self._curve_active = False
                self._curve_direction = 0.0
                self._curve_hits = 0
        elif magnitude >= enter and direction != 0.0:
            if direction == self._curve_candidate_direction:
                self._curve_hits += 1
            else:
                self._curve_candidate_direction = direction
                self._curve_hits = 1
            if self._curve_hits >= int(self.get_parameter('curve_enter_frames').value):
                self._curve_active = True
                self._curve_direction = direction
        else:
            self._curve_candidate_direction = 0.0
            self._curve_hits = 0

        if self._curve_active:
            ratio = self.get_parameter('curve_max_steering_ratio').value
            peak_target = self._curve_direction * curve_release_target(
                magnitude, enter, exit_error, float(np.clip(ratio, 0.0, 1.0)))
            return raw_steering + self._curve_weight * (peak_target - raw_steering)
        return raw_steering

    def _rate_limit_steering(self, target, now_ns):
        """진입/반전은 빠르게, 같은 방향의 조향 해제는 천천히 적용한다."""
        previous = self._applied_steering
        if self._last_steering_ns:
            delta_sec = (now_ns - self._last_steering_ns) / 1e9
        else:
            delta_sec = 0.05
        self._last_steering_ns = now_ns
        self._applied_steering = limited_steering(
            previous, target, delta_sec,
            self.get_parameter('steering_attack_rate').value,
            self.get_parameter('steering_release_rate').value,
            self.get_parameter('steering_reversal_rate').value)
        if abs(self._applied_steering) < 1e-4:
            self._applied_steering = 0.0
        return self._applied_steering

    def _mode_error(self, result):
        return tracking_error(
            result, self.get_parameter('lateral_gain').value,
            self.get_parameter('heading_gain').value,
            self.get_parameter('left_reference_gain').value,
            self.get_parameter('right_reference_gain').value,
            self.get_parameter('preview_heading_gain').value, self._curve_weight)

    def _reset_steering(self):
        self._last_valid_curve = False
        self._geometry_hits = 0
        self._curve_weight = 0.0
        self._curve_active = False
        self._curve_direction = 0.0
        self._curve_candidate_direction = 0.0
        self._curve_hits = 0
        self._applied_steering = self._held_steering = 0.0
        self._last_steering_ns = 0
        self._last_valid_cmd = None
        self._last_valid_ns = 0

    def _update_zigzag(self, detected, now_ns):
        self._zigzag_active = self._zigzag_tracker.update(
            detected, now_ns / 1e9,
            self.get_parameter('zigzag_confirm_sec').value,
            self.get_parameter('zigzag_gap_sec').value,
            self.get_parameter('zigzag_hold_sec').value)
        self.zigzag_pub.publish(Bool(data=self._zigzag_active))

    def _publish_result(self, source, result):
        model = Vector3Stamped()
        model.header = source.header
        model.header.frame_id = 'lane_model' if result.valid else 'lane_invalid'
        model.vector.x = result.lateral_error
        model.vector.y = result.heading_error
        model.vector.z = result.confidence
        self.model_pub.publish(model)
        if not self.get_parameter('publish_debug').value or result.debug_image is None:
            return
        ok, encoded = cv2.imencode('.jpg', result.debug_image)
        if ok:
            debug = CompressedImage(header=source.header, format='jpeg', data=encoded.tobytes())
            self.debug_pub.publish(debug)

    def _watchdog_callback(self):
        if self.viewer:
            message = self.viewer.poll_message()
            if message:
                if 'unavailable' in message:
                    self.get_logger().warn(message)
                else:
                    self.get_logger().info(message)
        if self._last_image_ns == 0 or ((self.get_clock().now().nanoseconds - self._last_image_ns) / 1e9
                                       > self.get_parameter('command_timeout_sec').value):
            self._reset_steering()
            self._publish_command(Twist())

    def _safe_stop(self, reason):
        self._reset_steering()
        self._publish_command(Twist())
        self._warn(reason)

    def _warn(self, text):
        now = self.get_clock().now().nanoseconds
        if now - self._last_warning_ns > 2_000_000_000:
            self.get_logger().warn(text)
            self._last_warning_ns = now


def main(args=None):
    argv = sys.argv[1:] if args is None else args
    params = os.path.join(get_package_share_directory('jetracer_ros2'), 'params', 'params.yaml')
    rclpy.init(args=[sys.argv[0], '--ros-args', '--params-file', params, '--'] + list(argv))
    node = LaneFollowingNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node._publish_command(Twist())
        node.close_viewer()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
