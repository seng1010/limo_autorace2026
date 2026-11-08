"""Camera lane follower for the 2026 autorace mission stack.

This node never publishes ``/cmd_vel``.  It publishes a proposed command on
``inference/cmd_vel`` and the mission manager is the sole vehicle commander.
"""

import os
import sys
from dataclasses import fields
from typing import Optional

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist, Vector3Stamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import Bool, Float32, String

from jetracer_ros2.lane_pipeline_ks import LanePipeline, LanePipelineConfig
from jetracer_ros2.m6_zigzag import M6Config, M6ZigzagGuide
from jetracer_ros2.utils.ackermann_utils import inner_angle_to_omega


VALID_MODES = {'center', 'left', 'right', 'heading_hold'}

# M6Config 중 노드의 다른 파라미터에서 가져오는 값 (m6_ 파라미터로 따로 만들지 않음)
M6_SHARED = {'image_width', 'image_height', 'lane_width_px', 'lookahead_y_ratio',
             'lateral_gain', 'heading_gain'}


class LaneFollowingNode(Node):
    def __init__(self):
        super().__init__('lane_following_node')
        self._declare_parameters()
        self.pipeline = LanePipeline(self._pipeline_config())
        self._camera_matrix: Optional[np.ndarray] = None
        self._distortion: Optional[np.ndarray] = None
        self._last_image_ns = 0
        self._last_warning_ns = 0
        self._zigzag_hits = 0
        self._zigzag_clears = 0
        self._zigzag_active = False
        self._mode = 'center'
        self._held_steering = 0.0
        self._invalid_frames = 0
        self._last_valid_cmd: Optional[Twist] = None
        self._last_valid_ns = 0
        # ---- M6 서행 유도선(지그재그) ----
        self.m6 = M6ZigzagGuide(self._m6_config())
        self._stop_line_distance: Optional[float] = None
        self._stop_line_ns = 0

        self.cmd_pub = self.create_publisher(
            Twist, self.get_parameter('output_topic').value, 10)
        self.model_pub = self.create_publisher(
            Vector3Stamped, '/lane/model', 10)
        self.debug_pub = self.create_publisher(
            CompressedImage, '/lane/debug/compressed', 2)
        self.zigzag_pub = self.create_publisher(Bool, '/lane/zigzag', 10)
        self.mode_state_pub = self.create_publisher(String, '/lane/mode_state', 10)
        self.m6_state_pub = self.create_publisher(String, '/lane/m6_state', 10)
        self.event_pub = self.create_publisher(
            String, self.get_parameter('m6_event_topic').value, 10)
        self.stop_line_sub = self.create_subscription(
            Float32, self.get_parameter('m6_stop_line_topic').value,
            self._stop_line_callback, 10)
        if self.get_parameter('m6_require_mission').value:
            # 매니저가 'M6' 를 알려줄 때만 M6 모드를 허용한다
            self.m6.set_enabled(False)
            self.mission_sub = self.create_subscription(
                String, self.get_parameter('m6_mission_topic').value,
                self._mission_callback, 10)
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
            f"{self.get_parameter('output_topic').value}")

    def _declare_parameters(self):
        values = {
            'image_topic': '/camera/color/image_raw/compressed',
            'camera_info_topic': '/camera/color/camera_info',
            'output_topic': 'inference/cmd_vel',
            'mode_topic': 'mission/lane_mode',
            'use_camera_info': True,
            'publish_debug': True,
            'command_timeout_sec': 0.30,
            'minimum_confidence': 0.48,
            'normal_speed': 0.25,
            'zigzag_speed': 0.12,
            'lateral_gain': 0.95,
            'heading_gain': 0.70,
            'max_steer_angle': 0.42,
            'heading_hold_decay': 0.98,
            'invalid_hold_sec': 0.25,
            'reset_after_invalid_frames': 5,
            'zigzag_enter_frames': 3,
            'zigzag_exit_frames': 5,
            'image_width': 640,
            'image_height': 480,
            'bev_src': [0.30, 0.52, 0.70, 0.52, 0.94, 0.92, 0.06, 0.92],
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
            'morphology_kernel': 3,
            'windows': 10,
            'window_margin': 55,
            'recenter_min_pixels': 35,
            'fit_min_points': 120,
            'polynomial_degree': 3,
            'ransac_iterations': 60,
            'ransac_residual_px': 7.0,
            'lane_width_px': 360.0,
            'lane_width_tolerance': 0.35,
            'max_lane_width_change_px': 70.0,
            'max_fit_jump_px': 90.0,
            'lookahead_y_ratio': 0.62,
            'smoothing_alpha': 0.35,
            'zigzag_min_step_px': 8.0,
            'zigzag_min_span_px': 55.0,
            'zigzag_min_direction_changes': 1,
            # ---- M6 서행 유도선 (자세한 설명은 m6_zigzag.py 의 M6Config) ----
            'm6_enable': True,
            'm6_require_mission': False,      # True: 매니저가 M6 차례라고 알려줄 때만 동작
            'm6_mission_topic': 'mission/active',
            'm6_event_topic': 'mission/event',
            'm6_stop_line_topic': 'mission/stop_line_distance',
            'm6_publish_done': True,          # 끝나면 mission/event 로 'M6_DONE' 발행
            'm6_speed': 0.15,                 # M6 구간 속도 [m/s] (규정: 서행 구간 0.2 이하)
        }
        for item in fields(M6Config):
            if item.name not in M6_SHARED:
                values['m6_' + item.name] = item.default
        for name, default in values.items():
            self.declare_parameter(name, default)

    def _pipeline_config(self):
        p = lambda name: self.get_parameter(name).value
        return LanePipelineConfig(
            width=p('image_width'), height=p('image_height'),
            src_points=tuple(p('bev_src')), dst_points=tuple(p('bev_dst')),
            detect_white=p('detect_white'),
            white_lightness_min=p('white_lightness_min'),
            white_saturation_max=p('white_saturation_max'),
            detect_yellow=p('detect_yellow'),
            yellow_hue_min=p('yellow_hue_min'), yellow_hue_max=p('yellow_hue_max'),
            yellow_saturation_min=p('yellow_saturation_min'),
            yellow_lightness_min=p('yellow_lightness_min'),
            gradient_min=p('gradient_min'), morphology_kernel=p('morphology_kernel'),
            windows=p('windows'), window_margin=p('window_margin'),
            recenter_min_pixels=p('recenter_min_pixels'), fit_min_points=p('fit_min_points'),
            polynomial_degree=p('polynomial_degree'),
            ransac_iterations=p('ransac_iterations'), ransac_residual_px=p('ransac_residual_px'),
            lane_width_px=p('lane_width_px'), lane_width_tolerance=p('lane_width_tolerance'),
            max_lane_width_change_px=p('max_lane_width_change_px'),
            max_fit_jump_px=p('max_fit_jump_px'), lookahead_y_ratio=p('lookahead_y_ratio'),
            smoothing_alpha=p('smoothing_alpha'), zigzag_min_step_px=p('zigzag_min_step_px'),
            zigzag_min_span_px=p('zigzag_min_span_px'),
            zigzag_min_direction_changes=p('zigzag_min_direction_changes'))

    def _m6_config(self):
        p = lambda name: self.get_parameter(name).value
        config = M6Config(
            image_width=p('image_width'), image_height=p('image_height'),
            lane_width_px=p('lane_width_px'), lookahead_y_ratio=p('lookahead_y_ratio'),
            lateral_gain=p('lateral_gain'), heading_gain=p('heading_gain'))
        for item in fields(M6Config):
            if item.name not in M6_SHARED:
                setattr(config, item.name, type(item.default)(p('m6_' + item.name)))
        return config

    def _stop_line_callback(self, msg: Float32):
        self._stop_line_distance = float(msg.data)
        self._stop_line_ns = self.get_clock().now().nanoseconds

    def _mission_callback(self, msg: String):
        enabled = msg.data.strip().upper() == 'M6'
        self.m6.set_enabled(enabled)
        self.get_logger().info(f'mission {msg.data} -> M6 mode {"allowed" if enabled else "off"}')

    def _recent_stop_line(self, now_ns):
        if self._stop_line_distance is None or now_ns - self._stop_line_ns > 500_000_000:
            return None
        return self._stop_line_distance

    def _update_m6(self, result, now_ns):
        """M6 판단. M6 전용 조향이면 그 결과를, 아니면 None 을 돌려준다."""
        if not self.get_parameter('m6_enable').value or result.binary_image is None:
            return None
        seeds = (result.left_fit, result.right_fit) if result.valid else (None, None)
        try:
            out = self.m6.update(result.binary_image, now_ns / 1e9, seeds,
                                 self._held_steering, self._recent_stop_line(now_ns))
        except (cv2.error, ValueError, TypeError, np.linalg.LinAlgError) as error:
            self._warn(f'M6 processing failed: {error}')
            return None
        if out.just_entered:
            self.get_logger().info('M6 zigzag: ACTIVE (slow straight guidance)')
        if out.send_done:
            self.get_logger().info(f'M6 done: {out.reason}')
            if self.get_parameter('m6_publish_done').value:
                self.event_pub.publish(String(data='M6_DONE'))
        if out.just_exited:
            self.get_logger().info(f'M6 zigzag: mode off ({out.reason})')
            # M6 동안 쌓인 차선 기준은 지그재그 기준이라 일반 주행에 맞지 않는다
            self.pipeline.reset()
        self.m6_state_pub.publish(String(data=out.state))
        return out

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
            self._mode = mode
            self.get_logger().info(f'lane mode -> {mode}')
        self.mode_state_pub.publish(String(data=self._mode))

    def _image_callback(self, msg: CompressedImage):
        self._last_image_ns = self.get_clock().now().nanoseconds
        image = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            self._safe_stop('image decode failed')
            return
        if self._camera_matrix is not None and self._distortion is not None:
            image = cv2.undistort(image, self._camera_matrix, self._distortion)
        try:
            result = self.pipeline.process(image)
        except (cv2.error, ValueError, TypeError, np.linalg.LinAlgError) as error:
            self._safe_stop(f'lane processing failed: {error}')
            return

        now_ns = self.get_clock().now().nanoseconds
        m6 = self._update_m6(result, now_ns)
        self._update_zigzag(result.valid and result.zigzag, m6 is not None and m6.active)
        if m6 is not None and m6.active:
            # M6: 지그재그 선마다 직선을 맞춘 결과로 천천히, 부드럽게 직진
            cmd = Twist()
            cmd.linear.x = self.get_parameter('m6_speed').value
            cmd.angular.z = inner_angle_to_omega(
                m6.steering * self.get_parameter('max_steer_angle').value, cmd.linear.x)
            self._held_steering = m6.steering
            self._last_valid_cmd = cmd
            self._last_valid_ns = now_ns
            self._invalid_frames = 0
            self.cmd_pub.publish(cmd)
            self._publish_result(msg, result, m6)
            return

        usable = result.valid and result.confidence >= self.get_parameter('minimum_confidence').value
        if usable:
            self._invalid_frames = 0
        else:
            self._invalid_frames += 1
            reset_after = int(self.get_parameter('reset_after_invalid_frames').value)
            if reset_after > 0 and self._invalid_frames % reset_after == 0:
                # 이전 차선 기준이 오래되면 급변 검사(max_fit_jump_px)에 계속 걸려
                # 영영 복구하지 못하므로, 연속 실패 시 기준을 초기화한다.
                self.pipeline.reset()
                self._warn(f'lane lost for {self._invalid_frames} frames, pipeline reset')

        cmd = Twist()
        if usable:
            error = self._mode_error(result)
            steering = float(np.clip(-error, -1.0, 1.0))
            self._held_steering = steering
            speed = (self.get_parameter('zigzag_speed').value
                     if self._zigzag_active else self.get_parameter('normal_speed').value)
            cmd.linear.x = speed
            cmd.angular.z = inner_angle_to_omega(
                steering * self.get_parameter('max_steer_angle').value, speed)
            self._last_valid_cmd = cmd
            self._last_valid_ns = now_ns
        elif self._mode == 'heading_hold' and abs(self._held_steering) > 1e-3:
            self._held_steering *= self.get_parameter('heading_hold_decay').value
            cmd.linear.x = self.get_parameter('zigzag_speed').value
            cmd.angular.z = inner_angle_to_omega(
                self._held_steering * self.get_parameter('max_steer_angle').value,
                cmd.linear.x)
        elif (self._last_valid_cmd is not None
              and (now_ns - self._last_valid_ns) / 1e9 <= self.get_parameter('invalid_hold_sec').value):
            # 반사·그림자 등으로 잠깐 놓친 프레임은 마지막 명령을 유지해 덜컥거림을 막는다.
            cmd = self._last_valid_cmd
        else:
            self._warn(f'lane invalid: {result.reason}, confidence={result.confidence:.2f}')
        self.cmd_pub.publish(cmd)
        self._publish_result(msg, result, m6)

    def _mode_error(self, result):
        lateral = result.lateral_error
        if self._mode in ('left', 'right'):
            y = self.get_parameter('image_height').value * self.get_parameter('lookahead_y_ratio').value
            width = self.get_parameter('lane_width_px').value
            target_x = self.get_parameter('image_width').value / 2.0
            if self._mode == 'left' and result.left_fit is not None:
                lateral = (np.polyval(result.left_fit, y) + width / 2.0 - target_x) / (target_x)
            elif self._mode == 'right' and result.right_fit is not None:
                lateral = (np.polyval(result.right_fit, y) - width / 2.0 - target_x) / target_x
        return (self.get_parameter('lateral_gain').value * lateral
                + self.get_parameter('heading_gain').value * result.heading_error)

    def _update_zigzag(self, detected, m6_active=False):
        if detected:
            self._zigzag_hits += 1
            self._zigzag_clears = 0
            if self._zigzag_hits >= self.get_parameter('zigzag_enter_frames').value:
                self._zigzag_active = True
        else:
            self._zigzag_clears += 1
            self._zigzag_hits = 0
            if self._zigzag_clears >= self.get_parameter('zigzag_exit_frames').value:
                self._zigzag_active = False
        self.zigzag_pub.publish(Bool(data=self._zigzag_active or m6_active))

    def _publish_result(self, source, result, m6=None):
        model = Vector3Stamped()
        model.header = source.header
        model.header.frame_id = 'lane_model' if result.valid else 'lane_invalid'
        model.vector.x = result.lateral_error
        model.vector.y = result.heading_error
        model.vector.z = result.confidence
        self.model_pub.publish(model)
        if not self.get_parameter('publish_debug').value:
            return
        image = result.debug_image
        if image is None and result.binary_image is not None:
            image = result.binary_image
        if image is None:
            return
        if m6 is not None:
            image = self.m6.draw(image, m6)
        ok, encoded = cv2.imencode('.jpg', image)
        if ok:
            debug = CompressedImage(header=source.header, format='jpeg', data=encoded.tobytes())
            self.debug_pub.publish(debug)

    def _watchdog_callback(self):
        if self._last_image_ns == 0 or ((self.get_clock().now().nanoseconds - self._last_image_ns) / 1e9
                                       > self.get_parameter('command_timeout_sec').value):
            self.cmd_pub.publish(Twist())

    def _safe_stop(self, reason):
        self.cmd_pub.publish(Twist())
        self._warn(reason)

    def _warn(self, text):
        now = self.get_clock().now().nanoseconds
        if now - self._last_warning_ns > 2_000_000_000:
            self.get_logger().warn(text)
            self._last_warning_ns = now


def main(args=None):
    argv = sys.argv if args is None else args
    params = os.path.join(get_package_share_directory('jetracer_ros2'), 'params', 'params.yaml')
    rclpy.init(args=[argv[0], '--ros-args', '--params-file', params])
    node = LaneFollowingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
