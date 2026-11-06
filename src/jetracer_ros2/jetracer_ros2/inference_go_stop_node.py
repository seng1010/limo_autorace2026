import math
import os
import sys

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool

from jetracer_ros2.utils.ackermann_utils import inner_angle_to_omega

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401
    HAS_TRT = True
except ImportError:
    HAS_TRT = False

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# go_stop.engine(또는 .pth) 하나로 조향(steering)과 정지판단(stop)을 동시에
# 예측하는 멀티태스크 모델 전용 노드. section별 조향 모델이 따로 필요 없다 —
# 이 엔진 하나로 바로 주행 + 정지판단을 다 처리한다 (inference_node.py와는
# 완전히 별개, 서로 안 건드림).
#
# 학습(train_go_stop.py)은 go_stop/go/, go_stop/stop/ 폴더의 모든 이미지에서
# 파일명 cx로 조향 라벨을, 폴더(go/stop)로 정지 라벨을 동시에 뽑아서 출력 2개짜리
# 모델로 학습한다. 이 노드는 그 두 출력을 그대로 씀:
#   output[0] = steering ∈[-1,1] (train.py와 동일한 라벨 관례)
#   output[1] = stop raw logit (여기서 sigmoid를 씌워 확률로 씀)
#
# 안전장치 (오탐 방지): stop_thresh 이상 확률이 stop_consec 프레임 연속으로 나오고
# 노드 시작 후 stop_min_time초가 지났으면 정지 확정 — /cmd_vel 영구 정지 +
# parking_start_topic 발행. stop_dry_run:=true면 실제로 안 멈추고 로그만 (재무장돼서
# 계속 관찰 가능). 정지 확정 후에도 모델 추론(조향 계산) 자체는 계속 돈다 — 그냥
# 그 출력을 cmd_vel로 내보내지만 않을 뿐이다.
#
# 사용법:
#   ros2 run jetracer_ros2 inference_go_stop_node
#   ros2 run jetracer_ros2 inference_go_stop_node --ros-args -p stop_dry_run:=true


class _GoStopTRTModel:
    def __init__(self, engine_path: str, logger):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        self.in_name = self.engine.get_tensor_name(0)
        self.out_name = self.engine.get_tensor_name(1)
        in_shape = self.engine.get_tensor_shape(self.in_name)
        out_shape = self.engine.get_tensor_shape(self.out_name)

        self.h_in = cuda.pagelocked_empty(trt.volume(in_shape), dtype=np.float32)
        self.d_in = cuda.mem_alloc(self.h_in.nbytes)
        self.h_out = cuda.pagelocked_empty(trt.volume(out_shape), dtype=np.float32)
        self.d_out = cuda.mem_alloc(self.h_out.nbytes)
        self.stream = cuda.Stream()

        self.context.set_tensor_address(self.in_name, int(self.d_in))
        self.context.set_tensor_address(self.out_name, int(self.d_out))

        logger.info(f'GoStop TRT engine loaded  in={in_shape}  out={out_shape}  [{engine_path}]')

    def infer(self, input_tensor: np.ndarray):
        np.copyto(self.h_in, input_tensor.ravel())
        cuda.memcpy_htod_async(self.d_in, self.h_in, self.stream)
        self.context.execute_async_v3(self.stream.handle)
        cuda.memcpy_dtoh_async(self.h_out, self.d_out, self.stream)
        self.stream.synchronize()
        return float(self.h_out[0]), float(self.h_out[1])  # steering, stop_logit


class _GoStopTorchModel:
    def __init__(self, pth_path: str, device: str, logger):
        import torch
        import torch.nn as nn
        import torchvision.models as models
        self.device = torch.device(device)
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model.load_state_dict(torch.load(pth_path, map_location=self.device, weights_only=True))
        model.eval()
        self.model = model.to(self.device)
        logger.info(f'GoStop PyTorch model loaded  [{pth_path}]')

    def infer(self, input_tensor: np.ndarray):
        import torch
        t = torch.from_numpy(input_tensor).to(self.device)
        with torch.no_grad():
            out = self.model(t)
        vals = out.squeeze(0).cpu().numpy()
        return float(vals[0]), float(vals[1])


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class InferenceGoStopNode(Node):
    def __init__(self):
        super().__init__('inference_go_stop_node')

        self.declare_parameter('engine_dir', '/home/wego/third_impact/src/jetracer_ros2/jetracer_engines')
        self.declare_parameter('linear_x', 1.0)
        self.declare_parameter('max_steer_angle', 0.42)
        self.declare_parameter('crop_ratio', 0.0)
        self.declare_parameter('stop_engine_path', '')  # 비우면 engine_dir/go_stop.(engine|pth) 자동 탐색
        self.declare_parameter('stop_thresh', 0.85)       # sigmoid 확률이 이 이상이어야 stop 후보로 침
        self.declare_parameter('stop_consec', 5)           # 연속 이 프레임 수만큼 후보여야 확정
        self.declare_parameter('stop_min_time', 3.0)       # 노드 시작 후 이 시간(초) 전에는 무시
        self.declare_parameter('stop_dry_run', False)      # true면 실제로 안 멈추고 로그만
        self.declare_parameter('parking_start_topic', '/parking_start')

        engine_dir = self.get_parameter('engine_dir').value
        self.linear_x = self.get_parameter('linear_x').value
        self.max_steer_angle = self.get_parameter('max_steer_angle').value
        self.crop_ratio = self.get_parameter('crop_ratio').value
        stop_engine_path = self.get_parameter('stop_engine_path').value
        self.stop_thresh = self.get_parameter('stop_thresh').value
        self.stop_consec = self.get_parameter('stop_consec').value
        self.stop_min_time = self.get_parameter('stop_min_time').value
        self.stop_dry_run = self.get_parameter('stop_dry_run').value
        parking_start_topic = self.get_parameter('parking_start_topic').value

        self.model = self._load_model(engine_dir, stop_engine_path)

        self._consec_count = 0
        self._start_time = self.get_clock().now()
        self._driving_enabled = True

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.parking_start_pub = self.create_publisher(Bool, parking_start_topic, 10)
        self.image_sub = self.create_subscription(
            CompressedImage, '/camera/color/image_raw/compressed',
            self.image_callback, 10)

        mode = 'DRY-RUN(로그만)' if self.stop_dry_run else '실제 정지'
        self.get_logger().info(
            f'GoStop ready — model_loaded={self.model is not None}  '
            f'thresh={self.stop_thresh}  consec={self.stop_consec}  '
            f'min_time={self.stop_min_time}s  모드={mode}')

    def _load_model(self, engine_dir: str, explicit_path: str):
        if explicit_path:
            engine_path = explicit_path
            pth_path = explicit_path
        else:
            engine_path = os.path.join(engine_dir, 'go_stop.engine')
            pth_path = os.path.join(engine_dir, 'go_stop.pth')

        if HAS_TRT and os.path.exists(engine_path):
            try:
                return _GoStopTRTModel(engine_path, self.get_logger())
            except Exception as e:
                self.get_logger().error(f'go_stop TRT 로드 실패: {e}')

        if HAS_TORCH and os.path.exists(pth_path):
            import torch
            try:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                return _GoStopTorchModel(pth_path, device, self.get_logger())
            except Exception as e:
                self.get_logger().error(f'go_stop PyTorch 로드 실패: {e}')

        self.get_logger().warn(
            f'go_stop 모델을 못 찾았습니다 ({engine_path} / {pth_path}).')
        return None

    def image_callback(self, msg: CompressedImage):
        if self.model is None:
            return

        np_arr = np.frombuffer(msg.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            return

        # inference_node.py와 동일한 전처리 (crop_ratio, 224×224, ImageNet 정규화)
        h, w = image.shape[:2]
        crop_y = int(h * self.crop_ratio)
        crop = image[crop_y:h, :]
        inp = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
        inp = inp[:, :, ::-1].astype(np.float32) / 255.0
        inp = (inp - _IMAGENET_MEAN) / _IMAGENET_STD
        inp = inp.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

        try:
            steer_raw, stop_logit = self.model.infer(inp)
        except Exception as e:
            self.get_logger().warn(f'go_stop 추론 오류: {e}')
            return

        # --- 조향: inference_node.py와 동일한 방식 ---
        steer_raw = float(np.clip(steer_raw, -1.0, 1.0))
        inner_angle = steer_raw * self.max_steer_angle
        angular_z = inner_angle_to_omega(inner_angle, self.linear_x)

        cmd = Twist()
        if self._driving_enabled:
            cmd.linear.x = self.linear_x
            cmd.angular.z = angular_z
        # else: 정지 확정 후에는 계속 0(Twist() 기본값)만 내보낸다.
        self.cmd_pub.publish(cmd)

        # --- 정지 판단 (조향 계산과 무관하게 계속 돈다) ---
        self._update_stop_decision(_sigmoid(stop_logit))

    def _update_stop_decision(self, prob: float):
        if not self._driving_enabled:
            return  # 이미 정지 확정됨

        elapsed = (self.get_clock().now() - self._start_time).nanoseconds / 1e9
        if prob >= self.stop_thresh and elapsed >= self.stop_min_time:
            self._consec_count += 1
        else:
            self._consec_count = 0

        self.get_logger().info(
            f'[stop-debug] prob={prob:.3f} (thresh={self.stop_thresh})  '
            f'consec={self._consec_count}/{self.stop_consec}  elapsed={elapsed:.1f}s',
            throttle_duration_sec=0.5)

        if self._consec_count >= self.stop_consec:
            self._on_stop_confirmed(prob)

    def _on_stop_confirmed(self, prob: float):
        if self.stop_dry_run:
            self.get_logger().warn(
                f'[DRY-RUN] STOP 확정됐을 것 (prob={prob:.3f}) — 실제로는 안 멈춤')
            self._consec_count = 0  # 재무장 — 계속 관찰 가능
            return

        self._driving_enabled = False
        self.get_logger().warn(
            f'STOP 확정 (prob={prob:.3f}) — cmd_vel 영구 정지, /parking_start 발행')
        self.cmd_pub.publish(Twist())
        self.parking_start_pub.publish(Bool(data=True))


def main(args=None):
    argv = sys.argv if args is None else args

    pkg = get_package_share_directory('jetracer_ros2')
    params_path = os.path.join(pkg, 'params', 'params.yaml')
    ros_args = ['--ros-args', '--params-file', params_path]

    rclpy.init(args=[argv[0]] + ros_args)
    node = InferenceGoStopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
