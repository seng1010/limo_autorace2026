"""
JetRacer 추론(자율주행) 노드.

구간(section)마다 하나씩 TensorRT 엔진(section_1.engine ... section_5.engine)을
미리 로드해두고, /current_section(Int32) 토픽으로 어느 구간 모델을 쓸지 실시간으로
바꿀 수 있다. 예측 결과를 곧바로 /cmd_vel로 publish한다 (중간 토픽 없음).

TensorRT가 없는 개발 PC에서는 PyTorch(.pth) 모델로 자동 대체(fallback)한다.

전처리 파이프라인 (데이터 수집 때와 반드시 동일해야 함, 비율은 params.yaml의 crop_ratio):
    CompressedImage → decode(640×480) → 하단 크롭 → resize(224,224)
    → ImageNet 정규화 → (1,3,224,224)
    ※ 중간에 다른 크기로 리사이즈하지 않음 — 딱 한 번만 리사이즈해 품질 손실 최소화

모델 출력: -1~1 사이의 float 하나 (회귀).
    train.py 라벨 관례와 동일: +1=완전좌회전, -1=완전우회전
    이 값에 max_steer_angle을 곱해서 "의도한 실제 조향각(inner_angle)"을 구하고,
    그걸 다시 limo_base가 이해하는 angular.z로 변환해서 보낸다
    (data_collection_node.py와 완전히 동일한 방식 — ackermann_utils.py 공용 모듈 사용).

section_id:=N 을 주면 시작할 때부터 해당 구간 모델로 추론한다
(원래는 /current_section 토픽으로만 바꿀 수 있었음, 그 방식은 그대로 유지됨).

사용법:
    ros2 run jetracer_ros2 inference_node
    ros2 run jetracer_ros2 inference_node section_id:=2
"""

import os
import re
import sys

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32
from cv_bridge import CvBridge
import cv2
import numpy as np
from ament_index_python.packages import get_package_share_directory

from jetracer_ros2.utils.ackermann_utils import inner_angle_to_omega

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401
    HAS_TRT = True
except ImportError:
    HAS_TRT = False

try:
    import torch
    import torchvision.models as models
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ------------------------------------------------------------------ #
# TensorRT wrapper
# ------------------------------------------------------------------ #

class _TRTModel:
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

        logger.info(f'TRT engine loaded  in={in_shape}  out={out_shape}  [{engine_path}]')

    def infer(self, input_tensor: np.ndarray) -> float:
        np.copyto(self.h_in, input_tensor.ravel())
        cuda.memcpy_htod_async(self.d_in, self.h_in, self.stream)
        self.context.execute_async_v3(self.stream.handle)
        cuda.memcpy_dtoh_async(self.h_out, self.d_out, self.stream)
        self.stream.synchronize()
        return float(self.h_out[0])


# ------------------------------------------------------------------ #
# PyTorch fallback wrapper
# ------------------------------------------------------------------ #

class _TorchModel:
    def __init__(self, pth_path: str, device: str, logger):
        import torch
        import torchvision.models as models
        self.device = torch.device(device)
        model = models.resnet18(weights=None)
        import torch.nn as nn
        model.fc = nn.Linear(model.fc.in_features, 1)
        model.load_state_dict(torch.load(pth_path, map_location=self.device, weights_only=True))
        model.eval()
        self.model = model.to(self.device)
        logger.info(f'PyTorch model loaded  [{pth_path}]')

    def infer(self, input_tensor: np.ndarray) -> float:
        import torch
        t = torch.from_numpy(input_tensor).to(self.device)
        with torch.no_grad():
            out = self.model(t)
        return float(out.item())


# ------------------------------------------------------------------ #
# Inference node
# ------------------------------------------------------------------ #

class InferenceNode(Node):
    def __init__(self):
        super().__init__('inference_node')
        self.br = CvBridge()

        self.declare_parameter('engine_dir', '/home/wego/third_impact/src/jetracer_engines')
        self.declare_parameter('num_sections', 5)
        self.declare_parameter('linear_x', 1.0)   # 자율주행 고정 속도 [m/s] (수집 때와 동일해야 함)
        self.declare_parameter('max_steer_angle', 0.42)  # 학습 라벨과 동일한 값을 써야 함
        # data_collector_node의 crop_ratio와 반드시 같은 값을 써야 한다 (수집/추론 전처리 일치).
        self.declare_parameter('crop_ratio', 0.0)

        engine_dir = self.get_parameter('engine_dir').value
        num_sections = self.get_parameter('num_sections').value
        self.linear_x = self.get_parameter('linear_x').value
        self.max_steer_angle = self.get_parameter('max_steer_angle').value
        self.crop_ratio = self.get_parameter('crop_ratio').value

        self.current_section = 1
        self.models: dict = {}
        self._load_models(engine_dir, num_sections)

        self.image_sub = self.create_subscription(
            CompressedImage, '/camera/color/image_raw/compressed',
            self.image_callback, 10)
        self.section_sub = self.create_subscription(
            Int32, '/current_section',
            self.section_callback, 10)

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.debug_pub = self.create_publisher(
            CompressedImage, '/inference/debug/compressed', 10)

        loaded = list(self.models.keys())
        self.get_logger().info(f'InferenceNode ready  loaded_sections={loaded}')
        if not loaded:
            self.get_logger().warn(
                f'No models loaded — check engine_dir ({engine_dir}) for section_N.engine/.pth files.')

    # ------------------------------------------------------------------ #

    def _load_models(self, engine_dir: str, num_sections: int):
        # 디렉토리에서 section_N.engine / section_N.pth 파일을 자동 스캔
        candidates: set = set()
        if os.path.isdir(engine_dir):
            for fname in os.listdir(engine_dir):
                m = re.match(r'^section_(\d+)\.(engine|pth)$', fname)
                if m:
                    candidates.add(int(m.group(1)))
        # num_sections 범위도 병합 (하위 호환)
        candidates |= set(range(1, num_sections + 1))

        for sec in sorted(candidates):
            # Prefer TRT engine
            if HAS_TRT:
                path = os.path.join(engine_dir, f'section_{sec}.engine')
                if os.path.exists(path):
                    try:
                        self.models[sec] = _TRTModel(path, self.get_logger())
                        continue
                    except Exception as e:
                        self.get_logger().error(
                            f'TRT load failed section={sec}: {e}')

            # Fall back to PyTorch .pth
            if HAS_TORCH:
                pth_path = os.path.join(engine_dir, f'section_{sec}.pth')
                if os.path.exists(pth_path):
                    try:
                        device = 'cuda' if torch.cuda.is_available() else 'cpu'
                        self.models[sec] = _TorchModel(
                            pth_path, device, self.get_logger())
                    except Exception as e:
                        self.get_logger().error(
                            f'PyTorch load failed section={sec}: {e}')

    def section_callback(self, msg: Int32):
        """/current_section 토픽으로 구간이 바뀌면 그 구간 전용 모델로 전환한다."""
        self.current_section = msg.data
        self.get_logger().info(f'Switched to section {self.current_section}')

    def image_callback(self, msg: CompressedImage):
        """카메라 프레임마다 현재 구간 모델로 조향을 예측해서 /cmd_vel로 내보낸다."""
        model = self.models.get(self.current_section)
        if model is None:
            return

        np_arr = np.frombuffer(msg.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            return

        # Pre-process: 640×480 → crop_ratio만큼 상단 크롭 → 224×224 (단 한 번만 리사이즈)
        h, w = image.shape[:2]
        crop_y = int(h * self.crop_ratio)
        crop = image[crop_y:h, :]                          # e.g. 640×192
        inp = cv2.resize(crop, (224, 224),
                         interpolation=cv2.INTER_LINEAR)   # 224×224
        inp = inp[:, :, ::-1].astype(np.float32) / 255.0  # BGR→RGB + 정규화
        inp = (inp - _IMAGENET_MEAN) / _IMAGENET_STD
        inp = inp.transpose(2, 0, 1)[np.newaxis].astype(np.float32)  # (1,3,224,224)

        try:
            raw = model.infer(inp)
        except Exception as e:
            self.get_logger().warn(f'Inference error: {e}')
            return

        # raw ∈ [-1, 1] matches train.py's label convention: +1=full-left, -1=full-right
        raw = float(np.clip(raw, -1.0, 1.0))
        inner_angle = raw * self.max_steer_angle

        # limo_base derives steering from r = linear.x / angular.z, so the
        # intended wheel angle has to be converted into the angular.z that
        # reproduces it (see ackermann_utils.inner_angle_to_omega). Left/right
        # asymmetry compensation is not reapplied here — it's already baked
        # into the model weights from the boosted training labels.
        angular_z = inner_angle_to_omega(inner_angle, self.linear_x)

        cmd = Twist()
        cmd.linear.x = self.linear_x
        cmd.angular.z = angular_z
        self.cmd_pub.publish(cmd)

        self._publish_debug(crop, raw)

    def _publish_debug(self, crop: np.ndarray, raw: float):
        debug = crop.copy()
        h, w = debug.shape[:2]
        cx = w // 2
        cy = h - 10
        # 조향 방향 시각화: 중심에서 예측 방향으로 선 그리기 (raw>0=left)
        bar_x = int(cx - raw * cx)
        bar_x = max(0, min(w - 1, bar_x))
        cv2.line(debug, (cx, cy), (bar_x, cy), (0, 255, 0), 3)
        cv2.circle(debug, (cx, cy), 5, (255, 0, 0), -1)
        cv2.circle(debug, (bar_x, cy), 5, (0, 0, 255), -1)
        cv2.putText(debug,
                    f'sec:{self.current_section}  raw:{raw:+.3f}',
                    (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        self.debug_pub.publish(self.br.cv2_to_compressed_imgmsg(debug))


# ------------------------------------------------------------------ #

def _extract_plain_arg(argv, name, default):
    """`name:=value` 형태로 넘어온 인자를 찾는다.

    ros2 run으로 `--ros-args` 없이 바로 넘긴 `section_id:=2` 같은 인자는
    rclpy가 파라미터로 자동 인식하지 않으므로 직접 파싱한다.
    """
    pattern = re.compile(rf'^{re.escape(name)}:=(.*)$')
    for arg in argv:
        m = pattern.match(arg)
        if m:
            return m.group(1)
    return default


def main(args=None):
    argv = sys.argv if args is None else args
    section_id = int(_extract_plain_arg(argv, 'section_id', 1))

    pkg = get_package_share_directory('jetracer_ros2')
    params_path = os.path.join(pkg, 'params', 'params.yaml')
    ros_args = ['--ros-args', '--params-file', params_path]

    rclpy.init(args=[argv[0]] + ros_args)
    node = InferenceNode()

    node.current_section = section_id
    if section_id in node.models:
        node.get_logger().info(f'Starting with section={section_id}')
    else:
        node.get_logger().warn(
            f'section={section_id} 모델이 로드되지 않았습니다 (engine_dir 확인).')

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
