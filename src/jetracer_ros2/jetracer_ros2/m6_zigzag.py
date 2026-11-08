"""M6 서행 유도선(지그재그 차선) 구간 처리. ROS에 의존하지 않는다.

역할
  1. 감지: 버드아이뷰(BEV) 이진 영상에서 '선 하나하나가 톱니처럼 흔들리는지' 본다.
     (양쪽 지그재그가 같은 방향이든 거울처럼 반대 방향이든 모두 감지된다.)
  2. 주행: M6 모드에서는 지그재그 선마다 '직선'을 맞춰 톱니를 평균내고,
     그 직선들의 가운데를 작고 부드러운 조향으로 따라간다 (= 직전 헤딩 유지).
  3. 상태: IDLE / ARMED / ACTIVE 와 끝 신호(M6_DONE)를 관리한다.

좌표 규칙은 lane_pipeline 과 같다.
  x: 영상 오른쪽 +, y: 영상 아래쪽 +, lateral/heading error 는 영상 오른쪽이 +.
  선 모델은 x = a * y + b (a = dx/dy).

길이 관련 값은 모두 lane_width_px(정상 차선 폭, BEV 픽셀)의 비율로 정한다.
실제 BEV 배율이 바뀌어도 lane_width_px 만 맞으면 그대로 동작하게 하기 위함이다.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np


IDLE = 'IDLE'        # 대기 (끝난 직후 쿨다운, 또는 M6 차례가 아님)
ARMED = 'ARMED'      # 지그재그를 찾는 중 (일반 주행)
ACTIVE = 'ACTIVE'    # M6 전용 주행 중


@dataclass
class M6Config:
    image_width: int = 640
    image_height: int = 480
    lane_width_px: float = 360.0
    lookahead_y_ratio: float = 0.62

    # ---- 감지 ----
    row_step: int = 4                    # 몇 픽셀 간격으로 행을 볼지
    row_top_ratio: float = 0.04          # 이 높이부터
    row_bottom_ratio: float = 0.98       # 이 높이까지 사용
    band_half_ratio: float = 0.20        # 선 주변 탐색 폭(한쪽) / 차선폭
    accept_ratio: float = 0.13           # 예측선에서 이만큼 안쪽 점만 선으로 인정 / 차선폭
    max_run_ratio: float = 0.12          # 한 행에서 덩어리 폭이 이보다 넓으면 선 아님(가로선 등)
    ignore_center_ratio: float = 0.50    # 차선 가운데 무시 폭(양쪽 합) / 차선폭 (마름모·글자)
    min_rows_ratio: float = 0.22         # 선으로 인정할 최소 행 비율 (주행용)
    detect_rows_ratio: float = 0.35      # 지그재그 판단에 필요한 최소 행 비율 (짧은 조각으로는 판단 안 함)
    min_amplitude_ratio: float = 0.030   # 톱니 최소 진폭 / 차선폭  (도면 기준 실제 약 0.078)
    max_amplitude_ratio: float = 0.18    # 톱니 최대 진폭 / 차선폭  (이보다 크면 노이즈)
    max_half_period_ratio: float = 0.35  # 톱니 반 주기(꺾이는 간격) 최대 / 차선폭
                                         #  도면 기준 실제 약 0.17. S자 커브는 이보다 훨씬 길어서 걸러진다
    enter_flips: int = 3                 # 시작: 화면 전체에서 '좌↔우 꺾임'이 이 횟수 이상
    enter_near_flips: int = 1            #   + 화면 아래쪽(가까운 곳)에서도 이 횟수 이상
    enter_near_ratio: float = 0.5        #   '아래쪽' = 화면 높이의 이 비율
                                         #  → 멀리 보일 때가 아니라 지그재그 바로 앞에서 켜진다
    stay_flips: int = 1                  # 유지할 때 필요한 횟수 (끝을 늦게 판단하도록)

    # ---- 상태 전환 ----
    enter_frames: int = 3                # 연속 N프레임 감지 → ACTIVE
    exit_frames: int = 5                 # 연속 N프레임 '선은 보이는데 지그재그 아님' → 끝
    min_active_sec: float = 1.0          # 켜진 뒤 이 시간 동안은 끝내지 않음
    max_active_sec: float = 20.0         # 안전장치: 이보다 길면 강제로 끝
    lost_exit_sec: float = 1.0           # 선이 아예 안 보이는 상태가 이만큼 지속되면 M6 모드 해제
    cooldown_sec: float = 20.0           # 끝난 뒤 이 시간 동안 다시 켜지지 않음

    # ---- 정지선으로 M6_DONE 일찍 보내기 ----
    use_stop_line: bool = True
    exit_stop_line_m: float = 0.8        # 정지선이 이 거리 안이면 M6_DONE
    stop_line_confirm: int = 3           # 연속 N번 확인
    stop_line_clear_msgs: int = 5        # 시작 가로선을 지나쳤다고 볼 '정지선 없음' 연속 횟수
    stop_line_arm_sec: float = 6.0       # 위 조건이 안 와도 이 시간이 지나면 정지선 판단 허용

    # ---- 조향 ----
    lateral_gain: float = 0.95           # lane_following_node 와 같은 값 사용
    heading_gain: float = 0.70
    steer_gain: float = 1.0              # 평소 대비 조향 세기 배율 (직선 맞춤이라 원래 흔들림이 없음)
    max_steer: float = 0.45              # 최대 조향 (-1~1 중, 평소 최대의 45%)
    max_steer_rate: float = 0.06         # 프레임당 조향 변화 한계
    blend_sec: float = 0.5               # 진입 직후 일반 조향 → M6 조향으로 바꾸는 시간
    lost_steer_decay: float = 0.90       # 선을 놓치면 조향을 서서히 0(직진)으로
    model_alpha: float = 0.40            # 직선 모델 부드럽게 (0~1, 클수록 새 값 반영)
    max_jump_ratio: float = 0.15         # 한 프레임에 선이 이보다 많이 튀면 무시 / 차선폭
    side_hold_frames: int = 5            # 한쪽 선이 안 보여도 이 프레임 동안은 이전 값 사용


@dataclass
class SideFit:
    """한쪽 선의 이번 프레임 측정 결과."""
    ok: bool = False
    a: float = 0.0
    b: float = 0.0
    points: Optional[np.ndarray] = None     # (N, 2) [x, y]
    amplitude: float = 0.0
    flips: int = 0           # 보이는 전체 구간에서 센 좌↔우 꺾임 횟수
    flips_near: int = 0      # 그중 가까운(화면 아래쪽) 구간에 있는 꺾임 횟수


@dataclass
class M6Output:
    state: str = ARMED
    active: bool = False
    steering: float = 0.0           # -1~1, lane_following_node 의 steering 과 같은 의미
    lateral_error: float = 0.0
    heading_error: float = 0.0
    zigzag: bool = False            # 이번 프레임 지그재그 보임 (enter 기준)
    send_done: bool = False         # 이번 프레임에 M6_DONE 을 보내야 함
    just_entered: bool = False
    just_exited: bool = False
    reason: str = ''
    left: SideFit = field(default_factory=SideFit)
    right: SideFit = field(default_factory=SideFit)
    left_model: Optional[Tuple[float, float]] = None
    right_model: Optional[Tuple[float, float]] = None


class M6ZigzagGuide:
    def __init__(self, config: M6Config):
        self.config = config
        self.state = ARMED
        self._enabled = True                 # 미션 조건 (require_mission 일 때 외부에서 설정)
        self._hits = 0
        self._absent = 0
        self._entered_at: Optional[float] = None
        self._lost_since: Optional[float] = None
        self._cooldown_until = -1.0
        self._done_sent = False
        self._steer = 0.0
        self._blend_from = 0.0
        self._models: List[Optional[Tuple[float, float]]] = [None, None]
        self._model_age = [999, 999]
        self._width = config.lane_width_px
        self._stop_clear = 0
        self._stop_hits = 0
        self._stop_armed = False
        self.last_output = M6Output()

    # ------------------------------------------------------------------ 외부
    def set_enabled(self, enabled: bool) -> None:
        """M6 차례인지 (require_mission 사용 시). False 면 ACTIVE 도 해제."""
        self._enabled = bool(enabled)
        if not self._enabled and self.state == ACTIVE:
            self._leave(None, 'mission changed')

    def reset(self) -> None:
        enabled = self._enabled
        self.__init__(self.config)
        self._enabled = enabled

    # ------------------------------------------------------------------ 메인
    def update(
        self,
        binary: np.ndarray,
        now_sec: float,
        seed_fits: Tuple[Optional[np.ndarray], Optional[np.ndarray]] = (None, None),
        normal_steering: float = 0.0,
        stop_line_distance: Optional[float] = None,
    ) -> M6Output:
        """한 프레임 처리.

        binary: lane_pipeline 결과의 BEV 이진 영상 (H x W, 0/255)
        seed_fits: lane_pipeline 이 찾은 (left_fit, right_fit) 다항식. 없으면 None.
        normal_steering: 일반 주행이 이번 프레임에 낼 조향 (진입 시 부드럽게 이어 붙이는 용도)
        stop_line_distance: 정지선까지 거리[m], 안 보이면 음수, 정보 없음 None
        """
        cfg = self.config
        out = M6Output()

        if self.state == IDLE and now_sec >= self._cooldown_until and self._enabled:
            self.state = ARMED
        if self.state != ACTIVE and not self._enabled:
            self.state = IDLE

        left, right = self._measure(binary, seed_fits)
        out.left, out.right = left, right
        visible = left.ok or right.ok
        enter_ok = any(s.ok and s.flips >= cfg.enter_flips and s.flips_near >= cfg.enter_near_flips
                       for s in (left, right))
        stay_ok = any(s.ok and s.flips >= cfg.stay_flips for s in (left, right))
        out.zigzag = enter_ok

        if self.state == ARMED:
            self._hits = self._hits + 1 if enter_ok else 0
            if self._hits >= cfg.enter_frames:
                self._enter(now_sec, normal_steering)
                out.just_entered = True

        if self.state == ACTIVE:
            self._update_models(left, right)
            self._update_stop_line(now_sec, stop_line_distance, out)
            active_for = now_sec - self._entered_at

            if visible:
                self._lost_since = None
                self._absent = 0 if stay_ok else self._absent + 1
            elif self._lost_since is None:
                self._lost_since = now_sec

            if active_for >= cfg.max_active_sec:
                self._leave(now_sec, 'timeout', out, done=True)
            elif (self._lost_since is not None
                  and now_sec - self._lost_since >= cfg.lost_exit_sec):
                self._leave(now_sec, 'lines lost', out, done=False)
            elif active_for >= cfg.min_active_sec and self._absent >= cfg.exit_frames:
                self._leave(now_sec, 'zigzag ended', out, done=True)
            else:
                self._steer_step(now_sec, out)

        out.state = self.state
        out.active = self.state == ACTIVE
        out.left_model, out.right_model = self._models[0], self._models[1]
        self.last_output = out
        return out

    # ------------------------------------------------------------------ 상태
    def _enter(self, now_sec: float, normal_steering: float) -> None:
        self.state = ACTIVE
        self._entered_at = now_sec
        self._absent = 0
        self._lost_since = None
        self._done_sent = False
        self._blend_from = float(normal_steering)
        self._steer = float(normal_steering)
        self._stop_clear = 0
        self._stop_hits = 0
        self._stop_armed = False

    def _leave(self, now_sec: Optional[float], reason: str,
               out: Optional[M6Output] = None, done: bool = False) -> None:
        self.state = IDLE if done else ARMED
        self._hits = 0
        self._models = [None, None]
        self._model_age = [999, 999]
        if done and now_sec is not None:
            self._cooldown_until = now_sec + self.config.cooldown_sec
        if out is not None:
            out.just_exited = True
            out.reason = reason
            if done and not self._done_sent:
                out.send_done = True
                self._done_sent = True

    def _update_stop_line(self, now_sec: float, distance: Optional[float], out: M6Output) -> None:
        """정지선이 가까우면 M6_DONE 을 미리 보낸다 (조향은 지그재그가 끝날 때까지 계속).

        지그재그 시작 부분에도 가로선이 있으므로, 켜진 직후 보이는 선은 무시한다:
        '정지선 없음'이 몇 번 연속 나온 뒤(= 시작 선을 지나침)부터 판단한다.
        """
        cfg = self.config
        if not cfg.use_stop_line or self._done_sent or distance is None:
            return
        if not self._stop_armed:
            far = distance < 0.0 or distance > cfg.exit_stop_line_m + 0.2
            self._stop_clear = self._stop_clear + 1 if far else 0
            if (self._stop_clear >= cfg.stop_line_clear_msgs
                    or now_sec - self._entered_at >= cfg.stop_line_arm_sec):
                self._stop_armed = True
            return
        if 0.0 <= distance <= cfg.exit_stop_line_m:
            self._stop_hits += 1
        else:
            self._stop_hits = 0
        if self._stop_hits >= cfg.stop_line_confirm:
            out.send_done = True
            out.reason = 'stop line ahead'
            self._done_sent = True

    # ------------------------------------------------------------------ 측정
    def _row_ys(self) -> np.ndarray:
        cfg = self.config
        top = int(cfg.image_height * cfg.row_top_ratio)
        bottom = int(cfg.image_height * cfg.row_bottom_ratio)
        return np.arange(bottom, top, -cfg.row_step)

    def _seeds(self, binary, seed_fits, ys) -> List[Optional[np.ndarray]]:
        """각 행에서 선이 있을 것으로 예상되는 x (왼쪽, 오른쪽)."""
        cfg = self.config
        seeds: List[Optional[np.ndarray]] = [None, None]
        for side in (0, 1):
            model = self._models[side]
            if model is not None and self._model_age[side] <= cfg.side_hold_frames:
                seeds[side] = model[0] * ys + model[1]
            elif seed_fits[side] is not None:
                seeds[side] = np.polyval(seed_fits[side], ys)
        if seeds[0] is None or seeds[1] is None:
            hist = self._histogram_seeds(binary)
            for side in (0, 1):
                if seeds[side] is None and hist[side] is not None:
                    seeds[side] = np.full(ys.shape, float(hist[side]))
        # 두 예상선이 너무 가깝거나 순서가 뒤집히면 둘 중 하나는 틀린 것: 버린다
        if seeds[0] is not None and seeds[1] is not None:
            gap = float(np.median(seeds[1] - seeds[0]))
            if gap < 0.5 * cfg.lane_width_px:
                seeds = [None, None]
                hist = self._histogram_seeds(binary)
                for side in (0, 1):
                    if hist[side] is not None:
                        seeds[side] = np.full(ys.shape, float(hist[side]))
        return seeds

    def _histogram_seeds(self, binary) -> List[Optional[float]]:
        cfg = self.config
        h, w = binary.shape[:2]
        lower = binary[int(h * 0.4):, :] > 0
        hist = lower.sum(axis=0).astype(np.float32)
        k = max(3, int(cfg.lane_width_px * 0.15))
        hist = np.convolve(hist, np.ones(k, np.float32) / k, mode='same')
        minimum = 0.06 * lower.shape[0]
        seeds: List[Optional[float]] = [None, None]
        mid = w // 2
        for side, (lo, hi) in enumerate(((0, mid), (mid, w))):
            part = hist[lo:hi]
            if part.size and float(part.max()) >= minimum:
                seeds[side] = float(np.argmax(part) + lo)
        return seeds

    def _measure(self, binary, seed_fits) -> Tuple[SideFit, SideFit]:
        if binary is None or binary.size == 0:
            return SideFit(), SideFit()
        if binary.ndim == 3:
            binary = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)
        ys = self._row_ys()
        seeds = self._seeds(binary, seed_fits, ys)
        center = None
        if seeds[0] is not None and seeds[1] is not None:
            center = 0.5 * (seeds[0] + seeds[1])
        elif seeds[0] is not None:
            center = seeds[0] + 0.5 * self._width
        elif seeds[1] is not None:
            center = seeds[1] - 0.5 * self._width
        return (self._measure_side(binary, ys, seeds[0], center, side=0),
                self._measure_side(binary, ys, seeds[1], center, side=1))

    def _measure_side(self, binary, ys, seed, center, side) -> SideFit:
        cfg = self.config
        if seed is None:
            return SideFit()
        w = binary.shape[1]
        lane = cfg.lane_width_px
        band = cfg.band_half_ratio * lane
        accept = cfg.accept_ratio * lane
        max_run = cfg.max_run_ratio * lane
        ignore = 0.5 * cfg.ignore_center_ratio * lane
        tight = (self._models[side] is not None
                 and self._model_age[side] <= cfg.side_hold_frames)
        xs_out, ys_out = [], []
        for y, x_pred, i in zip(ys, seed, range(len(ys))):
            lo = int(max(0, x_pred - band))
            hi = int(min(w, x_pred + band + 1))
            if center is not None:
                # 차선 가운데(마름모, '천천히' 글자) 영역은 보지 않는다
                if side == 0:
                    hi = min(hi, int(center[i] - ignore))
                else:
                    lo = max(lo, int(center[i] + ignore + 1))
            if hi - lo < 3:
                continue
            row = binary[int(y), lo:hi] > 0
            if not row.any():
                continue
            # 연속된 덩어리(run)들 중 예측선에 가장 가까운 것 하나만 쓴다
            padded = np.concatenate(([False], row, [False]))
            edges = np.flatnonzero(padded[1:] != padded[:-1])
            starts, ends = edges[0::2], edges[1::2]
            widths = ends - starts
            good = widths <= max_run
            if not good.any():
                continue
            centers = lo + 0.5 * (starts[good] + ends[good] - 1)
            best = centers[np.argmin(np.abs(centers - x_pred))]
            # 내 직선 모델로 예측했으면(정확) 좁게, 아니면 탐색 폭 전체를 인정한다.
            # 좁게 보면 선 바깥의 흰 경계선, 안쪽 글자가 섞이지 않는다.
            limit = accept if tight else band
            if abs(best - x_pred) <= limit:
                xs_out.append(best)
                ys_out.append(float(y))
        return self._fit_side(np.asarray(xs_out), np.asarray(ys_out), len(ys))

    def _fit_side(self, xs, ys, total_rows) -> SideFit:
        cfg = self.config
        lane = cfg.lane_width_px
        if xs.size < max(6, int(cfg.min_rows_ratio * total_rows)):
            return SideFit()
        # 1) 직선 맞추고, 톱니보다 많이 벗어난 점(글자·잡음)은 빼고 다시 맞춘다
        keep = np.ones(xs.size, bool)
        a, b = np.polyfit(ys, xs, 1)
        for _ in range(2):
            residual = xs - (a * ys + b)
            keep = np.abs(residual) <= max(cfg.accept_ratio * lane, 1.0)
            if keep.sum() < max(6, int(cfg.min_rows_ratio * total_rows)):
                return SideFit()
            a, b = np.polyfit(ys[keep], xs[keep], 1)
        xk, yk = xs[keep], ys[keep]
        # 2) 지그재그 판단: 완만한 곡선 성분(2차)을 빼고 남은 흔들림을 본다
        order = np.argsort(yk)
        xk, yk = xk[order], yk[order]
        trend = np.polyval(np.polyfit(yk, xk, 2), yk)
        residual = xk - trend
        if residual.size >= 3:
            residual = np.convolve(residual, np.ones(3) / 3.0, mode='same')
            residual[0], residual[-1] = residual[1], residual[-2]
        amplitude = 0.5 * float(np.percentile(residual, 90) - np.percentile(residual, 10))
        min_amp = cfg.min_amplitude_ratio * lane
        flips = flips_near = 0
        enough = xk.size >= cfg.detect_rows_ratio * total_rows
        if enough and min_amp <= amplitude <= cfg.max_amplitude_ratio * lane:
            threshold = max(0.5 * min_amp, 0.4 * amplitude)
            at = self._flip_positions(residual, yk, threshold)
            # 꺾이는 간격이 길면(완만한 S자 커브) 지그재그가 아니다
            if len(at) >= 2 and float(np.median(np.diff(at))) <= cfg.max_half_period_ratio * lane:
                flips = len(at)
                near_y = cfg.image_height * (1.0 - cfg.enter_near_ratio)
                flips_near = int(np.sum(np.asarray(at) >= near_y))
            elif len(at) == 1:
                flips = 1
                flips_near = int(at[0] >= cfg.image_height * (1.0 - cfg.enter_near_ratio))
        return SideFit(ok=True, a=float(a), b=float(b),
                       points=np.column_stack((xk, yk)),
                       amplitude=amplitude, flips=flips, flips_near=flips_near)

    @staticmethod
    def _flip_positions(residual: np.ndarray, ys: np.ndarray, threshold: float) -> List[float]:
        """기준선 위/아래를 '확실히'(threshold 이상) 넘나든 위치(y)들. 작은 떨림은 무시."""
        positions: List[float] = []
        sign = 0
        for value, y in zip(residual, ys):
            new_sign = 1 if value > threshold else (-1 if value < -threshold else 0)
            if new_sign != 0 and new_sign != sign:
                if sign != 0:
                    positions.append(float(y))
                sign = new_sign
        return positions

    # ------------------------------------------------------------------ 주행
    def _update_models(self, left: SideFit, right: SideFit) -> None:
        cfg = self.config
        y_eval = cfg.image_height * cfg.lookahead_y_ratio
        max_jump = cfg.max_jump_ratio * cfg.lane_width_px
        for side, fit in enumerate((left, right)):
            self._model_age[side] += 1
            if not fit.ok:
                if self._model_age[side] > cfg.side_hold_frames:
                    self._models[side] = None
                continue
            previous = self._models[side]
            if previous is None:
                self._models[side] = (fit.a, fit.b)
                self._model_age[side] = 0
                continue
            x_prev = previous[0] * y_eval + previous[1]
            x_new = fit.a * y_eval + fit.b
            if abs(x_new - x_prev) > max_jump:
                if self._model_age[side] > cfg.side_hold_frames:
                    self._models[side] = (fit.a, fit.b)   # 오래 못 봤으면 새 값을 믿는다
                    self._model_age[side] = 0
                continue
            k = cfg.model_alpha
            self._models[side] = (k * fit.a + (1 - k) * previous[0],
                                  k * fit.b + (1 - k) * previous[1])
            self._model_age[side] = 0
        if self._models[0] is not None and self._models[1] is not None:
            width = ((self._models[1][0] - self._models[0][0]) * y_eval
                     + self._models[1][1] - self._models[0][1])
            if 0.6 * cfg.lane_width_px <= width <= 1.6 * cfg.lane_width_px:
                self._width = 0.8 * self._width + 0.2 * width

    def _center_model(self) -> Optional[Tuple[float, float]]:
        left, right = self._models
        if left is not None and right is not None:
            return 0.5 * (left[0] + right[0]), 0.5 * (left[1] + right[1])
        if left is not None:
            return left[0], left[1] + 0.5 * self._width
        if right is not None:
            return right[0], right[1] - 0.5 * self._width
        return None

    def _steer_step(self, now_sec: float, out: M6Output) -> None:
        cfg = self.config
        center = self._center_model()
        if center is None:
            target = self._steer * cfg.lost_steer_decay
            out.reason = out.reason or 'holding heading (lines lost)'
        else:
            y_eval = cfg.image_height * cfg.lookahead_y_ratio
            center_x = center[0] * y_eval + center[1]
            half = cfg.image_width * 0.5
            out.lateral_error = float(np.clip((center_x - half) / half, -1.5, 1.5))
            out.heading_error = float(np.arctan(-center[0]))
            error = cfg.lateral_gain * out.lateral_error + cfg.heading_gain * out.heading_error
            target = float(np.clip(-cfg.steer_gain * error, -cfg.max_steer, cfg.max_steer))
        # 진입 직후에는 일반 조향에서 서서히 넘어온다
        if cfg.blend_sec > 0.0:
            k = float(np.clip((now_sec - self._entered_at) / cfg.blend_sec, 0.0, 1.0))
            target = (1.0 - k) * self._blend_from + k * target
        step = float(np.clip(target - self._steer, -cfg.max_steer_rate, cfg.max_steer_rate))
        self._steer += step
        out.steering = self._steer

    # ------------------------------------------------------------------ 디버그
    def draw(self, image: np.ndarray, out: Optional[M6Output] = None) -> np.ndarray:
        """BEV 디버그 영상 위에 M6 정보를 그린다 (복사본 반환)."""
        cfg = self.config
        out = out or self.last_output
        canvas = image.copy()
        if canvas.ndim == 2:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        colors = ((255, 255, 0), (255, 0, 255))
        for side, fit in enumerate((out.left, out.right)):
            if fit.points is not None:
                for x, y in fit.points.astype(int):
                    cv2.circle(canvas, (int(x), int(y)), 2, colors[side], -1)
        ys = np.array([0, cfg.image_height - 1], dtype=np.float32)
        for model in (out.left_model, out.right_model):
            if model is not None:
                xs = model[0] * ys + model[1]
                cv2.line(canvas, (int(xs[0]), 0), (int(xs[1]), int(ys[1])), (0, 255, 0), 2)
        color = (0, 0, 255) if out.active else ((0, 200, 255) if out.state == ARMED else (160, 160, 160))
        text = (f'M6 {out.state} zz={int(out.zigzag)} '
                f'L a={out.left.amplitude:.0f} f={out.left.flips_near}/{out.left.flips} '
                f'R a={out.right.amplitude:.0f} f={out.right.flips_near}/{out.right.flips}')
        cv2.putText(canvas, text, (10, cfg.image_height - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        if out.active or out.reason:
            cv2.putText(canvas, f'steer={out.steering:+.2f} {out.reason}', (10, cfg.image_height - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return canvas
