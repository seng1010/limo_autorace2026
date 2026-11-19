"""매니저가 `ros2 run` 으로 띄운 미션 노드를 실제로 끄기 (ROS 없음).

문제 (11/19 경기장 로그 claude_1140, claude_192952)
  매니저는 subprocess.Popen(['ros2', 'run', ...]) 로 노드를 띄운다. 이 Popen 은 `ros2 run`
  래퍼이고 실제 노드는 그 자식 프로세스다. 래퍼에만 SIGINT 를 보내면 래퍼는 "자식도 같은
  신호를 받았겠지"(터미널 Ctrl+C 기준) 하고 자식을 기다리기만 하고, 2초 뒤 래퍼만 terminate
  되면 실제 노드는 부모 없이 계속 돈다. M8 강제 탈출 뒤에도 콘 노드 로그가 끝까지 나온 이유.

해결
  끌 때 그 순간의 자식·손자 PID 를 /proc 에서 찾아 같이 신호를 보낸다.
  - 프로세스 그룹/세션은 바꾸지 않는다 → 터미널 Ctrl+C 가 예전처럼 전부에 간다
  - 언제 끄는지는 매니저가 정한다 (지금과 같음: 그 미션을 떠날 때, 매니저 종료 때)
  - 2초 안에 안 꺼지면 SIGTERM, 5초면 SIGKILL
"""

import os
import signal
from typing import Dict, Iterable, List, Optional


def _read_stat(pid: int, proc: str = '/proc'):
    try:
        with open(os.path.join(proc, str(pid), 'stat')) as handle:
            text = handle.read()
    except OSError:
        return None
    # 'pid (comm) state ppid ...'  comm 에 공백·괄호가 있을 수 있어 마지막 ')' 기준으로 자른다
    rest = text.rsplit(')', 1)[-1].split()
    if len(rest) < 2:
        return None
    try:
        return rest[0], int(rest[1])
    except ValueError:
        return None


def descendants(pid: int, proc: str = '/proc') -> List[int]:
    """pid 의 자식·손자… (pid 자신은 제외). 살아 있는 프로세스만."""
    children: Dict[int, List[int]] = {}
    try:
        entries = os.listdir(proc)
    except OSError:
        return []
    for entry in entries:
        if not entry.isdigit():
            continue
        stat = _read_stat(int(entry), proc)
        if stat is None or stat[0] == 'Z':
            continue
        children.setdefault(stat[1], []).append(int(entry))
    found, queue = [], list(children.get(pid, []))
    while queue:
        child = queue.pop(0)
        if child in found:
            continue
        found.append(child)
        queue.extend(children.get(child, []))
    return found


def alive(pid: int, proc: str = '/proc') -> bool:
    stat = _read_stat(pid, proc)
    return stat is not None and stat[0] != 'Z'


def send(pids: Iterable[int], sig) -> None:
    for pid in pids:
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError):
            pass


class StoppingProcess:
    """끄는 중인 `ros2 run` 래퍼 + 끄기 시작할 때 찾은 자식들."""

    def __init__(self, popen, now_sec: float, term_after_sec: float = 2.0,
                 kill_after_sec: float = 5.0):
        self.popen = popen
        self.since = float(now_sec)
        self.term_after_sec = float(term_after_sec)
        self.kill_after_sec = float(kill_after_sec)
        # 래퍼가 먼저 죽으면 자식은 init 으로 넘어가 다시 못 찾는다 → 지금 기억해 둔다
        self.children = descendants(popen.pid) if popen.poll() is None else []
        self._termed = False
        send(self.children, signal.SIGINT)        # 실제 노드: Ctrl+C 와 같은 정상 종료
        if popen.poll() is None:
            popen.send_signal(signal.SIGINT)

    def done(self) -> bool:
        return self.popen.poll() is not None and not any(alive(pid) for pid in self.children)

    def step(self, now_sec: float) -> bool:
        """주기마다 호출. 다 꺼졌으면 True."""
        if self.done():
            return True
        waited = now_sec - self.since
        if waited >= self.kill_after_sec:
            send([pid for pid in self.children if alive(pid)], signal.SIGKILL)
            if self.popen.poll() is None:
                self.popen.kill()
            return True
        if waited >= self.term_after_sec and not self._termed:
            self._termed = True
            send([pid for pid in self.children if alive(pid)], signal.SIGTERM)
            if self.popen.poll() is None:
                self.popen.terminate()
        return False


def other_instances(name: str, exclude: Iterable[int] = (), proc: str = '/proc') -> List[int]:
    """이름이 같은 노드가 이미 떠 있는지 (이전 주행에서 남은 고아 확인용, 경고만)."""
    skip = set(exclude) | {os.getpid()}
    found = []
    try:
        entries = os.listdir(proc)
    except OSError:
        return []
    for entry in entries:
        if not entry.isdigit() or int(entry) in skip:
            continue
        try:
            with open(os.path.join(proc, entry, 'cmdline'), 'rb') as handle:
                argv = handle.read().split(b'\0')
        except OSError:
            continue
        # 실제 노드: [python3, .../lib/jetracer_ros2/<name>, ...]. ros2 run 래퍼는 [python3, .../ros2, run, ...]
        if any(os.path.basename(arg.decode(errors='ignore')) == name for arg in argv[:2]) \
                and alive(int(entry), proc):
            found.append(int(entry))
    return found


def wait_all(stopping: List[StoppingProcess], clock, timeout_sec: float = 5.0) -> None:
    """매니저 종료 때: 전부 꺼질 때까지 (최대 timeout) 기다린다."""
    import time
    deadline = clock() + timeout_sec
    pending: List[Optional[StoppingProcess]] = list(stopping)
    while pending and clock() < deadline:
        pending = [item for item in pending if not item.step(clock())]
        if pending:
            time.sleep(0.05)
    for item in pending:
        item.step(item.since + item.kill_after_sec)
