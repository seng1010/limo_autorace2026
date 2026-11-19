"""매니저가 띄운 `ros2 run` 래퍼 + 실제 노드(자식)를 같이 끄는지, 실제 프로세스로 확인.

가짜 래퍼는 ros2run 과 같은 식으로 동작한다: 자식을 띄우고, SIGINT 를 받으면
"자식도 같은 신호를 받았겠지" 하고 계속 기다린다. 자식은 SIGINT 를 받으면 끝난다 (rclpy 노드처럼).
"""

import signal
import subprocess
import sys
import textwrap
import time

import pytest

from jetracer_ros2 import process_tree

WRAPPER = textwrap.dedent('''
    import subprocess, sys
    child = subprocess.Popen([sys.executable, '-c',
        'import time\\nwhile True: time.sleep(0.05)'])
    print(child.pid, flush=True)
    while child.poll() is None:
        try:
            child.wait()
        except KeyboardInterrupt:
            pass        # ros2run: "the subprocess will also receive the signal"
''')


def _spawn():
    wrapper = subprocess.Popen([sys.executable, '-c', WRAPPER], stdout=subprocess.PIPE, text=True)
    child = int(wrapper.stdout.readline())
    return wrapper, child


def _wait(predicate, timeout=6.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


@pytest.fixture
def spawned():
    wrapper, child = _spawn()
    yield wrapper, child
    for pid in (child, wrapper.pid):
        try:
            import os
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    wrapper.wait(timeout=2)


def test_old_way_leaves_real_node_running(spawned):
    """예전 _stop_process: 래퍼에만 SIGINT, 2초 뒤 래퍼 terminate -> 노드는 고아로 남음 (11/19 로그와 같음)."""
    wrapper, child = spawned
    assert process_tree.descendants(wrapper.pid) == [child]
    wrapper.send_signal(signal.SIGINT)
    time.sleep(0.5)
    assert wrapper.poll() is None and process_tree.alive(child)
    wrapper.terminate()
    wrapper.wait(timeout=2)
    time.sleep(0.3)
    assert process_tree.alive(child)            # 버그 재현


def test_stopping_process_stops_wrapper_and_node(spawned):
    wrapper, child = spawned
    stopping = process_tree.StoppingProcess(wrapper, time.monotonic())
    assert stopping.children == [child]
    assert _wait(lambda: stopping.step(time.monotonic()))
    assert not process_tree.alive(child) and wrapper.poll() is not None


def test_stopping_process_escalates_when_node_ignores_sigint():
    stubborn = textwrap.dedent('''
        import signal, time
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        while True: time.sleep(0.05)
    ''')
    wrapper_code = WRAPPER.replace("'import time\\nwhile True: time.sleep(0.05)'", repr(stubborn))
    wrapper = subprocess.Popen([sys.executable, '-c', wrapper_code], stdout=subprocess.PIPE, text=True)
    child = int(wrapper.stdout.readline())
    time.sleep(0.2)
    stopping = process_tree.StoppingProcess(wrapper, time.monotonic(), term_after_sec=0.3,
                                            kill_after_sec=1.0)
    assert _wait(lambda: stopping.step(time.monotonic()), timeout=3)
    assert _wait(lambda: not process_tree.alive(child), timeout=2)


def test_wait_all_for_manager_shutdown(spawned):
    wrapper, child = spawned
    process_tree.wait_all([process_tree.StoppingProcess(wrapper, time.monotonic())],
                          time.monotonic, timeout_sec=5)
    assert not process_tree.alive(child) and wrapper.poll() is not None


def test_other_instances_finds_node_by_executable_name(tmp_path):
    exe = tmp_path / 'mission_cone_avoid_node'
    exe.write_text('import time\nwhile True: time.sleep(0.05)\n')
    proc = subprocess.Popen([sys.executable, str(exe)])
    try:
        assert _wait(lambda: proc.pid in process_tree.other_instances('mission_cone_avoid_node'), 3)
        assert proc.pid not in process_tree.other_instances('mission_traffic_light_node')
    finally:
        proc.kill()
        proc.wait()
