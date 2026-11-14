"""Field recordings must identify both the loaded module and source checkout."""

from pathlib import Path

from jetracer_ros2.runtime_identity import find_source_checkout, runtime_identity


def test_source_checkout_is_found_from_colcon_build_copy(tmp_path):
    checkout = tmp_path / '2026_ws' / 'src' / 'limo_autorace2026'
    (checkout / '.git').mkdir(parents=True)
    (checkout / 'src/jetracer_ros2').mkdir(parents=True)
    (checkout / 'src/jetracer_ros2/package.xml').touch()
    module = tmp_path / '2026_ws' / 'build' / 'jetracer_ros2' / 'jetracer_ros2' / 'lane_following_node.py'
    module.parent.mkdir(parents=True)
    module.touch()
    assert find_source_checkout(module) == checkout.resolve()


def test_unmatched_module_does_not_claim_a_build_revision(tmp_path):
    module = tmp_path / 'unknown.py'
    module.touch()
    loaded, checkout = runtime_identity(module)
    assert loaded == str(module.resolve())
    assert checkout.startswith('unknown')
