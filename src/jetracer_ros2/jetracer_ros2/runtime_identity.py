"""Best-effort runtime path and source-checkout identity for field recordings.

The checkout HEAD is deliberately not labelled a build revision: colcon may
still be running an older installed copy until the workspace is rebuilt.
"""

from pathlib import Path
import subprocess


def find_source_checkout(module_path):
    module = Path(module_path).resolve()
    for parent in module.parents:
        if (parent / '.git').exists() and (parent / 'src/jetracer_ros2/package.xml').is_file():
            return parent
    for parent in module.parents:
        if parent.name not in ('build', 'install'):
            continue
        source_dir = parent.parent / 'src'
        if not source_dir.is_dir():
            continue
        candidates = sorted(
            child for child in source_dir.iterdir()
            if child.is_dir() and (child / '.git').exists()
            and (child / 'src/jetracer_ros2/package.xml').is_file())
        if len(candidates) == 1:
            return candidates[0]
    return None


def runtime_identity(module_path):
    """Return installed module path and checkout HEAD without conflating them."""
    module = Path(module_path).resolve()
    checkout = find_source_checkout(module)
    if checkout is None:
        return str(module), 'unknown (no matching source checkout)'
    try:
        result = subprocess.run(
            ['git', '-C', str(checkout), 'rev-parse', '--short=12', 'HEAD'],
            check=True, capture_output=True, text=True, timeout=2)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return str(module), f'unknown ({checkout})'
    return str(module), f'{result.stdout.strip()} ({checkout}; rebuild not verified)'
