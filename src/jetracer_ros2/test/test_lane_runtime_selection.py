"""Resolve the actual launch choices without ROS; never starts a process."""
import importlib.util
from pathlib import Path
import sys
import types

import pytest


@pytest.mark.parametrize('lane3', ['simple', 'full', 'gap'])
@pytest.mark.parametrize('team', ['false', 'true'])
@pytest.mark.parametrize('enabled', ['false', 'true'])
def test_exactly_one_selected_lane_runtime(monkeypatch, lane3, team, enabled):
    args = dict(lane3=lane3, team_lane=team, lane=enabled)
    fields = {
        'launch': dict(LaunchDescription=lambda actions: actions),
        'launch.actions': dict(DeclareLaunchArgument=lambda *a, **kw: None),
        'launch.conditions': dict(IfCondition=lambda expression: bool(expression)),
        'launch.substitutions': dict(LaunchConfiguration=lambda name: args.get(name, 'M5'),
            PythonExpression=lambda parts: eval(''.join(parts), {'__builtins__': {}})),
        'launch_ros.actions': dict(Node=lambda **kw: kw),
    }
    for name, attrs in fields.items():
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, mod)
    path = Path(__file__).parents[1]/'launch/all_in_node.launch.py'
    spec = importlib.util.spec_from_file_location('_test_lane_runtime_launch', path)
    launch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launch)
    lane_names = {'simple_lane', 'lane3_gap', 'lane_following_node'}
    chosen = [n for n in launch.generate_launch_description() if isinstance(n, dict)
              and n.get('executable') in lane_names and n.get('condition', True)]
    if enabled == 'false':
        assert chosen == []
        return
    assert len(chosen) == 1
    expected = ('lane_following_node' if team == 'true' else
                'lane3_gap' if lane3 == 'gap' else 'simple_lane')
    assert chosen[0]['executable'] == expected
    if expected == 'simple_lane':
        assert chosen[0]['arguments'] == ['drive', '--topic', 'inference/cmd_vel', ['l3', lane3]]
    if expected == 'lane3_gap':
        assert chosen[0]['arguments'] == ['drive', '--topic', 'inference/cmd_vel']


def test_default_lane3_choice_matches_first_full_run():
    # The runtime test above covers all combinations; also pin the declared default.
    import ast
    tree = ast.parse((Path(__file__).parents[1]/'launch/all_in_node.launch.py').read_text())
    declaration = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                       and isinstance(n.func, ast.Name) and n.func.id == 'DeclareLaunchArgument'
                       and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == 'lane3')
    values = {k.arg: ast.literal_eval(k.value) for k in declaration.keywords}
    assert values['default_value'] == 'simple'
    assert values['choices'] == ['simple', 'full', 'gap']
