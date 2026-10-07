"""Isolated validation and fork-branch publication for three contribution fixes."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import tomllib

BASE = '0aa579a6c77ef61e230512c7116e87b5b1186d8f'
ROOT = Path.cwd()
SOURCE = 'src/fmdtools/sim/sample.py'
HEADER = '''#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Regression tests for DESCRIPTION.

Copyright © 2024, United States Government, as represented by the Administrator
of the National Aeronautics and Space Administration. All rights reserved.

The "Fault Model Design tools - fmdtools version 2" software is licensed
under the Apache License, Version 2.0 (the "License"); you may not use this
file except in compliance with the License. You may obtain a copy of the
License at http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""
'''
MAPPING_TEST = r'''
import copy
import unittest

import numpy as np

from fmdtools.define.block.function import Function
from fmdtools.define.container.parameter import Parameter
from fmdtools.define.container.state import State
from fmdtools.sim import propagate
from fmdtools.sim.sample import ParameterDomain, ParameterSample


class InputParameter(Parameter, readonly=True):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


class Accumulation(State):
    total: float = 0.0


class MappedFunction(Function):
    container_p = InputParameter
    container_s = Accumulation

    def dynamic_behavior(self):
        self.s.total += self.p.x + 10*self.p.y + 100*self.p.z

    def classify(self, **kwargs):
        return {'total': self.s.total}


def mapped_domain(factory=dict):
    domain = ParameterDomain(factory)
    domain.add_variables('x', 'y', 'z', var_map=False)
    domain.add_variables('z', 'x', var_map=lambda z, x: (z + 30, x + 10))
    domain.add_variable('y', var_map=lambda y: (y + 20,))
    return domain


class TestNamedParameterMaps(unittest.TestCase):
    def test_independent_groups_follow_names_not_registration_order(self):
        domain = mapped_domain()
        self.assertEqual(domain.get_map_vars(1, 2, 3), [11, 22, 33])
        self.assertEqual(domain.get_param_kwargs(1, 2, 3), {'x': 11, 'y': 22, 'z': 33})
        self.assertEqual(domain(1, 2, 3), {'x': 11, 'y': 22, 'z': 33})
        maps = domain.var_maps.copy()
        domain.var_maps = dict(reversed(list(maps.items())))
        self.assertEqual(domain.get_map_vars(1, 2, 3), [11, 22, 33])
        self.assertEqual(domain.variables, {'x': (), 'y': (), 'z': ()})

    def test_replacing_a_group_does_not_shift_or_duplicate_variables(self):
        domain = ParameterDomain(dict)
        domain.add_variables('x', 'y')
        domain.add_variable('x', var_map=lambda x: (2*x,))
        self.assertEqual(domain.get_map_vars(3, 4), [6, 4])
        self.assertEqual(domain(3, 4), {'x': 6, 'y': 4})
        domain.add_variables('y', 'x', var_map=lambda y, x: (y+10, x+20))
        self.assertEqual(domain.get_map_vars(3, 4), [23, 14])

    def test_passthrough_partial_inputs_and_constants_keep_order(self):
        domain = ParameterDomain(dict)
        domain.add_variables('x', 'y', 'z')
        domain.add_constant('fixed', 99)
        for values in [(), (1,), (1, 2), (1, 2, 3)]:
            with self.subTest(values=values):
                self.assertEqual(domain.get_map_vars(*values), list(values))
        self.assertEqual(domain.get_param_kwargs(1, 2, 3), {'fixed': 99, 'x': 1, 'y': 2, 'z': 3})
        domain = ParameterDomain(dict)
        domain.add_variable('x', var_map=False)
        domain.add_variable('y', var_map=lambda y: (2*y,))
        self.assertEqual(domain(3, 4), {'x': 3, 'y': 8})

    def test_mapping_arity_is_checked_without_mutating_domains(self):
        for mapper in [lambda x, y: (x,), lambda x, y: (x, y, 99)]:
            domain = ParameterDomain(dict)
            domain.add_variables('x', 'y', var_map=mapper)
            before = copy.deepcopy(domain.variables)
            with self.assertRaisesRegex(ValueError, 'mapped value'):
                domain.get_map_vars(1, 2)
            self.assertEqual(domain.variables, before)
        domain = ParameterDomain(dict)
        domain.add_variables('x', 'y', var_map=lambda x, y: (item for item in (y, x)))
        self.assertEqual(domain.get_map_vars(1, 2), [2, 1])

    def test_sample_simulations_receive_the_intended_transformed_parameters(self):
        domain = mapped_domain(InputParameter)
        sample = ParameterSample(domain, seed=11)
        sample.add_variable_replicates([(1, 2, 3), (4, 5, 6)], replicates=2, weight=.8)
        original = copy.deepcopy([s.asdict() for s in sample.scenarios()])
        model = MappedFunction(sp={'end_time': 3.})
        result, history = propagate.parameter_sample(model, sample, showprogress=False)
        for scenario, values in zip(sample.scenarios(), [(11, 22, 33)]*2 + [(14, 25, 36)]*2):
            self.assertEqual(tuple(scenario.p[k] for k in ('x','y','z')), values)
            step = values[0] + 10*values[1] + 100*values[2]
            self.assertEqual(result[scenario.name+'.tend.classify.total'], 3*step)
            np.testing.assert_allclose(history[scenario.name+'.s.total'], np.arange(4)*step)
        self.assertAlmostEqual(sum(s.prob for s in sample.scenarios()), .8)
        self.assertEqual([s.asdict() for s in sample.scenarios()], original)
        self.assertEqual(model.s.total, 0.)


if __name__ == '__main__':
    unittest.main()
'''
LIMIT_TEST = r'''
import copy
import unittest

import numpy as np

from fmdtools.define.block.function import ExampleFunction
from fmdtools.sim import propagate
from fmdtools.sim.sample import FaultDomain, FaultSample


def domain(limit, seed=7):
    result = FaultDomain(ExampleFunction(sp={'end_time': 3.}))
    result.add_fault_space('examplefunction', 'low', {'s.x': {1., 3., 5.}, 's.y': {2., 4.}}, n=limit, seed=seed)
    return result


def definitions(value):
    return {key: fault.asdict() for key, fault in value.faults.items()}


class TestFaultSpaceSampleLimits(unittest.TestCase):
    def test_numpy_limits_match_builtin_integers_and_never_exceed_them(self):
        for seed in (0, 7, 19):
            for count in (1, 2, 5, 12, 20):
                expected = domain(count, seed)
                for dtype in (np.int32, np.int64, np.uint64):
                    with self.subTest(seed=seed, count=count, dtype=dtype):
                        actual = domain(dtype(count), seed)
                        self.assertEqual(definitions(actual), definitions(expected))
                        self.assertLessEqual(len(actual.faults), count)

    def test_zero_is_a_noop_for_existing_domains_and_supplied_ranges(self):
        for zero in (0, np.int64(0), np.uint64(0)):
            with self.subTest(zero=zero):
                target = domain('all')
                before = definitions(target)
                ranges = {'s.x': {1., 2.}, 's.y': (0., 2., 3)}
                original = copy.deepcopy(ranges)
                target.add_fault_space('examplefunction', 'low', ranges, n=zero)
                self.assertEqual(definitions(target), before)
                self.assertEqual(ranges, original)
                self.assertEqual(domain(zero).faults, {})

    def test_invalid_limits_fail_before_changing_the_domain_or_ranges(self):
        for limit in (-1, np.int64(-1), 1.5, 2., True, np.bool_(True), None, '2', 'ALL', [2], np.array([2])):
            with self.subTest(limit=repr(limit)):
                target = domain(2)
                before = definitions(target)
                ranges = {'s.x': {1., 2.}}
                original = copy.deepcopy(ranges)
                with self.assertRaisesRegex(ValueError, 'non-negative integer'):
                    target.add_fault_space('examplefunction', 'low', ranges, n=limit)
                self.assertEqual(definitions(target), before)
                self.assertEqual(ranges, original)

    def test_all_and_large_limits_keep_full_space_and_explicit_probabilities(self):
        self.assertEqual(definitions(domain('all')), definitions(domain(100)))
        target = FaultDomain(ExampleFunction())
        target.add_fault_space('examplefunction', 'low', {'s.x': {1., 2.}}, n=np.int64(10), prob=.25)
        self.assertEqual(len(target.faults), 2)
        self.assertTrue(all(fault.prob == .25 for fault in target.faults.values()))

    def test_real_fault_samples_match_the_equivalent_builtin_limit(self):
        outputs = []
        for count in (3, np.int64(3)):
            selected = domain(count)
            sample = FaultSample(selected, def_mdl_phasemap=False)
            sample.add_fault_times([1.])
            result, history = propagate.fault_sample(selected.mdl, sample, showprogress=False)
            outputs.append((sample, result, history))
        left, right = outputs
        self.assertEqual([s.asdict() for s in left[0].scenarios()], [s.asdict() for s in right[0].scenarios()])
        self.assertEqual(set(left[1]), set(right[1]))
        self.assertEqual(set(left[2]), set(right[2]))
        for index in (1, 2):
            for key in left[index]:
                np.testing.assert_array_equal(left[index][key], right[index][key])


if __name__ == '__main__':
    unittest.main()
'''
INDEX_TEST = r'''
import copy
import unittest

import numpy as np

from fmdtools.analyze.history import History
from fmdtools.analyze.result import Result
from fmdtools.define.block.function import Function
from fmdtools.define.container.parameter import Parameter
from fmdtools.define.container.state import State
from fmdtools.sim import propagate
from fmdtools.sim.sample import ParameterDomain, ParameterHistSample, ParameterResultSample


class InputParameter(Parameter, readonly=True):
    x: float = 1.0


class TotalState(State):
    total: float = 0.0


class HistoryFunction(Function):
    container_p = InputParameter
    container_s = TotalState

    def dynamic_behavior(self):
        self.s.total += self.p.x

    def classify(self, **kwargs):
        return {'total': self.s.total}


def history_sample(factory=dict, arrays=True):
    values = np.array([2., 5., 9.]) if arrays else [2., 5., 9.]
    history = History({'case.signal': values, 'case.time': np.array([0., 2., 4.])})
    params = ParameterDomain(factory)
    params.add_variable('x')
    return ParameterHistSample(history, 'signal', paramdomain=params, seed=7), history


class TestHistorySampleIndices(unittest.TestCase):
    def test_invalid_index_types_raise_instead_of_becoming_parameter_values(self):
        for arrays in (False, True):
            for index in ('bad', [], {}, object(), True, np.bool_(False), 1., np.float64(1.), np.nan):
                with self.subTest(arrays=arrays, index=repr(index)):
                    sample, history = history_sample(arrays=arrays)
                    before = copy.deepcopy(history)
                    with self.assertRaisesRegex(TypeError, 'integer or None'):
                        sample.get_param_ins(rep='case', t=index)
                    with self.assertRaisesRegex(TypeError, 'integer or None'):
                        sample.add_hist_scenario(rep='case', t=index)
                    self.assertEqual(sample.scenarios(), [])
                    for name in history:
                        np.testing.assert_array_equal(history[name], before[name])

    def test_integer_indices_include_zero_negative_and_numpy_values(self):
        for arrays in (False, True):
            for index, expected in ((0, 2.), (1, 5.), (-1, 9.), (np.int64(2), 9.), (np.int32(0), 2.)):
                with self.subTest(arrays=arrays, index=index):
                    sample, _ = history_sample(arrays=arrays)
                    self.assertEqual(sample.get_param_ins(rep='case', t=index), [expected])
                    sample.add_hist_scenario(rep='case', t=index)
                    scenario = sample.scenarios()[0]
                    self.assertEqual(scenario.p, {'x': expected})
                    self.assertEqual(scenario.inputparams['t'], index)
                    self.assertEqual(scenario.r, {'seed': 7})

    def test_none_preserves_whole_values_and_result_only_sampling(self):
        sample, _ = history_sample()
        np.testing.assert_array_equal(sample.get_param_ins(rep='case', t=None)[0], [2., 5., 9.])
        domain = ParameterDomain(dict)
        domain.add_variable('x')
        result = ParameterResultSample(Result({'case.signal': 7.}), 'signal', paramdomain=domain)
        self.assertEqual(result.get_param_ins(rep='case'), [7.])
        result.add_res_scenario(rep='case')
        self.assertEqual(result.scenarios()[0].p, {'x': 7.})

    def test_out_of_range_indices_still_raise_without_adding_a_scenario(self):
        for index in (-4, 3):
            sample, _ = history_sample()
            with self.assertRaises(IndexError):
                sample.add_hist_scenario(rep='case', t=index)
            self.assertEqual(sample.scenarios(), [])

    def test_valid_history_parameters_drive_real_simulation_outputs(self):
        sample, history = history_sample(InputParameter)
        sample.add_hist_times('default', 'case', ts=[0, 2])
        model = HistoryFunction(sp={'end_time': 3.})
        result, histories = propagate.parameter_sample(model, sample, showprogress=False)
        for scenario, value in zip(sample.scenarios(), (2., 9.)):
            self.assertEqual(result[scenario.name+'.tend.classify.total'], 3*value)
            np.testing.assert_allclose(histories[scenario.name+'.s.total'], np.arange(4)*value)
        np.testing.assert_array_equal(history['case.signal'], [2., 5., 9.])
        self.assertEqual(model.s.total, 0.)


if __name__ == '__main__':
    unittest.main()
'''
SPECS = [
    dict(key='named-maps', branch='fix/named-parameter-maps-20261007-ah', title='Align parameter mappings with their named variables', test='tests/test_named_parameter_maps.py', code=MAPPING_TEST),
    dict(key='fault-limits', branch='fix/fault-space-count-validation-20261007-ah', title='Honor zero and NumPy integer fault-space limits', test='tests/test_fault_space_sample_limits.py', code=LIMIT_TEST),
    dict(key='history-indices', branch='fix/reject-invalid-history-indices-20261007-ah', title='Reject invalid history indices before creating parameter scenarios', test='tests/test_history_sample_indices.py', code=INDEX_TEST),
]


def run(args, cwd=ROOT, env=None, check=True):
    process = subprocess.run(args, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(process.stdout, flush=True)
    if check and process.returncode:
        raise RuntimeError(f'Command failed with {process.returncode}: {args}')
    return process


def patch_source(key, text):
    if key == 'named-maps':
        old = '''        """Get the mapped variables for x."""
        x_mapped = []
        i = 0
        for var_group in self.var_maps:
            x_map = self.var_maps[var_group](*x[i:i+len(var_group)])
            x_mapped.extend(x_map)
            i += len(var_group)
        return x_mapped'''
        new = '''        """Map inputs by variable name and return them in domain order.

        Groups receive the original values of their named inputs. Later groups
        may replace earlier mappings of the same variable. Unmapped supplied
        values pass through unchanged; each map returns one value per input.
        """
        inputs = dict(zip(self.variables, x))
        mapped = dict(inputs)
        for var_group, mapper in self.var_maps.items():
            names = [name for name in var_group if name in inputs]
            values = list(mapper(*(inputs[name] for name in names)))
            if len(values) != len(names):
                raise ValueError("Each parameter map must return one mapped value "
                                 "per supplied variable.")
            mapped.update(zip(names, values))
        return [mapped[name] for name in inputs]'''
    elif key == 'fault-limits':
        old = '''        # determine overall state combinations to sample from
        for state, vals in dist_ranges.items():'''
        new = '''        if not (isinstance(n, str) and n == 'all'):
            if (isinstance(n, (bool, np.bool_))
                    or not isinstance(n, (int, np.integer)) or n < 0):
                raise ValueError("n must be 'all' or a non-negative integer.")
            n = int(n)
            if n == 0:
                return

        # determine overall state combinations to sample from
        for state, vals in dist_ranges.items():'''
    else:
        old = '''        elif is_numeric(t):
            return self.res_to_sample[comp_group].get(rep).get(var)[t]
        else:
            return Exception("Invalid option for t: "+str(t))'''
        new = '''        elif (isinstance(t, (int, np.integer))
              and not isinstance(t, (bool, np.bool_))):
            return self.res_to_sample[comp_group].get(rep).get(var)[t]
        else:
            raise TypeError("History index t must be an integer or None.")'''
    assert text.count(old) == 1, (key, 'source mismatch')
    return text.replace(old, new, 1)


def main():
    assert os.environ.get('GITHUB_REPOSITORY') == 'sylvesterkaczmarek/fmdtools'
    assert os.environ.get('GITHUB_REF_NAME') == 'validation/fmdtools-risk-20261007-ah'
    deps = tomllib.loads((ROOT/'pyproject.toml').read_text())['project']['dependencies']
    run([sys.executable, '-m', 'pip', 'install', *deps, 'pytest-subtests', 'ruff'])
    run(['git', 'fetch', '--depth=1', 'origin', BASE])
    workspace = Path(tempfile.mkdtemp(prefix='fmdtools-ah-'))
    launcher = workspace/'run_pytest.py'
    launcher.write_text('''import importlib.util
from pathlib import Path
import sys
if __name__ == '__main__':
    root = Path(sys.argv[1])
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root/'src'))
    spec = importlib.util.spec_from_file_location('fmdtools_examples', root/'examples/__init__.py', submodule_search_locations=[str(root/'examples')])
    module = importlib.util.module_from_spec(spec)
    sys.modules['fmdtools_examples'] = module
    spec.loader.exec_module(module)
    import pytest
    raise SystemExit(pytest.main(sys.argv[2:]))
''')
    env = {**os.environ, 'MPLBACKEND': 'Agg'}
    manifest = {}
    for spec in SPECS:
        candidate = workspace/spec['key']
        run(['git', 'worktree', 'add', '--no-checkout', '--detach', str(candidate), BASE])
        run(['git', 'sparse-checkout', 'init', '--cone'], cwd=candidate)
        run(['git', 'sparse-checkout', 'set', 'src', 'tests', 'examples/pump', 'examples/tank', 'examples/multirotor_drone'], cwd=candidate)
        run(['git', 'checkout', '--detach', BASE], cwd=candidate)
        test = candidate/spec['test']
        test.write_text(HEADER.replace('DESCRIPTION', spec['title'].lower()) + spec['code'])
        run([sys.executable, '-m', 'ruff', 'check', '--isolated', '--select', 'E9,F63,F7,F82,F401,I', '--fix', spec['test']], cwd=candidate)
        run([sys.executable, '-m', 'ruff', 'format', '--isolated', spec['test']], cwd=candidate)
        command = [sys.executable, str(launcher), str(candidate), '-o', 'addopts=', '--noconftest', '-q', spec['test']]
        baseline = run(command, cwd=candidate, env=env, check=False)
        assert baseline.returncode == 1 and 'FAILED' in baseline.stdout and 'ERROR collecting' not in baseline.stdout, spec['key']
        source = candidate/SOURCE
        source.write_text(patch_source(spec['key'], source.read_text()))
        regression = run(command, cwd=candidate, env=env)
        related = ['tests/test_sample.py', 'tests/test_parameter_sample_defaults.py', 'tests/test_parameter_sample_seeds.py', 'tests/test_callable_parameter_domains.py', 'tests/test_discrete_grid_values.py', 'tests/test_comparison_groups.py', 'tests/test_nested_comparison_inputs.py']
        related = [name for name in related if (candidate/name).exists()]
        suite = run(command + related, cwd=candidate, env=env)
        run([sys.executable, '-m', 'ruff', 'check', '--isolated', '--select', 'E9,F63,F7,F82', SOURCE], cwd=candidate)
        run(['git', 'diff', '--check'], cwd=candidate)
        changed = set(run(['git', 'diff', '--name-only'], cwd=candidate).stdout.splitlines()) | set(run(['git', 'ls-files', '--others', '--exclude-standard'], cwd=candidate).stdout.splitlines())
        assert changed == {SOURCE, spec['test']}, changed
        manifest[spec['key']] = {'branch': spec['branch'], 'base': BASE, 'source': str(candidate), 'title': spec['title'], 'test': spec['test'], 'baseline': baseline.stdout, 'regression': regression.stdout, 'related': suite.stdout, 'hashes': {name: hashlib.sha256((candidate/name).read_bytes()).hexdigest() for name in changed}}
    # Publish only after all three candidates have passed their checks.
    for spec in SPECS:
        candidate = Path(manifest[spec['key']]['source'])
        run(['git', 'add', '--', SOURCE, spec['test']], cwd=candidate)
        run(['git', '-c', 'user.name=Sylvester Kaczmarek', '-c', 'user.email=16242628+sylvesterkaczmarek@users.noreply.github.com', 'commit', '-s', '-m', spec['title']], cwd=candidate)
        remote = run(['git', 'ls-remote', 'origin', 'refs/heads/'+spec['branch']], cwd=candidate)
        assert not remote.stdout.strip(), 'A prepared branch already exists; inspect it before retrying.'
        run(['git', 'push', 'origin', 'HEAD:refs/heads/'+spec['branch']], cwd=candidate)
        manifest[spec['key']]['commit'] = run(['git', 'rev-parse', 'HEAD'], cwd=candidate).stdout.strip()
    print('VALIDATED_BRANCHES_JSON=' + json.dumps(manifest), flush=True)
    summary = os.environ.get('GITHUB_STEP_SUMMARY')
    if summary:
        with open(summary, 'a') as target:
            target.write('# Validated contributions\n\n')
            for entry in manifest.values():
                target.write(f"- {entry['title']} `{entry['commit']}`\n")


if __name__ == '__main__':
    main()
