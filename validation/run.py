"""Validate two isolated contributions, then push their tested branches."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

BASE = '51e3de7bdeb0cdee2f0e8c56ffffd62aad702272'
ROOT = Path(os.environ['GITHUB_WORKSPACE']).resolve()
LOGS = ROOT / 'validation-results'
LOGS.mkdir(exist_ok=True)
PYTHON = sys.executable


def command(args, cwd, label, env=None, allow_failure=False):
    result = subprocess.run(args, cwd=cwd, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=300)
    (LOGS / f'{label}.log').write_text(result.stdout)
    print(f'\n=== {label} exit={result.returncode} ===\n{result.stdout}', flush=True)
    if result.returncode and not allow_failure:
        raise RuntimeError(f'{label} failed')
    return result.returncode


def pytest_run(path, label, targets, env, repo=True, expect_failure=False):
    xml = LOGS / f'{label}.xml'
    args = [PYTHON, '-m', 'pytest', '-q', '--tb=short', f'--junitxml={xml}']
    if repo:
        args += ['-o', 'addopts=', '--testtype=custom']
    args += targets
    code = command(args, path, label, env=env, allow_failure=True)
    root = ET.parse(xml).getroot()
    suites = [root] if root.tag == 'testsuite' else list(root.findall('testsuite'))
    counts = {key: sum(int(s.get(key, 0)) for s in suites)
              for key in ('tests', 'failures', 'errors', 'skipped')}
    counts['passed'] = counts['tests'] - counts['failures'] - counts['errors'] - counts['skipped']
    failures = [f"{t.get('classname')}::{t.get('name')}"
                for t in root.iter('testcase')
                if t.find('failure') is not None or t.find('error') is not None]
    counts.update(exit_code=code, failed_nodes=failures)
    if expect_failure:
        assert code == 1 and counts['failures'] > 0 and counts['errors'] == 0, counts
    return counts


def environment(path):
    return dict(os.environ, PYTHONPATH=str(path / 'src'), MPLBACKEND='Agg')


def check_upstream():
    current = subprocess.check_output(
        ['git', 'ls-remote', 'https://github.com/nasa/fmdtools.git', 'refs/heads/dev'],
        text=True, timeout=40).split()[0]
    assert current == BASE, f'Upstream changed to {current}; rebase before submitting.'


check_upstream()
plans = [
    ('ci', 'fix/bootstrap-constant-slices', 'Handle constant outputs along the bootstrap axis',
     'src/fmdtools/analyze/common.py', 'tests/test_bootstrap_constant_slices.py',
     '''    if len(np.shape(vals)) > 1:\n        val = vals[axis, 0]\n    else:\n        val = np.array(vals).flatten()[0]''',
     '''    val = np.take(vals, [0], axis=axis)'''),
    ('join', 'fix/join-repeated-phase-labels', 'Retain all intervals when joining same-named phases',
     'src/fmdtools/analyze/phases.py', 'tests/test_join_repeated_labels.py',
     '''        phases = {c: phasemaps[i].phases[c] for i, c in enumerate(combo)}\n        intervals = [i for i in phases.values()]''',
     '''        intervals = [phasemaps[i].phases[c] for i, c in enumerate(combo)]'''),
]
records = []
for label, branch, title, source, test, old, new in plans:
    path = ROOT.parent / f'contribution-{label}'
    command(['git', 'worktree', 'add', '-b', branch, str(path), BASE], ROOT, f'{label}-worktree')
    if not records:
        command([PYTHON, '-m', 'pip', 'install', '-e', str(path), 'pytest', 'pytest-cov',
                 'nbmake', 'ruff', 'build', 'pymoo', 'deap', 'pathos', 'adjustText'],
                path, 'dependencies')
    env = environment(path)
    template = (ROOT / 'validation' / f'test_{label}.py').read_text()
    (path / test).write_text(template)
    command([PYTHON, '-m', 'ruff', 'format', test], path, f'{label}-format')
    original_test = (path / test).read_text()
    before = pytest_run(path, f'{label}-before', [test], env, expect_failure=True)
    file = path / source
    content = file.read_text()
    assert content.count(old) == 1, 'Expected source block changed.'
    file.write_text(content.replace(old, new))
    after = pytest_run(path, f'{label}-after', [test], env)
    assert after['exit_code'] == 0 and after['tests'] == before['tests']
    unit = pytest_run(path, f'{label}-unit', ['tests/'], env)
    assert unit['exit_code'] == 0
    docs = pytest_run(path, f'{label}-doctests', ['--doctest-modules', 'src/fmdtools'], env)
    baseline_docs = None
    if docs['exit_code']:
        pristine = ROOT.parent / f'baseline-{label}'
        command(['git', 'worktree', 'add', '--detach', str(pristine), BASE], ROOT,
                f'{label}-baseline-worktree')
        baseline_docs = pytest_run(pristine, f'{label}-baseline-doctests',
                                   ['--doctest-modules', 'src/fmdtools'], environment(pristine))
        assert set(docs['failed_nodes']) == set(baseline_docs['failed_nodes']), (docs, baseline_docs)
    command([PYTHON, '-m', 'ruff', 'check', '--select', 'E9,F63,F7,F82', source, test],
            path, f'{label}-lint')
    command([PYTHON, '-m', 'compileall', '-q', source, test], path, f'{label}-syntax')
    generated = subprocess.check_output(['git', 'diff', '--name-only'], cwd=path, text=True).splitlines()
    for name in generated:
        if name != source:
            assert name.startswith('docs-source/figures/'), name
            subprocess.check_call(['git', 'restore', '--', name], cwd=path)
    assert (path / test).read_text() == original_test
    dist = LOGS / f'{label}-dist'
    command([PYTHON, '-m', 'build', '--outdir', str(dist)], path, f'{label}-build', env=env)
    wheel = next(dist.glob('*.whl'))
    installed = ROOT.parent / f'installed-{label}'
    command([PYTHON, '-m', 'pip', 'install', '--no-deps', '--target', str(installed), str(wheel)],
            path, f'{label}-install')
    testdir = ROOT.parent / f'installed-tests-{label}'
    testdir.mkdir()
    shutil.copy2(path / test, testdir / Path(test).name)
    install_env = dict(env, PYTHONPATH=str(installed))
    verify = f'import fmdtools; print(fmdtools.__file__); assert fmdtools.__file__.startswith({str(installed)!r})'
    command([PYTHON, '-c', verify], testdir, f'{label}-installed-import', env=install_env)
    installed_result = pytest_run(testdir, f'{label}-installed-tests', [Path(test).name],
                                  install_env, repo=False)
    assert installed_result['exit_code'] == 0
    command(['git', 'diff', '--check'], path, f'{label}-whitespace')
    command(['git', 'diff', '--', source], path, f'{label}-source-diff')
    subprocess.check_call(['git', 'config', 'user.name', 'Sylvester Kaczmarek'], cwd=path)
    subprocess.check_call(['git', 'config', 'user.email', '16242628+sylvesterkaczmarek@users.noreply.github.com'], cwd=path)
    subprocess.check_call(['git', 'add', '--', source, test], cwd=path)
    staged = subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=path, text=True).splitlines()
    assert set(staged) == {source, test}, staged
    command(['git', 'commit', '-m', title], path, f'{label}-commit')
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=path, text=True).strip()
    records.append(dict(label=label, branch=branch, title=title, sha=sha, base=BASE,
                        before=before, after=after, unit=unit, doctests=docs,
                        baseline_doctests=baseline_docs, installed=installed_result,
                        path=str(path)))
    (LOGS / 'results.json').write_text(json.dumps(records, indent=2))

# No branch is published until both candidates pass all required gates.
check_upstream()
assert os.environ['GITHUB_REPOSITORY'] == 'sylvesterkaczmarek/fmdtools'
for record in records:
    command(['git', 'push', 'origin', 'HEAD:refs/heads/' + record['branch']],
            Path(record['path']), record['label'] + '-push')
print('VALIDATION_RESULTS ' + json.dumps(records), flush=True)
