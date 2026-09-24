"""Revalidate the bootstrap correction while preserving constant-data behavior."""
from pathlib import Path
import os
import subprocess

root = Path(os.environ['GITHUB_WORKSPACE'])
base = '51e3de7bdeb0cdee2f0e8c56ffffd62aad702272'
subprocess.run(['git', 'fetch', '--depth', '1', 'origin', base], cwd=root,
               check=True, timeout=60)

# Adjust only our unpublished regression fixture. A globally constant array
# has always required return_anyway; distinct constant columns previously
# returned degenerate basic-bootstrap intervals and must continue to do so.
test_path = root / 'validation/test_ci.py'
tests = test_path.read_text()
old = '    values = np.tile([2., 7.], (4, 1))'
assert tests.count(old) == 1
tests = tests.replace(old, '    values = np.full((4, 2), 2.)')
old = '        np.testing.assert_array_equal(entry, [2., 7.])'
assert tests.count(old) == 1
tests = tests.replace(old, '        np.testing.assert_array_equal(entry, [2., 2.])')
tests += '''

@pytest.mark.parametrize('axis', [0, 1, -1])
def test_distinct_constant_outputs_keep_degenerate_bounds(axis):
    values = np.tile([2., 7.], (4, 1))
    if axis != 0:
        values = values.T
    actual = calc_metric_ci(values, axis=axis, n_resamples=499,
                            rng=np.random.default_rng(37))
    expected = bootstrap([values], np.average, axis=axis, method='basic',
                         n_resamples=499, rng=np.random.default_rng(37))
    np.testing.assert_array_equal(actual[0], [2., 7.])
    np.testing.assert_array_equal(actual[1], expected.confidence_interval.low)
    np.testing.assert_array_equal(actual[2], expected.confidence_interval.high)
'''
test_path.write_text(tests)

script = (root / 'validation/run.py').read_text()
script = script.replace("'fix/bootstrap-constant-slices'", "'fix/bootstrap-constant-slices-compatible'")
old = 'records = []\nfor label, branch, title, source, test, old, new in plans:'
assert script.count(old) == 1
script = script.replace(old, 'plans = plans[:1]\n' + old)
old = '    file.write_text(content.replace(old, new))'
new = '''    changed = content.replace(old, new)
    old_guard = "    if not np.all(vals_vary):"
    assert changed.count(old_guard) == 1
    changed = changed.replace(old_guard,
        "    # Preserve the existing policy for globally identical data.\\n"
        "    if not np.all(vals == vals.flat[0]):")
    file.write_text(changed)'''
assert script.count(old) == 1
script = script.replace(old, new)
# Print only useful tails in the job view; retain complete logs in the artifact.
script = script.replace("{result.stdout}', flush=True)", "{result.stdout[-7000:]}', flush=True)")
compile(script, 'validated_bootstrap_runner.py', 'exec')
exec(compile(script, 'validated_bootstrap_runner.py', 'exec'))
