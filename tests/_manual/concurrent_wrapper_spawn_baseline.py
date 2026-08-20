# Manual baseline marker — NOT part of the automated suite.
# Excluded from default collection via norecursedirs in pyproject.toml.
# Meaningful only when run by hand against a git stash of the pre-fix tree to
# demonstrate the concurrent-wrapper-spawn regression-trap behavior.


def test_concurrent_wrapper_spawn_baseline():
    pass
