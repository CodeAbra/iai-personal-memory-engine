"""Native submodule identity: fully-qualified names for pickle and inspect.

Each native submodule must report a dotted ``__name__`` matching its
``sys.modules`` key, so that classes defined inside it carry a resolvable
``__module__`` and round-trip through pickle and ``inspect``.
"""

import importlib
import inspect
import pickle

import pytest

SUBMODULES = ["embed", "graph", "hd", "store", "engine"]


def _import(name):
    return importlib.import_module(f"iai_mcp_native.{name}")


@pytest.mark.parametrize("sub", SUBMODULES)
def test_submodule_name_is_fully_qualified(sub):
    mod = _import(sub)
    assert mod.__name__ == f"iai_mcp_native.{sub}", (
        f"submodule {sub} reports a short __name__ ({mod.__name__!r}); "
        "pickle and inspect need the dotted form"
    )


@pytest.mark.parametrize("sub", SUBMODULES)
def test_dotted_import_resolves(sub):
    # The dotted import statement itself must work, not only the
    # from-package form, via the sys.modules registration.
    mod = importlib.import_module(f"iai_mcp_native.{sub}")
    assert mod is not None


def _submodule_classes(mod):
    """Classes defined in the module (their __module__ is the module name)."""
    return [
        obj
        for _name, obj in inspect.getmembers(mod, inspect.isclass)
        if getattr(obj, "__module__", None) == mod.__name__
    ]


@pytest.mark.parametrize("sub", SUBMODULES)
def test_submodule_class_references_pickle_round_trip(sub):
    # A class defined in a native submodule must pickle and unpickle by its
    # qualified module path: pickle locates the class via __module__ +
    # __qualname__, which only resolves when the class carries the dotted
    # module name. The pickled payload is our own class reference produced and
    # consumed in-process, so there is no untrusted-data exposure.
    mod = _import(sub)
    classes = _submodule_classes(mod)
    if not classes:
        pytest.skip(f"{sub} submodule defines no native classes")
    for cls in classes:
        restored = pickle.loads(pickle.dumps(cls))
        assert restored is cls, f"{cls!r} did not round-trip through pickle"


@pytest.mark.parametrize("sub", SUBMODULES)
def test_inspect_resolves_submodule_classes(sub):
    mod = _import(sub)
    classes = [
        obj
        for _name, obj in inspect.getmembers(mod, inspect.isclass)
        if getattr(obj, "__module__", None) == mod.__name__
    ]
    for cls in classes:
        assert inspect.getmodule(cls) is mod, (
            f"inspect.getmodule failed to resolve {cls!r} back to {mod.__name__}"
        )
