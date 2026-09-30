"""Every declared setting is read by the engine.

A setting is read when the engine looks it up by name (a subscript load,
``.get``, ``.pop``, ``.setdefault`` or an ``in`` test), takes it as a named
parameter of one of its functions, or uses it inside ``_run_flat()`` for
anything other than copying it into the settings. The modules that only declare
or validate settings do not count. A setting that is accepted and then ignored
fails, whether it is declared in ``defaults.py`` or only in a config class.
"""
import ast
import dataclasses
import inspect
import pathlib

import pytest

import neural_mi
from neural_mi import config
from neural_mi.defaults import BASE_PARAMS_SCHEMA, MODE_KWARGS_SCHEMA, PROCESSOR_PARAMS_SCHEMA

DECLARED = {'defaults.py', 'config.py', 'validation.py'}
ROUTING = {'run', '_run_flat'}


def _copies_a_setting(node):
    """``settings['k'] = k`` or ``_inject(settings, 'k', k)``."""
    if isinstance(node, ast.Assign):
        return (all(isinstance(t, ast.Subscript) for t in node.targets)
                and isinstance(node.value, ast.Name))
    return (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
            and getattr(node.value.func, 'id', None) == '_inject')


def _uses_in(function):
    """The names ``function`` loads outside the lines that only copy settings."""
    used = set()

    def visit(node):
        if _copies_a_setting(node):
            return
        if (isinstance(node, ast.If) and not node.orelse
                and all(_copies_a_setting(s) for s in node.body)):
            return
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            used.add(node.id)
        for child in ast.iter_child_nodes(node):
            visit(child)

    for statement in function.body:
        visit(statement)
    return used


def _engine_reads():
    read = set()
    root = pathlib.Path(neural_mi.__file__).parent
    for path in root.rglob('*.py'):
        if path.name in DECLARED:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if (isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load)
                    and isinstance(node.slice, ast.Constant)):
                read.add(node.slice.value)
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                  and node.func.attr in ('get', 'pop', 'setdefault') and node.args
                  and isinstance(node.args[0], ast.Constant)):
                read.add(node.args[0].value)
            elif (isinstance(node, ast.Compare) and isinstance(node.left, ast.Constant)
                  and any(isinstance(op, (ast.In, ast.NotIn)) for op in node.ops)):
                read.add(node.left.value)
            elif isinstance(node, ast.FunctionDef):
                params = {a.arg for a in node.args.args + node.args.kwonlyargs}
                if path.name == 'run.py' and node.name in ROUTING:
                    read |= params & _uses_in(node)
                else:
                    read |= params
    return read


READ = _engine_reads()


class _Marker:
    def __repr__(self):
        return 'MARKER'


def _config_settings():
    """(class, field, key) for every key a config class lowers a field into."""
    out = []
    for name, cls in inspect.getmembers(config, inspect.isclass):
        if not dataclasses.is_dataclass(cls) or cls.__module__ != config.__name__:
            continue
        marks = {f.name: _Marker() for f in dataclasses.fields(cls)}
        obj = cls(**marks)
        lowered = {}
        for method, fn in inspect.getmembers(obj, callable):
            if method.startswith('to_'):
                for key, value in (fn() or {}).items():
                    for field, mark in marks.items():
                        if value is mark:
                            lowered.setdefault(field, set()).add(key)
        for field in marks:
            for key in sorted(lowered.get(field, {None})):
                out.append((name, field, key))
    return out


SETTINGS = ([('base', k) for k in BASE_PARAMS_SCHEMA]
            + [(f'mode {m}', k) for m, schema in MODE_KWARGS_SCHEMA.items() for k in schema]
            + [(f'processor {p}', k) for p, schema in PROCESSOR_PARAMS_SCHEMA.items()
               for k in schema])
CONFIG_SETTINGS = _config_settings()


@pytest.mark.parametrize('where, key', SETTINGS, ids=[f'{w}:{k}' for w, k in SETTINGS])
def test_every_declared_setting_is_read(where, key):
    assert key in READ, (
        f"The {where} setting '{key}' is declared but nothing in the engine reads it. "
        f"Wire it up or remove it from defaults.py, its config class and PARAMETERS.md."
    )


@pytest.mark.parametrize('cls, field, key', CONFIG_SETTINGS,
                         ids=[f'{c}.{f}' for c, f, _ in CONFIG_SETTINGS])
def test_every_config_field_reaches_the_engine(cls, field, key):
    assert key is not None, f"{cls}.{field} is dropped by every to_* method of {cls}."
    assert key in READ, (
        f"{cls}.{field} becomes '{key}', and nothing in the engine reads it. Wire it up "
        f"or remove the field and its row in PARAMETERS.md."
    )
