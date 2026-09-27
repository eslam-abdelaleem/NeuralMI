# tests/test_docs_drift.py
"""The reference documents describe the code as it is.

reference/PARAMETERS.md lists every setting with its default, and
reference/MESSAGES.md is keyed by the text the library prints. Both are checked
against the code here, so a renamed field, a changed default or a reworded
message fails the suite until the document follows.
"""
import ast
import dataclasses
import inspect
import pathlib
import re

import pytest

import neural_mi as nmi
from neural_mi import config as cfg
from neural_mi.defaults import BASE_PARAMS_SCHEMA, MODE_KWARGS_SCHEMA, PROCESSOR_PARAMS_SCHEMA

REFERENCE = pathlib.Path(__file__).resolve().parents[1] / 'reference'
_MISSING = object()


# ---------------------------------------------------------------------------
# PARAMETERS.md
# ---------------------------------------------------------------------------

def _parameter_tables():
    """``{section: {name: default_cell}}`` for every table in PARAMETERS.md.

    A section is the backticked name of the nearest heading above a table, or
    the bold processor name for the processor tables.
    """
    tables, section = {}, None
    for line in (REFERENCE / 'PARAMETERS.md').read_text().split('\n'):
        heading = re.match(r'^#{2,3} `([A-Za-z_]+)(?:\(\))?`', line)
        processor = re.match(r"^\*\*`'(\w+)'`\*\*", line)
        if heading:
            section = heading.group(1)
        elif processor:
            section = f"processor:{processor.group(1)}"
        elif line.startswith('| `') and section is not None:
            cells = [c.strip() for c in line.strip('|').split('|')]
            for name in re.findall(r'`(\w+)`', cells[0]):
                assert name not in tables.setdefault(section, {}), (section, name)
                tables[section][name] = cells[1]
    return tables


def _literal(cell):
    """The default a cell states, or _MISSING when it names no literal value."""
    m = re.fullmatch(r'`(.+)`', cell)
    if not m:
        return _MISSING
    try:
        return eval(m.group(1), {'__builtins__': {}}, {'range': range})
    except Exception:
        return _MISSING


TABLES = _parameter_tables()
CONFIGS = [cfg.Model, cfg.Training, cfg.Split, cfg.Estimator, cfg.Output, cfg.Processing,
           cfg.Rigorous, cfg.Precision, cfg.Lag, cfg.Transfer, cfg.Conditional, cfg.Interaction,
           cfg.Pairwise, cfg.Dimensionality, cfg.Sweep]
# Config fields stored under another name in the parameter schema.
SCHEMA_NAME = {'Split': {'mode': 'split_mode', 'gap_fraction': 'split_gap_fraction'},
               'Estimator': {'name': 'estimator_name', 'params': 'estimator_params'},
               'Output': {'units': 'output_units'}}
MODE_OF = {'Rigorous': 'rigorous', 'Precision': 'precision', 'Lag': 'lag', 'Transfer': 'transfer',
           'Conditional': 'conditional', 'Interaction': 'interaction', 'Pairwise': 'pairwise',
           'Dimensionality': 'dimensionality', 'Sweep': 'sweep'}


@pytest.mark.parametrize('cls', CONFIGS, ids=lambda c: c.__name__)
def test_every_config_field_is_documented_once(cls):
    fields = {f.name for f in dataclasses.fields(cls)}
    documented = set(TABLES.get(cls.__name__, {}))
    assert documented == fields, (
        f"PARAMETERS.md `{cls.__name__}`: missing {sorted(fields - documented)}, "
        f"not a field {sorted(documented - fields)}")


@pytest.mark.parametrize('cls', CONFIGS, ids=lambda c: c.__name__)
def test_every_documented_default_matches_the_code(cls):
    rename = SCHEMA_NAME.get(cls.__name__, {})
    mode_schema = MODE_KWARGS_SCHEMA.get(MODE_OF.get(cls.__name__), {})
    for name, cell in TABLES[cls.__name__].items():
        stated = _literal(cell)
        if stated is _MISSING:
            continue
        if cls.__name__ in MODE_OF:
            actual = mode_schema.get(name, {}).get('default', _MISSING)
            if actual is _MISSING and name == 'gamma_range':
                actual = inspect.signature(
                    __import__('neural_mi.analysis.rigorous', fromlist=['x'])
                    .run_rigorous_analysis).parameters['gamma_range'].default
        else:
            actual = BASE_PARAMS_SCHEMA.get(rename.get(name, name), {}).get('default', _MISSING)
        if actual is _MISSING:
            continue
        assert stated == actual, f"`{cls.__name__}.{name}`: documented {stated!r}, code {actual!r}"


def test_run_arguments_are_documented():
    params = inspect.signature(nmi.run).parameters
    config_args = {'processing', 'model', 'training', 'split', 'estimator', 'output', *MODE_OF.values()}
    run_args = {n for n, p in params.items() if p.kind is not p.VAR_KEYWORD} - config_args
    assert set(TABLES['run']) == run_args
    for name, cell in TABLES['run'].items():
        stated = _literal(cell)
        if stated is not _MISSING:
            assert stated == params[name].default, name


@pytest.mark.parametrize('processor', sorted(PROCESSOR_PARAMS_SCHEMA))
def test_every_processor_parameter_is_documented(processor):
    documented = set(TABLES[f"processor:{processor}"])
    assert documented == set(PROCESSOR_PARAMS_SCHEMA[processor])


def test_processor_defaults_match_the_code():
    from neural_mi.data import handler
    source = inspect.getsource(handler.create_single_dataset)
    code_defaults = {k: ast.literal_eval(v)
                     for k, v in re.findall(r"\.get\('(\w+)', ([^)]+)\)", source)}
    for processor in PROCESSOR_PARAMS_SCHEMA:
        for name, cell in TABLES[f"processor:{processor}"].items():
            stated = _literal(cell)
            if stated is not _MISSING and name in code_defaults:
                assert stated == code_defaults[name], (processor, name)


# ---------------------------------------------------------------------------
# MESSAGES.md
# ---------------------------------------------------------------------------

PACKAGE = pathlib.Path(nmi.__file__).resolve().parent
HOLE = '\x00'


def _text(node, env):
    """The literal text of a message argument, with HOLE for every value."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return re.sub(r'%[-+ #0-9.]*[sdrfgxe]', HOLE, node.value)
    if isinstance(node, ast.JoinedStr):
        return ''.join(v.value if isinstance(v, ast.Constant) else HOLE for v in node.values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _text(node.left, env), _text(node.right, env)
        if left is None and right is None:
            return None
        return (HOLE if left is None else left) + (HOLE if right is None else right)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
        return _text(node.left, env)
    if isinstance(node, ast.Name):
        return env.get(node.id)
    return None


def _is_warning_call(func, aliases):
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        if func.attr == 'warn' and func.value.id in ('warnings', '_warnings'):
            return 'warning'
        if func.value.id in ('logger', '_logger') and func.attr in ('warning', 'info', 'error'):
            return 'warning' if func.attr == 'warning' else 'other'
    if isinstance(func, ast.Name) and func.id in aliases:
        return 'warning'
    return None


def _emitted_messages():
    """``[(level, 'file:line', text)]`` for every message the package emits.

    Read from the source: every ``warnings.warn`` and ``logger`` call, with the
    literal parts of its message kept and each formatted value replaced by a
    placeholder. A local chosen between logger methods (``log = logger.warning
    if ... else logger.debug``) counts as a warning.
    """
    found = {}
    for path in sorted(PACKAGE.rglob('*.py')):
        tree = ast.parse(path.read_text())
        for scope in ast.walk(tree):
            if not isinstance(scope, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            assigns = [n for n in ast.walk(scope) if isinstance(n, ast.Assign)
                       and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)]
            aliases = {n.targets[0].id for n in assigns
                       if any(isinstance(a, ast.Attribute) and a.attr == 'warning'
                              for a in ast.walk(n.value))}
            env = {}
            for n in assigns:
                text = _text(n.value, env)
                if text is not None:
                    env[n.targets[0].id] = HOLE if n.targets[0].id in env else text
            for node in ast.walk(scope):
                if not isinstance(node, ast.Call) or not node.args:
                    continue
                level = _is_warning_call(node.func, aliases)
                if level is None:
                    continue
                where = f"{path.relative_to(PACKAGE.parent)}:{node.lineno}"
                found[where] = (level, where, _text(node.args[0], env))
    return list(found.values())


def _documented_keys():
    doc = (REFERENCE / 'MESSAGES.md').read_text()
    return [a or b for a, b in re.findall(r'\*\*``(.+?)``\*\*|\*\*`([^`]+?)`\*\*', doc)]


def _key_matches(key, message):
    """A key's parts between its ``...`` placeholders appear in order in the message."""
    parts = [re.escape(part) for part in re.sub(r'\s+', ' ', key).split('...')]
    return re.search('.*?'.join(parts), re.sub(r'\s+', ' ', message), re.S) is not None


MESSAGES = _emitted_messages()
KEYS = _documented_keys()


def test_every_message_argument_is_readable():
    unreadable = [where for _, where, text in MESSAGES if text is None]
    assert not unreadable, f"messages whose text could not be read from the source: {unreadable}"


@pytest.mark.parametrize('key', KEYS)
def test_every_documented_message_is_emitted(key):
    assert any(_key_matches(key, text) for _, _, text in MESSAGES if text), (
        f"MESSAGES.md documents `{key}`, and no message in neural_mi/ contains it")


def test_every_warning_is_documented():
    missing = [f"{where}: {text.replace(HOLE, '...')[:100]}"
               for level, where, text in MESSAGES
               if level == 'warning' and text and not any(_key_matches(k, text) for k in KEYS)]
    assert not missing, "warnings without an entry in MESSAGES.md:\n" + "\n".join(missing)
