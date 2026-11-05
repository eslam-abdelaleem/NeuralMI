# tests/test_mode_registry.py
"""Each mode is declared once, in analysis/modes.py (MODES), and everything else agrees with it.

USING.md's table of what each mode accepts is checked against the
declarations, run() takes one config per mode that has one, and run.py tests
what a mode can do by asking its declaration, not by listing modes by name.
The contract test (tests/test_contract.py) checks each producer against the
axis its mode declares.
"""
import ast
import dataclasses
import inspect
import pathlib
import re

import pytest

import neural_mi as nmi
from neural_mi.analysis.modes import MODES, Mode

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _table():
    """USING.md's rows of what each mode accepts: {mode: (repeats, grid, rigorous)}."""
    text = (ROOT / 'reference' / 'USING.md').read_text()
    section = text[text.index('### What each mode accepts'):]
    rows = {}
    for line in section.splitlines()[4:]:
        if not line.startswith('|'):
            break
        cells = [c.strip() for c in line.strip('|').split('|')]
        for mode in re.findall(r'`(\w+)`', cells[0]):
            rows[mode] = tuple(cells[1:])
    return rows


def test_usings_table_of_what_each_mode_accepts_is_the_registry():
    assert _table() == {name: mode.table for name, mode in MODES.items()}


def test_run_takes_one_config_per_mode_that_has_one():
    parameters = inspect.signature(nmi.run).parameters
    for name, mode in MODES.items():
        if mode.config is not None:
            assert name in parameters, name
            assert mode.config.__name__ in str(parameters[name].annotation), name


def test_a_mode_that_trains_one_network_says_what_to_use_instead():
    for name, mode in MODES.items():
        if mode.one_network is not None:
            assert mode.table[:2] == ('refused', 'refused'), name
            assert all(mode.one_network), name


@pytest.mark.parametrize('name', list(MODES))
def test_every_declaration_is_complete(name):
    mode = MODES[name]
    assert isinstance(mode, Mode) and callable(mode.produce)
    assert (mode.terms is None) == (not mode.terms), name
    if mode.reads_embeddings:
        assert mode.critic is not None, name


def _names_a_mode(node) -> bool:
    """A literal mode name, or a literal list of them."""
    if isinstance(node, ast.Constant):
        return node.value in MODES
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return any(_names_a_mode(e) for e in node.elts)
    return False


def test_no_module_branches_on_a_modes_name():
    """Nothing in the package compares a mode with a mode's name, and no module keeps a table keyed by
    mode names outside the registry: each such fact is a field of the mode's declaration."""
    found = []
    for path in sorted((ROOT / 'neural_mi').rglob('*.py')):
        tree = ast.parse(path.read_text())
        where = path.relative_to(ROOT).as_posix()
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                sides = [node.left, *node.comparators]
                named = [s for s in sides if isinstance(s, (ast.Name, ast.Attribute))
                         and (s.id if isinstance(s, ast.Name) else s.attr) == 'mode']
                if named and any(_names_a_mode(s) for s in sides):
                    found.append(f"{where}:{node.lineno}")
        if where == 'neural_mi/analysis/modes.py':
            continue
        for node in tree.body:
            keys = [k for k in node.value.keys if k is not None] if (
                isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict)) else []
            if len(keys) > 1 and all(_names_a_mode(k) for k in keys):
                found.append(f"{where}:{node.lineno} (a table keyed by mode names)")
    assert not found, f"branches on a mode's name: {found}"


def test_every_field_is_read_somewhere():
    """A declared fact that nothing reads would be a second, idle copy of the truth."""
    source = '\n'.join(path.read_text() for path in (ROOT / 'neural_mi').rglob('*.py'))
    for field in dataclasses.fields(Mode):
        if field.name in ('produce', 'table'):
            continue
        assert re.search(rf'\.{field.name}\b', source), field.name
