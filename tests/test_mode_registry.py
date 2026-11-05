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


def test_run_asks_a_modes_declaration_instead_of_listing_modes():
    """No membership test against a literal list of mode names, and no module-level table of them,
    is left in run.py: each such fact is a field of the mode's declaration."""
    tree = ast.parse((ROOT / 'neural_mi' / 'run.py').read_text())
    listed = [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Compare)
              and isinstance(node.left, ast.Name) and node.left.id == 'mode'
              and any(isinstance(c, (ast.Tuple, ast.List, ast.Set)) for c in node.comparators)]
    tables = [node.targets[0].id for node in tree.body if isinstance(node, ast.Assign)
              and isinstance(node.targets[0], ast.Name) and isinstance(node.value, (ast.Tuple, ast.Dict))
              and any(isinstance(e, ast.Constant) and e.value in MODES
                      for e in (node.value.elts if isinstance(node.value, ast.Tuple) else node.value.keys))]
    assert not listed, f"run.py lists modes by name at lines {listed}"
    assert not tables, f"run.py keeps tables of modes: {tables}"


def test_every_field_is_read_somewhere():
    """A declared fact that nothing reads would be a second, idle copy of the truth."""
    source = '\n'.join(path.read_text() for path in (ROOT / 'neural_mi').rglob('*.py'))
    for field in dataclasses.fields(Mode):
        if field.name in ('produce', 'table'):
            continue
        assert re.search(rf'\.{field.name}\b', source), field.name
