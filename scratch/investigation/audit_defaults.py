"""Find default values that disagree with BASE_PARAMS_SCHEMA.

1. `<something>.get('<schema key>', <literal>)` whose literal differs from the schema default.
2. Function parameters named like a schema key whose literal default differs from the schema's.
Only literal defaults are compared (numbers, strings, bools, None).
"""
import ast
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(root))
from neural_mi.defaults import BASE_PARAMS_SCHEMA, MODE_KWARGS_SCHEMA  # noqa: E402

schema = {k: v['default'] for k, v in BASE_PARAMS_SCHEMA.items() if 'default' in v}
for mode, spec in MODE_KWARGS_SCHEMA.items():
    for k, v in spec.items():
        if 'default' in v:
            schema.setdefault(k, v['default'])


def literal(node):
    try:
        return True, ast.literal_eval(node)
    except Exception:
        return False, None


def differs(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return a is not b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) != float(b)
    return a != b


rows = []
for path in sorted((root / 'neural_mi').rglob('*.py')):
    tree = ast.parse(path.read_text())
    rel = path.relative_to(root)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'get'
                and len(node.args) == 2 and isinstance(node.args[0], ast.Constant)
                and node.args[0].value in schema):
            ok, val = literal(node.args[1])
            key = node.args[0].value
            if ok and differs(val, schema[key]):
                rows.append((str(rel), node.lineno, key, repr(val), repr(schema[key]), '.get fallback'))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            pos = args.args[len(args.args) - len(args.defaults):]
            pairs = list(zip(pos, args.defaults)) + [(a, d) for a, d in zip(args.kwonlyargs, args.kw_defaults) if d is not None]
            for a, d in pairs:
                if a.arg in schema:
                    ok, val = literal(d)
                    if ok and differs(val, schema[a.arg]):
                        rows.append((str(rel), node.lineno, a.arg, repr(val), repr(schema[a.arg]),
                                     f'param of {node.name}()'))

print(f"{'file:line':45s} {'key':24s} {'here':>10s} {'schema':>10s}  where")
for f, ln, k, here, sch, where in sorted(rows):
    print(f"{f + ':' + str(ln):45s} {k:24s} {here:>10s} {sch:>10s}  {where}")
print(f"\n{len(rows)} disagreements")
