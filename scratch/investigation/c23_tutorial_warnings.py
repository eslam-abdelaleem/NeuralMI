"""Every warning the tutorials' saved outputs show, behind C23.

For each code cell of the five tutorials and the README figures notebook: the
section it sits in, each kind of warning it shows (its first sentence, numbers
replaced by #) and how often the library says it was raised (its 'was raised N
times' line, else 1). The notebooks are read at a git revision, by default HEAD.

Usage: python scratch/investigation/c23_tutorial_warnings.py <repo> [revision]
"""
import json
import re
import subprocess
import sys

NOTEBOOKS = ['tutorials/01_Why_Information.ipynb', 'tutorials/02_Your_First_Number.ipynb',
             'tutorials/03_Which_Quantity.ipynb', 'tutorials/04_Preparing_Your_Data.ipynb',
             'tutorials/05_Estimator_And_Architecture.ipynb', 'docs/figures/readme_figures.ipynb']
WARNING = re.compile(r'^\S+:\d+: (\w*Warning): (.*)$')
COUNT = re.compile(r"^'(.*)' was raised (\d+) times in this (?:cell|call) and shown once\.$")
NUMBER = re.compile(r'[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?')


def kind(text):
    return NUMBER.sub('#', text.split('. ')[0])[:95]


def load(repo, revision, path):
    out = subprocess.run(['git', '-C', repo, '-c', 'core.fileMode=false', 'show', f'{revision}:{path}'],
                         capture_output=True, text=True)
    return json.loads(out.stdout) if out.returncode == 0 else None


def catalog(notebook):
    """(cell, section, kind, count) for every kind of warning a cell shows."""
    rows, section = [], ''
    for i, cell in enumerate(notebook['cells']):
        source = ''.join(cell['source'])
        if cell['cell_type'] == 'markdown':
            for line in source.splitlines():
                if line.startswith('#'):
                    section = line.strip('# ').strip()
            continue
        shown, counted = {}, {}
        for output in cell.get('outputs', []):
            if not (output['output_type'] == 'stream' and output.get('name') == 'stderr'):
                continue
            for line in ''.join(output['text']).split('\n'):
                match = WARNING.match(line.split('\r')[-1])
                if not match:
                    continue
                text = match.group(2)
                repeat = COUNT.match(text)
                if repeat:
                    # A count names its warning by a snippet cut at 90 characters.
                    snippet = NUMBER.sub('#', repeat.group(1).rstrip('.').rstrip())
                    hits = [k for k, t in shown.items() if NUMBER.sub('#', t).startswith(snippet)]
                    key = hits[0] if hits else kind(repeat.group(1))
                    counted[key] = counted.get(key, 0) + int(repeat.group(2))
                else:
                    shown.setdefault(kind(text), text)
        for k in shown:
            rows.append((i, section, k, max(1, counted.get(k, 0))))
    return rows


def main(repo, revision='HEAD'):
    kinds = {}
    for path in NOTEBOOKS:
        notebook = load(repo, revision, path)
        if notebook is None:
            continue
        print(f'\n## {path.split("/")[-1]} at {revision}')
        for cell, section, k, n in catalog(notebook):
            print(f'  cell {cell:>2} [{section[:38]}] x{n:<3} {k}')
            kinds.setdefault(k, set()).add(path.split('/')[-1][:2])
    print(f'\n{len(kinds)} kinds of warning:')
    for k, where in sorted(kinds.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        print(f'  {",".join(sorted(where)):<15} {k}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'HEAD')
