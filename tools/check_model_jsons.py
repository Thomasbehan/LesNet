from pathlib import Path
import json


def find_model_jsons(root: Path):
    for p in root.rglob('*.json'):
        yield p


def contains_deprecated_keys(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return False
    s = json.dumps(data)
    return any(k in s for k in ('"renorm"', '"renorm_clipping"', '"renorm_momentum"'))


if __name__ == '__main__':
    root = Path('.')
    problems = []
    for p in find_model_jsons(root):
        if contains_deprecated_keys(p):
            problems.append(str(p))
    if problems:
        print('Found deprecated keys in the following JSON files:')
        for p in problems:
            print(' -', p)
        raise SystemExit(2)
    print('No deprecated renorm keys found in JSON files.')
