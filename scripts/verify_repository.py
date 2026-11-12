from pathlib import Path, PurePosixPath, PureWindowsPath
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]

def main():
    manifest = json.loads((ROOT / 'MANIFEST_SHA256.json').read_text(encoding='utf-8'))
    failures = []
    for relative, expected in manifest['sha256'].items():
        name = PurePosixPath(relative)
        path = (ROOT / name).resolve()
        if name.is_absolute() or PureWindowsPath(relative).drive or '..' in name.parts or '\\' in relative or not path.is_relative_to(ROOT.resolve()):
            failures.append(relative + ': invalid path')
            continue
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            failures.append(relative + ': missing or changed')
    print(json.dumps({'passed': not failures, 'verified_files': len(manifest['sha256']), 'failures': failures}, indent=2))
    if failures:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
