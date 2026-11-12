from pathlib import Path, PurePosixPath, PureWindowsPath
import argparse
import hashlib
import http.client
import json
import os
import tempfile
import urllib.error
import urllib.request
ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()

def destination(relative):
    relative = PurePosixPath(relative)
    if relative.is_absolute() or PureWindowsPath(str(relative)).drive or '..' in relative.parts or ('\\' in str(relative)):
        raise ValueError('Expected a project-relative destination')
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError('Destination leaves the reproduction directory')
    return path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', choices=['facebook', 'all', 'provenance'], default='all')
    parser.add_argument('--check', action='store_true', help='Verify local files without network access')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'input/upstream_manifest.json').read_text(encoding='utf-8'))
    failures = []
    entries = [e for e in manifest['files'] if args.only == 'provenance' and e['role'] == 'provenance' or (args.only == 'facebook' and e['path'] == 'input/fb_proua_full_osf.RDS') or (args.only == 'all' and e['role'] == 'analysis_input')]
    if not entries:
        raise SystemExit('The manifest contains no files for the requested selection.')
    for entry in entries:
        path = destination(entry['path'])
        if path.exists():
            if digest(path) == entry['sha256']:
                print('OK', entry['path'])
            else:
                failures.append(entry['path'] + ': existing file has an unexpected SHA256; not overwritten')
            continue
        if args.check:
            failures.append(entry['path'] + ': missing')
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            print('DOWNLOAD', entry['path'], flush=True)
            request = urllib.request.Request(entry['download_url'], headers={'User-Agent': 'ukraine-news-engagement reproducibility-materials/1.0'})
            with urllib.request.urlopen(request, timeout=120) as response:
                with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.download-', delete=False) as target:
                    temporary = Path(target.name)
                    for chunk in iter(lambda: response.read(1024 * 1024), b''):
                        target.write(chunk)
            if digest(temporary) != entry['sha256']:
                raise ValueError('Downloaded SHA256 differs from the scientific snapshot')
            os.replace(temporary, path)
            temporary = None
            print('VERIFIED', entry['path'])
        except (OSError, ValueError, urllib.error.URLError, http.client.HTTPException) as error:
            failures.append(entry['path'] + ': ' + str(error))
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    if failures:
        print('\n'.join(failures))
        print('Manual alternative: obtain the named files from the source pages in the manifest,\nplace them at the listed paths, then repeat this command with --check.\nA matching SHA256 is required; changed upstream files are never silently accepted.')
        raise SystemExit(1)
    print(f'Verified {len(entries)} upstream files; no scientific estimate was recomputed.')
if __name__ == '__main__':
    main()
