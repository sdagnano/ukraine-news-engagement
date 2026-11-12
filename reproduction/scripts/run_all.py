from pathlib import Path
import subprocess, sys, hashlib, json
root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / 'audit/v4_input_hashes.json').read_text())
for name, digest in manifest.items():
    assert hashlib.sha256((root / 'input' / name).read_bytes()).hexdigest() == digest, name
if '--verify-inputs-only' in sys.argv:
    print('All input hashes match.')
    raise SystemExit(0)
stages = ['validate_new_methods.py', 'cell_models.py', 'scale_and_influence.py', 'hierarchical_sources.py', 'ppml_heterogeneity.py', 'measurement_sensitivity.py', 'working_covariance.py', 'sample_sensitivity.py', 'empirical_calibration.py', 'diagnose_dgp.py', 'make_assets.py']
for name in stages:
    print('RUN', name, flush=True)
    subprocess.run([sys.executable, '-X', 'utf8', str(root / 'analysis' / name)], cwd=root, check=True)
print('Analysis completed.')
