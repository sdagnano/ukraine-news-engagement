from pathlib import Path
import argparse
import os
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
STAGES = [('scripts/run_all.py', []), ('analysis/r6_inference_audit.py', ['all']), ('analysis/r6_inference_verify.py', []), ('analysis/r6_inference_report.py', []), ('analysis/r6_margins.py', ['--part', 'all']), ('analysis/r6_margins_report.py', []), ('analysis/r6_labels.py', ['--phase', 'all']), ('analysis/r6_labels_OLS.py', []), ('analysis/r6_assets.py', []), ('analysis/r7_reproduce_composition.py', ['--write']), ('analysis/r7_reproduce_composition.py', ['--check']), ('analysis/r7_assets.py', []), ('scripts/verify_submission.py', ['--analysis-only', '--skip-manifest'])]

def run(relative, arguments):
    print('RUN', relative, *arguments, flush=True)
    environment = os.environ.copy()
    for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
        environment[key] = '1'
    subprocess.run([sys.executable, '-X', 'utf8', str(ROOT / relative), *arguments], cwd=ROOT, env=environment, check=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full', action='store_true', help='Repeat all expensive fits and simulations')
    parser.add_argument('--list-stages', action='store_true', help='List full stages without running them')
    args = parser.parse_args()
    if args.list_stages:
        for relative, arguments in STAGES:
            print(relative, *arguments)
        return
    if not args.full:
        run('scripts/verify_submission.py', [])
        print('Verification complete. Use --full only to repeat the expensive fits and simulations.')
        return
    for relative, arguments in STAGES:
        run(relative, arguments)
    print('Scientific reproduction complete. No manuscript compilation or manifest rewrite was performed.')
if __name__ == '__main__':
    main()
