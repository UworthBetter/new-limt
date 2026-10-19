"""One-shot sequential unseal wrapper. Exclusive attempt marker forbids retries.

The plan contains analysis_args and bench_args (argument arrays), analysis_out,
and inputs (path/sha256 records). It cannot replace either frozen executable.
Any failed production attempt commits only its own log files. A new attempt
requires a separately registered DL entry and a new plan/attempt directory.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
from new_gate_b_common import preflight,sha256_file,write_json


def run_once(commands,directory,primary_out,inputs=()):
    directory=Path(directory);preflight([directory]);directory.mkdir(parents=True)
    journal=dict(started_at=datetime.now(timezone.utc).isoformat(),inputs=list(inputs),stages=[])
    for item in inputs:
        if not Path(item['path']).is_file() or sha256_file(item['path'])!=item['sha256']:
            journal.update(passed=False,failure='unseal input hash mismatch',failed_input=item['path'])
            write_json(directory/'new_unseal_log.json',journal)
            return journal
    prior=None
    for name,command in commands:
        try:
            result=subprocess.run(command,capture_output=True,text=True)
        except OSError as exc:
            result=subprocess.CompletedProcess(command,1,'',type(exc).__name__)
        stdout=directory/f'new_{name}_stdout.txt';stderr=directory/f'new_{name}_stderr.txt'
        stdout.write_text(result.stdout,encoding='utf-8');stderr.write_text(result.stderr,encoding='utf-8')
        stage=dict(stage=name,returncode=result.returncode,time=datetime.now(timezone.utc).isoformat(),
                   stdout_sha256=sha256_file(stdout),stderr_sha256=sha256_file(stderr))
        if name=='analysis' and result.returncode==0:
            if not Path(primary_out).is_file():
                stage.update(returncode=1,failure='analysis returned success without output')
            else:
                prior=sha256_file(primary_out);stage['analysis_output_sha256']=prior
        if name=='bench_eval' and prior is not None:
            stage['analysis_output_unchanged']=Path(primary_out).is_file() and sha256_file(primary_out)==prior
            if not stage['analysis_output_unchanged']:stage['returncode']=1
        journal['stages'].append(stage)
        if stage['returncode']!=0:break
    journal['passed']=len(journal['stages'])==2 and all(x['returncode']==0 for x in journal['stages'])
    write_json(directory/'new_unseal_log.json',journal)
    return journal


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--attempt-dir',type=Path,required=True);a=p.parse_args()
    plan=json.loads(a.plan.read_text(encoding='utf-8'));src=a.root/'src'
    analysis_args=plan['analysis_args']
    if '--out' not in analysis_args or Path(analysis_args[analysis_args.index('--out')+1]).resolve()!=Path(plan['analysis_out']).resolve():
        raise SystemExit('analysis_args --out must identify analysis_out')
    preflight([plan['analysis_out']])
    commands=[('analysis',[sys.executable,str(src/'new_gate_b_analysis_v3_1.py'),'--run',*plan['analysis_args']]),
              ('bench_eval',[sys.executable,str(src/'new_bench_eval_v1.py'),'--run',*plan['bench_args']])]
    inputs=list(plan['inputs'])+[dict(path=str(a.plan),sha256=sha256_file(a.plan))]
    record=run_once(commands,a.attempt_dir,plan['analysis_out'],inputs)
    if not record['passed']:
        paths=[str(p.resolve()) for p in a.attempt_dir.glob('new_*')]
        subprocess.run(['git','-C',str(a.root),'add','--',*paths],check=True)
        subprocess.run(['git','-C',str(a.root),'commit','-m','Record failed one-shot Gate B unseal','--',*paths],check=True)
        raise SystemExit('unseal failed; logs committed; rerun requires a DL entry')
    print('WROTE '+str(a.attempt_dir/'new_unseal_log.json'))


if __name__=='__main__':main()
