"""CPU reselection reuses cpu_baselines.choose_arm (line 78), choose_rule (69).

Replay implementation remains the v2 parameterized copy. Only rule selection is
bound to the original CPU module; its immutable hash is checked before import.
"""
from pathlib import Path
from new_gate_b_common import sha256_file

CPU_SHA='4c585a0fda0d93055aae78e5c2099cfe08ae701d52132803ac77460456e940da'


def main():
    path=Path(__file__).with_name('cpu_baselines.py')
    if sha256_file(path)!=CPU_SHA:raise SystemExit('CPU baseline source changed')
    from cpu_baselines import choose_arm
    from new_rule_grid_bench_v2p import main as grid_main
    grid_main(chooser=choose_arm)


if __name__=='__main__':main()
