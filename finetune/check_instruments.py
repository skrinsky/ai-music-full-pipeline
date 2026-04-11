import numpy as np
from pathlib import Path
d = Path('finetune/runs/noto_data')
insts = np.load(d / 'train_insts.npy').flatten()
for i, c in sorted(zip(*np.unique(insts, return_counts=True)), key=lambda x: -x[1]):
    print(f'  program {i:3d}: {c:6d} events  ({100*c/len(insts):.1f}%)')
