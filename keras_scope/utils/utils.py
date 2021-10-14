from pathlib import Path
import numpy as np

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def save_dataset(ct_inputs, brain_masks, ids, params, out_path):
    np.savez_compressed(out_path,
                        params=params, ids=ids,
                        ct_inputs=ct_inputs, brain_masks=brain_masks)