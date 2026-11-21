import sys; sys.path.insert(0, "/Users/amil/Downloads/hackathon-watermark/modal_t1/tracks/postproc")
from common import *
gt, meta, acc = load_train(); full = {e['image_name']: e['full_watermark'] for e in meta}
z = np.load('tm_mf_train.npz'); names = z['names'].tolist()
probs = load_probs('train', 'trustmark')
feats = {k: z[k] for k in z.files if k != 'names'}

feats['net'] = np.stack([probs[n] for n in names])
for k, v in feats.items():
    r = []
    for reg in (True, False):
        a = [auc(gt[n], v[i]) for i, n in enumerate(names) if full[n] == reg and 0 < gt[n].mean() < 1]
        r.append(f'{"full" if reg else "part"} {np.mean(a):.3f}')
    print(f'{k:10s}', ' | '.join(r))
