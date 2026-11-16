# make_view_index.py: argv[1] = official model.safetensors.index.json, argv[2] = the view's index.
# Keeps the 93,374 names of layers 0-39, embed, norm and head; leaves out vision, MTP, the vision router
# bias and the FP8 Engram embed tables. Names in shards 47 and 48 point at the compact .selected files.
# Expected output sha256: 24918e69c2e922bb87ef5abec921bce8bc0f666ac51d00ea3572e52fd4b31ca9. Run with python3 -I.
import sys, json, re
def keep(n):
    return n in ('embed.weight', 'norm.weight', 'head.weight') or (
        re.fullmatch(r'layers\.(?:[0-9]|[1-3][0-9])\..+', n) is not None and
        not n.endswith(('ffn.gate.bias_vl', 'engram.embed.weight', 'engram.embed.scale')))
wm = json.load(open(sys.argv[1]))['weight_map']
tail = ('model-00047-of-00048.safetensors', 'model-00048-of-00048.safetensors')
m = {n: f + '.selected.safetensors' if f in tail else f for n, f in wm.items() if keep(n)}
assert len(m) == 93374
open(sys.argv[2], 'w').write(json.dumps({'metadata': {'total_size': 298624517568}, 'weight_map': m}, indent=2, sort_keys=True) + '\n')
