# extract_tail.py: stdin = one whole official safetensors shard (47 or 48); argv[1] = compact output.
# Keeps the shard's four Engram q_weight, k_weight, wkv.weight and wkv.scale tensors, in their original
# order and bytes, and drops the FP8 engram.embed table (the IQ4_NL GGUF replaces it). Prints the sha256
# of the whole input stream, which must equal the shard's LFS hash. Run with python3 -I.
import sys, json, re, struct, hashlib
def keep(n):
    return n in ('embed.weight', 'norm.weight', 'head.weight') or (
        re.fullmatch(r'layers\.(?:[0-9]|[1-3][0-9])\..+', n) is not None and
        not n.endswith(('ffn.gate.bias_vl', 'engram.embed.weight', 'engram.embed.scale')))
src, whole = sys.stdin.buffer, hashlib.sha256()
def read(k):
    b = src.read(k); whole.update(b); return b
n = struct.unpack('<Q', read(8))[0]
hdr = json.loads(read(n)); hdr.pop('__metadata__', None)
sel = sorted((t['data_offsets'][0], t['data_offsets'][1], name) for name, t in hdr.items() if keep(name))
assert len(sel) == 4, sel
out, at = {}, 0
for lo, hi, name in sel:
    out[name] = {'dtype': hdr[name]['dtype'], 'shape': hdr[name]['shape'], 'data_offsets': [at, at + hi - lo]}
    at += hi - lo
h = json.dumps(out, sort_keys=True, separators=(',', ':')).encode(); h += b' ' * ((-len(h)) % 8)
pos, parts = 0, {}
while True:
    b = read(4 << 20)
    if not b: break
    for lo, hi, name in sel:
        a, z = max(lo, pos), min(hi, pos + len(b))
        if a < z: parts.setdefault(name, bytearray()).extend(b[a - pos:z - pos])
    pos += len(b)
with open(sys.argv[1], 'wb') as f:
    f.write(struct.pack('<Q', len(h)) + h)
    for lo, hi, name in sel:
        assert len(parts[name]) == hi - lo; f.write(parts[name])
print('stream sha256', whole.hexdigest(), 'bytes', 8 + n + pos)
