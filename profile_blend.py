"""A placement profile for --expert-profile from more than one workload (core/place/native_decode_profile.h).

The planner keeps on the cards, and then in the RAM cache, the experts the profile's `decode` counts call
hottest. A profile recorded on one family of prompts places another family badly: prompts of different kinds
route to different experts. This tool sums decode counts of several --expert-stats captures, each times a
whole-number weight, into one profile. Placement only: which experts a row routes to and every unit's bytes
are untouched, so replies are the base engine's whatever profile is given.

  profile_blend.py write OUT.json W:BASE.json [W:STATS.json ...]
      BASE is a sealed decode profile (an expert-stats/1 file with `source_identity`, accepted by
      --expert-profile on this container); its identity, metadata and non-decode sections are kept. Every
      other file is a capture (--expert-stats FILE, written at shutdown; ab_window.sh's Xst part) or
      another profile of the same container. W is a whole number from 1 to 1000: the file's decode
      counts are taken W times, so a file of T decode tokens weighs W x T tokens in the result. Refuses a
      file that is not a complete pure-decode record of the same model geometry (format, final, model,
      arch, tp, container, layers, experts, top-k, map; no mixed steps; no unmapped, unclassified or
      dropped routing; per layer the counts sum to top-k x tokens) and an OUT that exists.
  profile_blend.py write [--portable] [--container MODEL.rad] [--allow-mixed] OUT.json W:FILE.json ...
      --portable: OUT carries no path or stat (container "", device, inode, mtime_ns and ctime_ns 0), so it is
      accepted by --expert-profile on any machine whose container has the same size, metadata size and metadata
      sha256 (a byte-identical rebuild per RECIPE.md); the server logs "expert profile: portable (...)".
      --container MODEL.rad: the container the captures were recorded on (only its metadata header is read, never
      the weights). It seals a first file that is a raw capture (strict: this machine's path and stat; portable
      with --portable), and checks raw captures against a portable first file. Every sealed file must name the
      same container generation (size, metadata size, metadata sha256; and path and stat when both are strict).
      --allow-mixed: take captures that also hold mixed steps (prefill and decode rows in one step, as batched
      serving runs them): only the pure decode steps' counts are used; the decode rows of mixed steps are left out.
      --mixed-share F (0 to 1; implies --allow-mixed): APPROXIMATE. A mixed step's histogram cannot be split into its
      prefill and decode rows (one histogram a layer a step, counted on the card), so its counts are scaled to
      F x its decode rows, i.e. those rows are taken to route as the step's average row does (mostly prefill rows).
      Useful when batched serving leaves few pure decode steps; placement only, replies are unchanged. The folded
      capture records "mixed_folded" and its mixed sections are zeroed, so a later blend never folds them twice.
  profile_blend.py ident MODEL.rad
      The container's metadata sha256 (what a portable profile is matched by).
  profile_blend.py match MODEL.rad FILE.json
      Whether FILE (a sealed or portable profile) is for this container generation (size, metadata size, metadata
      sha256, model name); exit 0 if so. Reads the container's metadata header only.
  profile_blend.py show FILE.json [FILE.json ...]
      Each file's decode tokens and steps, and for each pair of files the overlap of their hottest
      experts: the share of file A's decode selections that the top N experts of file B hold, N the
      number a rank keeps on a card in the current plan (default 715 of 3840, --top N to change),
      ranked within each rank's 96 experts a layer. 1.00 on the diagonal is the in-sample bound.

Weights: a capture of a few hundred tokens is noisy, so weigh families by what will be served, not by how
many tokens each capture happens to hold; `show` prints the tokens to work from.
"""
import hashlib
import json
import os
import sys

FORMAT = "radiance-expert-stats/1"
NL, NE, TOPK, RANKS = 40, 384, 6, 4
PER = NE // RANKS
GEOMETRY = ("format", "model", "arch", "tp", "n_layers", "n_trunk_layers", "n_experts",
            "layer_experts", "top_k", "map")
MAX_WEIGHT = 1000
IDENTITY_FORMAT = "rad-metadata-prefix-sha256/1"
STAT = ("device", "inode", "mtime_ns", "ctime_ns")
PORTABLE_KEYS = ("format", "file_bytes", "metadata_bytes", "sha256")
MAX_METADATA = 256 << 20


class Refused(Exception):
    pass


def load(path):
    with open(path, "rb") as f:
        raw = f.read()
    try:
        doc = json.loads(raw)
    except ValueError as e:
        raise Refused("%s: not JSON (%s)" % (path, e))
    if not isinstance(doc, dict):
        raise Refused("%s: not an expert-stats object" % path)
    return doc, hashlib.sha256(raw).hexdigest()


def is_portable(doc):
    """container "" and every stat field 0 (core/place/native_decode_profile.h): only size and header hash bind it."""
    ident = doc.get("source_identity")
    return isinstance(ident, dict) and doc.get("container") == "" and all(ident.get(k) == 0 for k in STAT)


def make_portable(doc):
    doc["container"] = ""
    for k in STAT:
        doc["source_identity"][k] = 0
    for b in doc.get("blend") or []:
        if isinstance(b, dict) and "file" in b:
            b["file"] = os.path.basename(b["file"])
    return doc


def container_identity(path):
    """(source_identity, model name) of a .rad container: the sha256 of its first data_off bytes (the metadata the
    server hashes, native_profile_identity), its size and this machine's stat. Reads the header only."""
    import struct
    st = os.stat(path)
    with open(path, "rb") as f:
        head = f.read(256)
        if len(head) < 176:
            raise Refused("%s: not a .rad container (short header)" % path)
        magic, _version, file_bytes = struct.unpack_from("<IIQ", head, 0)
        (model_off,) = struct.unpack_from("<Q", head, 24)
        str_off, str_bytes = struct.unpack_from("<QQ", head, 56)
        (data_off,) = struct.unpack_from("<Q", head, 168)
        if magic != 0x31444152 or file_bytes != st.st_size or not 176 <= data_off <= min(file_bytes, MAX_METADATA):
            raise Refused("%s: not a whole .rad container (magic, size or metadata bound)" % path)
        f.seek(0)
        meta = f.read(data_off)
    if len(meta) != data_off:
        raise Refused("%s: short read of the metadata" % path)
    model = b""
    if model_off < str_bytes and str_off + str_bytes <= data_off:
        model = meta[str_off + model_off:str_off + str_bytes].split(b"\0", 1)[0]
    ident = {"format": IDENTITY_FORMAT, "file_bytes": file_bytes, "metadata_bytes": data_off,
             "device": st.st_dev, "inode": st.st_ino, "mtime_ns": st.st_mtime_ns, "ctime_ns": st.st_ctime_ns,
             "sha256": hashlib.sha256(meta).hexdigest()}
    return ident, model.decode("utf-8", "replace")


def fold_mixed(doc, share):
    """Mixed steps' counts into decode, scaled to round(share x mixed decode rows) tokens (largest remainder per
    layer, so every layer sums to top-k x tokens); the mixed sections are zeroed. Returns the tokens folded."""
    try:
        mixed_dec = int(doc.get("mixed_decode_tokens", 0))
        m = int(round(share * mixed_dec))
        sel = doc["selections"]["mixed"]
        if m <= 0 or any(sel[l] <= 0 for l in range(NL)):
            return 0
        for l in range(NL):
            row, total, target = doc["counts"]["mixed"][l], sel[l], TOPK * m
            exact = [c * target / total for c in row]
            got = [int(x) for x in exact]
            for e in sorted(range(NE), key=lambda e: (got[e] - exact[e], e))[:target - sum(got)]:
                got[e] += 1
            doc["counts"]["decode"][l] = [a + b for a, b in zip(doc["counts"]["decode"][l], got)]
            for key, add in (("selections", target), ("observed_tokens", m),
                             ("histograms", doc["histograms"]["mixed"][l]), ("passes", doc["passes"]["mixed"][l])):
                doc[key]["decode"][l] += add
        for key in ("counts",):
            doc[key]["mixed"] = [[0] * len(r) for r in doc[key]["mixed"]]
        for key in ("selections", "observed_tokens", "histograms", "passes"):
            doc[key]["mixed"] = [0] * len(doc[key]["mixed"])
        doc["tokens"]["decode"] += m
        doc["steps"]["decode"] += doc["steps"].get("mixed", 0)
        doc["tokens"]["mixed"], doc["steps"]["mixed"], doc["mixed_decode_tokens"] = 0, 0, 0
        doc["mixed_folded"] = {"share": share, "mixed_decode_tokens": mixed_dec, "folded_tokens": m,
                               "approximate": "a mixed step's whole histogram scaled to its decode rows"}
        return m
    except (KeyError, IndexError, TypeError) as e:
        raise Refused("mixed sections incomplete (%r)" % e)


def same_generation(a, b):
    return all(a.get(k) == b.get(k) for k in PORTABLE_KEYS)


def check(doc, path, allow_mixed=False):
    """A complete pure-decode record of a 40-layer, 384-expert, top-6 trunk; returns its decode tokens."""
    def need(ok, what):
        if not ok:
            raise Refused("%s: %s" % (path, what))
    try:
        need(doc.get("format") == FORMAT, "format is not %s" % FORMAT)
        need(doc.get("final") is True, "not final (written before shutdown)")
        need(doc.get("arch") == "deepseek41" and doc.get("tp") == 4 and doc.get("n_trunk_layers") == NL
             and doc.get("n_experts") == NE, "not a DeepSeek-V4.1 EP4 record (arch, tp, trunk layers, experts)")
        need(allow_mixed or (doc.get("mixed_decode_tokens", 0) == 0 and doc["tokens"].get("mixed", 0) == 0),
             "mixed steps: their decode rows are not separable from prefill rows (--allow-mixed takes the pure "
             "decode steps alone)")
        for kind in ("credited", "unmapped", "unclassified", "dropped"):
            row = doc["ranks"][kind]
            need(len(row) == RANKS and all((n > 0) if kind == "credited" else (n == 0) for n in row),
                 "ranks.%s: lost, unmapped or unclassified routing" % kind)
        tokens = doc["tokens"]["decode"]
        need(isinstance(tokens, int) and tokens > 0 and doc["steps"]["decode"] > 0, "no decode tokens")
        counts = doc["counts"]["decode"]
        for key in ("layer_experts", "top_k", "map"):
            need(len(doc[key]) >= NL, "%s is short" % key)
        for l in range(NL):
            need(doc["layer_experts"][l] == NE and doc["top_k"][l] == TOPK and doc["map"][l] == "container",
                 "layer %d: not 384 experts, top-6, ids from the container's directory" % l)
            row = counts[l]
            need(len(row) == NE and all(isinstance(c, int) and 0 <= c <= tokens for c in row),
                 "layer %d: counts are not 384 whole numbers up to the decode tokens" % l)
            need(sum(row) == TOPK * tokens == doc["selections"]["decode"][l]
                 and doc["observed_tokens"]["decode"][l] == tokens,
                 "layer %d: counts do not sum to top-k x decode tokens" % l)
            need(0 < doc["histograms"]["decode"][l] <= doc["passes"]["decode"][l],
                 "layer %d: no histogram, or more histograms than passes" % l)
    except (KeyError, IndexError, TypeError) as e:
        raise Refused("%s: incomplete record (%r)" % (path, e))
    return tokens


def parse_weighted(arg):
    w, sep, path = arg.partition(":")
    if not sep or not w.isdigit() or not 1 <= int(w) <= MAX_WEIGHT or not path:
        raise Refused("%s: give W:FILE with W a whole number from 1 to %d" % (arg, MAX_WEIGHT))
    return int(w), path


def write(out, args, portable=False, container=None, allow_mixed=False, mixed_share=0.0):
    if len(args) < 1:
        raise Refused("write: give OUT.json W:BASE.json [W:STATS.json ...]")
    if os.path.exists(out):
        raise Refused("%s exists" % out)
    parts = []
    for a in args:
        w, path = parse_weighted(a)
        doc, sha = load(path)
        if mixed_share > 0:
            folded = fold_mixed(doc, mixed_share)
            if folded:
                print("profile_blend.py: %s: %d decode tokens of mixed steps folded in at share %g (approximate)"
                      % (path, folded, mixed_share), file=sys.stderr)
        parts.append((w, path, doc, sha, check(doc, path, allow_mixed)))
    base = json.loads(json.dumps(parts[0][2]))
    held = None
    if container:
        held, model = container_identity(container)
        if model and base.get("model") != model:
            raise Refused("%s: model %r, the container's is %r" % (parts[0][1], base.get("model"), model))
    if not isinstance(base.get("source_identity"), dict):
        if held is None:
            raise Refused("%s: the first file must be a sealed profile (it has no source_identity), or give "
                          "--container MODEL.rad to seal it" % parts[0][1])
        if not portable and base.get("container") != os.path.abspath(container):
            print("profile_blend.py: note: the capture names the container %s, sealed with the stat of %s (the same "
                  "file seen through another path, e.g. a bind mount)" % (base.get("container"), container),
                  file=sys.stderr)
        base["source_identity"] = held
    base_ident = base["source_identity"]
    if held is not None and not same_generation(base_ident, held):
        raise Refused("%s: sealed for another container generation than %s (size or metadata sha256)"
                      % (parts[0][1], container))
    for w, path, doc, sha, tokens in parts[1:]:
        for key in GEOMETRY:
            if doc.get(key) != base.get(key):
                raise Refused("%s: %s differs from the base profile's" % (path, key))
        ident = doc.get("source_identity")
        if ident is not None:
            if not same_generation(ident, base_ident) or (
                    not is_portable(doc) and not is_portable(base) and
                    (ident != base_ident or doc.get("container") != base.get("container"))):
                raise Refused("%s: sealed for another container generation than the base profile" % path)
        elif is_portable(base):
            if held is None:
                raise Refused("%s: a raw capture beside a portable base: give --container MODEL.rad (the container it "
                              "was recorded on) to check it is the same generation" % path)
        elif doc.get("container") != base.get("container") and held is None:
            raise Refused("%s: container differs from the base profile's" % path)
    total = sum(w * t for w, _, _, _, t in parts)
    result = json.loads(json.dumps(base))
    for key in ("counts",):
        result[key]["decode"] = [[sum(w * d[key]["decode"][l][e] for w, _, d, _, _ in parts) for e in range(NE)]
                                 for l in range(NL)] + result[key]["decode"][NL:]
    for key in ("observed_tokens", "selections", "histograms", "passes"):
        result[key]["decode"] = [sum(w * d[key]["decode"][l] for w, _, d, _, _ in parts)
                                 for l in range(NL)] + result[key]["decode"][NL:]
    result["tokens"]["decode"] = total
    result["steps"]["decode"] = sum(w * d["steps"]["decode"] for w, _, d, _, _ in parts)
    result["blend"] = [{"weight": w, "file": os.path.basename(path), "sha256": sha, "decode_tokens": t}
                       for w, path, _, sha, t in parts]
    result["note"] = ("profile_blend.py: decode counts of %d files, each times its weight (blend); every other "
                      "section and the source identity are the first file's" % len(parts))
    if portable:
        make_portable(result)
    check(result, out, allow_mixed)
    with open(out, "x") as f:
        json.dump(result, f)
    print("wrote %s%s: %d decode tokens from %s" % (
        out, " (portable)" if is_portable(result) else "", total,
        ", ".join("%d x %s (%d)" % (w, os.path.basename(p), t) for w, p, _, _, t in parts)))


def match(container, path):
    doc, _ = load(path)
    check(doc, path, allow_mixed=True)
    ident = doc.get("source_identity")
    if not isinstance(ident, dict):
        raise Refused("%s: not a sealed profile (no source_identity)" % path)
    held, model = container_identity(container)
    if not same_generation(ident, held) or (model and doc.get("model") != model):
        raise Refused("%s: sealed for another container generation than %s (size, metadata sha256 or model)"
                      % (path, container))
    print("%s: %s profile for %s (metadata sha256 %s)" % (
        path, "portable" if is_portable(doc) else "strict", container, held["sha256"]))


def top_units(doc, top):
    """Per rank, its `top` hottest (layer, expert) units by decode count (ties: lower expert, lower layer)."""
    counts = doc["counts"]["decode"]
    kept = set()
    for r in range(RANKS):
        units = sorted(((l, PER * r + e) for l in range(NL) for e in range(PER)),
                       key=lambda u: (-counts[u[0]][u[1]], u[1], u[0]))
        kept.update(units[:top])
    return kept


def show(paths, top):
    docs = []
    for path in paths:
        doc, _ = load(path)
        tokens = check(doc, path, allow_mixed=True)
        docs.append((os.path.basename(path), doc, tokens))
        print("%s: %d decode tokens in %d steps%s%s" % (
            docs[-1][0], tokens, doc["steps"]["decode"],
            ", portable" if is_portable(doc) else ", sealed" if isinstance(doc.get("source_identity"), dict) else "",
            "" if not doc["tokens"].get("mixed") else ", %d mixed-step tokens left out (%d of them decode rows)" % (
                doc["tokens"]["mixed"], doc.get("mixed_decode_tokens", 0))))
    if len(docs) < 2:
        return
    kept = [top_units(d, top) for _, d, _ in docs]
    print("share of the row file's decode selections on the column file's hottest %d units a rank:" % top)
    print("%-28s %s" % ("", " ".join("%8s" % ("[%d]" % j) for j in range(len(docs)))))
    for i, (name, doc, tokens) in enumerate(docs):
        counts = doc["counts"]["decode"]
        row = []
        for j in range(len(docs)):
            held = sum(counts[l][e] for l, e in kept[j])
            row.append("%8.3f" % (held / (TOPK * NL * tokens)))
        print("[%d] %-24s %s" % (i, name[:24], " ".join(row)))


def main(argv):
    try:
        if len(argv) >= 3 and argv[1] == "write":
            args, opts = argv[2:], {"portable": False, "container": None, "allow_mixed": False, "mixed_share": 0.0}
            while args and args[0].startswith("--"):
                flag = args.pop(0)
                if flag == "--portable":
                    opts["portable"] = True
                elif flag == "--allow-mixed":
                    opts["allow_mixed"] = True
                elif flag == "--container" and args:
                    opts["container"] = args.pop(0)
                elif flag == "--mixed-share" and args:
                    try:
                        opts["mixed_share"] = float(args.pop(0))
                    except ValueError:
                        opts["mixed_share"] = -1.0
                    if not 0.0 <= opts["mixed_share"] <= 1.0:
                        raise Refused("--mixed-share takes a fraction from 0 to 1")
                    opts["allow_mixed"] = True
                else:
                    raise Refused("write: unknown option %s" % flag)
            if not args:
                raise Refused("write: give OUT.json W:BASE.json [W:STATS.json ...]")
            write(args[0], args[1:], **opts)
        elif len(argv) == 3 and argv[1] == "ident":
            print(container_identity(argv[2])[0]["sha256"])
        elif len(argv) == 4 and argv[1] == "match":
            match(argv[2], argv[3])
        elif len(argv) >= 3 and argv[1] == "show":
            args, top = argv[2:], 715
            if "--top" in args:
                i = args.index("--top")
                top = int(args[i + 1])
                args = args[:i] + args[i + 2:]
                if not 1 <= top <= NL * PER:
                    raise Refused("--top: 1 to %d units a rank" % (NL * PER))
            show(args, top)
        else:
            print(__doc__)
            return 2
    except Refused as e:
        print("profile_blend.py: refused: %s" % e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
