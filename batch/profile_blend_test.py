"""profile_blend_test.py -- profile_blend.py on synthetic --expert-stats records.

  python3 -I profile_blend_test.py [-v]
"""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import profile_blend as pb  # noqa: E402

IDENTITY = {"format": "rad-metadata-prefix-sha256/1", "file_bytes": 10, "metadata_bytes": 5, "device": 1,
            "inode": 2, "mtime_ns": 3, "ctime_ns": 4, "sha256": "ab" * 32}


def record(tokens, hot, steps=None, sealed=False, **over):
    """A pure-decode record of `tokens` tokens: every token routes each layer to the six experts hot .. hot+5."""
    row = [0] * 384
    for e in range(hot, hot + 6):
        row[e % 384] = tokens
    steps = steps or tokens
    doc = {
        "format": pb.FORMAT, "model": "deepseek41", "arch": "deepseek41", "container": "/models/m.rad", "tp": 4,
        "final": True, "n_layers": 40, "n_trunk_layers": 40, "n_experts": 384,
        "layer_experts": [384] * 40, "top_k": [6] * 40, "map": ["container"] * 40,
        "kinds": ["prefill", "decode", "mixed", "draft"],
        "steps": {"prefill": 2, "decode": steps, "mixed": 0, "draft": 0},
        "tokens": {"prefill": 30, "decode": tokens, "mixed": 0, "draft": 0}, "mixed_decode_tokens": 0,
        "ranks": {"credited": [5] * 4, "unmapped": [0] * 4, "unclassified": [0] * 4, "dropped": [0] * 4},
    }
    for key, per_layer in (("histograms", steps), ("passes", steps), ("observed_tokens", tokens),
                           ("selections", 6 * tokens)):
        doc[key] = {"prefill": [7] * 40, "decode": [per_layer] * 40, "mixed": [0] * 40, "draft": [0] * 40}
    doc["counts"] = {"prefill": [[1] * 384 for _ in range(40)], "decode": [list(row) for _ in range(40)],
                     "mixed": [[0] * 384 for _ in range(40)], "draft": [[0] * 384 for _ in range(40)]}
    if sealed:
        doc["source_identity"] = dict(IDENTITY)
    doc.update(over)
    return doc


class Blend(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir)

    def put(self, name, doc):
        path = os.path.join(self.dir, name)
        with open(path, "w") as f:
            json.dump(doc, f)
        return path

    def run_tool(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = pb.main(["profile_blend.py"] + list(argv))
        return rc, out.getvalue(), err.getvalue()

    def test_weighted_sum_is_a_valid_profile(self):
        base = self.put("base.json", record(10, 0, sealed=True))
        extra = self.put("extra.json", record(8, 100, steps=2))
        out = os.path.join(self.dir, "out.json")
        rc, text, err = self.run_tool("write", out, "3:" + base, "2:" + extra)
        self.assertEqual((rc, err), (0, ""))
        with open(out) as f:
            doc = json.load(f)
        self.assertEqual(doc["tokens"]["decode"], 3 * 10 + 2 * 8)
        self.assertEqual(doc["steps"]["decode"], 3 * 10 + 2 * 2)
        self.assertEqual(doc["counts"]["decode"][7][0], 30)
        self.assertEqual(doc["counts"]["decode"][7][100], 16)
        self.assertEqual(sum(doc["counts"]["decode"][39]), 6 * 46)
        self.assertEqual(doc["selections"]["decode"], [6 * 46] * 40)
        self.assertEqual(doc["observed_tokens"]["decode"], [46] * 40)
        self.assertEqual(doc["histograms"]["decode"], [34] * 40)
        self.assertEqual(doc["source_identity"], IDENTITY)
        self.assertEqual(doc["counts"]["prefill"][0][0], 1)          # the base's other sections are kept
        self.assertEqual([b["weight"] for b in doc["blend"]], [3, 2])
        self.assertEqual([b["file"] for b in doc["blend"]], ["base.json", "extra.json"])   # no directories
        self.assertEqual(pb.check(doc, out), 46)

    def test_refusals(self):
        base = self.put("base.json", record(10, 0, sealed=True))
        out = os.path.join(self.dir, "out.json")
        cases = {
            "the first file must be a sealed profile": ["1:" + self.put("raw.json", record(10, 0))],
            "container differs": ["1:" + base, "1:" + self.put("c.json", record(5, 0, container="/other.rad"))],
            "not final": ["1:" + base, "1:" + self.put("f.json", record(5, 0, final=False))],
            "mixed steps": ["1:" + base, "1:" + self.put("m.json", record(5, 0, mixed_decode_tokens=3))],
            "lost, unmapped": ["1:" + base, "1:" + self.put("u.json", record(
                5, 0, ranks={"credited": [5] * 4, "unmapped": [0, 1, 0, 0], "unclassified": [0] * 4,
                             "dropped": [0] * 4}))],
            "give W:FILE": ["0:" + base],
            "another container generation": ["1:" + base, "1:" + self.put("g.json", record(
                5, 0, source_identity=dict(IDENTITY, inode=99)))],
        }
        bad = record(5, 0)
        bad["counts"]["decode"][3][200] = 1
        cases["counts do not sum"] = ["1:" + base, "1:" + self.put("s.json", bad)]
        for why, args in cases.items():
            rc, text, err = self.run_tool("write", out, *args)
            self.assertEqual(rc, 1, why)
            self.assertIn(why, err)
            self.assertFalse(os.path.exists(out), why)
        self.assertEqual(self.run_tool("write", out, "1:" + base)[0], 0)
        rc, text, err = self.run_tool("write", out, "1:" + base)
        self.assertEqual(rc, 1)
        self.assertIn("exists", err)

    def fake_rad(self, name, salt=b""):
        """A .rad header the tool reads: magic, size, model name "deepseek41" in the string blob, data_off 512."""
        import struct
        head = bytearray(1024)
        struct.pack_into("<IIQ", head, 0, 0x31444152, 2, len(head))
        struct.pack_into("<Q", head, 24, 1)                     # model_name: offset 1 in the string blob
        struct.pack_into("<QQ", head, 56, 256, 64)              # str_off, str_bytes
        struct.pack_into("<Q", head, 168, 512)                  # data_off: the metadata the server hashes
        head[257:268] = b"deepseek41\0"
        head[300:300 + len(salt)] = salt
        path = os.path.join(self.dir, name)
        with open(path, "wb") as f:
            f.write(head)
        return path

    def test_portable(self):
        rad = self.fake_rad("m.rad")
        ident, model = pb.container_identity(rad)
        self.assertEqual(model, "deepseek41")
        self.assertEqual((ident["file_bytes"], ident["metadata_bytes"]), (1024, 512))
        import hashlib
        with open(rad, "rb") as f:
            self.assertEqual(ident["sha256"], hashlib.sha256(f.read(512)).hexdigest())
        capture = self.put("cap.json", record(10, 0, container=rad))
        # A raw capture is sealed by --container: strict (this machine) or portable.
        strict = os.path.join(self.dir, "strict.json")
        rc, text, err = self.run_tool("write", "--container", rad, strict, "1:" + capture)
        self.assertEqual(rc, 0, err)
        with open(strict) as f:
            doc = json.load(f)
        self.assertEqual(doc["source_identity"], ident)
        self.assertEqual(doc["container"], rad)
        self.assertFalse(pb.is_portable(doc))
        shared = os.path.join(self.dir, "shared.json")
        rc, text, err = self.run_tool("write", "--portable", "--container", rad, shared, "1:" + capture)
        self.assertEqual((rc, err), (0, ""))
        self.assertIn("(portable)", text)
        with open(shared) as f:
            doc = json.load(f)
        self.assertTrue(pb.is_portable(doc))
        self.assertEqual(doc["container"], "")
        self.assertEqual({k: doc["source_identity"][k] for k in pb.STAT}, dict.fromkeys(pb.STAT, 0))
        self.assertEqual({k: doc["source_identity"][k] for k in pb.PORTABLE_KEYS},
                         {k: ident[k] for k in pb.PORTABLE_KEYS})
        # A portable base blends with a capture of the same container (checked through --container) ...
        other = self.put("cap2.json", record(4, 100, container="/inside/docker/m.rad"))
        out = os.path.join(self.dir, "blend.json")
        rc, text, err = self.run_tool("write", out, "2:" + shared, "1:" + other)
        self.assertEqual(rc, 1)
        self.assertIn("give --container", err)
        rc, text, err = self.run_tool("write", "--portable", "--container", rad, out, "2:" + shared, "1:" + other)
        self.assertEqual((rc, err), (0, ""))
        with open(out) as f:
            doc = json.load(f)
        self.assertTrue(pb.is_portable(doc))
        self.assertEqual(doc["tokens"]["decode"], 24)
        self.assertEqual(doc["counts"]["decode"][0][100], 4)
        # ... and with a profile sealed strictly on another machine for the same generation.
        os.unlink(out)
        rc, text, err = self.run_tool("write", out, "1:" + shared, "1:" + strict)
        self.assertEqual((rc, err), (0, ""))
        rc, text, err = self.run_tool("match", rad, shared)
        self.assertEqual(rc, 0, err)
        self.assertIn("portable profile for", text)
        self.assertEqual(self.run_tool("match", rad, strict)[0], 0)
        # Another container generation is refused.
        rc, text, err = self.run_tool("match", self.fake_rad("m3.rad", salt=b"x"), shared)
        self.assertEqual(rc, 1)
        self.assertIn("another container generation", err)
        rad2 = self.fake_rad("m2.rad", salt=b"another")
        os.unlink(out)
        rc, text, err = self.run_tool("write", "--container", rad2, out, "1:" + shared)
        self.assertEqual(rc, 1)
        self.assertIn("another container generation", err)
        rc, text, err = self.run_tool("write", "--portable", "--container", rad2, out, "1:" + capture)
        self.assertEqual(rc, 0, err)
        os.unlink(out)
        rc, text, err = self.run_tool("write", out, "1:" + shared, "1:" + self.put(
            "g.json", record(5, 0, source_identity=dict(IDENTITY))))
        self.assertEqual(rc, 1)
        self.assertIn("another container generation", err)
        # Not a container at all.
        rc, text, err = self.run_tool("write", "--container", capture, out, "1:" + capture)
        self.assertEqual(rc, 1)
        self.assertIn("not a", err)

    def test_allow_mixed(self):
        base = self.put("base.json", record(10, 0, sealed=True))
        mixed = record(5, 0)
        mixed["tokens"]["mixed"], mixed["mixed_decode_tokens"] = 40, 3
        m = self.put("m.json", mixed)
        out = os.path.join(self.dir, "out.json")
        self.assertEqual(self.run_tool("write", out, "1:" + base, "1:" + m)[0], 1)
        rc, text, err = self.run_tool("write", "--allow-mixed", out, "1:" + base, "1:" + m)
        self.assertEqual((rc, err), (0, ""))
        with open(out) as f:
            self.assertEqual(json.load(f)["tokens"]["decode"], 15)
        rc, text, err = self.run_tool("show", m)
        self.assertIn("40 mixed-step tokens left out (3 of them decode rows)", text)

    def test_mixed_share(self):
        base = self.put("base.json", record(10, 0, sealed=True))
        mixed = record(5, 0)
        # 4 mixed steps of 50 rows (48 prefill + 2 decode each): 300 selections a layer, half on experts 200..205.
        mixed["tokens"]["mixed"], mixed["steps"]["mixed"], mixed["mixed_decode_tokens"] = 200, 4, 8
        row = [0] * 384
        for e in range(200, 206):
            row[e] = 25
        for e in range(300, 306):
            row[e] = 25
        mixed["counts"]["mixed"] = [list(row) for _ in range(40)]
        mixed["selections"]["mixed"] = [300] * 40
        mixed["observed_tokens"]["mixed"] = [200] * 40
        mixed["histograms"]["mixed"] = [4] * 40
        mixed["passes"]["mixed"] = [4] * 40
        m = self.put("m.json", mixed)
        out = os.path.join(self.dir, "out.json")
        rc, text, err = self.run_tool("write", "--mixed-share", "0.5", out, "1:" + base, "1:" + m)
        self.assertEqual(rc, 0, err)
        self.assertIn("4 decode tokens of mixed steps folded in at share 0.5 (approximate)", err)
        with open(out) as f:
            doc = json.load(f)
        self.assertEqual(doc["tokens"]["decode"], 10 + 5 + 4)       # half of the 8 mixed-step decode rows
        self.assertEqual(sum(doc["counts"]["decode"][0][200:206]) + sum(doc["counts"]["decode"][0][300:306]), 24)
        self.assertEqual(pb.check(doc, out), 19)
        # The folded capture's mixed sections are zero: blending it again folds nothing twice.
        folded = json.loads(json.dumps(mixed))
        self.assertEqual(pb.fold_mixed(folded, 0.5), 4)
        self.assertEqual(pb.fold_mixed(folded, 0.5), 0)
        self.assertEqual(folded["mixed_folded"]["folded_tokens"], 4)
        self.assertEqual(self.run_tool("write", "--mixed-share", "2", out + "2", "1:" + base)[0], 1)

    def test_show_overlap(self):
        a = self.put("a.json", record(10, 10, sealed=True))      # experts 10..15: rank 0
        b = self.put("b.json", record(20, 150))                   # experts 150..155: rank 1
        rc, text, err = self.run_tool("show", a, b, "--top", "240")
        self.assertEqual((rc, err), (0, ""))
        self.assertIn("a.json: 10 decode tokens in 10 steps, sealed", text)
        self.assertIn("b.json: 20 decode tokens in 20 steps", text)
        rows = [l.split() for l in text.splitlines() if l.startswith("[")]
        # Each file's own top 240 a rank (6 experts x 40 layers) hold all of its selections. In the rank
        # where a file counted nothing its top 240 are the rank's six lowest experts (the tie order), which
        # are not the other file's.
        self.assertEqual(rows[0][-2:], ["1.000", "0.000"])
        self.assertEqual(rows[1][-2:], ["0.000", "1.000"])


if __name__ == "__main__":
    unittest.main()
