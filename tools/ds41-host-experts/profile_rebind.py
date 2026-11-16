#!/usr/bin/env python3
"""Strip a placement profile (--expert-profile) for sharing, or bind one to your own copy of the model container.

A profile names the container it was recorded on by the SHA-256 of its metadata header (the first metadata_bytes
bytes, identical in every byte-identical rebuild per RECIPE.md), its size, and, in the strict form, the path and the
local file's device, inode and timestamps. The server accepts the portable form, container "" and those four 0, on
any machine whose container has the same size, metadata size and header hash (it logs "expert profile: portable
(header sha256 ...)"), so a shared profile needs no binding: `strip` makes one. `bind` is optional: it checks the
header hash and size against your container (reading only the header, never the weights) and fills in the path and
stat, which makes the server also refuse the profile if the file is replaced. The counts are untouched.

  profile_rebind.py strip PROFILE.json OUT.json
      Blanks the path and stat fields (and any `blend` file names): a portable profile.
  profile_rebind.py bind  PROFILE.json CONTAINER.rad OUT.json [--as PATH]
      PATH: the container's path as the server will be given it (e.g. inside Docker); default CONTAINER.rad's
      absolute path. The file's stat fields are read from CONTAINER.rad (a bind mount keeps them)."""
import hashlib, json, os, sys

STAT = ("device", "inode", "mtime_ns", "ctime_ns")


def die(msg):
    sys.exit("profile_rebind: " + msg)


def load(p):
    d = json.load(open(p))
    if d.get("format") != "radiance-expert-stats/1" or "source_identity" not in d:
        die("%s is not a sealed radiance-expert-stats/1 profile" % p)
    return d


def write(d, out):
    if os.path.exists(out):
        die("%s exists" % out)
    with open(out, "w") as f:
        json.dump(d, f, separators=(",", ":"))


def bind(prof, container, out, as_path):
    d = load(prof); ident = d["source_identity"]
    st = os.stat(container)
    if st.st_size != ident["file_bytes"]:
        die("container is %d bytes, the profile's is %d" % (st.st_size, ident["file_bytes"]))
    with open(container, "rb") as f:
        head = f.read(ident["metadata_bytes"])
    got = hashlib.sha256(head).hexdigest()
    if len(head) != ident["metadata_bytes"] or got != ident["sha256"]:
        die("container metadata sha256 %s differs from the profile's %s: not the same container" % (got, ident["sha256"]))
    ident.update(device=st.st_dev, inode=st.st_ino, mtime_ns=st.st_mtime_ns, ctime_ns=st.st_ctime_ns)
    d["container"] = as_path or os.path.abspath(container)
    write(d, out)
    print("bound to %s (metadata sha256 %s...)" % (d["container"], got[:16]))


def strip(prof, out):
    d = load(prof)
    d["container"] = ""
    for k in STAT:
        d["source_identity"][k] = 0
    for b in d.get("blend") or []:
        if isinstance(b, dict) and "file" in b:
            b["file"] = os.path.basename(b["file"])
    write(d, out)
    print("stripped -> %s" % out)


a = sys.argv[1:]
if len(a) >= 4 and a[0] == "bind":
    bind(a[1], a[2], a[3], a[5] if len(a) > 5 and a[4] == "--as" else None)
elif len(a) == 3 and a[0] == "strip":
    strip(a[1], a[2])
else:
    sys.exit(__doc__)
