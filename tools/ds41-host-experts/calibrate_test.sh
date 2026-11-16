#!/usr/bin/env bash
# calibrate_test.sh -- calibrate.sh end to end with no GPU, no model and no real server: a stub radiance (started
# through serve-batch.sh, as a real run starts it) answers /stats, /tokenize and /v1/completions, counts the requests,
# and on SIGINT writes a synthetic --expert-stats capture for the container it was given (a fake .rad header);
# calibrate.sh then blends it into a portable profile. Checks: the dry run, the capture launch (serve-batch.sh
# --capture), the warning under 2,000 decode tokens, mixed steps left out, --base with --weight auto, the server gone
# afterwards, and the refusals (existing OUT, another container's base). Exit 0 when every check passes.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
CAL=$HERE/calibrate.sh
T=$(mktemp -d /tmp/calibrate-test-XXXXXX)
trap 'pkill -INT -f "$T/stub-radiance" 2>/dev/null; rm -rf "$T"' EXIT
fail=0
ok()  { echo "ok   $*"; }
bad() { echo "FAIL $*"; fail=1; }
PORT=$((20000 + RANDOM % 20000))

python3 -I - "$T" <<'EOF'
import struct, sys
def rad(path, salt=b""):
    head = bytearray(4096)
    struct.pack_into("<IIQ", head, 0, 0x31444152, 2, len(head))
    struct.pack_into("<Q", head, 24, 1)
    struct.pack_into("<QQ", head, 56, 256, 64)
    struct.pack_into("<Q", head, 168, 2048)
    head[257:268] = b"deepseek41\0"
    head[600:600 + len(salt)] = salt
    open(path, "wb").write(head)
rad(sys.argv[1] + "/model.rad")
rad(sys.argv[1] + "/other.rad", b"another generation")
EOF
echo silu > "$T/silu.bin"
for i in 1 2 3 4 5; do echo "{\"prompt\": \"question $i\", \"system\": \"schema\"}"; done > "$T/prompts.jsonl"

cat > "$T/stub-radiance" <<'EOS'
#!/usr/bin/env python3
"""A stub server: the HTTP endpoints calibrate.sh and batch_client.py use; at SIGINT the --expert-stats capture."""
import http.server, json, os, signal, sys, threading
sys.path.insert(0, os.environ["TOOLS"])
import profile_blend_test as pbt
a = sys.argv[1:]
stats, model, port = a[a.index("--expert-stats") + 1], a[a.index("--model") + 1], int(a[a.index("--port") + 1])
open(os.environ["STUB_DIR"] + "/argv", "w").write("\n".join(a) + "\n")
env = {k: v for k, v in os.environ.items() if k.startswith("RADIANCE_EXPERIMENTAL_SCHED")}
open(os.environ["STUB_DIR"] + "/env", "w").write(json.dumps(env))
served = [0]
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *x): pass
    def reply(self, body):
        b = json.dumps(body).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        self.reply({"ok": True})
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/tokenize":
            self.reply({"tokens": [1] * len(body["content"].split())})
        else:
            served[0] += 1
            self.reply({"choices": [{"text": "SELECT 1"}], "usage": {"completion_tokens": body.get("max_tokens", 1)}})
srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), H)
def stop(sig, frame):
    tokens = int(os.environ.get("STUB_DECODE", "500")) * served[0]
    doc = pbt.record(tokens, 100, container=model)
    if os.environ.get("STUB_MIXED"):
        doc["tokens"]["mixed"], doc["steps"]["mixed"], doc["mixed_decode_tokens"] = 77, 4, 12
        row = [0] * 384
        for e in range(200, 206):
            row[e] = 77
        doc["counts"]["mixed"] = [list(row) for _ in range(40)]
        for key, v in (("selections", 6 * 77), ("observed_tokens", 77), ("histograms", 4), ("passes", 4)):
            doc[key]["mixed"] = [v] * 40
    json.dump(doc, open(stats, "w"))
    threading.Thread(target=srv.shutdown).start()
signal.signal(signal.SIGINT, stop)
srv.serve_forever()
EOS
chmod +x "$T/stub-radiance"
run() { TOOLS=$HERE STUB_DIR=$T RADIANCE_BIN=$T/stub-radiance MEMGUARD_MIB=0 "$CAL" --model "$T/model.rad" --silu-table "$T/silu.bin" \
          --port "$PORT" --ready-timeout 60 "$@"; }

# ---- 1. the dry run starts nothing and shows the capture launch and the portable blend
d=$(run "$T/prompts.jsonl" "$T/dry.json" --dry-run)
[[ $d == *"--expert-stats $T/dry.calibrate/capture.json"* && $d == *"--max-num-seqs 8 "* &&
   $d == *"env RADIANCE_EXPERIMENTAL_SCHED_BATCH_ADMIT=256"* && $d == *"env RADIANCE_EXPERIMENTAL_SCHED_BATCH_SPLIT=0"* &&
   $d == *"write --portable --container $T/model.rad $T/dry.json 1:$T/dry.calibrate/capture.json"* ]] &&
  ok "--dry-run: the capture launch (N8, Ba256, no Bs, --expert-stats) and the portable blend" || bad "--dry-run: $d"
[[ ! -e "$T/dry.calibrate" && ! -e "$T/dry.json" ]] && ok "--dry-run writes nothing" || bad "--dry-run wrote files"

# ---- 2. a small run: a portable profile, the warning under 2,000 tokens
out=$(STUB_DECODE=100 run "$T/prompts.jsonl" "$T/small.json" 2>&1); rc=$?
(( rc == 0 )) && [[ -s "$T/small.json" ]] && ok "a calibration run writes the profile" || bad "small run (rc $rc): $out"
[[ $out == *"captured 500 decode tokens"* && $out == *"WARNING: only 500 decode tokens"* ]] &&
  ok "it prints the decode tokens and warns under 2,000" || bad "no count/warning: $out"
python3 -I - "$T/small.json" "$T/model.rad" <<'EOF' && ok "the profile is portable, sealed with the container's header hash" || bad "not portable"
import hashlib, json, sys
d = json.load(open(sys.argv[1])); i = d["source_identity"]
h = hashlib.sha256(open(sys.argv[2], "rb").read(2048)).hexdigest()
assert d["container"] == "" and all(i[k] == 0 for k in ("device", "inode", "mtime_ns", "ctime_ns")), d["container"]
assert (i["sha256"], i["file_bytes"], i["metadata_bytes"], d["tokens"]["decode"]) == (h, 4096, 2048, 500), i
EOF
grep -qx -- '--expert-stats' "$T/argv" && grep -q '"RADIANCE_EXPERIMENTAL_SCHED_BATCH_ADMIT": "256"' "$T/env" &&
  ok "the server ran in capture mode" || bad "the server's launch: $(tr '\n' ' ' < "$T/argv")"
[[ $(wc -l < "$T/small.calibrate/replies.jsonl") == 5 ]] && ok "five replies kept in the work directory" || bad "replies"
pgrep -f "$T/stub-radiance" > /dev/null && bad "a server is left running" || ok "no server left running"

# ---- 3. --base with --weight auto and mixed steps; then serving with the result is the user's next step
out=$(STUB_DECODE=500 STUB_MIXED=1 run "$T/prompts.jsonl" "$T/blend.json" --base "$T/small.json" 2>&1); rc=$?
(( rc == 0 )) && ok "a run blended with --base" || bad "base run (rc $rc): $out"
[[ $out != *WARNING* && $out == *"captured 2500 decode tokens"* && $out == *"12 more decode tokens were in mixed steps"* ]] &&
  ok "2,500 tokens: no warning; the mixed steps' decode tokens reported" || bad "base run output: $out"
python3 -I - "$T/blend.json" <<'EOF' && ok "--weight auto: the capture x1 beside the 500-token base (rounded even share)" || bad "blend weights"
import json, sys
d = json.load(open(sys.argv[1]))
assert [b["weight"] for b in d["blend"]] == [1, 1] and d["tokens"]["decode"] == 3000 and d["container"] == "", d["blend"]
EOF
out=$(STUB_DECODE=100 run "$T/prompts.jsonl" "$T/w.json" --base "$T/blend.json" --weight 3 2>&1)
python3 -I -c "import json,sys; d=json.load(open('$T/w.json')); assert d['tokens']['decode']==3000+3*500" 2>/dev/null &&
  ok "--weight 3: the capture's counts three times" || bad "--weight 3: $out"

# ---- 4. refusals
out=$(run "$T/prompts.jsonl" "$T/small.json" 2>&1) && bad "an existing OUT accepted" ||
  { [[ $out == *exists* ]] && ok "an existing OUT refused" || bad "existing OUT: $out"; }
out=$(STUB_DECODE=100 TOOLS=$HERE STUB_DIR=$T RADIANCE_BIN=$T/stub-radiance MEMGUARD_MIB=0 "$CAL" --model "$T/other.rad" \
      --silu-table "$T/silu.bin" --port "$PORT" --ready-timeout 60 "$T/prompts.jsonl" "$T/other.json" --base "$T/small.json" 2>&1) &&
  bad "a base of another container accepted" ||
  { [[ $out == *"another container generation"* && ! -e "$T/other.json" && $out != *"starting the server"* ]] &&
      ok "a base for another container generation refused before the server starts" ||
    bad "other container: $out"; }
pgrep -f "$T/stub-radiance" > /dev/null && bad "a server is left running" || ok "no server left running after a refusal"
out=$(run "$T/prompts.jsonl" "$T/x.json" -- --capture y 2>&1) && bad "-- --capture accepted" || ok "-- --capture refused (calibrate.sh's own)"

# ---- 5. serve-batch.sh --auto-profile: the first run records a profile, the second uses it and refreshes it
AP=$T/auto
auto_run() {   # one served request, then SIGTERM to the launcher's pid; the blend runs after the server exits
  TOOLS=$HERE STUB_DIR=$T STUB_DECODE=300 STUB_MIXED=1 RADIANCE_BIN=$T/stub-radiance MEMGUARD_MIB=0 \
    "$HERE/serve-batch.sh" --model "$T/model.rad" --silu-table "$T/silu.bin" --port "$PORT" --auto-profile "$AP" \
    > "$T/auto.log" 2>&1 &
  local lp=$! t=0
  until curl -s -m 2 "http://127.0.0.1:$PORT/stats" > /dev/null 2>&1 || (( t > 100 )); do sleep 0.1; t=$((t + 1)); done
  python3 -I "$HERE/batch_client.py" "$T/prompts.jsonl" "$T/auto-replies.jsonl" --url "http://127.0.0.1:$PORT" \
    --concurrency 2 --max-tokens 4 > /dev/null
  kill -TERM "$lp"; wait "$lp"   # SIGTERM: a & job of a script ignores SIGINT; the launcher sends the server SIGINT
}
SHA=$(python3 -I "$HERE/profile_blend.py" ident "$T/model.rad")
AF=$AP/${SHA:0:16}.profile.json
auto_run
grep -q "no profile for this container" "$T/auto.log" && ! grep -qx -- '--expert-profile' "$T/argv" &&
  ok "auto-profile, first run: no profile yet, none used" || bad "auto first run: $(cat "$T/auto.log")"
grep -q "auto-profile: wrote $AF" "$T/auto.log" && [[ -s $AF ]] && ok "auto-profile, first run: $AF written" ||
  bad "auto first write: $(cat "$T/auto.log")"
python3 -I -c "import json,sys; d=json.load(open(sys.argv[1])); assert d['container']=='' and d['tokens']['decode']==1500+6 and d['mixed_folded']['folded_tokens']==6, d['tokens']" "$AF" &&
  ok "auto-profile: portable, mixed-step decode rows folded at share 0.5 (12 -> 6 tokens)" || bad "auto profile contents"
auto_run
grep -q "auto-profile: using $AF" "$T/auto.log" && grep -A1 -x -- '--expert-profile' "$T/argv" | grep -qx "$AF" &&
  ok "auto-profile, second run: the kept profile is the server's --expert-profile" || bad "auto second run: $(cat "$T/auto.log")"
python3 -I -c "import json,sys; d=json.load(open(sys.argv[1])); assert d['tokens']['decode']==2*1506, d['tokens']" "$AF" &&
  [[ -s ${AF%.profile.json}.prev.json ]] && ok "auto-profile, second run: previous x1 + this run x1, the previous kept" ||
  bad "auto second blend: $(cat "$T/auto.log")"
[[ -z $(ls -A "$AP" | grep capture) ]] && ok "auto-profile: no capture left behind" || bad "captures left: $(ls -A "$AP")"
pgrep -f "$T/stub-radiance" > /dev/null && bad "a server is left running" || ok "no server left running after auto runs"
d=$("$HERE/serve-batch.sh" --model M --silu-table S --auto-profile "$AP" --capture x --dry-run 2>&1) &&
  bad "--auto-profile with --capture accepted" || ok "--auto-profile with --capture refused"

(( fail == 0 )) && echo "calibrate_test: all passed"
exit $fail
