#!/usr/bin/env bash
# serve_batch_test.sh [ab_window.sh] -- serve-batch.sh against the window harness, with no GPU, no server and no model:
#   1. the harness's own arm loop (every action stubbed, as ab_window_ba_test.sh) builds the measured four-card batch
#      arm, and arm()'s own launch words (ab_window.sh launch_words) give the server's environment and flags;
#      serve-batch.sh --dry-run gives the same environment, each variable at its last assignment, and the same
#      flags, each at its last value (serve_batch_match.py compare), apart from the port, --radiance-home and
#      --served-model-name, and with serve_batch_match.py params' arguments all of them;
#   2. --streams 128 gives the 128-stream arm's pool (L147456, N128), --reserve-mib the mAN variable, --shared-head
#      the PchN floor, and no --profile / --mirror leaves those flags out;
#   3. a missing --silu-table or --model, and a reserve under 1024 MiB, are refused.
# Exit 0 when every check passes.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
W=${1:-$HERE/ab_window.sh}
SB=$HERE/serve-batch.sh
T=$(mktemp -d /tmp/serve-batch-test-XXXXXX)
trap 'rm -rf "$T"' EXIT
fail=0
ok()  { echo "ok   $*"; }
bad() { echo "FAIL $*"; fail=1; }

ARM=all-t40-f-M2048-H-p256-m229376-G-R-B6-W-A2-Rd13-Er-Es-Pa-K-Sc2-F8-Pp4-Ei-Pm56-Pe-Ss-Pu-Is-Pi8-Ns7-Fw-Mr32-Lc2048-P1-Qe64-Mb1-Rb-Vp-Cs-mA-Qd64-Qa0-Pc-Pch32-Pcs128-L294912-Pf-N256-C256-Ba2048-Bs-Qy60-Qyp60-Qyk-Qsp-Qt-Qz-Ee-Dt-Xqprof
ARMQW=${ARM%-Dt-Xqprof}-Qw-Dt-Xqprof; ARMQW=${ARMQW/-Qsp-Qt-Qz-/-Qsp-Qz-}   # with the WMMA P1 twin (Qw, inexact) in place of Qt: --fast-prefill 1
ARMDA=${ARM%-Xqprof}-Aw-Xqprof   # with the WMMA attention (Aw, inexact) too: --fast-attn 1
# The calibration run (serve-batch.sh --capture): 8 streams, Ba256 without Bs, routing counts (Xst), no profile.
ARMCAP=${ARM/-L294912-Pf-N256-C256-Ba2048-Bs-/-L9216-Pf-N8-C8-Ba256-}; ARMCAP=${ARMCAP%-Xqprof}-Xnp-Xst
ARMVL=${ARM%-Xqprof}-Qes-Kvl-mS16-Xqprof; ARMVL=${ARMVL/-L294912-/-Lk452-}   # VRAMLEND: --arena-split 1 --kv-lend 1 --ram-per-seq-mib 16, an Lk pool
ARMDM=${ARM%-Xqprof}-Dm384-Xqprof   # decode first at 384 ready streams (DmN, exact): --decode-min 384
mkdir -p "$T/home/kernels" "$T/mirror" "$T/profiles"; echo "{}" > "$T/profiles/prof.json"
echo "not a library" > "$T/home/kernels/libr4d.so"
echo '{}' > "$T/prompts.json"
LW=$(sed -n '/^launch_words() {/,/^}/p' "$W")
[[ $LW == *'"$BIN" --radiance-home'* ]] || bad "no launch_words in $W"
(
  set +u
  log() { echo "LOG: $*"; }
  power_set() { :; }; power_restore() { :; }
  vpack_gate() { return 0; }
  eval "$LW"
  arm() { local name=$1; shift; echo "ARM $name"; launch_words "$@"; printf '%s\n' "${LAUNCH[@]}"; }
  HX_THREADS=16 SILU=SILU RHOME=$T/home RADIANCE_HOME_DIR2=$T/home MIRROR=$T/mirror PROMPTS=$T/prompts.json
  BIN=BIN MODEL=MODEL PORT=18089 PROFILE=SEALED EXTRA_PROFILES=$T/profiles
  unset RADIANCE_TUNE
  SUM=$(sha256sum "$T/home/kernels/libr4d.so" | cut -d' ' -f1); LIB=$T/home/kernels/libr4d.so
  TALL_OK=1 TALL_WHY=test TALL_SUM=$SUM TALL_LIB=$LIB DK_OK=1 DK_WHY=test DK_SUM=$SUM DK_LIB=$LIB
  RB_OK=1 RB_WHY=test RB_SUM=$SUM RB_LIB=$LIB LCP_OK=1 LCP_WHY=test LCP_SUM=$SUM LCP_LIB=$LIB LCP_C=4096
  BAL_OK=1 BAL_WHY=test BAL_SUM=$SUM BAL_LIB=$LIB BAL_MAX_M=4096 P1_OK=1 P1_WHY=test P1_SUM=$SUM P1_LIB=$LIB
  P1T_OK=1 P1T_WHY=test P1T_SUM=$SUM P1T_LIB=$LIB P1W_OK=1 P1W_WHY=test P1W_SUM=$SUM P1W_LIB=$LIB
  DT_OK=1 DT_WHY=test DT_SUM=$SUM DT_LIB=$LIB AW_OK=1 AW_WHY=test AW_SUM=$SUM AW_LIB=$LIB
  OUT=$T/out; mkdir -p "$OUT"; CANCEL_FILE=/nonexistent/cancel; CANCEL=$CANCEL_FILE
  echo 'p1: PASS: test, full/span SwiGLU waves, also with the tail filled once' > "$OUT/devcheck-p1.log"
  echo 'p1tiled: PASS: test, also with the tail filled once' > "$OUT/devcheck-p1tiled.log"
  eval "$(sed -n '/^hotset_for() {/,/^}/p' "$W")"
  eval "$(sed -n '/^trace_base() {/,/^}/p' "$W")"
  eval "$(sed -n '/^ARM_PRESET_B2=/p;/^ARM_PRESET_B4=/p;/^arm_full() {/,/^}/p' "$W")"
  HX=(RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS=$HX_THREADS RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_SILU=$SILU)
  ARMS="$ARM $ARMQW $ARMDA $ARMCAP $ARMVL $ARMDM"
  s=$(grep -n '^for a in \$ARMS; do' "$W" | head -1 | cut -d: -f1)
  e=$(awk -v s="$s" 'NR > s && /^done$/ { print NR; exit }' "$W")
  eval "$(sed -n "${s},${e}p" "$W")"
) > "$T/loop.txt" 2>&1
grep -q "^ARM $ARM$" "$T/loop.txt" && ok "the harness launches the arm" ||
  { bad "the harness did not launch the arm: $(grep -m3 -E 'skip|unknown' "$T/loop.txt")"; }
awk -v a="ARM $ARM" '$0 == a { p = 1; next } /^ARM / { p = 0 } p && !/^LOG: /' "$T/loop.txt" > "$T/harness.txt"
grep -q "^ARM $ARMQW$" "$T/loop.txt" && ok "the harness launches the Qw arm" ||
  { bad "the harness did not launch the Qw arm: $(grep -m3 -E "arm $ARMQW" "$T/loop.txt")"; }
awk -v a="ARM $ARMQW" '$0 == a { p = 1; next } /^ARM / { p = 0 } p && !/^LOG: /' "$T/loop.txt" > "$T/harness-qw.txt"
grep -q "^ARM $ARMDA$" "$T/loop.txt" && ok "the harness launches the Dt-Aw arm" ||
  { bad "the harness did not launch the Dt-Aw arm: $(grep -m3 -E "arm $ARMDA" "$T/loop.txt")"; }
awk -v a="ARM $ARMDA" '$0 == a { p = 1; next } /^ARM / { p = 0 } p && !/^LOG: /' "$T/loop.txt" > "$T/harness-da.txt"
grep -q "^ARM $ARMDM$" "$T/loop.txt" && ok "the harness launches the Dm384 arm" ||
  { bad "the harness did not launch the Dm384 arm: $(grep -m3 -E "arm $ARMDM" "$T/loop.txt")"; }
awk -v a="ARM $ARMDM" '$0 == a { p = 1; next } /^ARM / { p = 0 } p && !/^LOG: /' "$T/loop.txt" > "$T/harness-dm.txt"

grep -q "^ARM $ARMCAP$" "$T/loop.txt" && ok "the harness launches the capture arm" ||
  { bad "the harness did not launch the capture arm: $(grep -m3 -E "arm $ARMCAP" "$T/loop.txt")"; }
awk -v a="ARM $ARMCAP" '$0 == a { p = 1; next } /^ARM / { p = 0 } p && !/^LOG: /' "$T/loop.txt" > "$T/harness-cap.txt"
awk -v a="ARM $ARMVL" '$0 == a { p = 1; next } /^ARM / { p = 0 } p && !/^LOG: /' "$T/loop.txt" > "$T/harness-vl.txt"
if [[ -s "$T/harness-vl.txt" ]] && python3 -I "$HERE/serve_batch_match.py" params "$T/harness-vl.txt" > "$T/params-vl.txt" &&
   mapfile -t P < "$T/params-vl.txt" &&
   [[ " ${P[*]} " == *" --arena-split 1 "* && " ${P[*]} " == *" --kv-lend 1 "* && " ${P[*]} " == *" --ram-per-seq-mib 16 "* && " ${P[*]} " == *" --ctx-per-stream 576 "* ]] &&
   RADIANCE_BIN=BIN "$SB" "${P[@]}" --dry-run > "$T/sb-vl.txt" && python3 -I "$HERE/serve_batch_match.py" compare "$T/harness-vl.txt" "$T/sb-vl.txt" > "$T/cmp-vl.txt"; then
  ok "the Qes-Kvl-mS16-Lk452 arm: params give --arena-split 1 --kv-lend 1 --ram-per-seq-mib 16 --ctx-per-stream 576, and serve-batch.sh with them: $(cat "$T/cmp-vl.txt")"
else bad "the VRAMLEND arm: $(grep -m2 "arm $ARMVL" "$T/loop.txt") $(tr '\n' ' ' < "$T/params-vl.txt" 2>/dev/null) $(tr '\n' ';' < "$T/cmp-vl.txt" 2>/dev/null)"; fi
dv=$("$SB" --model M --silu-table S --dry-run)
grep -qx 'env RADIANCE_EXPERIMENTAL_DS41_KV_LEND=0' <<< "$dv" && grep -qx 'env RADIANCE_EXPERIMENTAL_DS41_P1_LEND_ARENA_SPLIT=0' <<< "$dv" &&
  grep -qx 'env RADIANCE_EXPERIMENTAL_DS41_RAM_PER_SEQ_MIB=' <<< "$dv" && [[ $dv == *"--max-model-len 294912 "* ]] &&
  ok "--kv-lend, --arena-split off and no per-sequence term by default; the pool streams x 1152" || bad "the VRAMLEND defaults"
"$SB" --model M --silu-table S --kv-lend 2 --dry-run > /dev/null 2>&1 && bad "--kv-lend 2 accepted" || ok "--kv-lend 2 refused"
"$SB" --model M --silu-table S --kv-auto 8 --dry-run > /dev/null 2>&1 && bad "--kv-auto 8 accepted" || ok "--kv-auto under 16 refused"

RADIANCE_BIN=BIN "$SB" --model MODEL --silu-table SILU --profile "$T/profiles/prof.json" --mirror "$T/mirror" --exact --dry-run > "$T/sb.txt" ||
  bad "serve-batch.sh --dry-run failed"

M=$HERE/serve_batch_match.py
if python3 -I "$M" compare "$T/harness.txt" "$T/sb.txt" --port --radiance-home --served-model-name > "$T/cmp.txt"; then
  ok "$(cat "$T/cmp.txt") (the port, --radiance-home and the served name aside)"
else bad "serve-batch.sh against the arm: $(tr '\n' ';' < "$T/cmp.txt")"; fi
if python3 -I "$M" params "$T/harness.txt" > "$T/params.txt" && mapfile -t P < "$T/params.txt" &&
   RADIANCE_BIN=BIN "$SB" "${P[@]}" --dry-run > "$T/sb2.txt" && python3 -I "$M" compare "$T/harness.txt" "$T/sb2.txt" > "$T/cmp2.txt"; then
  ok "with serve_batch_match.py params' arguments: $(cat "$T/cmp2.txt"), nothing aside"
else bad "params: $(tr '\n' ' ' < "$T/params.txt") $(tr '\n' ';' < "$T/cmp2.txt" 2>/dev/null)"; fi

if python3 -I "$M" params "$T/harness-qw.txt" > "$T/params-qw.txt" && mapfile -t P < "$T/params-qw.txt" &&
   [[ " ${P[*]} " == *" --fast-prefill 1 "* ]] &&
   RADIANCE_BIN=BIN "$SB" "${P[@]}" --dry-run > "$T/sb3.txt" && python3 -I "$M" compare "$T/harness-qw.txt" "$T/sb3.txt" > "$T/cmp3.txt"; then
  ok "the Qw arm: params give --fast-prefill 1, and serve-batch.sh with them: $(cat "$T/cmp3.txt")"
else bad "the Qw arm's params: $(tr '\n' ' ' < "$T/params-qw.txt") $(tr '\n' ';' < "$T/cmp3.txt" 2>/dev/null)"; fi

if python3 -I "$M" params "$T/harness-da.txt" > "$T/params-da.txt" && mapfile -t P < "$T/params-da.txt" &&
   [[ " ${P[*]} " == *" --fast-dense 1 "* && " ${P[*]} " == *" --fast-attn 1 "* ]] &&
   RADIANCE_BIN=BIN "$SB" "${P[@]}" --dry-run > "$T/sb4.txt" && python3 -I "$M" compare "$T/harness-da.txt" "$T/sb4.txt" > "$T/cmp4.txt"; then
  ok "the Dt-Aw arm: params give --fast-dense 1 --fast-attn 1, and serve-batch.sh with them: $(cat "$T/cmp4.txt")"
else bad "the Dt-Aw arm's params: $(tr '\n' ' ' < "$T/params-da.txt") $(tr '\n' ';' < "$T/cmp4.txt" 2>/dev/null)"; fi
if python3 -I "$M" params "$T/harness-dm.txt" > "$T/params-dm.txt" && mapfile -t P < "$T/params-dm.txt" &&
   [[ " ${P[*]} " == *" --decode-min 384 "* ]] &&
   RADIANCE_BIN=BIN "$SB" "${P[@]}" --dry-run > "$T/sb5.txt" && python3 -I "$M" compare "$T/harness-dm.txt" "$T/sb5.txt" > "$T/cmp5.txt"; then
  ok "the Dm384 arm: params give --decode-min 384, and serve-batch.sh with them: $(cat "$T/cmp5.txt")"
else bad "the Dm384 arm's params: $(tr '\n' ' ' < "$T/params-dm.txt") $(tr '\n' ';' < "$T/cmp5.txt" 2>/dev/null)"; fi
[[ " $(tr '\n' ' ' < "$T/params.txt") " == *" --decode-min 0 "* ]] && ok "the arm without Dm: params give --decode-min 0" ||
  bad "the arm without Dm: $(tr '\n' ' ' < "$T/params.txt")"

# --capture against the capture arm: the same launch but for the stats file's path (the harness's xstats-<arm>.json).
if grep -q -- '^--expert-stats$' "$T/harness-cap.txt" &&
   RADIANCE_BIN=BIN "$SB" --model MODEL --silu-table SILU --mirror "$T/mirror" --capture "$T/cap.json" --dry-run > "$T/sb5.txt" &&
   python3 -I "$M" compare "$T/harness-cap.txt" "$T/sb5.txt" --port --radiance-home --served-model-name --expert-stats > "$T/cmp5.txt"; then
  ok "--capture against the capture arm (Ba256, no Bs, N8, Xst): $(cat "$T/cmp5.txt") (the stats path, port, home and name aside)"
else bad "--capture against the capture arm: $(tr '\n' ';' < "$T/cmp5.txt" 2>/dev/null)"; fi
dc=$("$SB" --model M --silu-table S --capture C.json --streams 4 --dry-run)
[[ $dc == *"--max-num-seqs 4 "* && $dc == *"--expert-stats C.json"* ]] && ok "--capture with --streams 4: N4" || bad "--capture --streams"
echo '{}' > "$T/exists.json"; echo model > "$T/model.rad"; echo silu > "$T/silu.bin"
de=$("$SB" --model "$T/model.rad" --silu-table "$T/silu.bin" --capture "$T/exists.json" 2>&1) &&
  bad "an existing --capture file accepted" || { [[ $de == *exists* ]] && ok "an existing --capture file refused" || bad "--capture refusal: $de"; }

# ---- 2. the parameters
d128=$("$SB" --model M --silu-table S --streams 128 --dry-run)
[[ $d128 == *"--max-model-len 147456 "* && $d128 == *"--max-num-seqs 128 "* ]] && ok "128 streams: L147456, N128" ||
  bad "128 streams: $(grep '^cmd' <<< "$d128")"
dres=$("$SB" --model M --silu-table S --reserve-mib 20480 --shared-head 64 --port 9000 --dry-run)
grep -qx 'env RADIANCE_EXPERIMENTAL_DS41_RAM_RESERVE_MIB=20480' <<< "$dres" && ok "--reserve-mib: the mAN variable" ||
  bad "--reserve-mib"
grep -qx 'env RADIANCE_EXPERIMENTAL_PREFIX_SHARED_HEAD=64' <<< "$dres" && ok "--shared-head 64: Pch64's floor" ||
  bad "--shared-head"
[[ $dres == *"--port 9000"* ]] && ok "--port" || bad "--port"
d0=$("$SB" --model M --silu-table S --dry-run)
if grep -q 'RAM_RESERVE_MIB' <<< "$d0" || [[ $d0 == *--expert-profile* || $d0 == *--expert-plane-mirror* ]]; then
  bad "the defaults add a reserve, a profile or a mirror"
else ok "the defaults: no reserve variable, no --expert-profile, no mirror"; fi
grep -qx 'env RADIANCE_EXPERIMENTAL_PREFIX_SHARED_HEAD=32' <<< "$d0" && ok "the shared-head floor's default: 32" ||
  bad "the shared-head default"

dh=$("$SB" --model M --silu-table S --dry-run)
grep -qx 'env RADIANCE_EXPERIMENTAL_DS41_P1_HYBRID=60' <<< "$dh" && grep -qx 'env RADIANCE_EXPERIMENTAL_DS41_P1_HYBRID_ADAPT=' <<< "$dh" &&
  ok "hybrid decode by default: Qy60, the share adapting" || bad "the hybrid default"
d0h=$("$SB" --model M --silu-table S --hybrid 0 --max-seq-len 4096 --dry-run)
grep -qx 'env RADIANCE_EXPERIMENTAL_DS41_P1_HYBRID=0' <<< "$d0h" && [[ $d0h == *"--max-seq-len 4096"* ]] &&
  ok "--hybrid 0: off (an arm without Qy); --max-seq-len 4096: SlN's flag" || bad "--hybrid 0 / --max-seq-len"
[[ $dh != *--max-seq-len* ]] && ok "no --max-seq-len by default" || bad "a --max-seq-len by default"

dq=$("$SB" --model M --silu-table S --tiled 0 --tail-once 0 --swiglu-span 0 --dry-run)
dx=$("$SB" --model M --silu-table S --reproducible --dry-run)
grep -qx 'env RADIANCE_EXPERIMENTAL_DS41_P1_TILED=1' <<< "$dx" && grep -qx 'env RADIANCE_EXPERIMENTAL_DS41_P1_TAIL_ONCE=1' <<< "$dh" &&
  grep -qx 'env RADIANCE_EXPERIMENTAL_DS41_P1_SWIGLU_SPAN=1' <<< "$dh" && ok "Qz, Qsp on by default; Qt on with --exact" || bad "the Qt/Qz/Qsp defaults"
grep -qx 'env RADIANCE_EXPERIMENTAL_DS41_P1_TILED=0' <<< "$dq" && grep -qx 'env RADIANCE_EXPERIMENTAL_DS41_P1_TAIL_ONCE=0' <<< "$dq" &&
  grep -qx 'env RADIANCE_EXPERIMENTAL_DS41_P1_SWIGLU_SPAN=0' <<< "$dq" && ok "--tiled 0 --tail-once 0 --swiglu-span 0: off" ||
  bad "--tiled / --tail-once / --swiglu-span"

grep -qx 'env RADIANCE_EXPERIMENTAL_DS41_P1_WMMA=1' <<< "$dh" && grep -qx 'env RADIANCE_EXPERIMENTAL_DS41_P1_TILED=0' <<< "$dh" &&
  grep -qx 'env RADIANCE_EXPERIMENTAL_DS41_P1_WMMA=0' <<< "$dx" && grep -qx 'env RADIANCE_EXPERIMENTAL_DS41_ATTN_WMMA=0' <<< "$dx" &&
  ok "--fast-prefill on by default (Qw, Qt off); --exact turns Qw and Aw off" ||
  bad "the --fast-prefill default"
dw=$("$SB" --model M --silu-table S --fast-prefill 1 --dry-run)
grep -qx 'env RADIANCE_EXPERIMENTAL_DS41_P1_WMMA=1' <<< "$dw" && grep -qx 'env RADIANCE_EXPERIMENTAL_DS41_P1_TILED=0' <<< "$dw" &&
  ok "--fast-prefill 1: Qw (..._P1_WMMA=1), the tiled twin's switch off (the core refuses both)" || bad "--fast-prefill 1"

grep -qx 'env RADIANCE_EXPERIMENTAL_DS41_DENSE_TILED=1' <<< "$dh" && grep -qx 'env RADIANCE_EXPERIMENTAL_DS41_ATTN_WMMA=1' <<< "$dh" &&
  ok "--fast-dense and --fast-attn on by default (Dt, Aw: ..._DENSE_TILED=1, ..._ATTN_WMMA=1)" || bad "the --fast-dense / --fast-attn defaults"
dda=$("$SB" --model M --silu-table S --fast-prefill 0 --fast-dense 1 --fast-attn 1 --dry-run)
grep -qx 'env RADIANCE_EXPERIMENTAL_DS41_DENSE_TILED=1' <<< "$dda" && grep -qx 'env RADIANCE_EXPERIMENTAL_DS41_ATTN_WMMA=1' <<< "$dda" &&
  grep -qx 'env RADIANCE_EXPERIMENTAL_DS41_P1_WMMA=0' <<< "$dda" && ok "--fast-prefill 0 --fast-dense 1 --fast-attn 1: Dt and Aw, Qw off" ||
  bad "--fast-dense 1 / --fast-attn 1"

grep -qx 'env RADIANCE_EXPERIMENTAL_SCHED_DECODE_MIN=0' <<< "$dh" && ok "--decode-min off by default (DmN: ..._DECODE_MIN=0)" ||
  bad "the --decode-min default"
ddm=$("$SB" --model M --silu-table S --decode-min 384 --dry-run)
grep -qx 'env RADIANCE_EXPERIMENTAL_SCHED_DECODE_MIN=384' <<< "$ddm" && ok "--decode-min 384: Dm384" || bad "--decode-min 384"

# ---- 3. refusals
"$SB" --model M --silu-table S --decode-min x --dry-run > /dev/null 2>&1 && bad "--decode-min x accepted" || ok "--decode-min x refused"
"$SB" --model M --silu-table S --fast-dense 2 --dry-run > /dev/null 2>&1 && bad "--fast-dense 2 accepted" || ok "--fast-dense 2 refused"
"$SB" --model M --silu-table S --fast-attn x --dry-run > /dev/null 2>&1 && bad "--fast-attn x accepted" || ok "--fast-attn x refused"
"$SB" --model M --silu-table S --fast-prefill 2 --dry-run > /dev/null 2>&1 && bad "--fast-prefill 2 accepted" || ok "--fast-prefill 2 refused"
"$SB" --model M --silu-table S --tiled 2 --dry-run > /dev/null 2>&1 && bad "--tiled 2 accepted" || ok "--tiled 2 refused"
"$SB" --model M --silu-table S --hybrid 401 --dry-run > /dev/null 2>&1 && bad "--hybrid 401 accepted" || ok "--hybrid above 400 refused"
"$SB" --model M --dry-run > /dev/null 2>&1 && bad "no --silu-table accepted" || ok "no --silu-table refused"
"$SB" --silu-table S --dry-run > /dev/null 2>&1 && bad "no --model accepted" || ok "no --model refused"
"$SB" --model M --silu-table S --reserve-mib 512 --dry-run > /dev/null 2>&1 && bad "a 512 MiB reserve accepted" ||
  ok "a reserve under 1024 MiB refused"
"$SB" --model "$T/none.rad" --silu-table S > /dev/null 2>&1 && bad "a missing container accepted" ||
  ok "a missing container refused before any start"

# ---- 4. a real start (a stub server): serve-batch.sh becomes the server (exec, the same pid), so the SIGINT a caller
# sends the pid it started reaches the server, and nothing is left behind (window H28's orphan).
cat > "$T/stub-radiance" <<'EOS'
#!/usr/bin/env python3
import os, signal, sys, time
d = os.environ["STUB_DIR"]
open(os.path.join(d, "pid"), "w").write("%d\n" % os.getpid())
def stop(sig, frame):
    open(os.path.join(d, "signal"), "w").write("%d\n" % sig)
    sys.exit(0)
signal.signal(signal.SIGINT, stop)
while True:
    time.sleep(0.1)
EOS
chmod +x "$T/stub-radiance"; echo model > "$T/model.rad"; echo silu > "$T/silu.bin"
STUB_DIR=$T RADIANCE_BIN=$T/stub-radiance "$SB" --model "$T/model.rad" --silu-table "$T/silu.bin" > "$T/live.log" 2>&1 &
lp=$!
for _ in $(seq 100); do [[ -s "$T/pid" ]] && break; sleep 0.1; done
[[ "$(cat "$T/pid" 2>/dev/null)" == "$lp" ]] && ok "the server runs as the launcher's own pid ($lp)" ||
  bad "the server's pid $(cat "$T/pid" 2>/dev/null) is not the launcher's $lp"
kill -INT "$lp"
for _ in $(seq 50); do kill -0 "$lp" 2>/dev/null || break; sleep 0.1; done
if kill -0 "$lp" 2>/dev/null; then bad "a SIGINT to the launcher's pid did not stop the server"; kill -KILL "$lp"
else ok "a SIGINT to the launcher's pid stops the server (signal $(cat "$T/signal" 2>/dev/null))"; fi
wait "$lp" 2>/dev/null
left=$(pgrep -f "$T/stub-radiance" | tr '\n' ' ')
[[ -z "$left" ]] && ok "nothing of the server left" || { bad "left behind: $left"; kill -KILL $left 2>/dev/null; }

(( fail == 0 )) && echo "serve_batch_test: all passed"
exit $fail
