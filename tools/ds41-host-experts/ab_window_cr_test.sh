#!/usr/bin/env bash
# ab_window.sh's CrG part (AB_REFILL=G: the arm's K requests kept in flight as a pool refilled G at a time):
# the request-json gate on synthetic results, the request launch stating AB_REFILL in every arm, and the arm
# loop with a stubbed arm(). No GPU, server or model is opened. Run under the box build lock like every other
# test.
set -u
W=${1:-$(dirname "$0")/ab_window.sh}
T=$(mktemp -d /tmp/ab-cr-test-XXXXXX)
trap 'rm -rf "$T"' EXIT
fail=0
ok() { echo "ok   $*"; }
bad() { echo "FAIL $*"; fail=1; }
eval "$(sed -n '/^refill_gate() {/,/^}/p' "$W")"
declare -F refill_gate >/dev/null || { echo "FAIL no refill_gate in $W"; exit 2; }

expect() { # status, description, command and arguments
  local want=$1 what=$2 why rc
  shift 2
  why=$("$@"); rc=$?
  if [[ $rc == "$want" ]]; then ok "$what${why:+ ($why)}"
  else bad "$what: exit $rc, expected $want${why:+ ($why)}"; fi
}
echo '{"requests": [], "waves": [{"wall_s": 1, "tokens": 2, "refill": 16}], "concurrency": 256, "refill": 16}' >"$T/pool16.json"
echo '{"requests": [], "waves": [{"wall_s": 1, "tokens": 2, "refill": 1}], "concurrency": 256, "refill": 1}' >"$T/pool1.json"
echo '{"requests": [], "waves": [{"wall_s": 1, "tokens": 2}], "concurrency": 256}' >"$T/waves.json"
expect 0 'Cr16 with a pool of 16'                   refill_gate "$T/pool16.json" 16
expect 1 'Cr16 with a pool of 1 refuses'            refill_gate "$T/pool1.json" 16
expect 1 'Cr16 with waves refuses'                  refill_gate "$T/waves.json" 16
expect 0 'no Cr part with waves'                    refill_gate "$T/waves.json" ''
expect 1 'no Cr part but a pool refuses'            refill_gate "$T/pool16.json" ''
expect 1 'no request json refuses'                  refill_gate "$T/none.json" ''

# The request launch: every arm states AB_REFILL (empty without the part), so a window's AB_REFILL cannot
# turn the pool on in an arm that does not name it.
if grep -qF 'env AB_CONCURRENCY=${ARM_CONC:-${AB_CONCURRENCY:-1}} AB_REFILL=${ARM_CR:-} ' "$W"; then
  ok 'the request launch states AB_REFILL=${ARM_CR:-}'
else bad 'the request launch does not state AB_REFILL from the arm'; fi

# The parser: CrG states G to arm(); without a concurrency of at least 2, past it, twice, without a number,
# beside F5, or with a receipt that sends replies back, the arm is skipped with the reason.
cat >"$T/prompts.json" <<'EOF'
{"benchmark_requests": [{"name": "a-0", "request": {"prompt": [1], "max_tokens": 2}}]}
EOF
cat >"$T/turns.json" <<'EOF'
{"benchmark_requests": [{"name": "t-0", "request": {"chat_turns": [{"user": "x"}]}},
  {"name": "t-1", "request": {"chat_turns": [{"user": "x"}, {"assistant_from": "t-0"}, {"user": "y"}]}}]}
EOF
H=$T/home
mkdir -p "$H/kernels"
B=all-t40-f-M16-H-p256-m98304-G-R-B1-W-A2-Rd13-Er-Es-Pa-K-Sc2-Pp4-Ei-Pm56-Pe-N256-C256-L720896
run_loop() {   # prompts, arms
  (
    set +u
    log() { echo "LOG: $*"; }
    power_set() { :; }; power_restore() { :; }; vpack_gate() { return 0; }
    arm() { echo "ARM $1 CR=${ARM_CR:-} CONC=${ARM_CONC:-}"; }
    HX_THREADS=16 SILU=/dev/null RHOME=$H RADIANCE_HOME_DIR2=$H MIRROR=$T MAX_SEQS=1
    PROMPTS=$1 OUT=$T/out CANCEL_FILE=/nonexistent/cancel CANCEL=$CANCEL_FILE
    mkdir -p "$OUT"
    unset RADIANCE_TUNE AB_CONCURRENCY
    ARMS=$2
    eval "$(sed -n '/^hotset_for() {/,/^}/p' "$W")"
    eval "$(sed -n '/^trace_base() {/,/^}/p' "$W")"
    eval "$(sed -n '/^ARM_PRESET_B2=/p;/^ARM_PRESET_B4=/p;/^arm_full() {/,/^}/p' "$W")"
    HX=(RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS=$HX_THREADS RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_SILU=$SILU)
    s=$(grep -n '^for a in \$ARMS; do' "$W" | head -1 | cut -d: -f1)
    e=$(awk -v s="$s" 'NR > s && /^done$/ { print NR; exit }' "$W")
    eval "$(sed -n "${s},${e}p" "$W")"
  ) 2>&1
}
run_loop "$T/prompts.json" "$B $B-Cr1 $B-Cr16 $B-Cr256 $B-Cr257 $B-Cr16-Cr8 ${B%-C256-L720896}-L720896-Cr4 $B-Cr0" >"$T/loop.log"
grep -qxF "ARM $B CR= CONC=256" "$T/loop.log" && ok 'plain arm states no Cr' || bad 'plain arm header'
for g in 1 16 256; do
  grep -qxF "ARM $B-Cr$g CR=$g CONC=256" "$T/loop.log" && ok "Cr$g states G=$g" || bad "Cr$g header"
done
for x in "$B-Cr257" "$B-Cr16-Cr8" "${B%-C256-L720896}-L720896-Cr4" "$B-Cr0"; do
  if grep -q "^ARM $x " "$T/loop.log"; then bad "$x launched"; else ok "$x skipped"; fi
done
grep -q 'Cr257 refills more slots than the arm.s concurrency 256' "$T/loop.log" && ok 'past-K reason logged' || bad 'past-K reason missing'
grep -q 'duplicate Cr part' "$T/loop.log" && ok 'duplicate reason logged' || bad 'duplicate reason missing'
grep -q 'Cr4 needs a concurrency of at least 2' "$T/loop.log" && ok 'no-concurrency reason logged' || bad 'no-concurrency reason missing'
run_loop "$T/turns.json" "$B-Cr1 $B" >"$T/turns.log"
if grep -q "^ARM $B-Cr1 " "$T/turns.log"; then bad 'Cr1 with replies sent back launched'
elif grep -q 'send earlier replies back (assistant_from)' "$T/turns.log"; then ok 'Cr1 with replies sent back skipped with the reason'
else bad 'Cr1 with replies sent back: no reason logged'; fi
grep -qxF "ARM $B CR= CONC=256" "$T/turns.log" && ok 'the same receipt in waves still runs' || bad 'waves arm on the turns receipt'

(( fail )) && { grep -E '^LOG|^ARM' "$T/loop.log" | head -20; echo 'ab_window_cr_test: FAILED'; exit 1; }
echo 'ab_window_cr_test: all checks passed'
