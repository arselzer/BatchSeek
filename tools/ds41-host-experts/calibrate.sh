#!/usr/bin/env bash
# calibrate.sh -- a placement profile (--expert-profile) for your own prompts, in one command.
#
#   calibrate.sh --model MODEL.rad --silu-table FILE PROMPTS.jsonl OUT.profile.json [--base PROFILE.json]
#                [--weight W|auto] [--streams 8] [--port 8000] [--max-tokens 128] [--work DIR]
#                [--ready-timeout 1800] [--dry-run] [-- more serve-batch.sh options]
#
# 1. starts serve-batch.sh --capture (the measured engine as a calibration run: 8 streams, batch admission without
#    the split, the routing counts written by --expert-stats), and waits until it answers;
# 2. sends PROMPTS.jsonl (batch_client.py's format: {"prompt", "system", "id", "max_tokens"} a line) --streams at a
#    time, greedy, --max-tokens a reply unless a line gives its own;
# 3. stops the server with SIGINT, which writes the final counts, and waits for it to exit;
# 4. profile_blend.py write --portable --container MODEL.rad: OUT is a portable profile (no path or stat; any machine
#    with the same container accepts it, the server logging "expert profile: portable (header sha256 ...)"). With
#    --base, the base (e.g. the shipped profile, portable or sealed for this container) is blended in: its counts once,
#    the capture's W times. --weight auto (default) gives the capture as many decode tokens as the base, an even blend;
#    a whole number W takes the capture's counts W times.
#
# Only decode routing counts are placement data. It prints how many decode tokens were captured and warns under
# 2,000 (a few hundred tokens are noisy: send more or longer prompts, or a larger --max-tokens). Steps that held
# prefill and decode rows together are left out of the counts (profile_blend.py --allow-mixed), and their decode
# tokens printed. The replies are kept in the work directory (--work DIR, default a new OUT.calibrate-XXXXXX beside
# OUT) with the server log and the raw capture.
#
# Then serve with it:  serve-batch.sh --model MODEL.rad --silu-table FILE --profile OUT.profile.json
#
# --dry-run prints the server's launch (serve-batch.sh --dry-run), the client and blend commands, and starts nothing.
# RADIANCE_BIN and RADIANCE_HOME reach serve-batch.sh as for any run. PYTHON (python3) runs the tools.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
die() { echo "calibrate.sh: $*" >&2; exit 2; }
say() { echo "calibrate.sh: $*"; }
MODEL="" SILU="" BASE="" WEIGHT=auto STREAMS=8 PORT=8000 MAXTOK=128 WORK="" READY=1800 STOPWAIT=${CALIBRATE_STOP_S:-900} DRY=0
POS=() EXTRA=()
while (( $# )); do
  case $1 in
    --model) MODEL=${2:?}; shift ;;
    --silu-table) SILU=${2:?}; shift ;;
    --base) BASE=${2:?}; shift ;;
    --weight) WEIGHT=${2:?}; shift ;;
    --streams) STREAMS=${2:?}; shift ;;
    --port) PORT=${2:?}; shift ;;
    --max-tokens) MAXTOK=${2:?}; shift ;;
    --work) WORK=${2:?}; shift ;;
    --ready-timeout) READY=${2:?}; shift ;;
    --dry-run) DRY=1 ;;
    --) shift; EXTRA=("$@"); break ;;
    -h|--help) sed -n '2,/^set -u/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
    -*) die "unknown option '$1' (--help)" ;;
    *) POS+=("$1") ;;
  esac
  shift
done
(( ${#POS[@]} == 2 )) || die "give PROMPTS.jsonl and OUT.profile.json (--help)"
PROMPTS=${POS[0]} OUT=${POS[1]}
[[ -n "$MODEL" ]] || die "--model MODEL.rad is required"
[[ -n "$SILU" ]] || die "--silu-table FILE is required (serve-batch.sh --help)"
for v in "--streams $STREAMS" "--port $PORT" "--max-tokens $MAXTOK" "--ready-timeout $READY"; do
  [[ ${v#* } =~ ^[0-9]+$ ]] && (( 10#${v#* } >= 1 )) || die "${v% *} takes a whole number of at least 1, not '${v#* }'"
done
[[ $WEIGHT == auto ]] || { [[ $WEIGHT =~ ^[0-9]+$ ]] && (( 10#$WEIGHT >= 1 && 10#$WEIGHT <= 1000 )); } ||
  die "--weight takes auto or a whole number from 1 to 1000, not '$WEIGHT'"
for a in "${EXTRA[@]}"; do
  case $a in --capture|--streams|--port|--model|--silu-table|--dry-run) die "$a is calibrate.sh's own option" ;; esac
done
PY=${PYTHON:-python3}
SB=$HERE/serve-batch.sh CLIENT=$HERE/batch_client.py BLEND=$HERE/profile_blend.py
[[ -e "$OUT" ]] && die "$OUT exists"
[[ -s "$PROMPTS" ]] || die "no prompts at $PROMPTS"
[[ -z "$BASE" || -s "$BASE" ]] || die "no base profile at $BASE"
[[ -n "$WORK" ]] || WORK=$(dirname "$OUT")/$(basename "$OUT" .json).calibrate
if (( DRY )); then
  CAP=$WORK/capture.json
  "$SB" --model "$MODEL" --silu-table "$SILU" --capture "$CAP" --streams "$STREAMS" --port "$PORT" "${EXTRA[@]}" --dry-run ||
    die "serve-batch.sh --dry-run refused"
  printf 'client'; printf ' %q' "$PY" "$CLIENT" "$PROMPTS" "$WORK/replies.jsonl" --url "http://127.0.0.1:$PORT" \
    --concurrency "$STREAMS" --max-tokens "$MAXTOK"; echo
  printf 'stop kill -INT <server pid>; wait for it to exit (%s s)\n' "$STOPWAIT"
  printf 'blend'; printf ' %q' "$PY" "$BLEND" write --portable --container "$MODEL" "$OUT"
  if [[ -n "$BASE" ]]; then printf ' %q' "1:$BASE" "$WEIGHT:$CAP"; else printf ' %q' "1:$CAP"; fi; echo
  exit 0
fi
[[ -f "$MODEL" ]] || die "no model container at $MODEL"
[[ -s "$SILU" ]] || die "no silu table at $SILU"
if [[ -e "$WORK" ]]; then WORK=$(mktemp -d "$WORK-XXXXXX") || die "cannot make a work directory beside $WORK"
else mkdir -p "$WORK" || die "cannot make $WORK"; fi
CAP=$WORK/capture.json LOG=$WORK/server.log
URL=http://127.0.0.1:$PORT
curl -s -m 3 "$URL/stats" > /dev/null 2>&1 && die "port $PORT already answers: stop that server or give --port"

# The base's decode tokens (for --weight auto), refused early if the base is not a profile for this container.
if [[ -n "$BASE" ]]; then
  "$PY" -I "$BLEND" match "$MODEL" "$BASE" > "$WORK/match.txt" 2>&1 || die "--base: $(tail -1 "$WORK/match.txt")"
  "$PY" -I "$BLEND" show "$BASE" > "$WORK/base.txt" 2>&1 || die "--base $BASE: $(tail -1 "$WORK/base.txt")"
fi

SPID=""
stop_server() {   # SIGINT: the server writes its final counts and exits
  [[ -n "$SPID" ]] && kill -0 "$SPID" 2>/dev/null || return 0
  kill -INT "$SPID" 2>/dev/null
  local t=0
  while kill -0 "$SPID" 2>/dev/null && (( t < STOPWAIT )); do sleep 1; t=$((t + 1)); done
  if kill -0 "$SPID" 2>/dev/null; then
    say "the server did not exit ${STOPWAIT} s after SIGINT; killing it (its last periodic counts are not final)"
    kill -KILL "$SPID" 2>/dev/null
  fi
  wait "$SPID" 2>/dev/null; SPID=""
}
trap 'stop_server' EXIT
trap 'say "interrupted"; exit 130' INT TERM

say "starting the server (log $LOG)"
"$SB" --model "$MODEL" --silu-table "$SILU" --capture "$CAP" --streams "$STREAMS" --port "$PORT" "${EXTRA[@]}" \
  > "$LOG" 2>&1 < /dev/null &
SPID=$!
t=0
until curl -s -m 3 "$URL/stats" > /dev/null 2>&1; do
  kill -0 "$SPID" 2>/dev/null || { SPID=""; die "the server exited during startup: $(tail -5 "$LOG")"; }
  (( t >= READY )) && die "the server did not answer within $READY s (--ready-timeout): $(tail -3 "$LOG")"
  sleep 2; t=$((t + 2))
done
say "serving after ${t} s; sending $(grep -c . "$PROMPTS") prompts, $STREAMS at a time"
"$PY" -I "$CLIENT" "$PROMPTS" "$WORK/replies.jsonl" --url "$URL" --concurrency "$STREAMS" --max-tokens "$MAXTOK"
crc=$?
(( crc == 0 )) || say "batch_client.py exit $crc: some requests failed ($WORK/replies.jsonl); their routing is still counted"
say "stopping the server (SIGINT)"
stop_server
[[ -s "$CAP" ]] || die "no capture at $CAP (server log: $(tail -3 "$LOG"))"

# What was captured; the blend's options and weight.
read -r DEC MIXED MIXDEC FINAL BASEDEC < <("$PY" -I - "$CAP" "$WORK/base.txt" <<'EOF'
import json, re, sys
d = json.load(open(sys.argv[1]))
t = d.get("tokens", {})
base = 0
try:
    m = re.search(r": (\d+) decode tokens", open(sys.argv[2]).read())
    base = int(m.group(1)) if m else 0
except OSError:
    pass
print(t.get("decode", 0), t.get("mixed", 0), d.get("mixed_decode_tokens", 0), int(d.get("final") is True), base)
EOF
) || die "unreadable capture $CAP"
(( FINAL == 1 )) || die "the capture is not final (the server did not stop cleanly): $CAP"
say "captured $DEC decode tokens in pure decode steps$( (( MIXED > 0 )) && echo "; $MIXDEC more decode tokens were in mixed steps and are left out")"
(( DEC >= 2000 )) || say "WARNING: only $DEC decode tokens; under ~2,000 the counts are noisy. Send more prompts or longer replies (--max-tokens)."
(( DEC > 0 )) || die "no decode tokens captured"
BLENDOPT=(--portable --container "$MODEL")
(( MIXED > 0 )) && BLENDOPT+=(--allow-mixed)
PARTS=()
if [[ -n "$BASE" ]]; then
  if [[ $WEIGHT == auto ]]; then
    WEIGHT=$(( (BASEDEC + DEC / 2) / DEC )); (( WEIGHT >= 1 )) || WEIGHT=1; (( WEIGHT <= 1000 )) || WEIGHT=1000
    say "--weight auto: the capture x$WEIGHT ($((WEIGHT * DEC)) tokens) beside the base's $BASEDEC"
  fi
  PARTS=("1:$BASE" "$((10#$WEIGHT)):$CAP")
else
  PARTS=("1:$CAP")
fi
"$PY" -I "$BLEND" write "${BLENDOPT[@]}" "$OUT" "${PARTS[@]}" || die "profile_blend.py refused (above); the capture is kept at $CAP"
say "done: serve with  $SB --model $MODEL --silu-table $SILU --profile $OUT"
