#!/usr/bin/env bash
# ab_window_p1_test.sh [ab_window.sh] -- ab_window.sh's P1 parts (P1, QmN, QlK, QeK, QhMS) with no GPU, no
# server and no model:
#   1. lcp_check on synthetic server logs: a P1 arm is measured only when the card path's startup line
#      states its threshold and chunk, a Ql arm only with its lent-ring line, a Qe arm only with its
#      lent-arena line naming its K, and an arm without a part only without its line;
#   2. the arm loop, every action stubbed (as ab_window_vp_test.sh): the adopted large-chunk arm with P1
#      on top of it (B6, Ql10, M1024, Mr32, Mb1, Cs) is launched with the switches it names and the rest
#      stated off; the adopted arm itself states P1 off; P1 at the default threshold with M16, Ql12 on
#      B6's ring, Qh without Ql and Ql without P1 are skipped; Qm17 with M16 is launched; P1 at Lc2048 with
#      Qe64 and M2048 is launched with the lent arena on, Qh beside Qe alone too; Qe at M1024 beside Lc2048
#      and Qe without P1 are skipped; Lc4096 with explicit Pci4096 works cold or cached, while
#      a missing/short interval, another prefix-only part without Pc, or Lc4112 are refused;
#   3. balanced-kernel receipts state their actual requested-height coverage; an M2048 balance
#      oracle cannot authorize Mb+Lc4096 even when the independent LCP oracle covered 4096;
#   4. SgK / SgaK (prompt-lookup speculation) beside Lc, P1 and Cs: sg_gate on synthetic logs (an Sg arm
#      is measured only with the file stager's, the scheduler's, the card path's and the consumer skip's
#      admission lines naming its K, an arm without the part only without them, and at exit only with no
#      pass outside the decode band and no verify pass over 16 rows); in the arm loop the adopted fast
#      arm with Sg8, Sga8 or Sg15 on top is launched with the speculation's arguments and every other
#      switch as without it, with N2 Sg7 too, and Sg16, N2 with Sg8 and N8 with Sg2 (more than 16 decode
#      rows a step) are skipped with the reason;
#   5. QdN (decode passes on the cards, ..._P1_DECODE_ROWS): qd_gate on synthetic logs (a Qd arm is measured only
#      with rank 0's decode line naming its N and chunk, and at exit only when every rank's stats line counts its
#      decode passes and some rank armed one; an arm without the part only without the line and with none armed),
#      alone and through lcp_check; in the arm loop the fast arm with Qd200 is launched with the switch at 200
#      and every arm without the part states it 0, while Qd above the chunk, Qd16, Qd beside Tb and Qd without
#      P1 are skipped with the reason;
#   6. QyP / QyfP (hybrid decode, ..._P1_HYBRID): qy_gate on synthetic logs (a Qy arm is measured only with rank 0's
#      hybrid line naming its share and mode, and at exit only when every rank's stats line counts its hybrid passes
#      and some rank split one; an arm without the part only without the line and with none split), alone and
#      through lcp_check; in the arm loop Qd64-Qy25 and Qd64-Qyf30 on the fast arm are launched with the switch (and
#      the held share) and every other switch as the fast arm, every arm without the part states it 0 and the adapt
#      switch empty, while Qy without Qd, beside Sg8, Qv8 or Ax, Qy0, two Qy parts and Qy without P1 are skipped
#      with the reason.
#   6c. QypP (hybrid prefill, ..._P1_HYBRID_PREFILL): qyp_gate on synthetic logs (a Qyp arm is measured only with
#      rank 0's hybrid prefill line naming its share, and at exit only when every rank's stats line counts its prefill
#      splits and some rank split one; an arm without the part only without the line and with none split; the decode
#      hybrid's line and counts never stand in for it, nor its for them); in the arm loop Qyp30 on the fast arm (no Qd
#      needed) and Qd64-Qy60-Qyp30 are launched with the switch and every other switch as the fast arm (Qy60's
#      decode switch unchanged beside it), every arm without the part states it 0, while Qyp beside Sg8, Qv8 or Ax,
#      Qyp0, two Qyp parts and Qyp without P1 are skipped with the reason.
#   6d. Qyk (kernel costs, ..._HOST_EXPERTS_KERNEL_COST): qyk_gate on synthetic logs (the pool's line in a Qyk arm
#      only); in the arm loop Qd64-Qy60-Qyk and Qd64-Qy60-Qyp60-Qyk on the fast arm state it 1, every other arm 0,
#      and two Qyk parts are skipped.
#   6e. QycM (the hybrid split's calibration, ..._P1_HYBRID_CAL): qyc_gate on synthetic logs (rank 0's line naming M
#      once at startup; at exit every rank's calibration counts, with bit 1 some completion sample, with bit 2 the
#      coupled shares; an arm without the part only without the line); in the arm loop Qd64-Qy60-Qyp60-Qyk-Qyc3 on the
#      fast arm states it 3 and every other switch as the arm without it, every arm without the part states it 0,
#      while Qyc without Qy or Qyp, Qyc4 and two Qyc parts are skipped with the reason.
#   7. Ctp / CtpN (the prefill chain trace, ..._CHAIN_TRACE_PREFILL): the fast arm with Ctp traces one armed
#      prefill pass in 4 into chain-<arm>, with Qd64 Ct16 Ctp2 both traces at their N into the same files;
#      Ctp without P1 and Ctp0 are skipped with the reason; arms without the part do not set it; and
#      chain_report.py's selftest (the --split classes on a synthetic prefill pass, old files as decode).
#   8. Qt (the register-tiled P1 twin, ..._P1_TILED): qt_gate on synthetic logs (every rank's startup marker once
#      with the part, none without it); in the arm loop the fast arm with Qt (and with Qsp-Qt) is launched with
#      the switch at 1 and every other switch as the fast arm, every arm without the part states it 0, Qt without
#      P1 is skipped, and without a passing p1tiled receipt every Qt arm is refused while the fast arm still runs.
#   9. Ee (the next Engram layer's rows read ahead, ..._ENGRAM_EARLY): ee_gate on synthetic logs (an Ee arm is
#      measured only with rank 0's line, and at exit only when every rank's line counts its read-aheads, none
#      refused and some started; an arm without the part only without either line); in the arm loop the fast
#      arm with Ee (and with Qd64-Qy60-Ee) is launched with the switch and every other switch as without it,
#      and arms without the part do not set it.
#   8d. Dt / Aw (the tiled bf16f8 GEMM, exact; the WMMA sparse attention, inexact): the switches at 1 on the preset arm,
#      every other arm states them 0; Dt without Lc or F8, Aw without Lc, Aw with Kld before a reference skipped;
#      with a reference an Aw Kld arm scores against it (--kld-ref) and a Dt Kld arm records; without their receipts
#      refused while the fast arm runs; dt_gate / aw_gate on synthetic logs.
# Exit 0 when every check passes.
set -u
W=${1:-$(dirname "$0")/ab_window.sh}
T=$(mktemp -d /tmp/ab-p1-test-XXXXXX)
trap 'rm -rf "$T"' EXIT
fail=0
ok()  { echo "ok   $*"; }
bad() { echo "FAIL $*"; fail=1; }

# ---- 1. the server-log gate
eval "$(sed -n '/^lcp_check() {/,/^}/p' "$W")"
declare -F lcp_check >/dev/null || { echo "FAIL no lcp_check in $W"; exit 1; }
eval "$(sed -n '/^qd_gate() {/,/^}/p' "$W")"
declare -F qd_gate >/dev/null || { echo "FAIL no qd_gate in $W"; exit 1; }
eval "$(sed -n '/^qy_gate() {/,/^}/p' "$W")"
declare -F qy_gate >/dev/null || { echo "FAIL no qy_gate in $W"; exit 1; }
eval "$(sed -n '/^qyp_gate() {/,/^}/p' "$W")"
eval "$(sed -n '/^qyk_gate() {/,/^}/p' "$W")"
declare -F qyk_gate >/dev/null || { echo "FAIL no qyk_gate in $W"; exit 1; }
declare -F qyp_gate >/dev/null || { echo "FAIL no qyp_gate in $W"; exit 1; }
eval "$(sed -n '/^ee_gate() {/,/^}/p' "$W")"
declare -F ee_gate >/dev/null || { echo "FAIL no ee_gate in $W"; exit 1; }
OUT=$T/logs
mkdir -p "$OUT"
log() { echo "LOG: $*" >> "$OUT/window.log"; }
lc_line='I[0] large-chunk prefill: passes of up to 1024 tokens (RADIANCE_EXPERIMENTAL_DS41_LCP=1024): ...'
p1_line() { echo "I[0] ds41 p1: card experts on: every routed expert row of a pure prefill pass's layer of $1 to 1024 rows on the cards (RADIANCE_EXPERIMENTAL_DS41_P1_MIN_ROWS=$1; ...)"; }
lend_line() { echo "I[0] ds41 p1: lent ring: $1 of the ring's 12 slots (10 of them past the file buffers' alias bases; RADIANCE_EXPERIMENTAL_DS41_P1_LEND=$1) hold ..."; }
arena_line() { echo "I[0] ds41 p1: lent arena: $2 slots (RADIANCE_EXPERIMENTAL_DS41_P1_LEND_ARENA=$1; $2 fit) in the activation arena's top, ..."; }
mk() { local n=$1; shift; printf '%s\n' "$@" > "$OUT/$n.server.log"; }
mk p1def "$lc_line" "$(p1_line 320)"
mk p1lend "$lc_line" "$(p1_line 320)" "$(lend_line 10)"
mk p1q17 "$lc_line" "$(p1_line 17)"
mk plain "$lc_line"
mk p1arena "$lc_line" "$(p1_line 320)" "$(arena_line 64 18)"
mk p1both "$lc_line" "$(p1_line 320)" "$(lend_line 10)" "$(arena_line 64 18)"
expect() {   # want(0/1) what log LC P1 QM QL [QE]
  local want=$1 what=$2 rc
  ARM_LC=$4 ARM_P1=$5 ARM_QM=$6 ARM_QL=$7 ARM_QE=${8:-} lcp_check "$3" >/dev/null 2>&1; rc=$?
  if [[ $rc == "$want" ]]; then ok "$what"; else bad "$what: exit $rc, want $want ($(tail -1 "$OUT/window.log"))"; fi
}
expect 0 "P1 at the default threshold, its line"                      p1def  1024 1 320 ""
expect 1 "P1 at Qm17 on a server that took 320"                       p1def  1024 1 17  ""
expect 0 "P1 at Qm17, its line"                                       p1q17  1024 1 17  ""
expect 0 "P1 with Ql10, both lines"                                   p1lend 1024 1 320 10
expect 1 "P1 with Ql10, no lent-ring line"                            p1def  1024 1 320 10
expect 1 "P1 with Ql8 on a server that lent 10"                       p1lend 1024 1 320 8
expect 1 "P1 without Ql on a server that lends (the knob from the window)" p1lend 1024 1 320 ""
expect 1 "no P1 part on a server that runs the card path"             p1def  1024 "" 320 ""
expect 0 "the plain Lc arm"                                           plain  1024 "" 320 ""
expect 0 "P1 with Qe64, its line"                                     p1arena 1024 1 320 "" 64
expect 1 "P1 with Qe64, no lent-arena line"                           p1def  1024 1 320 "" 64
expect 1 "P1 with Qe32 on a server that took 64"                      p1arena 1024 1 320 "" 32
expect 1 "P1 without Qe on a server that lends the arena"             p1arena 1024 1 320 ""
expect 0 "P1 with Ql10 and Qe64, both lines"                          p1both 1024 1 320 10 64
expect 1 "P1 with Qe64 alone on a server that lends the ring too"     p1both 1024 1 320 "" 64

# SgK beside Lc: the admission's startup lines and the exit counts (sg_gate), alone and through lcp_check.
eval "$(sed -n '/^sg_gate() {/,/^}/p' "$W")"
declare -F sg_gate >/dev/null || { echo "FAIL no sg_gate in $W"; exit 1; }
sg_stager() { echo "I[0] large-chunk prefill: prompt-lookup speculation admitted (--num-speculative-tokens $1, no draft head): a verify step is a decode pass of at most $(( $1 + 1 )) rows, inside the decode band of 16 (...)"; }
sg_sched() { echo "I large-chunk prefill: prompt-lookup speculation admitted (--num-speculative-tokens $1, --max-num-seqs 1, no draft head): a step is one prompt's pure prefill chunk, ..."; }
sg_p1() { echo "I[0] ds41 p1: prompt-lookup speculation admitted (--num-speculative-tokens $1, no draft head): a verify step of at most $(( $1 + 1 )) decode rows never arms the card path; ..."; }
sg_cs() { echo "I consumer skip: prompt-lookup speculation admitted (--num-speculative-tokens $1, no draft head): a verify step of at most $(( $1 + 1 )) decode rows runs every layer on every row ..."; }
sg_exit() { echo "I large-chunk prefill, rank $1: prompt-lookup verify passes $2, $(( $2 * 5 )) rows (at most $3 a pass); passes outside the decode band $4"; }
mk sgall "$lc_line" "$(p1_line 320)" "$(sg_stager 8)" "$(sg_sched 8)" "$(sg_p1 8)" "$(sg_cs 8)"
mk sgnocs "$lc_line" "$(p1_line 320)" "$(sg_stager 8)" "$(sg_sched 8)" "$(sg_p1 8)"
mk sgnop1 "$lc_line" "$(p1_line 320)" "$(sg_stager 8)" "$(sg_sched 8)" "$(sg_cs 8)"
mk sgnosched "$lc_line" "$(p1_line 320)" "$(sg_stager 8)" "$(sg_p1 8)" "$(sg_cs 8)"
mk sgnostager "$lc_line" "$(p1_line 320)" "$(sg_sched 8)" "$(sg_p1 8)" "$(sg_cs 8)"
mk sgrefused "E[0] RADIANCE_EXPERIMENTAL_DS41_LCP=2048 (large-chunk prefill) refused: speculation is on (a window of 8): verify steps and DSpark's drafter are not covered"
for r in 0 1 2 3; do sg_exit $r 40 9 0; done > "$OUT/sgexit-good.server.log"
{ sg_exit 0 40 9 0; sg_exit 1 40 9 1; sg_exit 2 40 9 0; sg_exit 3 40 9 0; } > "$OUT/sgexit-outside.server.log"
{ sg_exit 0 40 9 0; sg_exit 1 40 17 0; } > "$OUT/sgexit-tall.server.log"
echo "I large-chunk prefill, rank 0: prompt-lookup verify passes 40 (an older line)" > "$OUT/sgexit-odd.server.log"
: > "$OUT/sgexit-none.server.log"
sgexpect() {   # want(0/1) what log K P1 CS stage
  local want=$1 what=$2 rc
  sg_gate "$OUT/$3.server.log" "$4" "$5" "$6" "$7" >/dev/null 2>&1; rc=$?
  if [[ $rc == "$want" ]]; then ok "$what"; else bad "$what: exit $rc, want $want"; fi
}
sgexpect 0 "Sg8 with P1 and Cs, every admission line"                 sgall      8 1 1 start
sgexpect 0 "Sg8 with P1, no Cs part: the consumer skip's line is not asked for" sgnocs 8 1 "" start
sgexpect 1 "Sg8 with Cs, no consumer skip admission line"             sgnocs     8 1 1 start
sgexpect 1 "Sg8 with P1, no card path admission line"                 sgnop1     8 1 1 start
sgexpect 1 "Sg8, no scheduler admission line"                         sgnosched  8 1 1 start
sgexpect 1 "Sg8, no file stager admission line"                       sgnostager 8 1 1 start
sgexpect 1 "Sg7 on a server that admitted depth 8"                    sgall      7 1 1 start
sgexpect 1 "Sg8 on a server that refused the speculation"             sgrefused  8 1 1 start
sgexpect 1 "no Sg part on a server that took the speculation"         sgall      "" 1 1 start
sgexpect 0 "no Sg part, no admission line"                            p1def      "" 1 1 start
sgexpect 0 "Sg8 at exit: every rank's verify passes inside the band"  sgexit-good 8 1 1 exit
sgexpect 1 "Sg8 at exit: a rank with a pass outside the band"         sgexit-outside 8 1 1 exit
sgexpect 1 "Sg8 at exit: a verify pass of 17 rows"                    sgexit-tall 8 1 1 exit
sgexpect 1 "Sg8 at exit: a line without its counts"                   sgexit-odd 8 1 1 exit
sgexpect 0 "Sg8 at exit: no verify pass ran (nothing to refuse)"      sgexit-none 8 1 1 exit
# ...and through lcp_check, which names the arm's K only beside an Lc part.
sglc() {   # want(0/1) what log LC SG CS
  local want=$1 what=$2 rc
  ARM_LC=$4 ARM_P1=1 ARM_QM=320 ARM_QL="" ARM_QE="" ARM_SG=$5 ARM_CS=$6 lcp_check "$3" >/dev/null 2>&1; rc=$?
  if [[ $rc == "$want" ]]; then ok "$what"; else bad "$what: exit $rc, want $want ($(tail -1 "$OUT/window.log"))"; fi
}
sglc 0 "lcp_check: the P1 arm with Sg8 and Cs, its lines"            sgall  1024 8 1
sglc 1 "lcp_check: the P1 arm with Sg8, no consumer skip line"       sgnocs 1024 8 1
sglc 1 "lcp_check: the P1 arm without Sg on a server that took it"   sgall  1024 "" 1
sglc 0 "lcp_check: the P1 arm without Sg, as before"                 p1def  1024 "" 1

# ---- 5a. QdN: qd_gate on synthetic logs, alone and through lcp_check.
qd_line() { echo "I[0] ds41 p1: decode passes on the cards: every routed expert row of a pure decode pass of $1 to ${2:-1024} rows on the cards, as a prefill pass's layer of the threshold's rows (RADIANCE_EXPERIMENTAL_DS41_P1_DECODE_ROWS=$1; ...)"; }
qd_stats() { echo "I[$1] ds41 p1 stats, rank $1: passes 40, ...; leases at pass end 0, verify rows 0, differ 0; touched experts 900: ...; decode passes armed $2, rows $(( $2 * 256 )) (at most 256 a pass), issue 20.500 s; decode passes of 17 to 199 rows on the CPU path 3"; }
mk qd200 "$lc_line" "$(p1_line 320)" "$(qd_line 200)"
mk qd200x2048 "$lc_line" "$(p1_line 320)" "$(qd_line 200 2048)"
{ cat "$OUT/qd200.server.log"; for r in 0 1 2 3; do qd_stats $r 12; done; } > "$OUT/qdexit.server.log"
{ cat "$OUT/qd200.server.log"; for r in 0 1 2 3; do qd_stats $r 0; done; } > "$OUT/qdidle.server.log"
{ cat "$OUT/qd200.server.log"; qd_stats 0 12; echo "I[1] ds41 p1 stats, rank 1: passes 40, leases at pass end 0, verify rows 0, differ 0"; } > "$OUT/qdodd.server.log"
{ cat "$OUT/p1def.server.log"; for r in 0 1 2 3; do qd_stats $r 5; done; } > "$OUT/qdstray.server.log"
qdexpect() {   # want(0/1) what log QD LC stage
  local want=$1 what=$2 rc
  qd_gate "$OUT/$3.server.log" "$4" "$5" "$6" >/dev/null 2>&1; rc=$?
  if [[ $rc == "$want" ]]; then ok "$what"; else bad "$what: exit $rc, want $want"; fi
}
qdexpect 0 "Qd200, its line"                                        qd200      200 1024 start
qdexpect 1 "Qd256 on a server that took 200"                        qd200      256 1024 start
qdexpect 1 "Qd200 at Lc2048 on a server whose line names 1024"      qd200      200 2048 start
qdexpect 0 "Qd200 at Lc2048, its line"                              qd200x2048 200 2048 start
qdexpect 1 "Qd200, no decode line"                                  p1def      200 1024 start
qdexpect 1 "no Qd part on a server that arms decode passes"         qd200      ""  1024 start
qdexpect 0 "no Qd part, no decode line"                             p1def      ""  1024 start
qdexpect 0 "Qd200 at exit: ranks armed decode passes"               qdexit     200 1024 exit
qdexpect 1 "Qd200 at exit: no rank armed one (the switch never acted)" qdidle  200 1024 exit
qdexpect 1 "Qd200 at exit: a stats line without its decode counts"  qdodd      200 1024 exit
qdexpect 1 "Qd200 at exit: no stats line"                           qd200      200 1024 exit
qdexpect 1 "no Qd part at exit, but a rank armed decode passes"     qdstray    ""  1024 exit
qdexpect 0 "no Qd part at exit, none armed"                         p1def      ""  1024 exit
qdlc() {   # want(0/1) what log QD
  local want=$1 what=$2 rc
  ARM_LC=1024 ARM_P1=1 ARM_QM=320 ARM_QL="" ARM_QE="" ARM_QD=$4 lcp_check "$3" >/dev/null 2>&1; rc=$?
  if [[ $rc == "$want" ]]; then ok "$what"; else bad "$what: exit $rc, want $want ($(tail -1 "$OUT/window.log"))"; fi
}
qdlc 0 "lcp_check: the P1 arm with Qd200, its line"                 qd200 200
qdlc 1 "lcp_check: the P1 arm with Qd200, no decode line"           p1def 200
qdlc 1 "lcp_check: the P1 arm without Qd on a server that arms decode" qd200 ""
qdlc 0 "lcp_check: the P1 arm without Qd, as before"                p1def ""

# ---- 6a. QyP / QyfP: qy_gate on synthetic logs, alone and through lcp_check.
qy_line() { echo "I[0] ds41 p1: hybrid decode on: in a decode pass armed for the cards (RADIANCE_EXPERIMENTAL_DS41_P1_DECODE_ROWS=64), each routed layer gives the CPU pool the RAM-cache units with the fewest rows while the pool's predicted time stays within $1% of the card's predicted link time for the rest (RADIANCE_EXPERIMENTAL_DS41_P1_HYBRID=$1; pool 78.0 GB/s, link 20.0 GB/s, the share $2); those units ..."; }
qy_adapt="adapts per rank from each layer's outcome, at most that"
qy_held="held (..._P1_HYBRID_ADAPT=0)"
qy_stats() { echo "I[$1] ds41 p1 stats, rank $1: passes 40, ...; leases at pass end 0, verify rows 0, differ 0; touched experts 900: VRAM residents 1, slab 2, ring 3, CPU pool 4; ...; decode passes armed 12, rows 3072 (at most 256 a pass), issue 20.500 s; decode passes of 17 to 63 rows on the CPU path 3; hybrid: passes split $2 of 12 armed, layers 400, units CPU 900 / card 3000 (fell back 0, layers without rows 0), CPU rows 2000, ..., share 25.0% (lowest 12.0%)"; }
mk qy25 "$lc_line" "$(p1_line 320)" "$(qd_line 64 2048)" "$(qy_line 25 "$qy_adapt")"
mk qyf30 "$lc_line" "$(p1_line 320)" "$(qd_line 64 2048)" "$(qy_line 30 "$qy_held")"
{ cat "$OUT/qy25.server.log"; for r in 0 1 2 3; do qy_stats $r 12; done; } > "$OUT/qyexit.server.log"
{ cat "$OUT/qy25.server.log"; for r in 0 1 2 3; do qy_stats $r 0; done; } > "$OUT/qyidle.server.log"
{ cat "$OUT/qy25.server.log"; qy_stats 0 12; qd_stats 1 12; } > "$OUT/qyodd.server.log"
{ cat "$OUT/qd200.server.log"; for r in 0 1 2 3; do qy_stats $r 5; done; } > "$OUT/qystray.server.log"
qyexpect() {   # want(0/1) what log QY QYF stage
  local want=$1 what=$2 rc
  qy_gate "$OUT/$3.server.log" "$4" "$5" "$6" >/dev/null 2>&1; rc=$?
  if [[ $rc == "$want" ]]; then ok "$what"; else bad "$what: exit $rc, want $want"; fi
}
qyexpect 0 "Qy25, its line"                                          qy25    25 ""  start
qyexpect 1 "Qy30 on a server that took 25"                            qy25    30 ""  start
qyexpect 1 "Qyf25 on a server whose share adapts"                     qy25    25 1   start
qyexpect 0 "Qyf30, its held line"                                     qyf30   30 1   start
qyexpect 1 "Qy30 on a server that holds the share"                    qyf30   30 ""  start
qyexpect 1 "Qy25, no hybrid line"                                     qd200   25 ""  start
qyexpect 1 "no Qy part on a server that splits"                       qy25    "" ""  start
qyexpect 0 "no Qy part, no hybrid line"                               qd200   "" ""  start
qyexpect 0 "Qy25 at exit: ranks split passes"                         qyexit  25 ""  exit
qyexpect 1 "Qy25 at exit: no rank split one (the switch never acted)" qyidle  25 ""  exit
qyexpect 1 "Qy25 at exit: a stats line without its hybrid counts"     qyodd   25 ""  exit
qyexpect 1 "Qy25 at exit: no stats line"                              qy25    25 ""  exit
qyexpect 1 "no Qy part at exit, but a rank split passes"              qystray "" ""  exit
qyexpect 0 "no Qy part at exit, none split"                           qdexit  "" ""  exit
qylc() {   # want(0/1) what log QY QYF
  local want=$1 what=$2 rc
  ARM_LC=2048 ARM_P1=1 ARM_QM=320 ARM_QL="" ARM_QE="" ARM_QD=64 ARM_QY=$4 ARM_QYF=$5 lcp_check "$3" >/dev/null 2>&1; rc=$?
  if [[ $rc == "$want" ]]; then ok "$what"; else bad "$what: exit $rc, want $want ($(tail -1 "$OUT/window.log"))"; fi
}
mk qy25lc "${lc_line//1024/2048}" "$(p1_line 320 | sed 's/320 to 1024/320 to 2048/')" "$(qd_line 64 2048)" "$(qy_line 25 "$qy_adapt")"
mk qd64lc "${lc_line//1024/2048}" "$(p1_line 320 | sed 's/320 to 1024/320 to 2048/')" "$(qd_line 64 2048)"
qylc 0 "lcp_check: the P1 arm with Qd64-Qy25, its line"              qy25lc 25 ""
qylc 1 "lcp_check: the P1 arm with Qd64-Qy25, no hybrid line"        qd64lc 25 ""
qylc 1 "lcp_check: the P1 arm without Qy on a server that splits"    qy25lc "" ""
qylc 0 "lcp_check: the P1 arm with Qd64 without Qy, as before"       qd64lc "" ""

# ---- 6c. QypP: qyp_gate on synthetic logs, alone and through lcp_check.
qyp_line() { echo "I[0] ds41 p1: hybrid prefill on: in a prefill pass armed for the cards (320 rows or more), each card-path routed layer gives the CPU pool the RAM-cache units with the fewest rows while the pool's predicted time stays within $1% of the card's predicted link time for the rest (RADIANCE_EXPERIMENTAL_DS41_P1_HYBRID_PREFILL=$1; pool 78.0 GB/s, link 20.0 GB/s, a share of its own that adapts per rank from each layer's outcome, at most that); ..."; }
qyp_stats() {   # rank, decode split, prefill split (empty: no prefill counts)
  echo "I[$1] ds41 p1 stats, rank $1: passes 40, ...; touched experts 900: VRAM residents 1, slab 2, ring 3, CPU pool 4; ...; decode passes armed 12, rows 3072 (at most 256 a pass), issue 20.500 s; decode passes of 17 to 63 rows on the CPU path 3; hybrid: passes split $2 of 12 armed, layers 400, units CPU 900 / card 3000 (fell back 0, layers without rows 0), CPU rows 2000, ..., share 25.0% (lowest 12.0%)${3:+; hybrid prefill: passes split $3 of 30 armed, layers 1200, units CPU 9000 / card 60000 (fell back 0, layers without rows 0, rows at the other arena level 0), CPU rows 20000, CPU 12.000 s (0.4000 s a pass) against issue 3.5000 s a pass, predicted CPU 9.000 s / card 70.000 s, gather wait 1.000 s, late layers 3 (wait 0.010 s), early layers 900, share 30.0% (lowest 20.0%)}"
}
mk qyp30 "$lc_line" "$(p1_line 320)" "$(qyp_line 30)"
mk qyp30qy "$lc_line" "$(p1_line 320)" "$(qd_line 64 2048)" "$(qy_line 60 "$qy_adapt")" "$(qyp_line 30)"
{ cat "$OUT/qyp30.server.log"; for r in 0 1 2 3; do qyp_stats $r 0 7; done; } > "$OUT/qypexit.server.log"
{ cat "$OUT/qyp30.server.log"; for r in 0 1 2 3; do qyp_stats $r 0 0; done; } > "$OUT/qypidle.server.log"
{ cat "$OUT/qyp30.server.log"; for r in 0 1 2 3; do qyp_stats $r 9 ""; done; } > "$OUT/qypnocount.server.log"
{ cat "$OUT/qyp30.server.log"; qyp_stats 0 0 5; qyp_stats 1 0 ""; } > "$OUT/qypodd.server.log"
{ cat "$OUT/qy25.server.log"; for r in 0 1 2 3; do qyp_stats $r 4 6; done; } > "$OUT/qypstray.server.log"
{ cat "$OUT/qy25.server.log"; for r in 0 1 2 3; do qyp_stats $r 4 ""; done; } > "$OUT/qydeconly.server.log"
qypexpect() {   # want(0/1) what log QYP stage
  local want=$1 what=$2 rc
  qyp_gate "$OUT/$3.server.log" "$4" "$5" >/dev/null 2>&1; rc=$?
  if [[ $rc == "$want" ]]; then ok "$what"; else bad "$what: exit $rc, want $want ($(qyp_gate "$OUT/$3.server.log" "$4" "$5"))"; fi
}
qypexpect 0 "Qyp30, its line"                                              qyp30      30 start
qypexpect 0 "Qyp30 beside Qy60, its line"                                  qyp30qy    30 start
qypexpect 1 "Qyp60 on a server that took 30"                               qyp30      60 start
qypexpect 1 "Qyp30, only the decode hybrid's line"                         qy25       30 start
qypexpect 1 "Qyp30, no hybrid line"                                        qd200      30 start
qypexpect 1 "no Qyp part on a server that splits prefill passes"           qyp30      "" start
qypexpect 0 "no Qyp part, the decode hybrid's line only"                   qy25       "" start
qypexpect 0 "Qyp30 at exit: ranks split prefill passes"                    qypexit    30 exit
qypexpect 1 "Qyp30 at exit: no rank split one (the switch never acted)"    qypidle    30 exit
qypexpect 1 "Qyp30 at exit: decode splits only, no prefill counts"         qypnocount 30 exit
qypexpect 1 "Qyp30 at exit: a stats line without its prefill counts"       qypodd     30 exit
qypexpect 1 "Qyp30 at exit: no stats line"                                 qyp30      30 exit
qypexpect 1 "no Qyp part at exit, but a rank split prefill passes"         qypstray   "" exit
qypexpect 0 "no Qyp part at exit, decode splits only"                      qydeconly  "" exit
qyexpect 0 "Qy25 at exit beside prefill counts: the decode gate unchanged" qypstray   25 ""  exit
qyplc() {   # want(0/1) what log QYP
  local want=$1 what=$2 rc
  ARM_LC=2048 ARM_P1=1 ARM_QM=320 ARM_QL="" ARM_QE="" ARM_QD="" ARM_QY="" ARM_QYF="" ARM_QYP=$4 lcp_check "$3" >/dev/null 2>&1; rc=$?
  if [[ $rc == "$want" ]]; then ok "$what"; else bad "$what: exit $rc, want $want ($(tail -1 "$OUT/window.log"))"; fi
}
mk qyp30lc "${lc_line//1024/2048}" "$(p1_line 320 | sed 's/320 to 1024/320 to 2048/')" "$(qyp_line 30)"
mk p1lc "${lc_line//1024/2048}" "$(p1_line 320 | sed 's/320 to 1024/320 to 2048/')"
qyplc 0 "lcp_check: the P1 arm with Qyp30, its line"                 qyp30lc 30
qyplc 1 "lcp_check: the P1 arm with Qyp30, no hybrid prefill line"   p1lc    30
qyplc 1 "lcp_check: the P1 arm without Qyp on a server that splits"  qyp30lc ""
qyplc 0 "lcp_check: the P1 arm without Qyp, as before"               p1lc    ""

# ---- 6d. Qyk: qyk_gate on synthetic logs.
mk qyk "$lc_line" "$(p1_line 320)" "I host experts: kernel costs (RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_KERNEL_COST=1): a job's further rows at the rate of the kernel it runs (the many-row one from 4 rows), and the hybrid split takes units by cost"
qykexpect() {   # want(0/1) what log QYK
  local want=$1 what=$2 rc
  qyk_gate "$OUT/$3.server.log" "$4" >/dev/null 2>&1; rc=$?
  if [[ $rc == "$want" ]]; then ok "$what"; else bad "$what: exit $rc, want $want"; fi
}
qykexpect 0 "Qyk, its line"                            qyk    1
qykexpect 1 "Qyk, no kernel-cost line"                 qyp30  1
qykexpect 1 "no Qyk part on a server with kernel costs" qyk   ""
qykexpect 0 "no Qyk part, no line"                     qyp30  ""

# Qsp's startup and useful-work gates, independent of ordinary P1 startup admission.
eval "$(sed -n '/^qsp_gate() {/,/^}/p' "$W")"
declare -F qsp_gate >/dev/null || { echo "FAIL no qsp_gate in $W"; exit 1; }
: > "$T/qsp-good.log"
for rank in 0 1 2 3; do
  echo "ds41 p1: rank $rank's wave SwiGLU spans on (test)" >> "$T/qsp-good.log"
  echo "ds41 p1 stats, rank $rank: test; wave SwiGLU spans 12, rows 20 of 200, full-replay fallbacks 0" >> "$T/qsp-good.log"
done
sed '/rank 3/d' "$T/qsp-good.log" > "$T/qsp-missing.log"
sed 's/spans 12/spans 0/' "$T/qsp-good.log" > "$T/qsp-idle.log"
sed 's/rows 20 of 200/rows 200 of 200/' "$T/qsp-good.log" > "$T/qsp-full.log"
: > "$T/qsp-off.log"
qexpect() {
  local want=$1 label=$2 rc; shift 2
  qsp_gate "$@" >/dev/null 2>&1; rc=$?
  if [[ $rc == "$want" ]]; then ok "$label"; else bad "$label: exit $rc, want $want"; fi
}
qexpect 0 'Qsp every-rank startup' "$T/qsp-good.log" 1 start
qexpect 0 'Qsp every-rank work reduction' "$T/qsp-good.log" 1 exit
qexpect 1 'Qsp missing rank' "$T/qsp-missing.log" 1 start
qexpect 1 'Qsp enabled without actual waves' "$T/qsp-idle.log" 1 exit
qexpect 1 'Qsp no reduced work' "$T/qsp-full.log" 1 exit
qexpect 1 'Qsp unrequested activation' "$T/qsp-good.log" '' start
qexpect 0 'Qsp explicitly off' "$T/qsp-off.log" '' exit

# Qt's startup gate: every rank's register-tiled marker once with the part, none without it.
eval "$(sed -n '/^qt_gate() {/,/^}/p' "$W")"
declare -F qt_gate >/dev/null || { echo "FAIL no qt_gate in $W"; exit 1; }
: > "$T/qt-good.log"
for rank in 0 1 2 3; do
  echo "I[$rank] ds41 p1: register-tiled twin on, rank $rank (RADIANCE_EXPERIMENTAL_DS41_P1_TILED=1): every P1 twin call of this rank on the register-tiled P1 twin, ..." >> "$T/qt-good.log"
done
sed '/rank 2 (/d' "$T/qt-good.log" > "$T/qt-missing.log"
cat "$T/qt-good.log" "$T/qt-good.log" > "$T/qt-twice.log"
: > "$T/qt-off.log"
texpect() {
  local want=$1 label=$2 rc; shift 2
  qt_gate "$@" >/dev/null 2>&1; rc=$?
  if [[ $rc == "$want" ]]; then ok "$label"; else bad "$label: exit $rc, want $want"; fi
}
texpect 0 'Qt every-rank startup marker' "$T/qt-good.log" 1
texpect 1 'Qt missing rank' "$T/qt-missing.log" 1
texpect 1 'Qt marker twice a rank' "$T/qt-twice.log" 1
texpect 1 'Qt unrequested activation' "$T/qt-good.log" ''
texpect 0 'Qt explicitly off' "$T/qt-off.log" ''
texpect 1 'Qt asked, no marker' "$T/qt-off.log" 1

# Qz's gates: every rank's tail-once marker at startup, and at exit every rank's skipped wave fills above 0.
eval "$(sed -n '/^qz_gate() {/,/^}/p' "$W")"
declare -F qz_gate >/dev/null || { echo "FAIL no qz_gate in $W"; exit 1; }
: > "$T/qz-good.log"
for rank in 0 1 2 3; do
  echo "I[$rank] ds41 p1: tail filled once a layer, rank $rank (RADIANCE_EXPERIMENTAL_DS41_P1_TAIL_ONCE=1): ..." >> "$T/qz-good.log"
  echo "I[$rank] ds41 p1 stats, rank $rank: passes 3; wave tail fills skipped 412" >> "$T/qz-good.log"
done
sed '/rank 1 (/d' "$T/qz-good.log" > "$T/qz-missing.log"
sed 's/skipped 412/skipped 0/' "$T/qz-good.log" > "$T/qz-idle.log"
zexpect() {
  local want=$1 label=$2 rc; shift 2
  qz_gate "$@" >/dev/null 2>&1; rc=$?
  if [[ $rc == "$want" ]]; then ok "$label"; else bad "$label: exit $rc, want $want"; fi
}
zexpect 0 'Qz every-rank startup' "$T/qz-good.log" 1 start
zexpect 0 'Qz every-rank skipped fills' "$T/qz-good.log" 1 exit
zexpect 1 'Qz missing rank' "$T/qz-missing.log" 1 start
zexpect 0 'Qz idle at startup' "$T/qz-idle.log" 1 start
zexpect 1 'Qz no fill skipped' "$T/qz-idle.log" 1 exit
zexpect 1 'Qz unrequested activation' "$T/qz-good.log" '' start
zexpect 0 'Qz explicitly off' "$T/qt-off.log" '' exit

# Qx's gates: rank 0's early-copies line naming K once at startup, and at exit some rank's bound early copies above 0.
eval "$(sed -n '/^qx_gate() {/,/^}/p' "$W")"
declare -F qx_gate >/dev/null || { echo "FAIL no qx_gate in $W"; exit 1; }
{
  echo "I ds41 p1: early ring copies on (RADIANCE_EXPERIMENTAL_DS41_P1_EARLY=12): behind each card-path layer's waves, ..."
  for rank in 0 1 2 3; do
    echo "I[$rank] ds41 p1 stats, rank $rank: passes 3; wave tail fills skipped 412; early copies 960 ($((rank * 100)) bound to their layer, 12 unread, 0 not in the RAM cache when posted)"
  done
} > "$T/qx-good.log"
sed -E 's/\(([0-9]+) bound/(0 bound/' "$T/qx-good.log" > "$T/qx-idle.log"
sed 's/P1_EARLY=12/P1_EARLY=8/' "$T/qx-good.log" > "$T/qx-other.log"
cat "$T/qx-good.log" "$T/qx-good.log" > "$T/qx-twice.log"
xexpect() {
  local want=$1 label=$2 rc; shift 2
  qx_gate "$@" >/dev/null 2>&1; rc=$?
  if [[ $rc == "$want" ]]; then ok "$label"; else bad "$label: exit $rc, want $want"; fi
}
xexpect 0 'Qx startup line' "$T/qx-good.log" 12 start
xexpect 0 'Qx bound copies at exit' "$T/qx-good.log" 12 exit
xexpect 1 'Qx line names another K' "$T/qx-other.log" 12 start
xexpect 1 'Qx line twice' "$T/qx-twice.log" 12 start
xexpect 0 'Qx idle at startup' "$T/qx-idle.log" 12 start
xexpect 1 'Qx never bound a copy' "$T/qx-idle.log" 12 exit
xexpect 1 'Qx unrequested activation' "$T/qx-good.log" '' start
xexpect 0 'Qx explicitly off' "$T/qt-off.log" '' exit
xexpect 1 'Qx asked, no line' "$T/qt-off.log" 12 start

# ---- 6e. Qyc's gates: rank 0's calibration line naming M once at startup; at exit every rank's calibration counts,
# some completion sample with bit 1, the coupled shares with bit 2.
eval "$(sed -n '/^qyc_gate() {/,/^}/p' "$W")"
declare -F qyc_gate >/dev/null || { echo "FAIL no qyc_gate in $W"; exit 1; }
{
  echo "I ds41 p1: hybrid calibration on (RADIANCE_EXPERIMENTAL_DS41_P1_HYBRID_CAL=3): the split's CPU rate learnt from ..."
  for rank in 0 1 2 3; do
    echo "I[$rank] ds41 p1 stats, rank $rank: passes 3; ...; hybrid prefill: passes split 9 of 9 armed, ..., share 40.0% (lowest 20.0%), pool done 1.000 s after the submits (0.1000 s a pass, 300 layers), cost 12.0 GB; hybrid calibration 3: pool rate decode 31.2 GB/s (0 samples), prefill 28.4 GB/s ($((rank * 100)) samples), model 78.0 GB/s; shares coupled, decode 60.0% (lowest 60.0%), prefill 85.0% (lowest 30.0%)"
  done
} > "$T/qyc-good.log"
sed -E 's/\(([0-9]+) samples\)/(0 samples)/g' "$T/qyc-good.log" > "$T/qyc-idle.log"
sed 's/; shares coupled, .*/; shares a rank'"'"'s own/' "$T/qyc-good.log" > "$T/qyc-uncoupled.log"
sed 's/P1_HYBRID_CAL=3/P1_HYBRID_CAL=1/; s/hybrid calibration 3:/hybrid calibration 1:/' "$T/qyc-uncoupled.log" > "$T/qyc-rate.log"
sed 's/P1_HYBRID_CAL=3/P1_HYBRID_CAL=2/; s/hybrid calibration 3:/hybrid calibration 2:/' "$T/qyc-idle.log" > "$T/qyc-couple.log"
grep -v 'rank 2:' "$T/qyc-good.log" > "$T/qyc-partial.log"
sed '3s/; hybrid calibration .*//' "$T/qyc-good.log" > "$T/qyc-onemissing.log"
cat "$T/qyc-good.log" "$T/qyc-good.log" > "$T/qyc-twice.log"
cexpect() {
  local want=$1 label=$2 rc; shift 2
  qyc_gate "$@" >/dev/null 2>&1; rc=$?
  if [[ $rc == "$want" ]]; then ok "$label"; else bad "$label: exit $rc, want $want"; fi
}
cexpect 0 'Qyc startup line' "$T/qyc-good.log" 3 start
cexpect 0 'Qyc counts at exit' "$T/qyc-good.log" 3 exit
cexpect 0 'Qyc partial ranks still every line counted' "$T/qyc-partial.log" 3 exit
cexpect 1 'Qyc line names another mask' "$T/qyc-rate.log" 3 start
cexpect 1 'Qyc line twice' "$T/qyc-twice.log" 3 start
cexpect 1 'Qyc3 never sampled' "$T/qyc-idle.log" 3 exit
cexpect 1 'Qyc3 shares not coupled' "$T/qyc-uncoupled.log" 3 exit
cexpect 1 'Qyc3 a rank without its counts' "$T/qyc-onemissing.log" 3 exit
cexpect 0 'Qyc1 sampled, uncoupled' "$T/qyc-rate.log" 1 exit
cexpect 0 'Qyc2 coupled, no samples needed' "$T/qyc-couple.log" 2 exit
cexpect 1 'Qyc unrequested activation' "$T/qyc-good.log" '' start
cexpect 0 'Qyc explicitly off' "$T/qt-off.log" '' exit
cexpect 1 'Qyc asked, no line' "$T/qt-off.log" 3 start

# ---- the balance receipt: an old generic PASS is not requested-height evidence
eval "$(sed -n '/^bal_receipt_bound() {/,/^}/p' "$W")"
declare -F bal_receipt_bound >/dev/null || { echo "FAIL no bal_receipt_bound in $W"; exit 1; }
receipt() {   # want status, description, requested C, expected maximum, receipt line(s)
  local want=$1 what=$2 requested=$3 maximum=$4 got rc
  printf '%s\n' "$5" > "$T/bal.log"
  got=$(bal_receipt_bound "$T/bal.log" "$requested"); rc=$?
  if [[ $rc == "$want" && ( $rc != 0 || $got == "$maximum" ) ]]; then ok "$what"
  else bad "$what: status $rc/max '$got', wanted $want/'$maximum'"; fi
}
receipt 0 'the 2048 receipt keeps its actual bound' 2048 2048 \
  'bal: byte coverage max_M=2048 requested_C=2048 heights=1,6,16,17,64,256,512,1024,2048'
receipt 0 'the 3072 boundary has an actual receipt' 3072 3072 \
  'bal: byte coverage max_M=3072 requested_C=3072 heights=1,6,16,17,64,256,512,1024,2048,3072'
receipt 0 'an intermediate requested C is actually present' 4000 4000 \
  'bal: byte coverage max_M=4000 requested_C=4000 heights=1,6,16,17,64,256,512,1024,2048,3072,4000'
receipt 0 'the 4096 receipt covers its upper boundary' 4096 4096 \
  'bal: byte coverage max_M=4096 requested_C=4096 heights=1,6,16,17,64,256,512,1024,2048,3072,4096'
receipt 1 'a generic legacy PASS has no actual bound' 4096 '' \
  "PASS: bal: the balanced gemv and twin give the plain kernel's bytes at 1..4096 tokens"
receipt 1 'a short maximum cannot certify the requested C' 4096 '' \
  'bal: byte coverage max_M=2048 requested_C=4096 heights=1,6,16,17,64,256,512,1024,2048'
receipt 1 'a claimed bound missing from the tested list refuses' 4096 '' \
  'bal: byte coverage max_M=4096 requested_C=4096 heights=1,6,16,17,64,256,512,1024,2048,3072'
receipt 1 'a receipt for another C refuses' 3072 '' \
  'bal: byte coverage max_M=4096 requested_C=4096 heights=1,6,16,17,64,256,512,1024,2048,3072,4096'
receipt 1 'duplicate tested heights refuse' 4096 '' \
  'bal: byte coverage max_M=4096 requested_C=4096 heights=1,6,16,17,64,256,512,1024,2048,2048,3072,4096'
receipt 1 'a trailing empty tested height refuses' 4096 '' \
  'bal: byte coverage max_M=4096 requested_C=4096 heights=1,6,16,17,64,256,512,1024,2048,3072,4096,'

# ---- 2. the arm loop, stubbed
H=$T/home
mkdir -p "$H/kernels"
echo "not a library" > "$H/kernels/libr4d.so"
echo '{}' > "$T/prompts.json"
pre=all-t40-f
mid=H-p256-m196608-G-R
post=W-A2-Rd13-Er-Es-Pa-K-Sc2-F8-Pp4-Ei-Pm56-Pe-Ss-Pu-Is-Pi8-Ns7-Fw-Mr32-Lc1024
REF=$pre-M1024-$mid-B1-$post-Cs-Mb1-Lm4096
NEW=$pre-M1024-$mid-B6-$post-P1-Ql10-Cs-Mb1-Lm4096
NEWG=$pre-M1024-$mid-B6-$post-P1-Ql10-Qh3000-Cs-Mb1-Lm4096
SHORTM=$pre-M16-$mid-B6-$post-P1-Cs-Mb1-Lm4096
Q17=$pre-M16-$mid-B6-$post-P1-Qm17-Cs-Mb1-Lm4096
OVERL=$pre-M1024-$mid-B6-$post-P1-Ql12-Cs-Mb1-Lm4096
GNOL=$pre-M1024-$mid-B6-$post-P1-Qh3000-Cs-Mb1-Lm4096
LNOP=$pre-M1024-$mid-B1-$post-Ql10-Cs-Mb1-Lm4096
post2k=${post%-Lc1024}-Lc2048
BIG=$pre-M2048-$mid-B6-$post2k-P1-Qe64-Cs-Mb1-Lm4096
BIGG=$pre-M2048-$mid-B6-$post2k-P1-Qe64-Qh3000-Cs-Mb1-Lm4096
BIGM=$pre-M1024-$mid-B6-$post2k-P1-Qe64-Cs-Mb1-Lm4096
ENOP=$pre-M1024-$mid-B1-$post-Qe64-Cs-Mb1-Lm4096
post4k=${post%-Lc1024}-Lc4096
COLD4K=$pre-M4096-$mid-B6-$post4k-P1-Qe64-Pci4096-Cs-Mb1-Lm8192
PC4K=$pre-M4096-$mid-B6-$post4k-P1-Qe64-Pc-Pci4096-Cs-Mb1-Lm8192
NOINT4K=$pre-M4096-$mid-B6-$post4k-P1-Qe64-Cs-Mb1-Lm8192
LOWINT4K=$pre-M4096-$mid-B6-$post4k-P1-Qe64-Pci2048-Cs-Mb1-Lm8192
NOPC4K=$pre-M4096-$mid-B6-$post4k-P1-Qe64-Pci4096-Pce-Cs-Mb1-Lm8192
OVER4K=$pre-M4112-$mid-B6-${post%-Lc1024}-Lc4112-P1-Qe64-Pci8192-Cs-Mb1-Lm8192
# The fast arm the windows run (Lc2048, P1, Qe64, Mb1, Rb, Vp, Cs) and prompt-lookup speculation on top of it.
FAST=$pre-M2048-$mid-B6-$post2k-P1-Qe64-Mb1-Rb-Vp-Cs
SG8=$FAST-Sg8
SGA8=$FAST-Sga8
SG15=$FAST-Sg15
SG16=$FAST-Sg16
N2SG7=$FAST-N2-Sg7
N2SG8=$FAST-N2-Sg8
N8SG2=$FAST-Sg2-N8
SG0=$FAST-Sg0
QD200=$FAST-Qd200
QD4096=$FAST-Qd4096
QD16=$FAST-Qd16
QDTB=$FAST-Qd200-Tb
QDNOP=$REF-Qd200
QY25=$FAST-Qd64-Qy25
QYF30=$FAST-Qd64-Qyf30
QYNOQD=$FAST-Qy25
QYSG=$FAST-Qd64-Qy25-Sg8
QYV=$FAST-Qd64-Qy25-Qv8
QYAX=$FAST-Qd64-Qy25-Ax
QY0=$FAST-Qd64-Qy0
QYTWO=$FAST-Qd64-Qy25-Qy30
QYNOP=$REF-Qd64-Qy25
CTP=$FAST-Ctp
CTP2=$FAST-Qd64-Ct16-Ctp2
CTPNOP=$REF-Ctp
CTP0=$FAST-Ctp0
EE=$FAST-Ee
EEQY=$FAST-Qd64-Qy60-Ee
QY60=$FAST-Qd64-Qy60
QYP30=$FAST-Qyp30
QYPQY=$FAST-Qd64-Qy60-Qyp30
QYPSG=$FAST-Qyp30-Sg8
QYPV=$FAST-Qyp30-Qv8
QYPAX=$FAST-Qyp30-Ax
QYP0=$FAST-Qyp0
QYPTWO=$FAST-Qyp30-Qyp60
QYPNOP=$REF-Qyp30
QYK=$FAST-Qd64-Qy60-Qyk
QYKP=$FAST-Qd64-Qy60-Qyp60-Qyk
QYKTWO=$FAST-Qd64-Qy60-Qyk-Qyk
ARMS="$REF $NEW $NEWG $SHORTM $Q17 $OVERL $GNOL $LNOP $BIG $BIGG $BIGM $ENOP $COLD4K $PC4K $NOINT4K $LOWINT4K $NOPC4K $OVER4K"
ARMS="$ARMS $FAST $SG8 $SGA8 $SG15 $SG16 $N2SG7 $N2SG8 $N8SG2 $SG0"
ARMS="$ARMS $QD200 $QD4096 $QD16 $QDTB $QDNOP"
ARMS="$ARMS $QY25 $QYF30 $QYNOQD $QYSG $QYV $QYAX $QY0 $QYTWO $QYNOP"
ARMS="$ARMS $CTP $CTP2 $CTPNOP $CTP0"
QT=$FAST-Qt
QTQSP=$FAST-Qsp-Qt
QTNOP=$REF-Qt
QZ=$FAST-Qz
QTQZ=$FAST-Qsp-Qt-Qz
QZNOP=$REF-Qz
ARMS="$ARMS $QT $QTQSP $QTNOP $QZ $QTQZ $QZNOP"
ARMS="$ARMS $EE $EEQY $QY60"
ARMS="$ARMS $QYP30 $QYPQY $QYPSG $QYPV $QYPAX $QYP0 $QYPTWO $QYPNOP $QYK $QYKP $QYKTWO"
# Qw: the WMMA P1 twin (inexact by design) on the N256-like preset (Qsp-Qt-Qz), beside Qv, without P1, with Kld (no
# reference yet in the loop: skipped; with KLD_REF set: --kld-ref), and the wave part Qw2 (unchanged: the wave size).
QW=$FAST-Qsp-Qt-Qz-Qw
QWV=$FAST-Qw-Qv8
QWNOP=$REF-Qw
QWKLD=$FAST-Qw-Kld
QW2=$FAST-Qw2
ARMS="$ARMS $QW $QWV $QWNOP $QWKLD $QW2"
# Dt / Aw: the tiled bf16f8 GEMM (exact) and the WMMA sparse attention (inexact) on the preset; Dt without Lc and
# without F8, Aw without Lc, and both with Kld.
DTAW=$FAST-Qsp-Qt-Qz-Dt-Aw
DTNOLC=$pre-M16-$mid-B6-${post%-Lc1024}-Dt
DTNOF8=${FAST/-F8-/-}-Dt
AWNOLC=$pre-M16-$mid-B6-${post%-Lc1024}-Aw
AWKLD=$FAST-Aw-Kld
DTKLD=$FAST-Dt-Kld
AWONLY=$FAST-Aw
ARMS="$ARMS $DTAW $DTNOLC $DTNOF8 $AWNOLC $AWKLD $DTKLD $AWONLY"
QX=$FAST-Qd64-Qy60-Qx12
QXNOP=$REF-Qx12
QXBIG=$FAST-Qx65
QXTWO=$FAST-Qx4-Qx8
ARMS="$ARMS $QX $QXNOP $QXBIG $QXTWO"
QYC=$FAST-Qd64-Qy60-Qyp60-Qyk-Qyc3
QYCBASE=$FAST-Qd64-Qy60-Qyp60-Qyk
QYCNOH=$FAST-Qyc1
QYCBIG=$FAST-Qd64-Qy60-Qyc4
QYCTWO=$FAST-Qd64-Qy60-Qyc1-Qyc2
ARMS="$ARMS $QYC $QYCBASE $QYCNOH $QYCBIG $QYCTWO"
run_loop() {   # actual balance-oracle bound; every model action remains stubbed
 (
  set +u
  log() { echo "LOG: $*"; }
  power_set() { :; }; power_restore() { :; }
  vpack_gate() { return 0; }
  arm() { echo "ARM $1"; shift; printf '    %s\n' "$@"; echo "  ARGS: ${ARGS[*]}"; echo "  ARM_SG: ${ARM_SG:-}"; echo "  ARM_QD: ${ARM_QD:-}"; echo "  ARM_QY: ${ARM_QY:-}"; echo "  ARM_QYF: ${ARM_QYF:-}"; echo "  ARM_QYP: ${ARM_QYP:-}"; echo "  ARM_QYK: ${ARM_QYK:-}"; echo "  ARM_QW: ${ARM_QW:-}"; echo "  ARM_DT: ${ARM_DT:-}"; echo "  ARM_AW: ${ARM_AW:-}"; echo "  ARM_QT: ${ARM_QT:-}"; echo "  ARM_EE: ${ARM_EE:-}"; }
  HX_THREADS=16 SILU=/dev/null RHOME=$H RADIANCE_HOME_DIR2=$H MIRROR=$T
  PROMPTS=$T/prompts.json
  unset RADIANCE_TUNE
  SUM=$(sha256sum "$H/kernels/libr4d.so" | cut -d' ' -f1); LIB=$H/kernels/libr4d.so
  TALL_OK=1 TALL_WHY=test TALL_SUM=$SUM TALL_LIB=$LIB DK_OK=1 DK_WHY=test DK_SUM=$SUM DK_LIB=$LIB
  RB_OK=1 RB_WHY=test RB_SUM=$SUM RB_LIB=$LIB LCP_OK=1 LCP_WHY=test LCP_SUM=$SUM LCP_LIB=$LIB LCP_C=4096
  BAL_OK=1 BAL_WHY=test BAL_SUM=$SUM BAL_LIB=$LIB BAL_MAX_M=$1 P1_OK=1 P1_WHY=test P1_SUM=$SUM P1_LIB=$LIB
  if [[ ${2:-} == no-p1t ]]; then P1T_OK="" P1T_WHY="no p1tiled check (test)" P1T_SUM="" P1T_LIB=""
  else P1T_OK=1 P1T_WHY=test P1T_SUM=$SUM P1T_LIB=$LIB; fi
  if [[ ${2:-} == no-p1w ]]; then P1W_OK="" P1W_WHY="no p1wmma check (test)" P1W_SUM="" P1W_LIB=""
  else P1W_OK=1 P1W_WHY=test P1W_SUM=$SUM P1W_LIB=$LIB; fi
  if [[ ${2:-} == no-fast ]]; then DT_OK="" DT_WHY="no dtiled check (test)" DT_SUM="" DT_LIB="" AW_OK="" AW_WHY="no attnwm check (test)" AW_SUM="" AW_LIB=""
  else DT_OK=1 DT_WHY=test DT_SUM=$SUM DT_LIB=$LIB AW_OK=1 AW_WHY=test AW_SUM=$SUM AW_LIB=$LIB; fi
  KLD_REF=${3:-}
  OUT=$T/out; mkdir -p "$OUT"; CANCEL_FILE=/nonexistent/cancel; CANCEL=$CANCEL_FILE
  eval "$(sed -n '/^hotset_for() {/,/^}/p' "$W")"
  eval "$(sed -n '/^trace_base() {/,/^}/p' "$W")"
  eval "$(sed -n '/^ARM_PRESET_B2=/p;/^ARM_PRESET_B4=/p;/^arm_full() {/,/^}/p' "$W")"
  HX=(RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS=$HX_THREADS RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_SILU=$SILU)
  s=$(grep -n '^for a in \$ARMS; do' "$W" | head -1 | cut -d: -f1)
  e=$(awk -v s="$s" 'NR > s && /^done$/ { print NR; exit }' "$W")
  eval "$(sed -n "${s},${e}p" "$W")"
) 2>&1
}
# Qsp's oracle line, as a passing `p1` device check writes it (the Qsp+Qt arm needs it).
mkdir -p "$T/out"
echo 'p1: PASS: moe_gemm_ggml_mxfp4_gemv_tall_p1 at M 17..4096: ... splits and full/span SwiGLU waves of 1-3 equal to one launch, also with the tail filled once (52 cases)' > "$T/out/devcheck-p1.log"
echo 'p1tiled: PASS: moe_gemm_ggml_mxfp4_gemv_tall_p1 served on the register-tiled P1 twin at M 17..4096: ... equal to one launch, also with the tail filled once (54 cases)' > "$T/out/devcheck-p1tiled.log"
echo '{"prompt": "a test document"}' > "$T/out/kld-corpus.jsonl"
out=$(run_loop 4096)
echo "$out" > "$T/loop.txt"
section() { awk -v a="ARM $1" '$0 == a { p = 1; next } /^ARM / { p = 0 } p' "$T/loop.txt"; }
# The last value an arm states for a variable (later entries win in the arm's environment).
val() { grep -oE "^    $2=.*" <<< "$1" | tail -1 | cut -d= -f2-; }
has() {   # what section var want
  local got
  got=$(val "$2" "$3")
  if [[ "$got" == "$4" ]]; then ok "$1: $3=$4"; else bad "$1: $3 is '${got}', want '$4'"; fi
}
sec=$(section "$REF")
if [[ -z "$sec" ]]; then bad "the adopted arm was not launched: $(grep -m3 "arm $REF" "$T/loop.txt")"
else
  has adopted "$sec" RADIANCE_EXPERIMENTAL_DS41_P1 0
  has adopted "$sec" RADIANCE_EXPERIMENTAL_DS41_P1_LEND 0
  has adopted "$sec" RADIANCE_EXPERIMENTAL_DS41_P1_LEND_ARENA 0
  has adopted "$sec" RADIANCE_EXPERIMENTAL_DS41_P1_MIN_ROWS ""
  has adopted "$sec" RADIANCE_EXPERIMENTAL_DS41_LCP 1024
  has adopted "$sec" RADIANCE_EXPERIMENTAL_DS41_MOE_BALANCE 1
fi
for a in "$NEW" "$NEWG"; do
  sec=$(section "$a")
  if [[ -z "$sec" ]]; then bad "arm $a not launched: $(grep -m3 "arm $a" "$T/loop.txt")"; continue; fi
  n=P1+Ql10
  [[ "$a" == *-Qh3000-* ]] && n=P1+Ql10+Qh3000
  has "$n" "$sec" RADIANCE_EXPERIMENTAL_DS41_P1 1
  has "$n" "$sec" RADIANCE_EXPERIMENTAL_DS41_P1_LEND 10
  has "$n" "$sec" RADIANCE_EXPERIMENTAL_DS41_P1_LEND_ARENA 0
  has "$n" "$sec" RADIANCE_EXPERIMENTAL_DS41_P1_MIN_ROWS ""
  has "$n" "$sec" RADIANCE_EXPERIMENTAL_DS41_FILE_BUFFER_UNITS 6
  has "$n" "$sec" RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_MISSES 1024
  has "$n" "$sec" RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_MAX_ROWS 32
  has "$n" "$sec" RADIANCE_EXPERIMENTAL_DS41_MOE_BALANCE 1
  has "$n" "$sec" RADIANCE_EXPERIMENTAL_DS41_CONSUMER_SKIP 1
  [[ "$sec" == *"--max-num-batched-tokens 1024"* ]] && ok "$n: chunks of 1024" || bad "$n: no --max-num-batched-tokens 1024"
done
has "P1+Ql10+Qh3000" "$(section "$NEWG")" RADIANCE_EXPERIMENTAL_DS41_P1_GRACE_MS 3000
has "P1+Ql10" "$(section "$NEW")" RADIANCE_EXPERIMENTAL_DS41_P1_GRACE_MS ""
sec=$(section "$Q17")
[[ -n "$sec" ]] && has "Qm17 with M16" "$sec" RADIANCE_EXPERIMENTAL_DS41_P1_MIN_ROWS 17 || bad "the Qm17 M16 arm was not launched"
for a in "$BIG" "$BIGG"; do
  sec=$(section "$a")
  if [[ -z "$sec" ]]; then bad "arm $a not launched: $(grep -m3 "arm $a" "$T/loop.txt")"; continue; fi
  n=P1+Lc2048+Qe64
  [[ "$a" == *-Qh3000-* ]] && n=P1+Lc2048+Qe64+Qh3000
  has "$n" "$sec" RADIANCE_EXPERIMENTAL_DS41_P1 1
  has "$n" "$sec" RADIANCE_EXPERIMENTAL_DS41_P1_LEND_ARENA 64
  has "$n" "$sec" RADIANCE_EXPERIMENTAL_DS41_P1_LEND 0
  has "$n" "$sec" RADIANCE_EXPERIMENTAL_DS41_LCP 2048
  has "$n" "$sec" RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_MISSES 2048
  [[ "$sec" == *"--max-num-batched-tokens 2048"* ]] && ok "$n: chunks of 2048" || bad "$n: no --max-num-batched-tokens 2048"
done
has "P1+Lc2048+Qe64+Qh3000" "$(section "$BIGG")" RADIANCE_EXPERIMENTAL_DS41_P1_GRACE_MS 3000
for a in "$COLD4K" "$PC4K"; do
  sec=$(section "$a")
  if [[ -z "$sec" ]]; then bad "4K arm $a was not launched"; continue; fi
  has "LCP4096" "$sec" RADIANCE_EXPERIMENTAL_DS41_LCP 4096
  has "LCP4096" "$sec" RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_MISSES 4096
  has "LCP4096" "$sec" RADIANCE_EXPERIMENTAL_DS41_P1_LEND_ARENA 64
  [[ "$sec" == *"--checkpoint-interval 4096"* ]] && ok "LCP4096: explicit interval 4096" || bad "LCP4096: missing interval 4096"
done
for a in "$SHORTM" "$OVERL" "$GNOL" "$LNOP" "$BIGM" "$ENOP" "$NOINT4K" "$LOWINT4K" "$NOPC4K" "$OVER4K"; do
  if [[ -n "$(section "$a")" ]]; then bad "arm $a was launched"
  elif grep -q "LOG: arm $a: .*skipped" "$T/loop.txt"; then ok "skipped: $(grep -m1 "LOG: arm $a: " "$T/loop.txt" | sed "s|LOG: arm $a: ||")"
  else bad "arm $a neither launched nor skipped"; fi
done
for a in "$REF" "$NEW" "$NEWG" "$Q17" "$BIG" "$BIGG" "$COLD4K" "$PC4K"; do
  if grep -q "LOG: arm $a: .*\(skipped\|refused\)" "$T/loop.txt"; then bad "arm $a: $(grep -m1 "LOG: arm $a: " "$T/loop.txt")"; fi
done

# Prompt-lookup speculation on the fast arm: launched with the speculation's arguments and the drafter left
# out, every other switch as the fast arm states it; skipped where a step of decode rows passes 16.
fast=$(section "$FAST")
[[ -n "$fast" ]] || bad "the fast arm was not launched: $(grep -m3 "arm $FAST" "$T/loop.txt")"
[[ "$fast" == *"  ARM_SG: "$'\n'* || "$fast" == *"  ARM_SG: " ]] && ok "the fast arm names no speculation" || bad "the fast arm's ARM_SG is not empty"
[[ "$fast" != *"--num-speculative-tokens"* ]] && ok "the fast arm adds no speculation argument" || bad "the fast arm passes --num-speculative-tokens"
sgarm() {   # arm, K, --max-num-seqs or empty, adaptive (non-empty)
  local a=$1 k=$2 n=$3 ad=$4 sec v
  sec=$(section "$a")
  if [[ -z "$sec" ]]; then bad "arm $a not launched: $(grep -m3 "arm $a" "$T/loop.txt")"; return; fi
  if grep -q "LOG: arm $a: .*\(skipped\|refused\)" "$T/loop.txt"; then bad "arm $a: $(grep -m1 "LOG: arm $a: " "$T/loop.txt")"; fi
  [[ "$sec" == *"--num-speculative-tokens $k --spec-ngram $k"* ]] && ok "Sg$k${n:+ N$n}: $k drafts by prompt lookup" || bad "Sg$k${n:+ N$n}: no --num-speculative-tokens $k --spec-ngram $k"
  if [[ -n "$ad" ]]; then
    [[ "$sec" == *"--draft-window adaptive"* ]] && ok "Sga$k: the adaptive window" || bad "Sga$k: no --draft-window adaptive"
  fi
  [[ -z "$n" || "$sec" == *"--max-num-seqs $n"* ]] && ok "Sg$k${n:+ N$n}: its sequences" || bad "Sg$k N$n: no --max-num-seqs $n"
  grep -qx "  ARM_SG: $k" <<< "$sec" && ok "Sg$k${n:+ N$n}: the arm is gated on its admission lines (ARM_SG=$k)" || bad "Sg$k${n:+ N$n}: ARM_SG is '$(grep '  ARM_SG:' <<< "$sec")'"
  has "Sg$k${n:+ N$n}" "$sec" RADIANCE_EXPERIMENTAL_DS41_LOOKUP_ALONE 1
  # Every switch the fast arm states, the speculating arm states alike.
  for v in RADIANCE_EXPERIMENTAL_DS41_LCP RADIANCE_EXPERIMENTAL_DS41_P1 RADIANCE_EXPERIMENTAL_DS41_P1_LEND_ARENA \
           RADIANCE_EXPERIMENTAL_DS41_CONSUMER_SKIP RADIANCE_EXPERIMENTAL_DS41_VRAM_PACKED RADIANCE_EXPERIMENTAL_DS41_MOE_BALANCE \
           RADIANCE_EXPERIMENTAL_DS41_ROUTER_BF16 RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_MISSES RADIANCE_EXPERIMENTAL_DS41_FILE_BUFFER_UNITS \
           RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_MAX_ROWS RADIANCE_EXPERIMENTAL_DS41_EARLY_ISSUE RADIANCE_EXPERIMENTAL_DS41_SKIP_POST_STOP \
           RADIANCE_EXPERIMENTAL_PREFIX_EORS RADIANCE_EXPERIMENTAL_DS41_PREFETCH_MULTIROW; do
    has "Sg$k${n:+ N$n} as the fast arm" "$sec" "$v" "$(val "$fast" "$v")"
  done
  [[ "$sec" == *"--max-num-batched-tokens 2048"* ]] && ok "Sg$k${n:+ N$n}: chunks of 2048" || bad "Sg$k${n:+ N$n}: no --max-num-batched-tokens 2048"
}
sgarm "$SG8" 8 "" ""
sgarm "$SGA8" 8 "" 1
sgarm "$SG15" 15 "" ""
sgarm "$N2SG7" 7 2 ""
for a in "$SG16" "$N2SG8" "$N8SG2"; do
  if [[ -n "$(section "$a")" ]]; then bad "arm $a was launched"
  elif grep -q "LOG: arm $a: Sg[0-9]* beside Lc2048: .*more than the 16 decode rows a step.*skipped" "$T/loop.txt"; then
    ok "skipped: $(grep -m1 "LOG: arm $a: " "$T/loop.txt" | sed "s|LOG: arm $a: ||" | cut -c1-110)..."
  else bad "arm $a neither launched nor skipped for its rows: $(grep -m1 "LOG: arm $a: " "$T/loop.txt")"; fi
done
# Sg0 is no speculation: launched, not gated.
sec=$(section "$SG0")
if [[ -z "$sec" ]]; then bad "the Sg0 arm was not launched: $(grep -m3 "arm $SG0" "$T/loop.txt")"
else grep -qx "  ARM_SG: " <<< "$sec" && ok "Sg0: no speculation, no gate" || bad "Sg0: ARM_SG is '$(grep '  ARM_SG:' <<< "$sec")'"; fi

# ---- 5b. QdN in the arm loop: Qd200 on the fast arm states the switch at 200 and every other switch as the fast
# arm; every arm without the part states it 0; Qd above the chunk, Qd16, Qd beside Tb and Qd without P1 skipped.
sec=$(section "$QD200")
if [[ -z "$sec" ]]; then bad "the Qd200 arm was not launched: $(grep -m3 "arm $QD200" "$T/loop.txt")"
else
  has "Qd200" "$sec" RADIANCE_EXPERIMENTAL_DS41_P1_DECODE_ROWS 200
  grep -qx "  ARM_QD: 200" <<< "$sec" && ok "Qd200: the arm is gated on its decode line (ARM_QD=200)" || bad "Qd200: ARM_QD is '$(grep '  ARM_QD:' <<< "$sec")'"
  for v in RADIANCE_EXPERIMENTAL_DS41_LCP RADIANCE_EXPERIMENTAL_DS41_P1 RADIANCE_EXPERIMENTAL_DS41_P1_LEND_ARENA \
           RADIANCE_EXPERIMENTAL_DS41_P1_MIN_ROWS RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_MISSES \
           RADIANCE_EXPERIMENTAL_DS41_FILE_BUFFER_UNITS RADIANCE_EXPERIMENTAL_SCHED_TBO; do
    has "Qd200 as the fast arm" "$sec" "$v" "$(val "$fast" "$v")"
  done
fi
for a in "$REF" "$NEW" "$BIG" "$FAST" "$SG8"; do
  has "no Qd part ($a)" "$(section "$a")" RADIANCE_EXPERIMENTAL_DS41_P1_DECODE_ROWS 0
done
grep -qx "  ARM_QD: " <<< "$fast" && ok "the fast arm names no Qd" || bad "the fast arm's ARM_QD is '$(grep '  ARM_QD:' <<< "$fast")'"
qdskip() {   # arm, the reason's pattern
  if [[ -n "$(section "$1")" ]]; then bad "arm $1 was launched"
  elif grep -q "LOG: arm $1: $2" "$T/loop.txt"; then ok "skipped: $(grep -m1 "LOG: arm $1: $2" "$T/loop.txt" | sed "s|LOG: arm $1: ||" | cut -c1-100)"
  else bad "arm $1 not skipped for '$2': $(grep -m1 "LOG: arm $1: " "$T/loop.txt")"; fi
}
qdskip "$QD4096" "Qd4096 is above the chunk Lc2048.*skipped"
qdskip "$QD16" "Qd16: Qd takes 17-4096.*skipped"
qdskip "$QDTB" "Qd200 with Tb.*skipped"
qdskip "$QDNOP" "the Q parts (RADIANCE_EXPERIMENTAL_DS41_P1_DECODE_ROWS=200) need P1.*skipped"

# ---- 6b. QyP / QyfP in the arm loop.
for spec in "$QY25:25:" "$QYF30:30:1"; do
  IFS=: read -r a n f <<< "$spec"
  sec=$(section "$a")
  if [[ -z "$sec" ]]; then bad "the Qy$n arm was not launched: $(grep -m3 "arm $a" "$T/loop.txt")"; continue; fi
  has "Qy${f:+f}$n" "$sec" RADIANCE_EXPERIMENTAL_DS41_P1_HYBRID "$n"
  has "Qy${f:+f}$n" "$sec" RADIANCE_EXPERIMENTAL_DS41_P1_HYBRID_ADAPT "${f:+0}"
  has "Qy${f:+f}$n" "$sec" RADIANCE_EXPERIMENTAL_DS41_P1_DECODE_ROWS 64
  grep -qx "  ARM_QY: $n" <<< "$sec" && ok "Qy${f:+f}$n: the arm is gated on its hybrid line (ARM_QY=$n)" || bad "Qy$n: ARM_QY is '$(grep '  ARM_QY:' <<< "$sec")'"
  grep -qx "  ARM_QYF: $f" <<< "$sec" && ok "Qy${f:+f}$n: ARM_QYF='$f'" || bad "Qy$n: ARM_QYF is '$(grep '  ARM_QYF:' <<< "$sec")'"
  for v in RADIANCE_EXPERIMENTAL_DS41_LCP RADIANCE_EXPERIMENTAL_DS41_P1 RADIANCE_EXPERIMENTAL_DS41_P1_LEND_ARENA \
           RADIANCE_EXPERIMENTAL_DS41_P1_MIN_ROWS RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_MISSES \
           RADIANCE_EXPERIMENTAL_DS41_FILE_BUFFER_UNITS RADIANCE_EXPERIMENTAL_SCHED_TBO RADIANCE_EXPERIMENTAL_DS41_P1_VERIFY; do
    has "Qy${f:+f}$n as the fast arm" "$sec" "$v" "$(val "$fast" "$v")"
  done
done
for a in "$REF" "$NEW" "$BIG" "$FAST" "$SG8" "$QD200"; do
  has "no Qy part ($a)" "$(section "$a")" RADIANCE_EXPERIMENTAL_DS41_P1_HYBRID 0
  has "no Qy part ($a)" "$(section "$a")" RADIANCE_EXPERIMENTAL_DS41_P1_HYBRID_ADAPT ""
done
grep -qx "  ARM_QY: " <<< "$fast" && ok "the fast arm names no Qy" || bad "the fast arm's ARM_QY is '$(grep '  ARM_QY:' <<< "$fast")'"
qdskip "$QYNOQD" "Qy25 splits the decode passes a Qd part arms, and there is none.*skipped"
qdskip "$QYSG" "Qy25 beside Sg8 (speculation.*skipped"
qdskip "$QYV" "Qy25 beside Qv8 (the host verify.*skipped"
qdskip "$QYAX" "Qy25 beside the aux expert worker.*skipped"
qdskip "$QY0" "Qy0: Qy takes 1-400.*skipped"
qdskip "$QYTWO" "two Qy parts.*skipped"
qdskip "$QYNOP" "the Q parts (.*RADIANCE_EXPERIMENTAL_DS41_P1_HYBRID=25.*) need P1.*skipped"
# ---- 6c. QypP in the arm loop.
for spec in "$QYP30:30:" "$QYPQY:30:60"; do
  IFS=: read -r a n qyn <<< "$spec"
  sec=$(section "$a")
  if [[ -z "$sec" ]]; then bad "the Qyp$n arm was not launched: $(grep -m3 "arm $a" "$T/loop.txt")"; continue; fi
  has "Qyp$n${qyn:+ beside Qy$qyn}" "$sec" RADIANCE_EXPERIMENTAL_DS41_P1_HYBRID_PREFILL "$n"
  has "Qyp$n${qyn:+ beside Qy$qyn}" "$sec" RADIANCE_EXPERIMENTAL_DS41_P1_HYBRID "${qyn:-0}"
  has "Qyp$n${qyn:+ beside Qy$qyn}" "$sec" RADIANCE_EXPERIMENTAL_DS41_P1_HYBRID_ADAPT ""
  has "Qyp$n${qyn:+ beside Qy$qyn}" "$sec" RADIANCE_EXPERIMENTAL_DS41_P1_DECODE_ROWS "$([[ -n "$qyn" ]] && echo 64 || echo 0)"
  grep -qx "  ARM_QYP: $n" <<< "$sec" && ok "Qyp$n: the arm is gated on its hybrid prefill line (ARM_QYP=$n)" || bad "Qyp$n: ARM_QYP is '$(grep '  ARM_QYP:' <<< "$sec")'"
  grep -qx "  ARM_QY: $qyn" <<< "$sec" && ok "Qyp$n: ARM_QY='$qyn'" || bad "Qyp$n: ARM_QY is '$(grep '  ARM_QY:' <<< "$sec")'"
  for v in RADIANCE_EXPERIMENTAL_DS41_LCP RADIANCE_EXPERIMENTAL_DS41_P1 RADIANCE_EXPERIMENTAL_DS41_P1_LEND_ARENA \
           RADIANCE_EXPERIMENTAL_DS41_P1_MIN_ROWS RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_MISSES \
           RADIANCE_EXPERIMENTAL_DS41_FILE_BUFFER_UNITS RADIANCE_EXPERIMENTAL_SCHED_TBO RADIANCE_EXPERIMENTAL_DS41_P1_VERIFY; do
    has "Qyp$n as the fast arm" "$sec" "$v" "$(val "$fast" "$v")"
  done
done
for a in "$REF" "$NEW" "$BIG" "$FAST" "$SG8" "$QD200" "$QY25" "$QY60"; do
  has "no Qyp part ($a)" "$(section "$a")" RADIANCE_EXPERIMENTAL_DS41_P1_HYBRID_PREFILL 0
  grep -qx "  ARM_QYP: " <<< "$(section "$a")" && ok "no Qyp part ($a): ARM_QYP empty" || bad "no Qyp part ($a): ARM_QYP is '$(grep '  ARM_QYP:' <<< "$(section "$a")")'"
done
qdskip "$QYPSG" "Qyp30 beside Sg8 (speculation.*skipped"
qdskip "$QYPV" "Qyp30 beside Qv8 (the host verify.*skipped"
qdskip "$QYPAX" "Qyp30 beside the aux expert worker.*skipped"
qdskip "$QYP0" "Qyp0: Qyp takes 1-400.*skipped"
qdskip "$QYPTWO" "two Qyp parts.*skipped"
qdskip "$QYPNOP" "the Q parts (.*RADIANCE_EXPERIMENTAL_DS41_P1_HYBRID_PREFILL=30.*) need P1.*skipped"
# ---- 6d. Qyk in the arm loop.
for spec in "$QYK::60" "$QYKP:60:60"; do
  IFS=: read -r a pf qyn <<< "$spec"
  sec=$(section "$a")
  if [[ -z "$sec" ]]; then bad "the Qyk arm $a was not launched: $(grep -m3 "arm $a" "$T/loop.txt")"; continue; fi
  has "Qyk${pf:+ Qyp$pf}" "$sec" RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_KERNEL_COST 1
  has "Qyk${pf:+ Qyp$pf}" "$sec" RADIANCE_EXPERIMENTAL_DS41_P1_HYBRID "$qyn"
  has "Qyk${pf:+ Qyp$pf}" "$sec" RADIANCE_EXPERIMENTAL_DS41_P1_HYBRID_PREFILL "${pf:-0}"
  grep -qx "  ARM_QYK: 1" <<< "$sec" && ok "Qyk: the arm is gated on the pool's line (ARM_QYK=1)" || bad "Qyk: ARM_QYK is '$(grep '  ARM_QYK:' <<< "$sec")'"
  for v in RADIANCE_EXPERIMENTAL_DS41_LCP RADIANCE_EXPERIMENTAL_DS41_P1 RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_MAX_ROWS \
           RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_MISSES RADIANCE_EXPERIMENTAL_DS41_FILE_BUFFER_UNITS; do
    has "Qyk as the fast arm" "$sec" "$v" "$(val "$fast" "$v")"
  done
done
for a in "$REF" "$FAST" "$QY60" "$QYP30" "$QYPQY"; do
  has "no Qyk part ($a)" "$(section "$a")" RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_KERNEL_COST 0
  grep -qx "  ARM_QYK: " <<< "$(section "$a")" && ok "no Qyk part ($a): ARM_QYK empty" || bad "no Qyk part ($a): ARM_QYK is '$(grep '  ARM_QYK:' <<< "$(section "$a")")'"
done
qdskip "$QYKTWO" "two Qyk parts.*skipped"
# ---- 7. Ctp: the prefill chain trace's switch and output prefix; skipped without P1 or at 0.
for x in "$CTP 4" "$CTP2 2"; do
  read -r a n <<< "$x"
  sec=$(section "$a")
  if [[ -z "$sec" ]]; then bad "the Ctp arm $a was not launched: $(grep -m3 "arm $a" "$T/loop.txt")"; continue; fi
  has "Ctp ($n)" "$sec" RADIANCE_EXPERIMENTAL_DS41_CHAIN_TRACE_PREFILL "$n"
  [[ "$(val "$sec" RADIANCE_EXPERIMENTAL_DS41_CHAIN_TRACE_OUT)" == */chain-$a ]] && ok "Ctp ($n): traced into chain-<arm>" ||
    bad "Ctp ($n): CHAIN_TRACE_OUT is '$(val "$sec" RADIANCE_EXPERIMENTAL_DS41_CHAIN_TRACE_OUT)'"
  has "Ctp ($n) keeps P1" "$sec" RADIANCE_EXPERIMENTAL_DS41_P1 1
done
sec=$(section "$CTP2")
has "Ct16 beside Ctp2" "$sec" RADIANCE_EXPERIMENTAL_DS41_CHAIN_TRACE 16
has "Qd64 beside Ctp2" "$sec" RADIANCE_EXPERIMENTAL_DS41_P1_DECODE_ROWS 64
has "Ctp alone sets no decode trace" "$(section "$CTP")" RADIANCE_EXPERIMENTAL_DS41_CHAIN_TRACE ""
for a in "$FAST" "$QD200" "$REF"; do
  has "no Ctp part ($a)" "$(section "$a")" RADIANCE_EXPERIMENTAL_DS41_CHAIN_TRACE_PREFILL ""
done
qdskip "$CTPNOP" "Ctp4 traces armed card-path prefill passes, which need P1.*skipped"
qdskip "$CTP0" "Ctp0: CtpN takes a positive N.*skipped"
if pyout=$(python3 -I "$(dirname "$W")/chain_report.py" --selftest 2>&1) && grep -q "^selftest ran" <<< "$pyout"; then
  ok "chain_report.py --selftest (the --split classes of a prefill pass)"
else bad "chain_report.py --selftest: $(tail -5 <<< "$pyout")"; fi
# ---- 8a. Qt in the arm loop: on the fast arm (and beside Qsp) the switch at 1 and every other switch as the fast
# arm; every arm without the part states it 0; Qt without P1 skipped; with no p1tiled receipt every Qt arm refused.
for a in "$QT" "$QTQSP"; do
  sec=$(section "$a")
  if [[ -z "$sec" ]]; then bad "the Qt arm $a was not launched: $(grep -m3 "arm $a" "$T/loop.txt")"; continue; fi
  has "Qt" "$sec" RADIANCE_EXPERIMENTAL_DS41_P1_TILED 1
  grep -qx "  ARM_QT: 1" <<< "$sec" && ok "Qt: the arm is gated on its startup markers (ARM_QT=1)" || bad "Qt: ARM_QT is '$(grep '  ARM_QT:' <<< "$sec")'"
  for v in RADIANCE_EXPERIMENTAL_DS41_LCP RADIANCE_EXPERIMENTAL_DS41_P1 RADIANCE_EXPERIMENTAL_DS41_P1_LEND_ARENA \
           RADIANCE_EXPERIMENTAL_DS41_P1_MIN_ROWS RADIANCE_EXPERIMENTAL_DS41_FILE_BUFFER_UNITS RADIANCE_EXPERIMENTAL_DS41_P1_DECODE_ROWS; do
    has "Qt as the fast arm" "$sec" "$v" "$(val "$fast" "$v")"
  done
done
has "Qsp+Qt" "$(section "$QTQSP")" RADIANCE_EXPERIMENTAL_DS41_P1_SWIGLU_SPAN 1
for a in "$REF" "$NEW" "$BIG" "$FAST" "$SG8" "$QD200"; do
  has "no Qt part ($a)" "$(section "$a")" RADIANCE_EXPERIMENTAL_DS41_P1_TILED 0
done
qdskip "$QTNOP" "Qt needs P1; the arm is skipped"
nop1t_out=$(run_loop 4096 no-p1t)
if grep -qxF "ARM $QT" <<< "$nop1t_out"; then bad "a Qt arm launched without a p1tiled receipt"
elif grep -qF "arm $QT: Qt refused: no p1tiled check (test)" <<< "$nop1t_out"; then ok 'Qt refused without a p1tiled receipt'
else bad "no p1tiled refusal for arm $QT: $(grep -m1 "arm $QT: " <<< "$nop1t_out")"; fi
grep -qxF "ARM $FAST" <<< "$nop1t_out" && ok 'the fast arm runs without a p1tiled receipt' || bad 'the fast arm needed a p1tiled receipt'

# ---- 8b. Qz in the arm loop: alone and with Qsp-Qt on the fast arm, the switch at 1 (Qt's too in the second);
# every arm without the part states it 0; Qz without P1 skipped; without the oracle's tail-once line refused.
for a in "$QZ" "$QTQZ"; do
  sec=$(section "$a")
  if [[ -z "$sec" ]]; then bad "the Qz arm $a was not launched: $(grep -m3 "arm $a" "$T/loop.txt")"; continue; fi
  has "Qz" "$sec" RADIANCE_EXPERIMENTAL_DS41_P1_TAIL_ONCE 1
  for v in RADIANCE_EXPERIMENTAL_DS41_LCP RADIANCE_EXPERIMENTAL_DS41_P1 RADIANCE_EXPERIMENTAL_DS41_P1_LEND_ARENA; do
    has "Qz as the fast arm" "$sec" "$v" "$(val "$fast" "$v")"
  done
done
has "Qsp+Qt+Qz" "$(section "$QTQZ")" RADIANCE_EXPERIMENTAL_DS41_P1_TILED 1
has "Qsp+Qt+Qz" "$(section "$QTQZ")" RADIANCE_EXPERIMENTAL_DS41_P1_SWIGLU_SPAN 1
has "Qz alone" "$(section "$QZ")" RADIANCE_EXPERIMENTAL_DS41_P1_TILED 0
for a in "$REF" "$FAST" "$QT" "$QD200"; do
  has "no Qz part ($a)" "$(section "$a")" RADIANCE_EXPERIMENTAL_DS41_P1_TAIL_ONCE 0
done
qdskip "$QZNOP" "Qz needs P1; the arm is skipped"
sed -i 's/, also with the tail filled once//' "$T/out/devcheck-p1.log"
noz_out=$(run_loop 4096)
if grep -qxF "ARM $QZ" <<< "$noz_out"; then bad "a Qz arm launched without the oracle's tail-once waves"
elif grep -qF "arm $QZ: Qz needs the p1 oracle's tail-once waves" <<< "$noz_out"; then ok 'Qz refused without the tail-once oracle line'
else bad "no tail-once refusal for arm $QZ: $(grep -m1 "arm $QZ: " <<< "$noz_out")"; fi

# ---- 8. Ee: ee_gate on synthetic logs, then the arm loop.
ee_line='radiance: deepseek41: Engram rows early (RADIANCE_EXPERIMENTAL_DS41_ENGRAM_EARLY=1): after each Engram layer'"'"'s host gather the next one'"'"'s table rows are asked for on a helper thread, each rank 1/4 of the step'"'"'s rows; the gathers themselves are unchanged'
ee_stats() { echo "radiance: deepseek41: rank $1: Engram rows early: $2 read-aheads started, 3 with nothing to ask for, 0 past the pass's last layer, $3 refused"; }
mk ee "$lc_line" "$ee_line"
mk eeoff "$lc_line"
{ cat "$OUT/ee.server.log"; for r in 0 1 2 3; do ee_stats $r 120 0; done; } > "$OUT/eeexit.server.log"
{ cat "$OUT/ee.server.log"; for r in 0 1 2 3; do ee_stats $r 0 0; done; } > "$OUT/eeidle.server.log"
{ cat "$OUT/ee.server.log"; for r in 0 1 2 3; do ee_stats $r 120 $((r == 2)); done; } > "$OUT/eeref.server.log"
{ cat "$OUT/ee.server.log"; ee_stats 0 120 0; echo "radiance: deepseek41: rank 1: Engram rows early: lots"; } > "$OUT/eeodd.server.log"
{ cat "$OUT/eeoff.server.log"; ee_stats 0 5 0; } > "$OUT/eestray.server.log"
eeexpect() {   # want(0/1) what log EE stage
  local want=$1 what=$2 rc
  ee_gate "$OUT/$3.server.log" "$4" "$5" >/dev/null 2>&1; rc=$?
  if [[ $rc == "$want" ]]; then ok "$what"; else bad "$what: exit $rc, want $want"; fi
}
eeexpect 0 "Ee, its line"                                   ee      1  start
eeexpect 1 "Ee, no line"                                    eeoff   1  start
eeexpect 1 "no Ee part on a server that reads early"        ee      "" start
eeexpect 0 "no Ee part, no line"                            eeoff   "" start
eeexpect 0 "Ee at exit: every rank started read-aheads"     eeexit  1  exit
eeexpect 1 "Ee at exit: none started (never acted)"         eeidle  1  exit
eeexpect 1 "Ee at exit: a refused read-ahead"               eeref   1  exit
eeexpect 1 "Ee at exit: a rank's line without its counts"   eeodd   1  exit
eeexpect 1 "Ee at exit: no rank line"                       ee      1  exit
eeexpect 1 "no Ee part at exit, but a rank line"            eestray "" exit
eeexpect 0 "no Ee part at exit, no line"                    eeoff   "" exit
for x in "$EE $FAST" "$EEQY $QY60"; do
  read -r a b <<< "$x"
  sec=$(section "$a"); base=$(section "$b")
  if [[ -z "$sec" ]]; then bad "the Ee arm $a was not launched: $(grep -m3 "arm $a" "$T/loop.txt")"; continue; fi
  if [[ -z "$base" ]]; then bad "the arm $b was not launched: $(grep -m3 "arm $b" "$T/loop.txt")"; continue; fi
  has "Ee ($a)" "$sec" RADIANCE_EXPERIMENTAL_DS41_ENGRAM_EARLY 1
  grep -qx "  ARM_EE: 1" <<< "$sec" && ok "Ee: the arm is gated on its lines (ARM_EE=1)" || bad "Ee: ARM_EE is '$(grep '  ARM_EE:' <<< "$sec")'"
  # Every other line the arm states, as the arm without Ee.
  if diff <(grep -v -e 'ENGRAM_EARLY' -e '^  ARM_EE:' -e "^ARM " -e 'CHAIN_TRACE_OUT' <<< "$sec") \
          <(grep -v -e 'ENGRAM_EARLY' -e '^  ARM_EE:' -e "^ARM " -e 'CHAIN_TRACE_OUT' <<< "$base") >/dev/null; then
    ok "Ee ($a): every other switch and argument as $b"
  else bad "Ee ($a): differs from $b beyond the switch: $(diff <(grep -v ENGRAM_EARLY <<< "$sec") <(grep -v ENGRAM_EARLY <<< "$base") | head -4 | tr '\n' ' ')"; fi
done
for a in "$REF" "$FAST" "$QD200" "$QY60" "$QY25"; do
  has "no Ee part ($a)" "$(section "$a")" RADIANCE_EXPERIMENTAL_DS41_ENGRAM_EARLY ""
  grep -qx "  ARM_EE: " <<< "$(section "$a")" && ok "no Ee part ($a): ARM_EE empty" || bad "no Ee part ($a): ARM_EE is '$(grep '  ARM_EE:' <<< "$(section "$a")")'"
done
# ---- 8c. Qx in the arm loop: on the fast arm with Qd64-Qy60 the switch at 12 and every other switch as the fast arm
# with Qd64-Qy60; every arm without the part states it 0; Qx without P1, Qx65 and two Qx parts skipped.
sec=$(section "$QX")
if [[ -z "$sec" ]]; then bad "the Qx arm $QX was not launched: $(grep -m3 "arm $QX" "$T/loop.txt")"
else
  has "Qx" "$sec" RADIANCE_EXPERIMENTAL_DS41_P1_EARLY 12
  has "Qx beside Qd64" "$sec" RADIANCE_EXPERIMENTAL_DS41_P1_DECODE_ROWS 64
  has "Qx beside Qy60" "$sec" RADIANCE_EXPERIMENTAL_DS41_P1_HYBRID 60
  for v in RADIANCE_EXPERIMENTAL_DS41_LCP RADIANCE_EXPERIMENTAL_DS41_P1 RADIANCE_EXPERIMENTAL_DS41_P1_LEND_ARENA \
           RADIANCE_EXPERIMENTAL_DS41_FILE_BUFFER_UNITS RADIANCE_EXPERIMENTAL_DS41_P1_TAIL_ONCE RADIANCE_EXPERIMENTAL_DS41_P1_TILED; do
    has "Qx as the fast arm" "$sec" "$v" "$(val "$fast" "$v")"
  done
fi
for a in "$REF" "$NEW" "$FAST" "$QD200" "$QY25" "$QZ" "$QT"; do
  has "no Qx part ($a)" "$(section "$a")" RADIANCE_EXPERIMENTAL_DS41_P1_EARLY 0
done
qdskip "$QXNOP" "Qx needs P1; the arm is skipped"
qdskip "$QXBIG" "Qx65: Qx takes 1-64 (units); the arm is skipped"
qdskip "$QXTWO" "two Qx parts; the arm is skipped"
# ---- 6e. Qyc in the arm loop: the mask at 3 and every other switch as the arm without it; every other arm states 0;
# Qyc without Qy or Qyp, Qyc4 and two Qyc parts skipped.
sec=$(section "$QYC")
base=$(section "$QYCBASE")
if [[ -z "$sec" || -z "$base" ]]; then bad "the Qyc arms $QYC / $QYCBASE were not launched: $(grep -m3 "arm $QYC" "$T/loop.txt")"
else
  has "Qyc" "$sec" RADIANCE_EXPERIMENTAL_DS41_P1_HYBRID_CAL 3
  has "Qyc beside Qy60" "$sec" RADIANCE_EXPERIMENTAL_DS41_P1_HYBRID 60
  has "Qyc beside Qyp60" "$sec" RADIANCE_EXPERIMENTAL_DS41_P1_HYBRID_PREFILL 60
  has "no Qyc part ($QYCBASE)" "$base" RADIANCE_EXPERIMENTAL_DS41_P1_HYBRID_CAL 0
  if diff <(grep -v 'HYBRID_CAL\|ARM_QYC\|^ARM \|Qyc' <<< "$sec") <(grep -v 'HYBRID_CAL\|ARM_QYC\|^ARM \|Qyc' <<< "$base") >/dev/null; then
    ok "Qyc: every other switch and argument as $QYCBASE"
  else bad "Qyc: differs from $QYCBASE beyond the switch: $(diff <(grep -v 'HYBRID_CAL\|ARM_QYC\|Qyc' <<< "$sec") <(grep -v 'HYBRID_CAL\|ARM_QYC\|Qyc' <<< "$base") | head -4 | tr '\n' ' ')"; fi
fi
for a in "$REF" "$NEW" "$FAST" "$QD200" "$QY25" "$QZ" "$QT" "$QX"; do
  has "no Qyc part ($a)" "$(section "$a")" RADIANCE_EXPERIMENTAL_DS41_P1_HYBRID_CAL 0
done
qdskip "$QYCNOH" "Qyc1 calibrates the hybrid split, which needs Qy or Qyp; the arm is skipped"
qdskip "$QYCBIG" "Qyc4: Qyc takes 1-3 (a mask); the arm is skipped"
qdskip "$QYCTWO" "two Qyc parts; the arm is skipped"

short_out=$(run_loop 2048)
for a in "$COLD4K" "$PC4K"; do
  if grep -qxF "ARM $a" <<< "$short_out"; then bad "Mb+Lc4096 launched after only the balance M2048 oracle"
  elif grep -qF "arm $a: Mb refused: its row bound 4096 exceeds the balance oracle's actual M2048" <<< "$short_out"; then
    ok 'Mb+Lc4096 refused above BAL_MAX_M despite the independent LCP4096 receipt'
  else bad "no actual-bound refusal for arm $a"; fi
done
grep -qxF "ARM $BIG" <<< "$short_out" && ok 'Mb+Lc2048 still runs on its M2048 balance receipt' ||
  bad 'the shorter balance receipt incorrectly stopped Mb+Lc2048'

# ---- 8c. Qw (the WMMA P1 twin, inexact by design): the switch at 1 on the preset arm with Qsp, Qt and Qz as they are;
# every other arm states it 0 (Qw2 too: the wave size, ..._P1_WAVE=2); Qw without P1, beside Qv, and with Kld before a
# reference skipped; with a reference, a Qw Kld arm scores against it (--kld-ref) instead of recording; without a
# p1wmma receipt refused; the startup gate (every rank's marker, none on a plain arm).
sec=$(section "$QW")
if [[ -z "$sec" ]]; then bad "the Qw arm $QW was not launched: $(grep -m3 "arm $QW" "$T/loop.txt")"
else
  has "Qw" "$sec" RADIANCE_EXPERIMENTAL_DS41_P1_WMMA 1
  for v in RADIANCE_EXPERIMENTAL_DS41_P1_TILED RADIANCE_EXPERIMENTAL_DS41_P1_TAIL_ONCE RADIANCE_EXPERIMENTAL_DS41_P1_SWIGLU_SPAN; do
    has "Qw on the Qsp-Qt-Qz preset" "$sec" "$v" 1
  done
  for v in RADIANCE_EXPERIMENTAL_DS41_LCP RADIANCE_EXPERIMENTAL_DS41_P1 RADIANCE_EXPERIMENTAL_DS41_P1_LEND_ARENA; do
    has "Qw as the fast arm" "$sec" "$v" "$(val "$fast" "$v")"
  done
  grep -qx "  ARM_QW: 1" <<< "$sec" && ok "Qw: the arm is gated on its startup markers (ARM_QW=1)" || bad "Qw: ARM_QW is '$(grep '  ARM_QW:' <<< "$sec")'"
fi
for a in "$REF" "$FAST" "$QT" "$QTQZ" "$QD200" "$QW2"; do
  has "no Qw part ($a)" "$(section "$a")" RADIANCE_EXPERIMENTAL_DS41_P1_WMMA 0
done
has "Qw2 is the wave size" "$(section "$QW2")" RADIANCE_EXPERIMENTAL_DS41_P1_WAVE 2
qdskip "$QWNOP" "Qw needs P1; the arm is skipped"
qdskip "$QWV" "Qw beside Qv8 (the host verify.*skipped"
qdskip "$QWKLD" "Qw with Kld scores against the window's Kld reference, and none has been recorded yet.*skipped"
ref_out=$(run_loop 4096 "" "$T/out/kld-plain")
sec=$(awk -v a="ARM $QWKLD" '$0 == a { p = 1; next } /^ARM / { p = 0 } p' <<< "$ref_out")
if [[ -z "$sec" ]]; then bad "the Qw Kld arm was not launched with a reference: $(grep -m2 "arm $QWKLD" <<< "$ref_out")"
elif grep -q -- "--kld-ref $T/out/kld-plain --kld-out $T/out/kld-$QWKLD.json" <<< "$sec" && ! grep -q -- "--kld-record" <<< "$sec"; then
  ok "Qw with Kld: scored against the reference (--kld-ref, --kld-out), not recorded"
else bad "Qw with Kld: $(grep '  ARGS:' <<< "$sec")"; fi
nop1w_out=$(run_loop 4096 no-p1w)
if grep -qxF "ARM $QW" <<< "$nop1w_out"; then bad "a Qw arm launched without a p1wmma receipt"
elif grep -qF "arm $QW: Qw refused: no p1wmma check (test)" <<< "$nop1w_out"; then ok 'Qw refused without a p1wmma receipt'
else bad "no p1wmma refusal for arm $QW: $(grep -m1 "arm $QW: " <<< "$nop1w_out")"; fi
grep -qxF "ARM $FAST" <<< "$nop1w_out" && ok 'the fast arm runs without a p1wmma receipt' || bad 'the fast arm needed a p1wmma receipt'
eval "$(sed -n '/^qw_gate() {/,/^}/p' "$W")"
for r in 0 1 2 3; do echo "ds41 p1: WMMA twin on (inexact), rank $r (RADIANCE_EXPERIMENTAL_DS41_P1_WMMA=1): every P1 twin call"; done > "$T/qw.log"
qw_gate "$T/qw.log" 1 > /dev/null && ok "qw_gate: every rank's marker passes a Qw arm" || bad "qw_gate refused four markers"
qw_gate "$T/qw.log" "" > /dev/null && bad "qw_gate passed a plain arm with WMMA markers" || ok "qw_gate: a plain arm with the markers is not measured"
sed -i '/rank 2/d' "$T/qw.log"
why=$(qw_gate "$T/qw.log" 1) && bad "qw_gate passed three markers" || { [[ $why == *"rank 2 lacks"* ]] && ok "qw_gate: a missing rank's marker is named" || bad "qw_gate: $why"; }
: > "$T/qw.log"
qw_gate "$T/qw.log" "" > /dev/null && ok "qw_gate: a plain arm without markers passes" || bad "qw_gate refused a plain arm"

# ---- 8d. Dt / Aw
sec=$(section "$DTAW")
if [[ -z "$sec" ]]; then bad "the Dt-Aw arm $DTAW was not launched: $(grep -m3 "arm $DTAW" "$T/loop.txt")"
else
  has "Dt" "$sec" RADIANCE_EXPERIMENTAL_DS41_DENSE_TILED 1
  has "Aw" "$sec" RADIANCE_EXPERIMENTAL_DS41_ATTN_WMMA 1
  for v in RADIANCE_EXPERIMENTAL_DS41_LCP RADIANCE_EXPERIMENTAL_DS41_P1 RADIANCE_EXPERIMENTAL_DS41_P1_WMMA RADIANCE_EXPERIMENTAL_DS41_DENSE_FP8; do
    has "Dt-Aw as the fast arm" "$sec" "$v" "$(val "$(section "$FAST")" "$v")"
  done
  grep -qx "  ARM_DT: 1" <<< "$sec" && grep -qx "  ARM_AW: 1" <<< "$sec" && ok "Dt-Aw: gated on both startup markers (ARM_DT=1, ARM_AW=1)" ||
    bad "Dt-Aw: $(grep -E '  ARM_(DT|AW):' <<< "$sec" | tr '\n' ' ')"
fi
for a in "$REF" "$FAST" "$QW" "$QT"; do
  has "no Dt part ($a)" "$(section "$a")" RADIANCE_EXPERIMENTAL_DS41_DENSE_TILED 0
  has "no Aw part ($a)" "$(section "$a")" RADIANCE_EXPERIMENTAL_DS41_ATTN_WMMA 0
done
qdskip "$DTNOLC" "Dt needs Lc.*skipped"
qdskip "$DTNOF8" "Dt needs F8.*skipped"
qdskip "$AWNOLC" "Aw needs Lc.*skipped"
qdskip "$AWKLD" "Aw with Kld scores against the window's Kld reference, and none has been recorded yet.*skipped"
sec=$(section "$DTKLD")
if [[ -n "$sec" ]] && grep -q -- "--kld-record $T/out/kld-$DTKLD" <<< "$sec"; then ok "Dt with Kld: recorded (exact), compared as any exact arm"
else bad "Dt with Kld: $(grep '  ARGS:' <<< "$sec")"; fi
sec=$(awk -v a="ARM $AWKLD" '$0 == a { p = 1; next } /^ARM / { p = 0 } p' <<< "$ref_out")
if [[ -z "$sec" ]]; then bad "the Aw Kld arm was not launched with a reference: $(grep -m2 "arm $AWKLD" <<< "$ref_out")"
elif grep -q -- "--kld-ref $T/out/kld-plain --kld-out $T/out/kld-$AWKLD.json" <<< "$sec" && ! grep -q -- "--kld-record" <<< "$sec"; then
  ok "Aw with Kld: scored against the reference (--kld-ref, --kld-out), not recorded"
else bad "Aw with Kld: $(grep '  ARGS:' <<< "$sec")"; fi
nofast_out=$(run_loop 4096 no-fast)
if grep -qxF "ARM $DTAW" <<< "$nofast_out" || grep -qxF "ARM $AWONLY" <<< "$nofast_out"; then bad "a Dt / Aw arm launched without its receipts"
elif grep -qF "arm $DTAW: Dt refused: no dtiled check (test)" <<< "$nofast_out" &&
     grep -qF "arm $AWONLY: Aw refused: no attnwm check (test)" <<< "$nofast_out"; then ok 'Dt and Aw refused without their receipts'
else bad "no dtiled / attnwm refusal: $(grep -m1 "arm $DTAW: " <<< "$nofast_out") $(grep -m1 "arm $AWONLY: " <<< "$nofast_out")"; fi
has "Aw alone" "$(section "$AWONLY")" RADIANCE_EXPERIMENTAL_DS41_ATTN_WMMA 1
has "Aw alone" "$(section "$AWONLY")" RADIANCE_EXPERIMENTAL_DS41_DENSE_TILED 0
grep -qxF "ARM $FAST" <<< "$nofast_out" && ok 'the fast arm runs without the dtiled / attnwm receipts' || bad 'the fast arm needed them'
eval "$(sed -n '/^fast_gate() {/,/^}/p;/^dt_gate() {/p;/^aw_gate() {/p' "$W")"
for r in 0 1 2 3; do
  echo "ds41 dense: tiled bf16f8 GEMM on (exact), rank $r (RADIANCE_EXPERIMENTAL_DS41_DENSE_TILED=1): 330 gemm_nt_bf16f8 ops"
  echo "ds41 attn: WMMA sparse attention on (inexact), rank $r (RADIANCE_EXPERIMENTAL_DS41_ATTN_WMMA=1): 43 attn_sparse_sink ops"
done > "$T/da.log"
dt_gate "$T/da.log" 1 > /dev/null && aw_gate "$T/da.log" 1 > /dev/null && ok "dt_gate / aw_gate: every rank's marker passes" ||
  bad "dt_gate / aw_gate refused four markers"
dt_gate "$T/da.log" "" > /dev/null && bad "dt_gate passed a plain arm with markers" || ok "dt_gate: a plain arm with the markers is not measured"
aw_gate "$T/da.log" "" > /dev/null && bad "aw_gate passed a plain arm with markers" || ok "aw_gate: a plain arm with the markers is not measured"
sed -i '/attn.*rank 1/d' "$T/da.log"
why=$(aw_gate "$T/da.log" 1) && bad "aw_gate passed three markers" || { [[ $why == *"rank 1 lacks"* ]] && ok "aw_gate: a missing rank's marker is named" || bad "aw_gate: $why"; }
dt_gate "$T/da.log" 1 > /dev/null && ok "dt_gate: unaffected by the attention markers" || bad "dt_gate refused after an attention line was removed"
: > "$T/da.log"
dt_gate "$T/da.log" "" > /dev/null && aw_gate "$T/da.log" "" > /dev/null && ok "dt_gate / aw_gate: a plain arm without markers passes" ||
  bad "dt_gate / aw_gate refused a plain arm"

(( fail )) && { echo "ab_window_p1_test: FAILED (loop output in $T/loop.txt, removed on exit)"; exit 1; }
echo "ab_window_p1_test: all checks passed"
