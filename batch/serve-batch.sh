#!/usr/bin/env bash
# The measured best four-card batch configuration of DeepSeek-V4.1-Flash (README "Batch jobs: quickstart"):
# four 16 GB RDNA4 cards at TP4, no fifth card, many concurrent streams of short prompts that share long heads.
# The environment and flags are the window harness's for the arm (ab_window.sh, 12 November)
#
#   all-t40-f-M2048-H-p256-m229376-G-R-B6-W-A2-Rd13-Er-Es-Pa-K-Sc2-F8-Pp4-Ei-Pm56-Pe-Ss-Pu-Is-Pi8-Ns7-Fw-Mr32-Lc2048-
#   P1-Qe64-Mb1-Rb-Vp-Cs-mA-Qd64-Qa0-Pc-Pch32-Pcs128-L294912-Pf-N256-C256-Ba2048-Bs-Qy60-Xqagent3
#
# taken from its stubbed dry run, with each variable resolved to its last assignment and each flag to its last value
# (the harness passes a few twice; the server keeps the last). What differs, on purpose:
#   - the shared-head floor is RADIANCE_EXPERIMENTAL_PREFIX_SHARED_HEAD=32 (--shared-head), the pending default; the
#     measured rows used 64 (Pch64). 32 ran in the same windows' Pch32 arms.
#   - the placement profile: the arm's (agent3) holds private captures and is not shipped. --profile names yours;
#     without it the server ranks experts by the container's own activation profile (unmeasured for batches).
#   - the plane mirror (R, Rd13) only with --mirror; the measured arm had one on a second drive.
#   - hybrid decode (Qy60, RADIANCE_EXPERIMENTAL_DS41_P1_HYBRID=60, the share adapting) is the default: exact and +28%
#     decode at 256 streams (window ab-20261113T014719Z); --hybrid 0 turns it off (the arm without Qy).
#   - --max-seq-len N (SlN) only when given.
#   - --tiled (Qt, ..._P1_TILED), --tail-once (Qz, ..._P1_TAIL_ONCE) and --swiglu-span (Qsp, ..._P1_SWIGLU_SPAN): the
#     prefill Phase 1 twin's variants, on by default: exact, prefill -19% and 256 NL-SQL 287 -> 250 s with hybrid
#     decode (window ab-20261115T070032Z); give 0 to turn one off.
#   - --hybrid-prefill P (Qyp, RADIANCE_EXPERIMENTAL_DS41_P1_HYBRID_PREFILL=P, with kernel costs Qyk,
#     ..._HOST_EXPERTS_KERNEL_COST=1): the CPU pool takes a share of each prefill pass's experts as hybrid decode does;
#     default 60: exact, 256 NL-SQL 257.5 -> 242.0 s, prefill -11.6% (A-K-P-P-A, window ab-20261116T031731Z);
#     --hybrid-prefill 0 turns both off.
#   - --engram-early (Ee, RADIANCE_EXPERIMENTAL_DS41_ENGRAM_EARLY=1): the next Engram layer's table rows read ahead; on by
#     default (exact, 256 NL-SQL 266 -> 250 s in an A-B-B-A, window ab-20261115T173507Z); --engram-early 0 turns it off.
#   - --fast-prefill (Qw, ..._P1_WMMA): the P1 twin's routed-expert GEMMs on the matrix cores (WMMA), ON by default.
#     NOT exact: the same bf16 weights and activations, another f32 summation order, so replies are not byte-identical
#     to the exact engine's and depend on the batch. BIRD 500 (window ab-20261116T104839Z): EX 62.4 exact vs 61.4 Qw+Dt
#     (McNemar p 0.33) and 61.8 with Aw (p 0.65); job 260.1 -> 243.6 / 239.5 s. --exact (or --fast-prefill 0) for the
#     exact, batch-invariant engine (FASTPREFILL-20261115.md).
#   - --fast-dense (Dt, ..._DENSE_TILED): the FP8-stored dense linears above 64 rows on the LDS-tiled WMMA form, ON by
#     default: EXACT (the m64 kernel's bits; FASTDENSE-20261115.md), 256 NL-SQL 267.2 -> 250.6 s, prefill -8.8%, every
#     reply identical and the KL run identical (window ab-20261116T041838Z); --fast-dense 0 turns it off.
#   - --fast-attn (Aw, ..._ATTN_WMMA): the sparse attention at every height on the matrix cores, ON by default. NOT
#     exact: q.k and sum p k summed in another f32 order (the softmax's exp, max and normaliser the exact kernel's), so
#     replies are not byte-identical to the exact engine's. --exact (or --fast-attn 0) turns it off (FASTDENSE-20261115.md).
#   - --engram-file A,B (Ef, ..._DS41_ENGRAM_FILE): the Engram tables read from the release's own FP8 shards
#     (model-00047/48-of-00048.safetensors) instead of the container's IQ4_NL conversion: the official weights, same
#     speed (BIRD 256 170.2 vs 172.2 s), NLL 0.688 vs 0.692 (window ab-20261116T114113Z). Recommended; off when not given.
#   - --reproducible (alias --exact): both batch-dependent modes off (--fast-prefill 0 --fast-attn 0): replies byte-identical whatever the batch.
#   - --hybrid-cal M (QycM, ..._P1_HYBRID_CAL): the hybrid split's CPU rate learnt from completions (1), one share for
#     every rank with the choice reserved under one lock (2), or both (3); exact (selection only). OFF (0) until window
#     H58 measures it (core/place/p1_hybrid.h "calibration").
#   - --early K (QxK, ..._P1_EARLY): up to K of the next card-path layer's units copied into the ring before its routing
#     counts, so the link works through attention and the router; exact (scheduling only). OFF (0) until window H55
#     measures it (OVERLAP-20261115.md).
#   - --pack W (BpW, RADIANCE_EXPERIMENTAL_SCHED_PACK_LOOKAHEAD): a batch whose next waiting prompt does not fit asks the next W
#     waiting prompts for one that does; exact (step composition only). OFF (0) until window H56 measures it.
#   - --capture FILE: a calibration run (calibrate.sh), not the measured arm: the routing counts written to FILE
#     (--expert-stats; at shutdown on SIGINT, and every 30 s), 8 streams unless --streams is given, and batch
#     admission at 256 rows without the batch split (Ba, no Bs), as the window harness's capture arms ran
#     (...-L9216-Pf-N8-C8-Ba-...-Xst): a step then holds prefill rows or decode rows, so its decode counts are a
#     placement profile's (profile_blend.py). Same exact engine otherwise; replies are the base engine's.
#   - --auto-profile DIR (off by default): a placement profile kept for you across runs. At start, a portable profile
#     DIR/<first 16 hex of the container's metadata sha256>.profile.json, if there is one for this container (and no
#     --profile), is used as --expert-profile; every run records its routing counts (--expert-stats into DIR) and,
#     after the server exits on SIGINT/SIGTERM, blends them into that file (the previous profile x1 + this run x1,
#     the previous kept as .prev.json; profile_blend.py write --portable). Batched serving runs mixed steps (prefill
#     and decode rows in one step, which one histogram cannot split): their counts are folded in at
#     AUTO_PROFILE_MIXED_SHARE (0.5) of their decode rows, approximate (profile_blend.py --mixed-share). Placement
#     only: replies are unchanged. The server then runs as this script's child, not as its pid: stop it with Ctrl-C
#     or SIGTERM to this script (both reach the server as SIGINT; a job started with & from a script ignores SIGINT,
#     so send SIGTERM there); it gets SIGINT if this script dies, where setpriv --pdeathsig is available.
#   - --pack-once 0|1 (Qo, ..._P1_PACK_ONCE): a drive miss the RAM cache admits is packed once, straight into the
#     packed RAM cache, and its ring copy reads the cache slots, instead of a pinned ready slot plus a second pack by
#     the insert threads; exact (the same bytes in the cache and on the card). OFF (0) until window H60 measures it.
#   - --arena-split 0|1 (Qes, ..._P1_LEND_ARENA_SPLIT): the lent arena's slots (Qe64) taken back only by a step at the
#     full plan, so an armed decode pass leaves their residents published (~38 more file-tier units a card stay on the
#     card through decode at N512). Residency only, exact. OFF (0) until window H57 measures it (VRAMLEND-20261116.md).
#   - --kv-lend 0|1 (Kvl, ..._KV_LEND): the KV pool's free tail (above the cache's held blocks less two chunks) lent
#     to expert slots carved from its FULL groups, taken back before the cache hands a block out. Residency only, exact.
#     OFF (0) until window H57 measures it (VRAMLEND-20261116.md).
#   - --kv-auto T (LkT): the KV pool from the streams: --ctx-per-stream = T x 5/4 rounded up to 16 tokens, T the
#     measured peak KV tokens a stream ((kv_tokens_used at the end + hit_cut_tokens) / streams from a run with
#     prefix_evictions 0; NL-SQL at 512 streams: 452 -> 576 a stream, L294912). 0 (default): --ctx-per-stream as given.
#     Placement only: VRAM the pool does not need holds static residents. OFF until H57 measures it.
#   - --ram-per-seq-mib N (mSN, ..._RAM_PER_SEQ_MIB): the RAM budget's per-sequence growth term (the engine's 32 MiB
#     when not given). Measured growth after serve() at 512 streams is <= 4.5 GiB in all (<= 8.8 MiB a sequence),
#     so 16 holds it twice over and gives the expert RAM cache 8 GiB more than 32 (VRAMLEND-20261116.md). Unset by
#     default.
#   - --decode-min N (DmN, RADIANCE_EXPERIMENTAL_SCHED_DECODE_MIN=N): with N or more streams ready to decode, a step
#     carries their decode rows ahead of waiting prompts (continuous serving keeps large decode batches instead of
#     prefilling every new prompt first); exact (step order only). OFF (0) until window H61 measures it.
#   - --served-model-name ds41 (a name only; --served-name sets another) and --radiance-home only when given
#     (RADIANCE_HOME otherwise).
#
# Usage: serve-batch.sh --model MODEL.rad --silu-table FILE [--streams 256] [--port 8000] [--reserve-mib N]
#                       [--ctx-per-stream 1152] [--profile FILE] [--mirror DIR] [--shared-head 32] [--served-name ds41]
#                       [--radiance-home DIR] [--hybrid 60] [--max-seq-len N] [--tiled 0|1] [--tail-once 0|1]
#                       [--swiglu-span 0|1] [--engram-early 0|1] [--hybrid-prefill 60] [--fast-prefill 0|1]
#                       [--fast-dense 0|1] [--fast-attn 0|1] [--reproducible] [--engram-file A,B] [--early K] [--pack W] [--pack-once 0|1] [--hybrid-cal M]
#                       [--arena-split 0|1] [--kv-auto T] [--ram-per-seq-mib N] [--kv-lend 0|1] [--decode-min N]
#                       [--capture FILE] [--auto-profile DIR] [--dry-run]
#                       [-- more radiance flags, e.g. --expert-stats FILE to record a profile; outside the measured set]
#   RADIANCE_BIN (default: radiance on PATH, else build/bin/radiance), RADIANCE_HOME as for any radiance run.
# --dry-run prints the environment and the command and starts nothing. Otherwise the script becomes the server (exec:
# same pid), beside a memory guard that stops it (SIGINT) if MemAvailable falls below MEMGUARD_MIB (3072; 0 off), as the
# window harness does: an exhausted host can hit an amdgpu bug on the OOM path that panics the kernel.
set -u
die() { echo "serve-batch.sh: $*" >&2; exit 2; }
EXTRA=() NAME=ds41 MODEL="" SILU="" STREAMS=256 PORT=8000 RESERVE="" CTX=1152 PROFILE="" MIRROR="" SHEAD=32 DRY=0 HOME_DIR="" HYBRID=60 SEQLEN="" TILED=1 TAILONCE=1 SPAN=1 ENGEARLY=1 HYBPF=60 FASTPF=1 FASTDN=1 FASTAT=1 EARLY=0 PACK=0 PACKONCE=0 HYBCAL=0 ASPLIT=0 KVAUTO=0 PERSEQ="" KVLEND=0 DECMIN=0 CAPTURE="" STREAMS_SET=0 AUTO="" FAST_SET=0 ENGFILE=""
while (( $# )); do
  case $1 in
    --model) MODEL=${2:?}; shift ;;
    --silu-table) SILU=${2:?}; shift ;;
    --streams) STREAMS=${2:?}; STREAMS_SET=1; shift ;;
    --capture) CAPTURE=${2:?}; shift ;;
    --auto-profile) AUTO=${2:?}; shift ;;
    --port) PORT=${2:?}; shift ;;
    --reserve-mib) RESERVE=${2:?}; shift ;;
    --ctx-per-stream) CTX=${2:?}; shift ;;
    --profile) PROFILE=${2:?}; shift ;;
    --mirror) MIRROR=${2:?}; shift ;;
    --shared-head) SHEAD=${2:?}; shift ;;
    --radiance-home) HOME_DIR=${2:?}; shift ;;
    --served-name) NAME=${2:?}; shift ;;
    --hybrid) HYBRID=${2:?}; shift ;;
    --max-seq-len) SEQLEN=${2:?}; shift ;;
    --tiled) TILED=${2:?}; shift ;;
    --tail-once) TAILONCE=${2:?}; shift ;;
    --engram-early) ENGEARLY=${2:?}; shift ;;
    --hybrid-prefill) HYBPF=${2:?}; shift ;;
    --swiglu-span) SPAN=${2:?}; shift ;;
    --fast-prefill) FASTPF=${2:?}; FAST_SET=1; shift ;;
    --fast-dense) FASTDN=${2:?}; shift ;;
    --fast-attn) FASTAT=${2:?}; FAST_SET=1; shift ;;
    --engram-file) ENGFILE=${2:?}; shift ;;
    --reproducible|--exact) FASTPF=0; FASTAT=0; FAST_SET=1 ;;
    --early) EARLY=${2:?}; shift ;;
    --pack) PACK=${2:?}; shift ;;
    --pack-once) PACKONCE=${2:?}; shift ;;
    --hybrid-cal) HYBCAL=${2:?}; shift ;;
    --arena-split) ASPLIT=${2:?}; shift ;;
    --kv-lend) KVLEND=${2:?}; shift ;;
    --kv-auto) KVAUTO=${2:?}; shift ;;
    --ram-per-seq-mib) PERSEQ=${2:?}; shift ;;
    --decode-min) DECMIN=${2:?}; shift ;;
    --dry-run) DRY=1 ;;
    --) shift; EXTRA=("$@"); break ;;
    -h|--help) sed -n '2,/^set -u/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown argument '$1' (--help)" ;;
  esac
  shift
done
# a calibration run is the exact engine unless the fast modes are named (its replies are the base engine's)
[[ -n "$CAPTURE" ]] && (( FAST_SET == 0 )) && { FASTPF=0; FASTAT=0; }
[[ -n "$CAPTURE" && -n "$AUTO" ]] && die "--capture and --auto-profile both record the counts; give one"
# A calibration run: few streams and batch admission without the split (see --capture above).
BA=2048 BSPLIT=1
if [[ -n "$CAPTURE" ]]; then
  (( STREAMS_SET )) || STREAMS=8
  BA=256 BSPLIT=0
fi
num() { [[ $2 =~ ^[0-9]+$ ]] && (( 10#$2 >= $3 )) || die "$1 takes a whole number of at least $3, not '$2'"; }
[[ -n "$MODEL" ]] || die "--model MODEL.rad is required"
[[ -n "$SILU" ]] || die "--silu-table FILE is required: the card's silu table (ds41_mxfp4_device_check LIBR4D.so swiglu FILE)"
num --streams "$STREAMS" 1; num --port "$PORT" 1; num --ctx-per-stream "$CTX" 1; num --shared-head "$SHEAD" 0
[[ -z "$RESERVE" ]] || num --reserve-mib "$RESERVE" 1024
num --hybrid "$HYBRID" 0; (( 10#$HYBRID <= 400 )) || die "--hybrid takes 0 (off) or a share of 1-400 percent, not '$HYBRID'"
num --hybrid-prefill "$HYBPF" 0; (( 10#$HYBPF <= 400 )) || die "--hybrid-prefill takes 0 (off) or a share of 1-400 percent, not '$HYBPF'"
[[ -z "$SEQLEN" ]] || num --max-seq-len "$SEQLEN" 64
num --early "$EARLY" 0; (( 10#$EARLY <= 64 )) || die "--early takes 0 (off) or 1-64 units, not '$EARLY'"
num --pack "$PACK" 0; (( 10#$PACK <= 256 )) || die "--pack takes 0 (off) or 1-256 waiting prompts, not '$PACK'"
num --hybrid-cal "$HYBCAL" 0; (( 10#$HYBCAL <= 3 )) || die "--hybrid-cal takes 0 (off) or a mask of 1-3, not '$HYBCAL'"
(( 10#$HYBCAL == 0 || 10#$HYBRID > 0 || 10#$HYBPF > 0 )) || die "--hybrid-cal $HYBCAL calibrates the hybrid split: it needs --hybrid or --hybrid-prefill above 0"
num --kv-auto "$KVAUTO" 0; (( 10#$KVAUTO == 0 || (10#$KVAUTO >= 16 && 10#$KVAUTO <= 1048576) )) || die "--kv-auto takes 0 (off) or 16-1048576 tokens a stream, not '$KVAUTO'"
[[ -z "$PERSEQ" ]] || { num --ram-per-seq-mib "$PERSEQ" 0; (( 10#$PERSEQ <= 1048576 )) || die "--ram-per-seq-mib takes 0-1048576, not '$PERSEQ'"; }
num --decode-min "$DECMIN" 0; (( 10#$DECMIN <= 1048576 )) || die "--decode-min takes 0 (off) or 1-1048576 streams, not '$DECMIN'"
for v in "--kv-lend $KVLEND" "--arena-split $ASPLIT" "--tiled $TILED" "--tail-once $TAILONCE" "--swiglu-span $SPAN" "--engram-early $ENGEARLY" "--fast-prefill $FASTPF" "--fast-dense $FASTDN" "--fast-attn $FASTAT" "--pack-once $PACKONCE"; do
  [[ ${v#* } == [01] ]] || die "${v% *} takes 0 or 1, not '${v#* }'"
done
STREAMS=$((10#$STREAMS)) CTX=$((10#$CTX))
# LkT: the pool from the measured peak tokens a stream, x 5/4, a whole number of 16-token blocks a stream.
(( 10#$KVAUTO > 0 )) && CTX=$(( (10#$KVAUTO * 5 + 63) / 64 * 16 ))
(( FASTPF == 1 )) && TILED=0   # Qw takes the register-tiled twin's calls; the core refuses both switches at once (H45)
if (( ! DRY )); then
  [[ -f "$MODEL" ]] || die "no model container at $MODEL"
  [[ -s "$SILU" ]] || die "no silu table at $SILU"
  [[ -z "$PROFILE" || -f "$PROFILE" ]] || die "no profile at $PROFILE"
  [[ -z "$MIRROR" || -d "$MIRROR" ]] || die "no mirror directory at $MIRROR"
  [[ -z "$CAPTURE" || ! -e "$CAPTURE" ]] || die "--capture $CAPTURE exists (the server would overwrite it)"
fi
BIN=${RADIANCE_BIN:-$(command -v radiance || echo build/bin/radiance)}

# The arm's environment (138 assignments). The empty ones are set empty, as the harness sets them, so that a value
# in the caller's environment cannot change the configuration.
ENV=(
  RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS=40
  RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_SILU="$SILU"
  RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_POLICY=all
  RADIANCE_EXPERIMENTAL_NATIVE_CACHE_DIRECT_FILL=1
  RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_MISSES=2048
  RADIANCE_EXPERIMENTAL_NATIVE_CACHE_PREGROW=1
  RADIANCE_EXPERIMENTAL_DS41_FILE_BUFFER_UNITS=6
  RADIANCE_EXPERIMENTAL_DS41_CACHE_WARM=1
  RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_ASYNC_MISSES=2
  RADIANCE_EXPERIMENTAL_DS41_ENGRAM_SPLIT=1
  RADIANCE_EXPERIMENTAL_DS41_PREFETCH_ATTN=1
  RADIANCE_EXPERIMENTAL_DS41_PACKED_CACHE=1
  RADIANCE_EXPERIMENTAL_DS41_CACHE_SECOND_CHANCE=2
  RADIANCE_EXPERIMENTAL_DS41_PREFETCH=pieces
  RADIANCE_EXPERIMENTAL_DS41_PREFETCH_K=4
  RADIANCE_EXPERIMENTAL_DS41_EARLY_ISSUE=1
  RADIANCE_EXPERIMENTAL_DS41_PREFETCH_MIB=56
  RADIANCE_EXPERIMENTAL_DS41_PREFETCH_ENGRAM=1
  RADIANCE_EXPERIMENTAL_DS41_SKIP_POST_STOP=1
  RADIANCE_EXPERIMENTAL_DS41_PREFETCH_URGENT=1
  RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_FUTEX=1
  RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_MAX_ROWS=32
  RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_KERNEL_COST=$(( 10#$HYBPF > 0 ? 1 : 0 ))   # Qyk: kernel costs, on with hybrid prefill
  RADIANCE_EXPERIMENTAL_DS41_LCP=2048
  RADIANCE_EXPERIMENTAL_DS41_VRAM_PACKED=1
  RADIANCE_EXPERIMENTAL_DS41_RAM_BUDGET=1
  RADIANCE_EXPERIMENTAL_DS41_RAM_PER_SEQ_MIB=${PERSEQ:+$((10#$PERSEQ))}   # mSN: the budget's per-sequence term (empty: 32)
  RADIANCE_EXPERIMENTAL_CACHE_HEADROOM_BACKED=1
  RADIANCE_EXPERIMENTAL_PREFIX_SHARED_HEAD="$SHEAD"
  RADIANCE_EXPERIMENTAL_SCHED_PREFILL_ALONE=1
  RADIANCE_EXPERIMENTAL_DS41_DENSE_FP8=1
  RADIANCE_EXPERIMENTAL_DS41_DENSE_FP8_EXCLUDE=
  RADIANCE_EXPERIMENTAL_DS41_ROUTER_BF16=1
  RADIANCE_EXPERIMENTAL_DS41_ROUTER_BF16_FORM=0
  RADIANCE_EXPERIMENTAL_DS41_N128=0
  RADIANCE_EXPERIMENTAL_DS41_PREFETCH_MULTIROW=0
  RADIANCE_EXPERIMENTAL_SCHED_TBO=0
  RADIANCE_EXPERIMENTAL_SCHED_BATCH_ADMIT=$BA       # Ba2048 (--capture: Ba, 256 rows)
  RADIANCE_EXPERIMENTAL_SCHED_BATCH_SPLIT=$BSPLIT   # Bs (--capture: off)
  RADIANCE_EXPERIMENTAL_SCHED_PACK_LOOKAHEAD=$((10#$PACK))   # BpW: pack lookahead, exact (--pack W; 0 off until H56 measures it)
  RADIANCE_EXPERIMENTAL_SCHED_DECODE_MIN=$((10#$DECMIN))   # DmN: decode first at N ready streams, exact (--decode-min N)
  RADIANCE_EXPERIMENTAL_DS41_PREFETCH_SCORE=int8
  RADIANCE_EXPERIMENTAL_DS41_PREFETCH_RESCORE=12
  RADIANCE_EXPERIMENTAL_DS41_PREFETCH_SCORE_CHECK=0
  RADIANCE_EXPERIMENTAL_DS41_PREFETCH_POST_LATE=0
  RADIANCE_EXPERIMENTAL_DS41_SIDE_STORE=7
  RADIANCE_EXPERIMENTAL_DS41_PREFETCH_RECORD=
  RADIANCE_EXPERIMENTAL_DS41_ROUTE_TRACE=
  RADIANCE_EXPERIMENTAL_DS41_CONSUMER_SKIP=1
  RADIANCE_EXPERIMENTAL_DS41_CONSUMER_SKIP_CKPT=512
  RADIANCE_EXPERIMENTAL_DS41_PREFILL_HASH=0
  RADIANCE_EXPERIMENTAL_PREFIX_EXACT_PUBLISH=0
  RADIANCE_EXPERIMENTAL_PREFIX_EORS=0
  RADIANCE_EXPERIMENTAL_DS41_RESUME_GRID=0
  RADIANCE_EXPERIMENTAL_PREFIX_SESSION_LRU=0
  RADIANCE_EXPERIMENTAL_CHECKPOINT_STORE_MIB=
  RADIANCE_EXPERIMENTAL_DS41_KV_REPLICATED=0
  RADIANCE_EXPERIMENTAL_DS41_DSPARK_CONF=0
  RADIANCE_EXPERIMENTAL_DS41_DSPARK_CONF_HEAD=
  RADIANCE_EXPERIMENTAL_DS41_DSPARK_LOAD=0
  RADIANCE_EXPERIMENTAL_DS41_DSPARK_CTX_DEFER=0
  RADIANCE_EXPERIMENTAL_DS41_DSPARK_CODEBOOK_RAM=0
  RADIANCE_EXPERIMENTAL_DS41_DSPARK_PROB_CAPTURE=0
  RADIANCE_EXPERIMENTAL_DS41_DSPARK_EARLY_VERIFY=off
  RADIANCE_EXPERIMENTAL_DS41_DSPARK_PREFIX_POLICY=off
  RADIANCE_EXPERIMENTAL_DS41_DRAFT_GEMV=0
  RADIANCE_EXPERIMENTAL_DS41_DSPARK_SHORT_BLOCK=0
  RADIANCE_EXPERIMENTAL_DS41_DSPARK_GATE=0
  RADIANCE_EXPERIMENTAL_DS41_DSPARK_FP8=0
  RADIANCE_EXPERIMENTAL_DS41_DRAFT_VRAM=0
  RADIANCE_EXPERIMENTAL_DS41_DRAFT_HOTSET=
  RADIANCE_EXPERIMENTAL_DS41_MOE_BALANCE=1
  RADIANCE_EXPERIMENTAL_DS41_LCP_PROFILE=0
  RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL=
  RADIANCE_EXPERIMENTAL_DS41_P1_SWIGLU_SPAN=$SPAN      # Qsp
  RADIANCE_EXPERIMENTAL_DS41_P1_TILED=$TILED           # Qt
  RADIANCE_EXPERIMENTAL_DS41_P1_TAIL_ONCE=$TAILONCE    # Qz
  RADIANCE_EXPERIMENTAL_DS41_P1_WMMA=$FASTPF           # Qw: the WMMA P1 twin, inexact (--fast-prefill)
  RADIANCE_EXPERIMENTAL_DS41_DENSE_TILED=$FASTDN       # Dt: the tiled bf16f8 GEMM, exact (--fast-dense)
  RADIANCE_EXPERIMENTAL_DS41_ATTN_WMMA=$FASTAT         # Aw: the WMMA sparse attention, inexact (--fast-attn)
  RADIANCE_EXPERIMENTAL_DS41_P1_EARLY=$((10#$EARLY))   # QxK: early ring copies, exact (--early K; 0 off until a window measures it)
  RADIANCE_EXPERIMENTAL_DS41_P1_PACK_ONCE=$PACKONCE    # Qo: admitted drive misses packed once, exact (--pack-once; 0 off until H60)
  RADIANCE_EXPERIMENTAL_DS41_P1_HYBRID_CAL=$((10#$HYBCAL))   # QycM: the hybrid split's calibration, exact (--hybrid-cal M; 0 off)
  RADIANCE_EXPERIMENTAL_DS41_P1=1
  RADIANCE_EXPERIMENTAL_DS41_P1_WAVE=
  RADIANCE_EXPERIMENTAL_DS41_P1_INFLIGHT=
  RADIANCE_EXPERIMENTAL_DS41_P1_READY=
  RADIANCE_EXPERIMENTAL_DS41_P1_AHEAD=0
  RADIANCE_EXPERIMENTAL_DS41_P1_VERIFY=
  RADIANCE_EXPERIMENTAL_DS41_P1_ALIGN=
  RADIANCE_EXPERIMENTAL_DS41_P1_MIN_ROWS=
  RADIANCE_EXPERIMENTAL_DS41_P1_LEND=0
  RADIANCE_EXPERIMENTAL_DS41_P1_GRACE_MS=
  RADIANCE_EXPERIMENTAL_DS41_P1_LEND_ARENA=64
  RADIANCE_EXPERIMENTAL_DS41_KV_LEND=$KVLEND             # Kvl: the KV pool's free tail lent, exact (--kv-lend; 0 off until H57)
  RADIANCE_EXPERIMENTAL_DS41_KV_LEND_MARGIN=
  RADIANCE_EXPERIMENTAL_DS41_P1_LEND_ARENA_SPLIT=$ASPLIT   # Qes: the arena's own take-back, exact (--arena-split; 0 off until H57)
  RADIANCE_EXPERIMENTAL_DS41_P1_DECODE_ROWS=64
  RADIANCE_EXPERIMENTAL_DS41_P1_HYBRID=$((10#$HYBRID))   # QyP: hybrid decode, the CPU's share P%; 0 off
  RADIANCE_EXPERIMENTAL_DS41_P1_HYBRID_ADAPT=            # empty: the share adapts (Qy, not Qyf)
  RADIANCE_EXPERIMENTAL_DS41_P1_HYBRID_PREFILL=$((10#$HYBPF))   # QypP: hybrid prefill, the CPU's share P%; 0 off (H50: exact, NL-SQL 256 -6.0%, prefill -11.6%)
  RADIANCE_EXPERIMENTAL_DS41_P1_LEND_HEAT=0
  RADIANCE_EXPERIMENTAL_DS41_P1_LEND_HEAT_HALF_ROWS=
  RADIANCE_EXPERIMENTAL_DS41_P1_LEND_HEAT_MARGIN_PCT=
  RADIANCE_EXPERIMENTAL_DS41_P1_LEND_HEAT_PRIOR_ROWS=
  RADIANCE_EXPERIMENTAL_DS41_P1_LEND_SLAB=0
  RADIANCE_EXPERIMENTAL_DS41_P1_LEND_AUDIT=0
  RADIANCE_EXPERIMENTAL_DS41_P1_LEND_REST_MS=
  RADIANCE_EXPERIMENTAL_DS41_LCP_ADMIT=
  RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_PREWAKE=0
  RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_PREWAKE_LEAD_US=60
  RADIANCE_EXPERIMENTAL_DS41_INSERT_NT=0
  RADIANCE_EXPERIMENTAL_DS41_INSERT_SLICE_KIB=256
  RADIANCE_EXPERIMENTAL_DS41_INSERT_WAIT_US=
  RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_LEAN_TILES=0
  RADIANCE_EXPERIMENTAL_DS41_AR_ROWS=0
  RADIANCE_EXPERIMENTAL_DS41_AG_ROWS=0
  RADIANCE_EXPERIMENTAL_DS41_AUX_EXPERTS=0
  RADIANCE_EXPERIMENTAL_DS41_AUX_DEVICE=
  RADIANCE_EXPERIMENTAL_DS41_AUX_POLICY=
  RADIANCE_EXPERIMENTAL_DS41_AUX_DEADLINE_US=2000
  RADIANCE_EXPERIMENTAL_DS41_AUX_HEADROOM_MIB=1024
  RADIANCE_EXPERIMENTAL_DS41_AUX_BACKOFF=64
  RADIANCE_EXPERIMENTAL_DS41_AUX_FORCE_LATE=0
  RADIANCE_EXPERIMENTAL_DS41_AUX_ROWS=
  RADIANCE_EXPERIMENTAL_DS41_AUX_SHARE=
  RADIANCE_EXPERIMENTAL_DS41_AUX_CONTROL=
  RADIANCE_EXPERIMENTAL_DS41_AUX_SHARE_ALLOW=
  RADIANCE_EXPERIMENTAL_DS41_AUX_SHARE_FOREIGN_MIB=
  RADIANCE_EXPERIMENTAL_DS41_AUX_SHARE_MIN_MIB=
  RADIANCE_EXPERIMENTAL_DS41_AUX_SHARE_FLOOR_MIB=
  RADIANCE_EXPERIMENTAL_DS41_AUX_SHARE_POLL_MS=
  RADIANCE_EXPERIMENTAL_DS41_AUX_SHARE_RELEASE_POLLS=
  RADIANCE_EXPERIMENTAL_DS41_AUX_SHARE_REACQUIRE_S=
  RADIANCE_EXPERIMENTAL_DS41_AUX_SHARE_FILL_MBPS=
  RADIANCE_EXPERIMENTAL_DS41_AUX_ONLINE=
  RADIANCE_EXPERIMENTAL_DS41_AUX_ONLINE_RATE=
  RADIANCE_EXPERIMENTAL_DS41_AUX_ONLINE_MAX=
  RADIANCE_EXPERIMENTAL_DS41_AUX_ONLINE_EVERY=
  RADIANCE_EXPERIMENTAL_DS41_AUX_ONLINE_HALFLIFE=
  RADIANCE_EXPERIMENTAL_DS41_AUX_ONLINE_PRIOR=
  RADIANCE_EXPERIMENTAL_DS41_AUX_ONLINE_MARGIN=
  RADIANCE_EXPERIMENTAL_DS41_AUX_ONLINE_SPARE=
  RADIANCE_EXPERIMENTAL_DS41_AUX_ONLINE_MBPS=
  RADIANCE_EXPERIMENTAL_DS41_AUX_ONLINE_CHECK=
  RADIANCE_EXPERIMENTAL_DS41_AUX_ONLINE_DRY=
  RADIANCE_EXPERIMENTAL_NATIVE_CACHE_MAPPED_SLABS=1
  RADIANCE_EXPERIMENTAL_NATIVE_CACHE_MAPPED_SOURCE=1
  RADIANCE_EXPERIMENTAL_DS41_MXFP4_PACK_SSE2=1
)
# mAN: the RAM budget's reserve (the engine's default 10240 MiB without it).
[[ -n "$RESERVE" ]] && ENV+=(RADIANCE_EXPERIMENTAL_DS41_RAM_RESERVE_MIB=$((10#$RESERVE)))
# Ee: the next Engram layer's rows read ahead (exact; window ab-20261115T173507Z: -6.1%); the harness states it only when on.
(( ENGEARLY == 1 )) && ENV+=(RADIANCE_EXPERIMENTAL_DS41_ENGRAM_EARLY=1)
# Ef: the release's FP8 Engram tables (--engram-file A,B); the harness states it only for Ef arms.
[[ -n "$ENGFILE" ]] && ENV+=(RADIANCE_EXPERIMENTAL_DS41_ENGRAM_FILE=$ENGFILE)

# The KV pool: --max-model-len is the pool the streams share, streams x tokens a stream (prompt plus reply). The
# measured arms: 256 x 1152 = 294912 and 128 x 1152 = 147456 for prompts of about 1K tokens and 128-token replies.
POOL=$((STREAMS * CTX))
# The pool is taken from the same VRAM as the experts kept on the GPUs. Measured on four 16 GB GPUs: 655,360 tokens
# left 444 resident experts a card, 720,896 left 335-399, and 1,179,648 or more left 43, so every prompt step read
# hundreds of experts from the drive (ab-20261116T210025Z: 213 s of prompt processing instead of 83 s). Size the
# pool from the tokens the streams really hold at once (prompt plus reply), not from the longest possible request.
if (( POOL > 786432 )); then
  echo "serve-batch: warning: a KV pool of $POOL tokens (--streams $STREAMS x --ctx-per-stream $CTX) takes VRAM the" >&2
  echo "  experts need; above about 786,432 tokens few experts stay on the GPUs and throughput drops sharply." >&2
  echo "  Lower --ctx-per-stream (or --streams) unless the requests really need it." >&2
fi
ARGS=(
  --kernels libr4d,libref --model "$MODEL" --served-model-name "$NAME"
  --tp 4 --tp-wire exact --placement expert_tiered --weights-disk-tier
  --expert-ram-cache-total-mib 229376        # a cap: the RAM budget sizes the cache from what startup leaves
  --host-pool-mib 256 --host-headroom-mib 8192 --gpu-headroom-mib 256
  --embedding-placement ram --vision-tower-placement vram --ngram-placement disk
  --max-model-len "$POOL" --max-num-batched-tokens 2048 --max-num-seqs "$STREAMS"
  --kv-cache-dtype bf16 --num-speculative-tokens 0 --spec-ngram 0 --deterministic
  --prefix-resume exact --checkpoint-slots 128
  --host 127.0.0.1 --port "$PORT"
)
[[ -n "$HOME_DIR" ]] && ARGS=(--radiance-home "$HOME_DIR" "${ARGS[@]}")
# --auto-profile: this container's kept profile (when there is one and no --profile), and this run's capture.
HERE=$(cd "$(dirname "$0")" && pwd)
AUTOFILE="" AUTOCAP="" AUTOUSED=""
if [[ -n "$AUTO" ]]; then
  if [[ -f "$MODEL" ]]; then
    SHA=$(python3 -I "$HERE/profile_blend.py" ident "$MODEL") || die "--auto-profile: cannot read $MODEL's metadata"
  else
    (( DRY )) || die "no model container at $MODEL"; SHA=SHA256OFTHECONTAINERMETADATA
  fi
  AUTOFILE=$AUTO/${SHA:0:16}.profile.json
  AUTOCAP=$AUTO/.capture-${SHA:0:16}-$(date -u +%Y%m%dT%H%M%SZ)-$$.json
  if (( ! DRY )); then
    mkdir -p "$AUTO" || die "--auto-profile: cannot make $AUTO"
    if [[ -n "$PROFILE" ]]; then
      echo "serve-batch.sh: auto-profile: --profile $PROFILE given; $AUTOFILE not used (this run still recorded into it)" >&2
    elif [[ -f "$AUTOFILE" ]]; then
      if why=$(python3 -I "$HERE/profile_blend.py" match "$MODEL" "$AUTOFILE" 2>&1); then
        PROFILE=$AUTOFILE AUTOUSED=1
        echo "serve-batch.sh: auto-profile: using $AUTOFILE ($(python3 -I "$HERE/profile_blend.py" show "$AUTOFILE" | head -1 | sed 's/^[^:]*: //'))" >&2
      else
        echo "serve-batch.sh: auto-profile: $AUTOFILE not used: ${why##*: }" >&2
      fi
    else
      echo "serve-batch.sh: auto-profile: no profile for this container in $AUTO yet; this run records one" >&2
    fi
  else
    [[ -z "$PROFILE" ]] && echo "# auto-profile: --expert-profile $AUTOFILE when it exists and matches"
  fi
  ARGS+=(--expert-stats "$AUTOCAP")
fi
[[ -n "$PROFILE" ]] && ARGS+=(--expert-profile "$PROFILE")
[[ -n "$SEQLEN" ]] && ARGS+=(--max-seq-len $((10#$SEQLEN)))
[[ -n "$CAPTURE" ]] && ARGS+=(--expert-stats "$CAPTURE")
[[ -n "$MIRROR" ]] && ARGS+=(--expert-plane-mirror "$MIRROR" --expert-plane-mirror-decode-share 0.13)
ARGS+=("${EXTRA[@]}")

if (( DRY )); then
  printf 'env %s\n' "${ENV[@]}"
  printf 'cmd'; printf ' %q' "$BIN" "${ARGS[@]}"; echo
  exit 0
fi
# The memory guard is its own background process watching this pid, which `exec` then hands to the server: the
# process a caller starts and signals is radiance itself (a SIGINT reaches it, and nothing of it outlives a SIGKILL).
GUARD=${MEMGUARD_MIB:-3072}
LPID=$$
if (( GUARD > 0 )); then
  ( while kill -0 "$LPID" 2>/dev/null; do
      avail=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
      if (( avail < GUARD )); then
        echo "serve-batch.sh: MemAvailable ${avail} MiB below ${GUARD} MiB; stopping the server" >&2
        kill -INT "$LPID"; break
      fi
      sleep 1
    done ) < /dev/null &
fi
if [[ -z "$AUTO" ]]; then
  exec env "${ENV[@]}" "$BIN" "${ARGS[@]}"
fi
# --auto-profile: the server is a child (signals forwarded, SIGINT if this script dies), then the blend.
PD=(); setpriv --help 2>/dev/null | grep -q -- --pdeathsig && PD=(setpriv --pdeathsig INT --)
"${PD[@]}" env "${ENV[@]}" "$BIN" "${ARGS[@]}" &
SPID=$!
trap 'kill -INT $SPID 2>/dev/null' INT
trap 'kill -INT $SPID 2>/dev/null' TERM   # the server's clean stop either way (a script's & job ignores SIGINT)
while :; do wait "$SPID"; rc=$?; kill -0 "$SPID" 2>/dev/null || break; done
trap - INT TERM
say() { echo "serve-batch.sh: auto-profile: $*" >&2; }
if [[ ! -s "$AUTOCAP" ]] || ! python3 -I -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("final") is True else 1)' "$AUTOCAP" 2>/dev/null; then
  say "no final capture (the server did not stop cleanly, exit $rc); $AUTOFILE unchanged"
  exit $rc
fi
TMP=$AUTO/.new-$$.profile.json; rm -f "$TMP"
PARTS=("1:$AUTOCAP"); [[ -f "$AUTOFILE" ]] && PARTS=("1:$AUTOFILE" "1:$AUTOCAP")
if msg=$(python3 -I "$HERE/profile_blend.py" write --portable --container "$MODEL" \
           --mixed-share "${AUTO_PROFILE_MIXED_SHARE:-0.5}" "$TMP" "${PARTS[@]}" 2>&1); then
  [[ -f "$AUTOFILE" ]] && mv -f "$AUTOFILE" "${AUTOFILE%.profile.json}.prev.json"
  mv -f "$TMP" "$AUTOFILE" && rm -f "$AUTOCAP"
  say "wrote $AUTOFILE ($(tr '\n' ' ' <<< "$msg" | sed 's/ *$//'))${AUTOUSED:+; this run used the previous one}"
else
  rm -f "$TMP"
  say "this run's counts not blended, capture kept at $AUTOCAP: $(tail -1 <<< "$msg")"
fi
exit $rc
