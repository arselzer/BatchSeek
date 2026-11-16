#!/usr/bin/env bash
# convert.sh: rebuild the DeepSeek-V4.1-Flash model.rad container from public inputs, CPU only.
# Follows README.md in this directory step by step. Run with --dry-run first.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: convert/convert.sh [--dry-run] [--work DIR] [--jobs N]

Rebuilds model.rad (sha256 12106cfa...25ab, 416,220,274,688 bytes) from DeepSeek's official
DeepSeek-V4.1-Flash-1017 release plus two files of ISTA-DASLab's community GGUF conversion.
Needs about 870 GB of free disk in DIR and downloads about 652 GB. CPU only; no GPU is used.

  --dry-run   print every step, command, size and the disk need; download and build nothing
  --work DIR  working directory for the clone, the inputs and the output (default: ./ds41-rebuild)
  --jobs N    parallel build jobs for the converter (default: 8; does not change the output)

Steps: 1 build the converter, 2 download and check the inputs, 3 build the input view
(streams shards 47 and 48 without keeping them), 4 convert, 5 check the output hashes.
Each step is skipped when its result is already there and checked, so a stopped run can resume.
An existing DIR/model.rad is never overwritten.
EOF
}

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DRY=0
WORK=$PWD/ds41-rebuild
JOBS=8
while [ $# -gt 0 ]; do
    case $1 in
        --dry-run) DRY=1 ;;
        --work) [ $# -ge 2 ] || { echo "convert.sh: --work needs a directory" >&2; exit 2; }; WORK=$2; shift ;;
        --jobs) [ $# -ge 2 ] && [[ $2 =~ ^[1-9][0-9]*$ ]] || { echo "convert.sh: --jobs needs a positive number" >&2; exit 2; }
                JOBS=$2; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "convert.sh: unknown argument '$1' (see --help)" >&2; exit 2 ;;
    esac
    shift
done
case $WORK in /*) ;; *) WORK=$PWD/$WORK ;; esac

# ------------------------------------------------------------------------------------------ pins
RADIANCE_URL=https://codeberg.org/StillDeadcode/radiance
RADIANCE_FALLBACK_URL=https://github.com/arselzer/radiance-ramcache   # its main branch contains the base commit
RADIANCE_REV=bd65d4e58f7c4233de09ad5245a82c9510eb8962
OFFICIAL_REPO=deepseek-ai/DeepSeek-V4.1-Flash-1017
OFFICIAL_REV=1bae0e1088993b9b82de08f59de5be5a34e42170
ISTA_REPO=ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF
ISTA_REV=cb02422c900df489cea5833e8a92171224059b3c
OFFICIAL_URL=https://huggingface.co/$OFFICIAL_REPO/resolve/$OFFICIAL_REV
ISTA_URL=https://huggingface.co/$ISTA_REPO/resolve/$ISTA_REV
SOURCE_DATE_EPOCH_PIN=1793102400      # 2026-10-27: the container embeds the converter's __DATE__
VIEW_PARENT=ds41-native-assets-v3-view-host-20261027T104300Z   # recorded in the container: keep the name
RECIPE=native-mxfp4-experts-bf16-nonexperts.recipe
PATCH=ds41-converter.patch

TARGET_SHA=12106cfa7a32c8c6a765c37367639a5cce34dbc26789e484f566d37c4aff25ab
TARGET_BYTES=416220274688
SOURCES_SHA=2903452d390c2d49be596e51820ba0ce1ee4efc80dcd2d81cc10e0b3aaedcb6b
INDEX_SHA=24918e69c2e922bb87ef5abec921bce8bc0f666ac51d00ea3572e52fd4b31ca9
NEED_DISK=870000000000                # inputs kept (~449 GB) + output (416.2 GB) + margin
DOWNLOAD_BYTES=652000000000

# Small official files: sha256, path in the repo, bytes.
SMALL_FILES=(
"8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879 config.json 3311"
"c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b tokenizer.json 6367257"
"6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547 tokenizer_config.json 801"
"2e84f45cf1dac8c7fcbb200e96667d4b913275690668ed496f24c7747207a809 inference/config.json 1982"
"74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8 model.safetensors.index.json 7470294"
)
# Official shards 02-43, kept in full (LFS sha256; 298.3 GB together).
SHARDS=(
"43616eebd88dd4f1f1b642772356fc160478d1d9d04edd240431e7d065d2eda7 model-00002-of-00048.safetensors"
"41611994fc9b27829b757d72257570d65a976fe65b8c6df8641adcd139d65ded model-00003-of-00048.safetensors"
"e8e901b3cb1ab46f32729c7eaab37bd774a09094f0a02c51a60e883f779f2c39 model-00004-of-00048.safetensors"
"c5df71c02d6a22bcb20a0dbbcda9c6b65aa104f44415102a9a6a03bee92bb9de model-00005-of-00048.safetensors"
"a5d42afde2aee33619fb7a674b4c737c38d1f1d47a636a26f3edfb3b54164167 model-00006-of-00048.safetensors"
"24856942add8f73f5e1fbaea1e9c9b6ff597fd1b4aa4651d789833dd3a5a2ab9 model-00007-of-00048.safetensors"
"7bf5f7afb5d2bacad49feffb049989fb365d6dbf96986e76265bd7a7a3469564 model-00008-of-00048.safetensors"
"e6ae47e9daca138fda8c04f833c8328eb21671c37d1c8e2c3f7a82aac26649f6 model-00009-of-00048.safetensors"
"835e9a8b0a28cc571cb1cf9e6014b3d5eb84ac41990a8aa4519e1ac435d7e23c model-00010-of-00048.safetensors"
"5df968edefe59033e451ff8512bef57da2cf97b57b4e7a1380118489dd02915b model-00011-of-00048.safetensors"
"4f97bf292f58178ecdf593b55252a401d6e9102f18625a4fccc52b8ee57e7be0 model-00012-of-00048.safetensors"
"1b8567854887396840eb5392ca763101308064a142cb36156159096e581659eb model-00013-of-00048.safetensors"
"e5ba6d7e4380728ad70b8208676075a898f80a64c3618060f1b81e4218a159d1 model-00014-of-00048.safetensors"
"f4b29d936024e68f44fb14d0ffffb15b42b27be04b4a0296acdae695afa94c30 model-00015-of-00048.safetensors"
"3aa67228351c80e708dab4ac7f6e56b19000649cce2cd655def3168a3de4115c model-00016-of-00048.safetensors"
"b704eec7b7aa1bad6c9d57fa7f61f2db28862e8e5fc2468fd9d872689c778c7d model-00017-of-00048.safetensors"
"1405cf03376739f46e72e6cdd4d404969f1b94c717bb01782a300e2fbbd17e29 model-00018-of-00048.safetensors"
"a84beaded1913e8d6119ac111f25040c8ec42e59f200dbe04c3a991cac512669 model-00019-of-00048.safetensors"
"038914f79cf5499da90f0bfb8e1b7761cc37d5f4dd132e2431a6d02c9fb9fdfe model-00020-of-00048.safetensors"
"2ba49f59c8cec144e41370f9f820f268899d0b5cd0501e881908ca65bccd2dc2 model-00021-of-00048.safetensors"
"52f12cceca5e2fdd0191dfbd33b808b6c7ce844cf9351c119413c5ba3bcd28c8 model-00022-of-00048.safetensors"
"d6e47e8b7147c64eb6619732694dda811e1a9f8af1894977919af9522724ca31 model-00023-of-00048.safetensors"
"9d611d65374d37f5e02fdae14e680cb69bf55af8a8478802ac465b97c65be22b model-00024-of-00048.safetensors"
"8db532a7cbb23cd4b5aa775636ca13af95b758501290fb935074348f660a29cd model-00025-of-00048.safetensors"
"2d6d7261937d2a98c7398a55ed48dbf129d9b4f188aa9148e46ea9f720255115 model-00026-of-00048.safetensors"
"8e76de6a5cdc7ac0923526d1c1a2e6459c55bbfa692793b29cdfe07b4f9fbc52 model-00027-of-00048.safetensors"
"3c0e5ad6a4e8e811c1a55fd8f5b59c9fa329b241c671997c28fc7a01944d5a77 model-00028-of-00048.safetensors"
"392a950bcb062b89f683abc52d49a17e92c66f80cd1cb6fb92b71091da75319b model-00029-of-00048.safetensors"
"0b44d1364e578fc26dc0cc86564e49a92871a51063fb7f2f3b51cb020de44ab0 model-00030-of-00048.safetensors"
"d3230ffc32e11c5a34505c5acaeaff642aed5c770b297695e1f7a94071d361ac model-00031-of-00048.safetensors"
"96d284016ac885415859346d971c0de43bd9e7947337edfb53c34462866ef1d7 model-00032-of-00048.safetensors"
"313953b3b6aaa6d4c60bbcfc177dda5d8d1380c9fe72d82560ba078f55278a8d model-00033-of-00048.safetensors"
"98eb27655774796bb3feadb35ec64a9849104b88c4c6ae7cabf6fc1ee317a056 model-00034-of-00048.safetensors"
"4a321a28c048389039e4595641fe0b1f130428c61fc8f655545afe0083d37cd9 model-00035-of-00048.safetensors"
"97c7910a8a14002dd941f45f6ddb2bd335eaf6fa05f4a76b2ff9285d6557b65d model-00036-of-00048.safetensors"
"d37ee013b805344619536e299c73adf92df03744aa0a242f2c5465590015e5be model-00037-of-00048.safetensors"
"7713bf7a78319eb1c6214b312e0a2da8cac88d77dce851d50f1113d78f8975a4 model-00038-of-00048.safetensors"
"658a77d47d217995b9b78612e69ee0d0e33f6c2240f4ec629fecdca3ecb03f4c model-00039-of-00048.safetensors"
"6a9dbcc8d5894ba614b5e962764f45f6d0df7f14f1287b17f173d1f2cd762186 model-00040-of-00048.safetensors"
"458308f08ed29b34850bd6b9d10bf769620729d3725c77f7bc0c1d14e01b47cc model-00041-of-00048.safetensors"
"3d283a4ed6eb2d5013bacc97c3500361926a3261957943434513304df52adfd7 model-00042-of-00048.safetensors"
"4ae12b941c3facf8616ce62dd45bc6868076c2aeb82106f98123ea6c5bb64182 model-00043-of-00048.safetensors"
)
# Official shards 47 and 48: streamed through extract_tail.py, not kept.
# whole-file LFS sha256, file, bytes, sha256 of the compact .selected file, its bytes.
TAILS=(
"76a05106ece259be83d1bfea0dcb54f23377864ee60fd321690f35e6698a2af5 model-00047-of-00048.safetensors 101535150936 55fa35c2489c1326bb5d451a31a5c2d8c111b8128c30727025d44516252c8915 157522312"
"9dc51433e195bc78b5ad846b68643cc0351d3b174e5e4309222d3b7402c83afd model-00048-of-00048.safetensors 101537926640 795f0bfe95b277e29f587939836d4c3a5e27429701c1de02a31c1138e677ecae 157522320"
)
# Community conversion (ISTA-DASLab, MIT): sha256, file, bytes. Order matters for --add.
ISTA_FILES=(
"4c65307992ab586c527ca1f6f64062aa96a33b23c02b7fed103f45f75e854798 DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00001-of-00004.gguf 39535776992"
"eb19f30d368d393784e1fc54fdfaffc427ebb2110bf3762fdf5a8fbb9a2d740a DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf 110595290816"
)

SRC=$WORK/radiance
BUILD=$SRC/build
INPUTS=$WORK/inputs
OFF=$INPUTS/official
ISTA=$INPUTS/ista
VIEW=$WORK/$VIEW_PARENT/input-view
TOK=$WORK/tok
OUT=$WORK/model.rad
SOURCES=$WORK/sources.jsonl

# -------------------------------------------------------------------------------------- helpers
step() { printf '\n== %s\n' "$*"; }
note() { printf '   %s\n' "$*"; }
die()  { printf 'convert.sh: ERROR: %s\n' "$*" >&2; exit 1; }
gb()   { awk -v b="$1" 'BEGIN { if (b < 1e8) printf "%d B", b; else printf "%.1f GB", b / 1e9 }'; }
# run CMD...: print it; execute unless --dry-run.
run() {
    printf '   $'; printf ' %q' "$@"; printf '\n'
    [ "$DRY" = 1 ] || "$@"
}
sha_of() { sha256sum "$1" | cut -d' ' -f1; }
# Checked files get a marker in $WORK/marks (never next to the file: the view is read by the converter),
# so a resumed run does not re-hash hundreds of GB.
mark_of() { local r=${1#"$WORK"/}; printf '%s/marks/%s.sha256ok' "$WORK" "${r//\//%}"; }
ok_mark() { [ -f "$1" ] && [ -f "$(mark_of "$1")" ] && [ "$(cat "$(mark_of "$1")")" = "$2" ]; }
set_mark() { mkdir -p "$WORK/marks"; echo "$2" > "$(mark_of "$1")"; }
# fetch URL DEST SHA [BYTES]: resumable download to DEST.part, sha256 check, then rename.
fetch() {
    local url=$1 dest=$2 sha=$3 bytes=${4:-}
    local size=${bytes:+ ($(gb "$bytes"))}
    if ok_mark "$dest" "$sha"; then note "have $(basename "$dest")$size, checked earlier"; return; fi
    note "$(basename "$dest")$size  sha256 $sha"
    run mkdir -p "$(dirname "$dest")"
    if [ ! -f "$dest" ]; then
        run curl -fL --retry 5 --retry-delay 10 -C - -o "$dest.part" "$url"
        run mv "$dest.part" "$dest"
    fi
    [ "$DRY" = 1 ] && { printf '   $ sha256sum %q  # must be %s\n' "$dest" "$sha"; return; }
    local got; got=$(sha_of "$dest")
    if [ "$got" != "$sha" ]; then
        mv "$dest" "$dest.bad"
        die "$(basename "$dest"): sha256 $got, expected $sha (moved to $dest.bad; delete it and rerun)"
    fi
    set_mark "$dest" "$sha"
}

# ------------------------------------------------------------------------------------ preflight
step "0. Preflight"
note "work directory: $WORK"
[ "$DRY" = 1 ] && note "DRY RUN: nothing is downloaded, built or written."
fail=0
[ "$(uname -s)" = Linux ] && [ "$(uname -m)" = x86_64 ] || { note "FAIL: needs Linux x86-64 (this is $(uname -s) $(uname -m))"; fail=1; }
CXX_BIN=${CXX:-/usr/bin/c++}
for t in git cmake make curl sha256sum python3 awk df "$CXX_BIN"; do
    command -v "$t" >/dev/null 2>&1 || { note "FAIL: '$t' not found"; fail=1; }
done
if command -v "$CXX_BIN" >/dev/null 2>&1; then
    note "compiler: $("$CXX_BIN" --version | head -n1)  (the original: GCC 15.2.0; other compilers are [unverified])"
fi
command -v cmake >/dev/null 2>&1 && note "cmake: $(cmake --version | head -n1)  (the original: 4.2.3)"
python3 -c 'import sys; sys.exit(sys.version_info < (3, 8))' 2>/dev/null || { note "FAIL: needs python3 >= 3.8"; fail=1; }
for f in "$PATCH" "$RECIPE" extract_tail.py make_view_index.py SHA256SUMS; do
    [ -f "$HERE/$f" ] || { note "FAIL: $f missing next to convert.sh"; fail=1; }
done
if [ -f "$HERE/SHA256SUMS" ]; then
    if (cd "$HERE" && sha256sum --quiet -c SHA256SUMS); then note "patch, recipe and helper scripts match SHA256SUMS"
    else note "FAIL: files in convert/ do not match SHA256SUMS"; fail=1; fi
fi
mem_kb=$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo 2>/dev/null || echo 0)
if [ "${mem_kb:-0}" -lt $((8 * 1024 * 1024)) ]; then
    note "FAIL: MemAvailable $((mem_kb / 1024 / 1024)) GiB; the original run started with 8 GiB or more"; fail=1
else note "MemAvailable $((mem_kb / 1024 / 1024)) GiB (8 GiB or more needed)"; fi
if [ -e "$OUT" ]; then
    note "FAIL: $OUT already exists; this script never overwrites a container. Move it away or use another --work."
    fail=1
fi

# Disk: everything still to be written must fit. Files already in place count against the need.
have=0
if [ -d "$INPUTS" ]; then have=$(du -sb "$INPUTS" 2>/dev/null | cut -f1); fi
need=$((NEED_DISK - have)); [ "$need" -lt 0 ] && need=0
probe=$WORK; while [ ! -d "$probe" ]; do probe=$(dirname "$probe"); done
free=$(( $(df -Pk "$probe" | awk 'NR == 2 { print $4 }') * 1024 ))
note "disk: total need about $(gb $NEED_DISK) (inputs kept ~449 GB + output 416.2 GB); already in $INPUTS: $(gb "$have")"
note "disk: still needed $(gb "$need"), free on $probe: $(gb "$free")"
if [ "$free" -lt "$need" ]; then note "FAIL: not enough free disk ($(gb "$free") free, $(gb "$need") needed)"; fail=1; fi
note "network: about $(gb $DOWNLOAD_BYTES) of downloads (shards 47 and 48, 101.5 GB each, are streamed and not kept)"
if [ "$fail" = 1 ]; then
    [ "$DRY" = 1 ] || die "preflight failed (see FAIL lines above)"
    note "(dry run: continuing to list the steps despite the FAIL lines; a real run stops here)"
fi
[ "$DRY" = 1 ] || mkdir -p "$WORK"

# ------------------------------------------------------------------------------ 1. converter
step "1. Build the converter (radiance $RADIANCE_REV + $PATCH, host only, RAD_WITH_HIP=OFF)"
if [ "$DRY" = 0 ] && [ -x "$BUILD/bin/rad-convert" ] && [ -f "$BUILD/.convert-sh-built" ] &&
   [ "$(cat "$BUILD/.convert-sh-built")" = "$(sha_of "$HERE/$PATCH")" ]; then
    note "have $BUILD/bin/rad-convert, built earlier from this patch"
else
    if [ "$DRY" = 1 ] || [ ! -d "$SRC/.git" ]; then
        run git clone "$RADIANCE_URL" "$SRC"
    fi
    # The base commit is on upstream's pull request 266, not necessarily on a branch a plain clone fetches.
    if [ "$DRY" = 1 ]; then
        note "if $RADIANCE_REV is not in the clone, fetch it, trying in turn:"
        run git -C "$SRC" fetch origin refs/pull/266/head
        run git -C "$SRC" fetch origin "$RADIANCE_REV"
        run git -C "$SRC" fetch "$RADIANCE_FALLBACK_URL" main
    elif ! git -C "$SRC" cat-file -e "$RADIANCE_REV^{commit}" 2>/dev/null; then
        git -C "$SRC" fetch origin refs/pull/266/head || true
        git -C "$SRC" cat-file -e "$RADIANCE_REV^{commit}" 2>/dev/null || git -C "$SRC" fetch origin "$RADIANCE_REV" || true
        git -C "$SRC" cat-file -e "$RADIANCE_REV^{commit}" 2>/dev/null || git -C "$SRC" fetch "$RADIANCE_FALLBACK_URL" main || true
        git -C "$SRC" cat-file -e "$RADIANCE_REV^{commit}" 2>/dev/null ||
            die "radiance commit $RADIANCE_REV not found in $RADIANCE_URL (pull request 266) or $RADIANCE_FALLBACK_URL"
    fi
    if [ "$DRY" = 0 ] && [ "$(git -C "$SRC" rev-parse HEAD)" = "$RADIANCE_REV" ] &&
       git -C "$SRC" apply --reverse --check "$HERE/$PATCH" 2>/dev/null; then
        note "patch already applied in $SRC"
    else
        if [ "$DRY" = 0 ] && [ -n "$(git -C "$SRC" status --porcelain)" ]; then
            die "$SRC has local changes that are not this patch; remove the directory and rerun"
        fi
        run git -C "$SRC" checkout --detach "$RADIANCE_REV"
        run git -C "$SRC" apply --check "$HERE/$PATCH"
        run git -C "$SRC" apply "$HERE/$PATCH"
    fi
    note "SOURCE_DATE_EPOCH=$SOURCE_DATE_EPOCH_PIN fixes __DATE__, which the container embeds; ccache is turned off"
    export SOURCE_DATE_EPOCH=$SOURCE_DATE_EPOCH_PIN CCACHE_DISABLE=1
    run cmake -S "$SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$CXX_BIN" -DRAD_WITH_HIP=OFF \
        -DRAD_WITH_FFMPEG=OFF -DRAD_BUILD_TESTS=ON -DRAD_ASAN=OFF -DRAD_TSAN=OFF
    run cmake --build "$BUILD" -j"$JOBS" --target rad_core libref libquant deepseek41 rad-convert rad-info
    [ "$DRY" = 1 ] || sha_of "$HERE/$PATCH" > "$BUILD/.convert-sh-built"
fi

# ------------------------------------------------------------------------------- 2. download
step "2. Download and check the inputs"
note "A. official $OFFICIAL_REPO @ $OFFICIAL_REV: 5 small files, shards 02-43 kept (298.3 GB)"
for e in "${SMALL_FILES[@]}"; do
    read -r sha f bytes <<<"$e"
    fetch "$OFFICIAL_URL/$f" "$OFF/$f" "$sha" "$bytes"
done
for e in "${SHARDS[@]}"; do
    read -r sha f <<<"$e"
    fetch "$OFFICIAL_URL/$f" "$OFF/shards/$f" "$sha"
done
note "B. community conversion $ISTA_REPO @ $ISTA_REV (2 files, 150.1 GB)"
for e in "${ISTA_FILES[@]}"; do
    read -r sha f bytes <<<"$e"
    fetch "$ISTA_URL/$f" "$ISTA/$f" "$sha" "$bytes"
done

# ----------------------------------------------------------------------------- 3. input view
step "3. Build the input view $VIEW_PARENT/input-view (the name is recorded in the container)"
run mkdir -p "$VIEW" "$TOK/inference"
for e in "${SHARDS[@]}"; do
    read -r _ f <<<"$e"
    if [ "$DRY" = 1 ] || [ ! -L "$VIEW/$f" ]; then run ln -sfn "$OFF/shards/$f" "$VIEW/$f"; fi
done
run cp "$OFF/config.json" "$VIEW/config.json"
for f in config.json tokenizer.json tokenizer_config.json inference/config.json; do
    run cp "$OFF/$f" "$TOK/$f"
done
note "derived index: the official index's 93,374 names of layers 0-39, embed, norm and head"
run python3 -I "$HERE/make_view_index.py" "$OFF/model.safetensors.index.json" "$VIEW/model.safetensors.index.json"
if [ "$DRY" = 1 ]; then printf '   $ sha256sum %q  # must be %s\n' "$VIEW/model.safetensors.index.json" "$INDEX_SHA"
else
    got=$(sha_of "$VIEW/model.safetensors.index.json")
    [ "$got" = "$INDEX_SHA" ] || die "derived index sha256 $got, expected $INDEX_SHA"
fi
note "compact tails: shards 47 and 48 are streamed through extract_tail.py and not kept [unverified: the"
note "streaming path through the full 101.5 GB shards was never re-run; the compact format was checked]"
for e in "${TAILS[@]}"; do
    read -r whole f bytes sel_sha sel_bytes <<<"$e"
    sel=$VIEW/$f.selected.safetensors part=$WORK/$f.selected.part   # .part outside the view
    if [ "$DRY" = 0 ] && ok_mark "$sel" "$sel_sha"; then note "have $(basename "$sel"), checked earlier"; continue; fi
    note "$f ($(gb "$bytes") streamed) -> $(basename "$sel") ($sel_bytes B, sha256 $sel_sha)"
    if [ "$DRY" = 1 ]; then
        printf '   $ curl -fsSL %q | python3 -I %q %q\n' "$OFFICIAL_URL/$f" "$HERE/extract_tail.py" "$part"
        printf '   # the printed stream sha256 must be %s; then sha256sum of the compact file\n' "$whole"
        continue
    fi
    msg=$(curl -fsSL "$OFFICIAL_URL/$f" | python3 -I "$HERE/extract_tail.py" "$part") ||
        die "streaming $f failed (network or extract_tail.py); rerun to retry"
    echo "   $msg"
    [ "$msg" = "stream sha256 $whole bytes $bytes" ] || die "$f: stream was '$msg', expected sha256 $whole and $bytes bytes"
    got=$(sha_of "$part")
    [ "$got" = "$sel_sha" ] || die "$(basename "$sel"): sha256 $got, expected $sel_sha"
    mv "$part" "$sel"; set_mark "$sel" "$sel_sha"
done

# --------------------------------------------------------------------------------- 4. convert
step "4. Convert (about 12 min on the original machine; writes $(gb $TARGET_BYTES))"
note "Environment and arguments as in the original run. Keep the --add order (IQ2_XS shard first) and"
note "OMP_NUM_THREADS=2: whether either changes the output is [unverified]."
run env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C TZ=UTC ROCR_VISIBLE_DEVICES=99 HIP_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES=-1 \
    OMP_NUM_THREADS=2 OMP_THREAD_LIMIT=2 OMP_MAX_ACTIVE_LEVELS=1 OMP_DYNAMIC=FALSE OMP_WAIT_POLICY=passive \
    "$BUILD/bin/rad-convert" "$VIEW" --home "$BUILD/radiance_home" --kernels libref --arch deepseek41 \
    --max-tok 4 --max-ctx 512 --max-spec 0 \
    --config "$TOK" --tokenizer "$TOK/tokenizer.json" \
    --set deepseek41.engram_source=ista_iq4_nl_v1 \
    --add "$ISTA/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00001-of-00004.gguf" \
    --add "$ISTA/DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf" \
    --recipe "$HERE/$RECIPE" \
    --sources "$SOURCES" -v -o "$OUT"

# ---------------------------------------------------------------------------------- 5. verify
step "5. Check the output"
if [ "$DRY" = 1 ]; then
    printf '   $ sha256sum %q  # must be %s\n' "$SOURCES" "$SOURCES_SHA"
    printf '   $ stat -c %%s %q  # must be %s\n' "$OUT" "$TARGET_BYTES"
    printf '   $ sha256sum %q  # must be %s (10-15 min on NVMe)\n' "$OUT" "$TARGET_SHA"
    run "$BUILD/bin/rad-info" "$OUT"
    printf '\nDry run done.%s\n' "$([ "$fail" = 1 ] && echo ' Preflight FAILED: fix the FAIL lines before a real run.')"
    exit "$fail"
fi
got=$(sha_of "$SOURCES")
[ "$got" = "$SOURCES_SHA" ] || note "MISMATCH: sources.jsonl sha256 $got, expected $SOURCES_SHA (the plan differs; model.rad will too)"
bytes=$(stat -c %s "$OUT")
[ "$bytes" = "$TARGET_BYTES" ] || die "model.rad has $bytes bytes, expected $TARGET_BYTES"
note "hashing model.rad (10-15 min on NVMe)"
got=$(sha_of "$OUT")
"$BUILD/bin/rad-info" "$OUT" || true
if [ "$got" = "$TARGET_SHA" ]; then
    printf '\nOK: %s is byte-identical to the published container (sha256 %s).\n' "$OUT" "$TARGET_SHA"
    echo "Please report a successful rebuild: nobody had rebuilt it before."
else
    die "model.rad sha256 $got, expected $TARGET_SHA. Compare rad-info's header with convert/README.md and report it."
fi
