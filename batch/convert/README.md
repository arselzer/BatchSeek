# Rebuilding model.rad

`convert.sh` rebuilds `model.rad`, the DeepSeek-V4.1-Flash container that every BatchSeek measurement used, from
public inputs, so you need not download it from
[Hugging Face](https://huggingface.co/aselzer/DeepSeek-V4.1-Flash-1017-radiance). Every input and the output are
pinned by sha256, so you can check that your rebuild is byte-identical.

**Nobody has rebuilt the container a second time yet.** The steps below are the ones that made it, written down
afterwards. Items marked **[unverified]** were not tested by a rebuild. If you try it, please report the result
either way.

```sh
convert/convert.sh --dry-run --work /big/disk/ds41-rebuild   # prints every step and checks disk, memory and tools
convert/convert.sh --work /big/disk/ds41-rebuild
```

## Target

| | |
|---|---|
| File | `model.rad` (RAD2), 416,220,274,688 bytes (387.64 GiB) |
| sha256 | `12106cfa7a32c8c6a765c37367639a5cce34dbc26789e484f566d37c4aff25ab` |
| `--sources` report | `sources.jsonl`, sha256 `2903452d390c2d49be596e51820ba0ce1ee4efc80dcd2d81cc10e0b3aaedcb6b` |

## What you need

- **Disk:** about 870 GB free: 449 GB of inputs that are kept and 416 GB of output.
- **Network:** about 652 GB of downloads. Official shards 47 and 48 (101.5 GB each) are streamed through and not kept.
- **Machine:** Linux x86-64, CPU only (no GPU; the converter is built with `RAD_WITH_HIP=OFF`), 8 GiB or more of
  free RAM, `git`, `cmake`, `make`, a C++ compiler, `curl`, `python3`, `sha256sum`. The original ran with GCC 15.2.0
  and CMake 4.2.3. Builds with other compilers were not tried **[unverified]**. The converter's peak memory use was
  not recorded.
- **Time:** the conversion took 11.5 minutes on the original machine. Downloading and hashing take far longer.

## Inputs

| Input | Source | Used for |
|---|---|---|
| `config.json`, `tokenizer.json`, `tokenizer_config.json`, `inference/config.json`, `model.safetensors.index.json` | official [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) @ `1bae0e10…` | config, tokenizer, tensor index |
| shards 02-43 (298.3 GB) | official, same revision | all weights except the Engram tables |
| shards 47 and 48 (streamed) | official, same revision | four small Engram tensors each |
| `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf` (110.6 GB) | **community conversion** [ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF) @ `cb02422c…` | the two Engram tables (layers 1 and 14), IQ4_NL |
| `…-IQ2_XS-00001-of-00004.gguf` (39.5 GB) | **community conversion**, same repo | only seven integer Engram controls; no IQ2 weights are read |
| radiance @ `bd65d4e5…` | [upstream radiance](https://codeberg.org/StillDeadcode/radiance), pull request 266 | the converter source |
| `ds41-converter.patch` | this directory | converter changes for this container |
| `native-mxfp4-experts-bf16-nonexperts.recipe` | this directory | the conversion recipe; the container embeds its text |

Both Hugging Face repositories are MIT-licensed. Shard 01 (vision) and shards 44-46 (MTP) are not used. The full
revisions and the sha256 of every file are in `convert.sh`.

The tensor names in the recipe are the ones upstream radiance's DeepSeek-V4.1 loader reads; some of them follow
upstream radiance's internal naming rather than DeepSeek's.

## What convert.sh does

0. **Preflight.** Checks the tools, 8 GiB of free RAM, free disk (870 GB minus inputs already downloaded), and that
   the patch, recipe and helper scripts match `SHA256SUMS`. It refuses to overwrite an existing `model.rad`.
1. **Build the converter.** Clones upstream radiance, checks out `bd65d4e58f7c4233de09ad5245a82c9510eb8962`
   (fetched from pull request 266 if a plain clone lacks it), applies `ds41-converter.patch`, and builds
   `rad-convert` and `rad-info` host-only with `SOURCE_DATE_EPOCH=1793102400` (the container embeds the
   converter's build date) and ccache off. Your binaries need not match the original's hashes; only the output must.
2. **Download.** Fetches the official small files, shards 02-43 and the two community GGUF files at pinned
   revisions, resumable, and checks each sha256.
3. **Build the input view.** The converter records the last two components of the checkpoint path, so the view
   must be named `ds41-native-assets-v3-view-host-20261027T104300Z/input-view`. It holds `config.json`, symlinks
   to shards 02-43, a derived index (`make_view_index.py`: 93,374 of the official 96,085 names), and compact copies
   of shards 47 and 48 (`extract_tail.py`, which reads each shard from the network, checks the whole stream's
   sha256 and keeps four Engram tensors byte for byte). The tokenizer files go in a separate `tok/`. The derived
   index and the compact format were checked against the original files; the streaming path through the full
   101.5 GB shards was not re-run **[unverified]**.
4. **Convert.** Runs `rad-convert` with the original environment and arguments (`--kernels libref --arch
   deepseek41 --set deepseek41.engram_source=ista_iq4_nl_v1`, two `--add` GGUF files, the recipe,
   `OMP_NUM_THREADS=2`). Whether the order of the two `--add` files or the thread count changes the output is
   **[unverified]**, so both are kept as they were.
5. **Verify.** Checks `sources.jsonl` (quick: a mismatch means the plan differs), the size and sha256 of
   `model.rad`, and prints its header with `rad-info`. Expect architecture `deepseek41`, `created by rad-convert
   Oct 27 2026 from ds41-native-assets-v3-view-host-20261027T104300Z/input-view`, a 1,205-line recipe, 162 metadata
   entries, 31,848 directory entries, 62,572 planes, 6 encodings, n_layers 40, n_expert 384, n_expert_used 6,
   n_vocab 129,280.

A stopped run can be restarted: checked downloads are remembered in `WORK/marks/` and not hashed again.

## Precision

The container keeps the release's native MXFP4 routed-expert codes and scales unchanged (30,720 expert weights,
288.8 GB); nothing is dequantized or requantized. It differs from the release elsewhere:

- The two Engram tables come from [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF)'s IQ4_NL GGUF instead of the official FP8 tables. The server can read
  the official FP8 tables instead (`--engram-file`, see the main README).
- The other 1,119 neural tensors are BF16 or F32 casts of the official weights (FP8→BF16, BF16→BF16, BF16→F32,
  F32→F32; never narrowed from F32 to BF16).
- The seven integer Engram controls (517,952 B) are copied unchanged from the IQ2_XS GGUF.

## Notes

- `ds41-converter.patch` is the converter tree that made the container, as a diff against `bd65d4e5`, with one code
  comment reworded since. Comments do not change the converter's output.
- After a rebuild, a short smoke test on the cards (see "From zero" in the main README) shows that the container
  loads. No reference reply is pinned here.
