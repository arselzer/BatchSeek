# BatchSeek

Large batch jobs (hundreds of requests at once) with DeepSeek-V4.1-Flash on four 16 GB AMD RDNA4 GPUs and 256 GB
of RAM: a fork of radiance tuned for large batches, plus the tools to run and measure such jobs (in `batch/`).

## Credits

This repository builds on an engine written by other people:

- **[radiance](https://codeberg.org/StillDeadcode/radiance)** by [Deadcode](https://codeberg.org/StillDeadcode) and contributors: the inference server,
  the HIP kernels and everything around them, including tcclaviger's and keptn42's all-reduce kernels and
  slydlake's and ggz14's MXFP4 matrix kernels. The full list is in radiance's
  [NOTICE](https://codeberg.org/StillDeadcode/radiance/src/branch/main/NOTICE).
- **[baKon](https://codeberg.org/bakon)**: DeepSeek-V4.1 in radiance ([pull request 266](https://codeberg.org/StillDeadcode/radiance/pulls/266)):
  the model across GPUs, the MXFP4 import, the Engram tables, and streaming experts from RAM and disk.
- **This repository** is a fork of radiance with changes for large batches (below), plus the batch tools in `batch/`:
  launcher, batch client, calibration, placement profiles, BIRD tools and model conversion.

Thanks also to DeepSeek for the open weights, to [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF)
for the IQ4_NL Engram tables in the converted model, and to the BIRD team for their benchmark and databases.

Unofficial; not affiliated with DeepSeek, BIRD or radiance.


## What this fork changes in radiance

The routed experts of DeepSeek-V4.1-Flash (about 289 GB) do not fit in four GPUs' 64 GB of VRAM. baKon's port
already keeps hot experts on the GPUs and streams the rest from RAM and disk. This fork changes how that path
behaves at hundreds of concurrent requests:

- **RAM expert cache with CPU-computed experts.** A packed RAM cache holds most of the experts that are not on the
  GPUs. The CPU computes some of them each step instead of sending them over PCIe (while generating and while
  prompt processing). The CPU kernels use the GPU's arithmetic, so the results are bit-identical to the GPU's.
- **Expert placement in VRAM.** Hot experts stay on the GPUs and follow live routing. The VRAM that the KV pool and
  other arenas do not need holds more experts.
- **Batch admission and scheduling.** Requests that arrive together have their prompts processed together in large
  chunks, and generation starts once the prompts are in, so each step generates a token for as many requests as
  possible. Each expert sent to a GPU serves every token in the batch that needs it.
- **Prefix cache and memory sizing.** Shared prompt heads (for example a database schema) are read once. The
  prefix cache's snapshot store is sized for short batch prompts, and a RAM budget sizes the expert cache from what
  the host actually has free.
- **Placement profiles.** baKon's port records how often each expert is used (`--expert-stats`) and keeps the
  busiest on the GPUs (`--expert-profile`). The fork also fills the RAM cache from the profile, adds `calibrate.sh`
  and profile tools, and makes profiles portable: the server then matches them to a container by its size and the
  sha256 of its metadata header, not by its path.
- **Kernels.** Faster prompt processing (prefill; register tiling) and a tiled form of the FP8 dense layers' matrix-unit kernel (based on radiance's
  tiled GEMM), both
  with unchanged results. Optionally, prompt processing and attention on the matrix units in another summation order: faster, but replies can then
  depend on the batch (see "Reproducible mode").
- **Reproducibility checks and measurement harness.** `tools/ds41-host-experts/ab_window.sh` runs A/B measurement
  windows. Each optimization (except the two opt-out fast modes below) was accepted only if replies, logits and
  prompt-processing hashes stayed byte-identical to the
  same engine with that optimization off.

Configuration, switches and measurements: [tools/ds41-host-experts/README.md](tools/ds41-host-experts/README.md).

## The batch tools (`batch/`)

- `serve-batch.sh` starts the fork's server with the measured batch configuration (`--help` explains every
  option; `--dry-run` prints the environment and command without starting anything).
- `batch_client.py` (optional; any OpenAI-API client works) sends a JSONL file of prompts N at a time, refilling as replies finish, and writes the replies
  in input order. Standard library only.
- `calibrate.sh` records a placement profile on your own prompts in one command.
- `profile_blend.py` blends, checks and shows placement profiles; `profiles/profile_rebind.py` makes one portable
  (`strip`) or ties it to your copy of the container (`bind`).
- `profiles/bird-train.profile.json` is a placement profile (counts of how often each expert is used; no prompts)
  recorded on 64 public BIRD training questions.
- `bird/` builds the BIRD request sets from the official downloads and scores replies.
- `batch/docs/charts/` has the charts below and the scripts that draw them.

## Results

Four RX 9070 XT (TP4), EPYC 7443 (24 cores), 256 GB DDR4-2666, one 2 TB NVMe drive.

![A 512-request job over time: requests waiting, processing prompts and generating, output tokens per second, KV cache, power](batch/docs/charts/timeline-512.svg)

One batch of 512 SQL questions from my own workload (not public), from the first request to the last reply.

| Requests at once | Output tokens/s, all together | Per request | vs. 1 request | Power (avg, W) | Wh per 1,000 tokens |
|---:|---:|---:|---:|---:|---:|
| 1 | 31 | 31 | 1× | ~580 | ~5.2 |
| 8 | 55 | 6.9 | 1.8× | 634 | 3.2 |
| 64 | 86 | 1.34 | 2.8× | 725 | 2.3 |
| 256 | 203 | 0.79 | 6.6× | 908 | 1.2 |
| **512** | **269** | **0.53** | **8.7×** | **965** | **1.0** |

Output tokens per second while generating, all requests together. Power is GPUs + CPU.

![Output tokens per second by number of requests at once](batch/docs/charts/throughput.svg)

Whole jobs, from the first request to the last reply, prompts included: the 512 SQL questions take 566 s with the best
configuration measured (the launcher's defaults: about 600 s).

![Energy per 1,000 requests by job](batch/docs/charts/energy.svg)

GPUs and CPU, from their own sensors.

Answer quality on 500 BIRD questions (execution accuracy, temperature 0, thinking off; two reference queries that
run past the 30-second limit count as wrong for everyone, as in BIRD's evaluator):

| Who runs the model | Execution accuracy | Cost for 500 questions |
|---|---:|---:|
| **This box** (default mode, public profile) | **61.6%** | **€0.014** at €0.30/kWh |
| DeepSeek API | 59.8% | €0.031 |
| DeepInfra (FP8) | 60.2% | €0.015 |
| Wafer (MXFP4) | 60.2% | €0.021 |
| *Other models* | | |
| Claude Haiku 5.5 | 62.0% | €0.10 |
| GPT-6 Luna | 55.8% | €0.063 |

API prices as charged through OpenRouter on 16 November 2026, at $1.12 per euro. Apart from GPT-6 Luna, the
accuracy differences are within noise. The other models ran with thinking off at their default sampling (they don't
accept temperature 0), with prompts written for DeepSeek.

The model file comes with a generic usage profile, taken from the quantization's calibration text. The same 256
BIRD questions with it and with a recorded profile:

| Expert usage profile | Job time |
|---|---:|
| Generic, from the model file | 221 s |
| Recorded from 64 public BIRD *training* questions (different databases) | **146 s** |

The measured numbers used the IQ4_NL Engram tables from [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF)'s GGUF release. The release's official FP8 tables
(`--engram-file`) ran at the same speed (170 s vs 172 s on 256 BIRD questions) with similar accuracy.

## Requirements

- A HIP build of this repository, built like upstream radiance (see "Build from source" in its README; the
  `deepseek41` plugin and `libr4d` must come from the same build), and the SiLU table its GPU check captures
  (`ds41_mxfp4_device_check ... swiglu silu_table_gfx1201.bin`; see `tools/ds41-host-experts/README.md`).
- The DeepSeek-V4.1-Flash container (`.rad`, 416 GB) on NVMe.
- Tested only on four 16 GB RX 9070 XT GPUs (gfx1201, TP4), a 24-core EPYC and 256 GB of RAM. Big batches use nearly
  all of that RAM: run jobs alone on the machine.
- Python 3 (standard library only, except `batch/docs/charts/decode_steps.py`, which needs numpy and scipy).

## From zero

Note (16 November 2026): the engine update behind the launcher's newest options (the faster default mode,
`--reproducible` and `--engram-file`) is being merged into `main` today; until then `main` runs the reproducible
configuration and ignores `--engram-file`.

```sh
git clone https://github.com/arselzer/BatchSeek && cd BatchSeek

# 1. Build (HIP for the cards it finds; everything lands in build/radiance_home)
./build.sh
make -C tools/ds41-host-experts mxfp4-check
# once per build: capture the card's SiLU table, which the launcher needs
HIP_VISIBLE_DEVICES=0 tools/ds41-host-experts/ds41_mxfp4_device_check build/radiance_home/kernels/libr4d.so \
  swiglu silu_table_gfx1201.bin

# 2. Get the model (8 parts of about 52 GB, plus the official FP8 Engram shards)
hf download aselzer/DeepSeek-V4.1-Flash-1017-radiance --local-dir model
cd model && cat model.rad.part-0{1..8} > model.rad && sha256sum -c SHA256SUMS --ignore-missing \
  && rm model.rad.part-* && cd ..

# 3. Serve (takes a few minutes to load; ready when http://localhost:8000/v1/models answers)
export RADIANCE_HOME=$PWD/build/radiance_home RADIANCE_BIN=$PWD/build/bin/radiance
cd batch
./serve-batch.sh --model ../model/model.rad --silu-table ../silu_table_gfx1201.bin --streams 512 \
  --profile profiles/bird-train.profile.json \
  --engram-file ../model/engram-fp8/model-00047-of-00048.safetensors,../model/engram-fp8/model-00048-of-00048.safetensors &

# 4. A first test with 8 questions, then the whole file
head -8 my-questions.jsonl > first8.jsonl && python3 batch_client.py first8.jsonl first8-answers.jsonl
python3 batch_client.py my-questions.jsonl answers.jsonl --concurrency 512
```

Disk: joining the parts needs room for the parts and `model.rad` at once (about 830 GB), plus 203 GB for
`engram-fp8/`. `--engram-file` is optional: leave it out (and skip `engram-fp8/`) to use the IQ4_NL Engram tables
bundled in `model.rad`; both ran at the same speed with similar accuracy in our tests. Or rebuild `model.rad` from
DeepSeek's release with `batch/convert/convert.sh` (CPU only; see [convert/README.md](batch/convert/README.md)).

The server speaks the OpenAI API, so any client works as long as it sends many requests at once (one request at a
time gets about 31 tokens/s instead of 269). `batch_client.py` is a small helper that does that for a JSONL file and
keeps the answers in order. `my-questions.jsonl` is your own file, with one `{"prompt": "...", "system": "...", "id": "..."}` a line; put text that prompts share (a
schema, instructions) in `system`, so the prefix cache processes it once. `batch_client.py --help` lists the rest.

**Long prompts.** Each request gets `--ctx-per-stream` tokens for prompt plus reply (default 1,152, enough for short
SQL questions). For longer prompts or replies, raise it and lower `--streams` so that streams x ctx-per-stream stays
at most about 1.2 million tokens (tested: `--streams 256 --ctx-per-stream 4608` for BIRD's 2026 prompts). `--dry-run` shows the resulting settings.

## Calibration

A profile recorded on prompts like yours places experts better. One command starts a calibration server, sends
your prompts, stops it and writes a portable profile:

```sh
./calibrate.sh --model model.rad --silu-table silu_table_gfx1201.bin my-prompts.jsonl my.profile.json \
  [--base profiles/bird-train.profile.json]
./serve-batch.sh --model model.rad --silu-table silu_table_gfx1201.bin --profile my.profile.json
```

`--base` blends a shipped profile in. Portable profiles carry no path; the server accepts them on any copy of the
same container (matched by its size and the sha256 of its metadata header). `serve-batch.sh --auto-profile DIR`
instead keeps a profile per container in DIR and refreshes it from every run.

## Reproducible vs default mode

By default, prompt processing and attention run on the GPUs' matrix units, summed in a different order, so a reply can change with
the batch around it. `--reproducible` (alias `--exact`) turns those two modes off: every reply is then
byte-identical whether a request runs alone or among hundreds of others, and the same whichever placement profile is
used. On 500 BIRD questions the default finished the job 8% faster (260 s to 240 s) at similar accuracy (61.6% vs
62.2%; the paired test found no significant difference). Calibration runs use the reproducible mode unless the fast modes are
named.

## Rebuilding the BIRD sets

The builders read your own download of BIRD (CC BY-SA 4.0; not redistributed here) and print the sha256 of each set
they write. Run them with `python3 -I`, since they read downloaded data.

1. BIRD Mini-Dev: download `minidev_0703.zip` (linked from https://github.com/bird-bench/mini_dev; sha256
   `aeb211c0e39010bbdae3838bb5e8bd27dc446ed77495b1709f85ccc9bf67f2be`) and unpack it into its own directory.
   ```sh
   python3 -I bird/make_bird_sets.py minidev/MINIDEV sets/
   ```
   The builder checks the questions file's sha256 and writes the measured sets byte for byte:
   ```
   7a04362f210f7649d922de42b650bbe53602c091241e04194713d1eca86fcda6  bird256-receipt.json
   f6e171ea32b6f73c6ae99ec0fc60bccb5cefdaca6909e70559b4186455e46746  bird500-receipt.json
   ```
2. BIRD training questions (for recording a profile on public prompts): unpack
   `https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip` and run the builder on the folder that holds
   `train.json` and `train_databases.zip`:
   ```sh
   python3 -I bird/make_bird_train_set.py train/ sets/birdtrain64-receipt.json
   ```
   This writes the 64-question set that `profiles/bird-train.profile.json` was recorded on:
   ```
   07e52383de992829824d2da76a6e09ed17f170db876cf8e1ae1c4592d8bee8f0  birdtrain64-receipt.json
   ```
3. `bird/make_birddev_set.py` builds a larger set from BIRD's full dev split, and `bird/make_birdhf_sets.py` the
   2026 revision with BIRD's own prompts (https://huggingface.co/datasets/birdsql/bird_mini_dev).

Send a set with `python3 batch_client.py sets/bird256-receipt.json replies.jsonl --receipt --concurrency 256`.

## Scoring

```sh
python3 -I bird/bird_ex.py replies.jsonl minidev/MINIDEV/mini_dev_sqlite.json minidev/MINIDEV/dev_databases
python3 -I bird/paired_ex.py a.jsonl b.jsonl minidev/MINIDEV/mini_dev_sqlite.json minidev/MINIDEV/dev_databases
python3 -I bird/compare_runs.py minidev/MINIDEV/mini_dev_sqlite.json minidev/MINIDEV/dev_databases A=a.jsonl B=b.jsonl
```

`bird_ex.py` runs each reply's SQL and the reference SQL read-only on the database and compares result sets (BIRD's
execution accuracy). `ex_official` counts reference queries that fail or time out on their own database as wrong
(the numbers above); `ex` leaves them out. `paired_ex.py`
compares two runs question by question with an exact McNemar test. `compare_runs.py` reports how often runs agree
on reply text, SQL and results.

## Tests

```sh
python3 batch_client_test.py
python3 profile_blend_test.py
./calibrate_test.sh                       # a stub server on localhost; no GPU, no model
./serve_batch_test.sh PATH/TO/ab_window.sh   # needs the harness in ../tools/ds41-host-experts/; skips without it
```

## What was not tested

Upstream radiance serves many other models and has many other features (other architectures, speculative decoding,
vision and audio input, Docker deployments, other GPUs and GPU counts). None of them were tested in this fork's
setup, and the fork's changes may affect them. For those, see
[upstream radiance](https://codeberg.org/StillDeadcode/radiance) and its README.

Also not covered:

- other hardware than four 16 GB RX 9070 XT and 256 GB of RAM;
- continuous serving with a long queue (the measurements are finite batches submitted together);
- long contexts at large batch sizes;
- chat: at large batches each request gets about 0.5-0.8 tokens/s, which suits jobs nobody waits on.

## License

Apache-2.0, as upstream radiance; `LICENSE` and `NOTICE` are upstream's. The model weights are not part of this
repository and keep their own license (MIT, DeepSeek).
