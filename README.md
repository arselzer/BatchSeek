# BatchSeek

Scripts for running large batch jobs (hundreds of requests at once) with DeepSeek-V4.1-Flash on four 16 GB AMD
RDNA4 cards and 256 GB of RAM.

## Credits

BatchSeek is a small kit around an engine written by other people:

- **[radiance](https://codeberg.org/StillDeadcode/radiance)** by [Deadcode](https://codeberg.org/StillDeadcode) and contributors: the inference server,
  the HIP kernels and everything around them, including tcclaviger's and keptn42's all-reduce kernels and
  slydlake's and ggz14's MXFP4 matrix kernels. The full list is in radiance's
  [NOTICE](https://codeberg.org/StillDeadcode/radiance/src/branch/main/NOTICE).
- **[baKon](https://codeberg.org/bakon)**: DeepSeek-V4.1 in radiance ([pull request 266](https://codeberg.org/StillDeadcode/radiance/pulls/266)):
  the model across cards, the MXFP4 import, the Engram tables, and streaming experts from RAM and disk.
- **[radiance-ramcache](https://github.com/arselzer/radiance-ramcache)**, a fork of radiance for this setup. It
  holds the engine changes: the RAM expert cache with the CPU computing part of each step, batch admission and
  scheduling for hundreds of requests, the reproducibility checks and the measurement harness.
- **This repository** adds only the tools around that engine: the batch launcher, the batch client, calibration,
  placement profiles and the BIRD tools.

Thanks also to DeepSeek for the open weights and to the BIRD team for their benchmark and databases.

Unofficial; not affiliated with DeepSeek, BIRD or radiance.

## What it does

- `serve-batch.sh` starts the fork's server with the measured batch configuration (`--help` explains every
  option; `--dry-run` prints the environment and command without starting anything).
- `batch_client.py` sends a JSONL file of prompts N at a time, refilling as replies finish, and writes the replies
  in input order. Standard library only.
- `calibrate.sh` records a placement profile on your own prompts in one command.
- `profile_blend.py` blends, checks and shows placement profiles; `profiles/profile_rebind.py` makes one portable
  (`strip`) or ties it to your copy of the container (`bind`).
- `profiles/bird-train.profile.json` is a placement profile (counts of how often each expert is used; no prompts)
  recorded on 64 public BIRD training questions.
- `bird/` builds the BIRD request sets from the official downloads and scores replies.
- `docs/charts/` has the charts below and the scripts that draw them.

## Results

Four RX 9070 XT (TP4), EPYC 7443 (24 cores), 256 GB DDR4-2666, one 2 TB NVMe drive.

![A 512-request job over time: requests waiting, prefilling and decoding, output tokens per second, KV cache, power](docs/charts/timeline-512.svg)

One batch of 512 SQL questions, from the first request to the last reply.

| Requests at once | Output tokens/s, all together | Per request | Speedup | Power (avg, W) | Wh per 1,000 tokens |
|---:|---:|---:|---:|---:|---:|
| 1 | 31 | 31 | 1× | ~580 | ~5.2 |
| 8 | 55 | 7 | 1.8× | 634 | 3.2 |
| 64 | 86 | 1.3 | 2.8× | 725 | 2.3 |
| 256 | 203 | 0.8 | 6.6× | 908 | 1.3 |
| **512** | **269** | **0.5** | **8.7×** | **965** | **1.0** |

Decode rate summed over all requests, in the reproducible mode. The rows come from different runs and
configurations, so the speedup is what the box achieves, not batching measured in isolation. Power is GPU boards plus
CPU package.

![Decode throughput by number of concurrent requests](docs/charts/throughput.svg)

Whole jobs, from the first request to the last reply, prompts included. "Earlier configuration" is the same engine
and hardware before most of the batch work:

| Job | Earlier configuration | Batched | Speedup |
|---|---:|---:|---:|
| 256 BIRD questions (public) | 1,025 s | 146 s | 7.0x |
| 512 SQL questions | | 566 s | |
| 256 COCO photos to object labels | | 532 s | |

![Energy per 1,000 requests by job](docs/charts/energy.svg)

GPU boards and CPU package, from their sensors.

Answer quality on 500 BIRD questions (execution accuracy, temperature 0, thinking off; two reference queries that
run past the 30-second limit count as wrong for everyone, as in BIRD's evaluator):

| Who runs the model | Execution accuracy | Cost for 500 questions |
|---|---:|---:|
| **This box** (default mode, public profile) | **61.6%** | **€0.014** at €0.30/kWh |
| DeepSeek API | 59.8% | €0.031 off-peak, €0.062 peak |
| DeepInfra (FP8) | 60.2% | €0.015 |
| Wafer (MXFP4) | 60.2% | €0.021 |
| *Other models* | | |
| Claude Haiku 5.5 | 62.0% | €0.10 |
| GPT-6 Luna | 55.8% | €0.063 |

API prices as charged through OpenRouter on 16 November 2026, at $1.12 per euro. Apart from GPT-6 Luna, the
accuracy differences are within noise. The other models ran with thinking off at their default sampling (they don't
accept temperature 0), with prompts written for DeepSeek.

The same BIRD 256 job with three placement profiles:

| Placement profile | BIRD 256 |
|---|---:|
| Built into the model file | 221 s |
| Recorded from 64 public BIRD *training* questions (different databases) | **146 s** |

The measured numbers used a community IQ4_NL conversion of the Engram tables. The engine is switching to the
release's official FP8 Engram tables, which measured the same speed (170 s vs 172 s) and a slightly lower NLL (0.688
vs 0.692 with IQ4_NL).

## Requirements

- A HIP build of the [radiance-ramcache](https://github.com/arselzer/radiance-ramcache) fork (its `deepseek41` plugin
  and `libr4d` from the same build), and the SiLU table its card check captures (`ds41_mxfp4_device_check ...
  swiglu silu_table_gfx1201.bin`; see `tools/ds41-host-experts/README.md` in the fork).
- The DeepSeek-V4.1-Flash container (`.rad`, 416 GB) on NVMe.
- Tested only on four 16 GB RX 9070 XT cards (gfx1201, TP4), a 24-core EPYC and 256 GB of RAM. Big batches use nearly
  all of that RAM: run jobs alone on the machine.
- Python 3 (standard library only, except `docs/charts/decode_steps.py`, which needs numpy and scipy).

## Quick start

The model (`model.rad`) and the release's FP8 Engram shards (`engram-fp8/`) are on Hugging Face:
[aselzer/DeepSeek-V4.1-Flash-1017-radiance](https://huggingface.co/aselzer/DeepSeek-V4.1-Flash-1017-radiance).

```sh
export RADIANCE_HOME=/path/to/radiance-ramcache/build/radiance_home
./serve-batch.sh --model model.rad --silu-table silu_table_gfx1201.bin --streams 512 \
  --profile profiles/bird-train.profile.json \
  --engram-file engram-fp8/model-00047-of-00048.safetensors,engram-fp8/model-00048-of-00048.safetensors
python3 batch_client.py questions.jsonl answers.jsonl --concurrency 512
```

`--engram-file` reads the release's official FP8 Engram tables (203 GB extra download). Leave it out to use the IQ4_NL
tables built into `model.rad` and save the space; both performed about the same in our tests.

`questions.jsonl` has one `{"prompt": "...", "system": "...", "id": "..."}` a line; put text that prompts share (a
schema, instructions) in `system`, so the prefix cache prefills it once. `batch_client.py --help` lists the rest.

## Calibration

A profile recorded on prompts like yours places experts better. One command starts a calibration server, sends
your prompts, stops it and writes a portable profile:

```sh
./calibrate.sh --model ds41.rad --silu-table silu_table_gfx1201.bin my-prompts.jsonl my.profile.json \
  [--base profiles/bird-train.profile.json]
./serve-batch.sh --model ds41.rad --silu-table silu_table_gfx1201.bin --profile my.profile.json
```

`--base` blends a shipped profile in. Portable profiles carry no path; the server accepts them on any copy of the
same container (matched by its size and the sha256 of its metadata header). `serve-batch.sh --auto-profile DIR`
instead keeps a profile per container in DIR and refreshes it from every run.

## Reproducible vs default mode

By default, prefill and attention run on the matrix cores, summed in a different order, so a reply can change with
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
./serve_batch_test.sh PATH/TO/ab_window.sh   # needs the fork's harness (tools/ds41-host-experts/); skips without it
```

## Known limits

- Tested on one machine: this model, four RX 9070 XT, 256 GB of RAM. radiance serves many other models and features;
  none were tested with this kit.
- Finite batches submitted together, not continuous serving. Keeping the server busy with a longer queue needs more
  work on scheduling and KV allocation.
- About 0.5-0.8 tokens/s per request at large batches: for jobs nobody waits on, not for chat.

## License

Apache-2.0, as radiance. See `LICENSE`.
