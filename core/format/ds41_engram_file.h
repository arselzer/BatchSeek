/* ds41_engram_file.h -- RADIANCE_EXPERIMENTAL_DS41_ENGRAM_FILE: DeepSeek-V4.1's Engram tables read
 * from a sidecar instead of the container.
 *
 * The native DeepSeek-V4.1-Flash container (deepseek41.engram_source=ista_iq4_nl_v1) holds each
 * Engram layer's table as IQ4_NL (per_layer_token_embd.l1.weight, .l14.weight: 4-bit codes, an f16
 * scale per 32 and a 16-entry codebook), 103 GiB of its 388 GiB. The release ships the same tables
 * as fp8_e4m3 codes with an e8m0 scale per 1x32 (layers.{1,14}.engram.embed.{weight,scale} in its
 * safetensors shards 47 and 48), 189 GiB. Writing a second 474 GiB container only to swap two
 * tensors costs a disk this box does not have, so the release's shards themselves are the sidecar:
 *
 *   RADIANCE_EXPERIMENTAL_DS41_ENGRAM_FILE=<a.safetensors>[,<b.safetensors>...]
 *
 * Each container table named here (per_layer_token_embd.l<L>.weight) is then answered from the
 * sidecar: rad_weight_encoding sees `affine:codes=fp8_e4m3[1x1],scale=e8m0[1x32]` (the plugin takes
 * codes and scale planes read in place, no codebook, and libref's embed_lookup_planes reads FP8
 * codes), and the loader publishes the planes' mapping on the mapped tier, where the host gather,
 * the row read-ahead (--row-prefetch) and the early read-ahead (..._ENGRAM_EARLY) read them as they
 * read the container's. Every other weight is the container's. The container's IQ4_NL tables are
 * not read (the sweep drops their cached pages with the rest of the blob).
 *
 * Refused at startup, with the reason, unless all of these hold:
 *   - the container keeps its Engram tables per layer (deepseek41.engram_source=ista_iq4_nl_v1),
 *     each of the spec's layers as per_layer_token_embd.l<L>.weight of [rows, dim];
 *   - the container's Engram hash constants (the token map, and each layer's multipliers, bucket
 *     sizes and offsets) are byte for byte the release's (sha256), since the ids they compute index
 *     the sidecar's rows; the release's offsets are absolute in the joined table, the container's
 *     (and so these) per layer;
 *   - the sidecar holds each layer's layers.<L>.engram.embed.weight as F8_E4M3 [rows, dim] and
 *     .scale as F8_E8M0 [rows, dim / group] exactly once over its files, at the container's rows;
 *   - no other use of the tables' bytes (a RAM or VRAM copy, a DSpark bank) is asked for: the
 *     engine checks that (Engine::read_metadata, the load).
 * The sidecar's other tensors (the shards' Engram projections, already in the container) are left
 * alone. Integrity of the payload is the download's (the release's LFS sha256 over each whole
 * shard); this check is the layout's, which is what makes the gather read the right rows. */
#pragma once
#include "radfile.h"
#include "safetensors.h"
#include "../build/rad_build.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace rad {

/* What the sidecar must match: the model's Engram layers, their rows and width, and the hash
 * constants the container must hold. */
struct Ds41EngramSpec {
    struct Layer { int64_t layer = 0; int64_t rows = 0; };
    std::vector<Layer> layers;
    int64_t dim = 256, group = 32;
    /* Container entry name -> sha256 (hex) of its one plane's bytes. */
    std::vector<std::pair<std::string, std::string>> controls;
};

/* DeepSeek-V4.1-Flash-1017's (revision 1bae0e10): layers 1 and 14, 384,006,168 and 384,016,682
 * rows of 256, group 32; the controls are data/engram_tables.py's from the release's config.json
 * and tokenizer.json, with layer 14's offsets made relative to its own table. */
const Ds41EngramSpec& ds41_engram_release_spec();

/* One container weight answered from the sidecar. */
struct Ds41EngramTable {
    std::string    name;                 /* the container's weight, per_layer_token_embd.l<L>.weight */
    int64_t        layer = 0;
    RadEncoding    enc{};                /* affine: codes fp8_e4m3 [1x1], scale e8m0 [1xgroup] */
    uint32_t       rank = 2;
    int64_t        shape[RAD_MAX_RANK] = {0};
    const uint8_t* plane[2] = { nullptr, nullptr };   /* codes, scale: into the sidecar's mapping */
    int64_t        plane_bytes[2] = { 0, 0 };
    std::string    file[2];              /* the sidecar files holding them */
    std::string    container_form;       /* what the container holds, for the log */
};

class Ds41EngramFile {
public:
    Ds41EngramFile() = default;
    Ds41EngramFile(const Ds41EngramFile&) = delete;
    Ds41EngramFile& operator=(const Ds41EngramFile&) = delete;

    /* `paths`: the sidecar's safetensors files, comma-separated. RAD_OK, or a negative status with
     * `why` set; nothing is kept open on a refusal. */
    int open(const std::string& paths, const RadFile& container, const Ds41EngramSpec& spec,
             std::string* why);

    /* The table answered for container weight `name`, or null. */
    const Ds41EngramTable* find(const std::string& name) const;
    const std::vector<Ds41EngramTable>& tables() const { return tables_; }
    int64_t bytes() const;

    /* `inner` with every table here answered from the sidecar. Borrows this object. */
    WeightSourceFn wrap(WeightSourceFn inner) const;

    /* One line for the log: the tables, their form here and in the container, the files. */
    std::string describe() const;

private:
    std::vector<std::unique_ptr<SafeTensorsFile>> files_;
    std::vector<Ds41EngramTable> tables_;
};

}  /* namespace rad */
