/* ds41_fp8_engram_arch_test.cpp -- deepseek41's Engram table in the release's own precision, on the
 * recording builder (arch_fakes.h) with DeepSeek-V4.1-Flash's geometry at TP4, no card or model file.
 *
 * The release ships each Engram table as `fp8_e4m3*e8m0[1x32]`: e4m3 codes and an e8m0 scale per 32
 * of a row's 256 elements (core/format/checkpoint.cpp imports it byte for byte; the two layers'
 * tables join along rows). A container that keeps it so must be served as it is:
 *   - the plugin takes the table as affine planes read in place (no relayout or copy of a table this
 *     size): codes [rows, 256] e4m3 and scale [rows, 8] e8m0, both on the table's own access class;
 *   - each Engram layer (1 and 14) gathers through embed_lookup_planes with group 32 over the whole
 *     joined table, without a codebook operand, into its host rows;
 *   - libref's embed_lookup_planes reads such codes (ref_test covers the values). */
#define RAD_ARCH_NO_EXPORTS 1

#include "arch_fakes.h"

#include "../arch/deepseek41/deepseek41.cpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int kWorld = 4;
constexpr int64_t kRows = 1000 + 1000;   /* ple.head_vocab_sizes below, joined */

MetaKV flash_meta() {
    RadModelMeta base{};
    base.arch_id = "deepseek41";
    base.name = "ds41-flash-geometry";
    base.n_layers = 40;
    base.n_embd = 5120;
    base.n_head = 64;
    base.n_head_kv = 1;
    base.head_dim = 512;
    base.n_vocab = 129280;
    base.n_expert = 384;
    base.n_expert_used = 6;
    base.n_expert_shared = 1;
    base.rms_eps = 1e-6f;
    base.rope_theta = 10000.f;
    base.rope_scale = 1.f;
    MetaKV k(base);
    std::string ratios;
    for (int l = 0; l < 40; ++l) ratios += (l ? "," : "") + std::to_string(l < 2 ? 0 : l < 20 ? 2 : 1);
    k.set("deepseek41.block_count", "40")
        .set("deepseek41.rope.dimension_count", "64")
        .set("deepseek41.attention.q_lora_rank", "1280")
        .set("deepseek41.attention.output_group_count", "8")
        .set("deepseek41.attention.output_lora_rank", "1024")
        .set("deepseek41.attention.sliding_window", "128")
        .set("deepseek41.expert_feed_forward_length", "2304")
        .set("deepseek41.expert_weights_scale", "1.5")
        .set("deepseek41.expert_weights_norm", "1")
        .set("deepseek41.swiglu_clamp_exp", "10")
        .set("deepseek41.attention.indexer.head_count", "32")
        .set("deepseek41.attention.indexer.key_length", "128")
        .set("deepseek41.attention.indexer.top_k", "512")
        .set("deepseek41.attention.indexer.candidate_source_layer", "20")
        .set("deepseek41.attention.indexer.candidate_top_k_blocks", "2048")
        .set("deepseek41.attention.indexer.candidate_block_size", "8")
        .set("deepseek41.hyper_connection.count", "4")
        .set("deepseek41.hyper_connection.sinkhorn_iterations", "20")
        .set("deepseek41.hyper_connection.epsilon", "1e-6")
        .set("deepseek41.attention.compress_rope_freq_base", "160000")
        .set("deepseek41.attention.compress_ratios", ratios.c_str())
        .set("deepseek41.attention.kv_source_layers", "2,8,14,20")
        .set("deepseek41.attention.indexer.source_layers", "2,8,14,20,24,28,32,36")
        .set("deepseek41.ple.layers", "1,14")
        .set("deepseek41.ple.ngram_size", "4")
        .set("deepseek41.ple.heads_per_ngram", "8")
        .set("deepseek41.embedding_length_per_layer_input", "256")
        .set("deepseek41.engram.pad_token_id", "2")
        .set("deepseek41.ple.head_vocab_sizes", "1000,1000");
    return k;
}

int64_t block_of(const std::string& n) {
    for (const char* p : { "kv_compressed_r", "kv_index_r" })
        if (n.rfind(p, 0) == 0) return std::atoll(n.c_str() + std::strlen(p));
    return 0;
}

const RecParam* param(const RecOp& o, const char* key) {
    for (const RecParam& p : o.p)
        if (p.key == key) return &p;
    return nullptr;
}

}  // namespace

TEST(ds41_release_fp8_engram_table_is_gathered_as_stored_planes) {
    for (int rank = 0; rank < kWorld; ++rank) {
        RadBuilder b;
        b.kv_block_of = block_of;
        b.encs.push_back({ "per_layer_token_embd", rad_enc_affine(RAD_F8E4M3, RAD_E8M0, 1, 32) });
        RadBuildCtx ctx = build_ctx(rank, kWorld);
        ctx.max_tok = 64;
        ctx.max_ctx = 4096;
        ctx.max_seqs = 1;
        MetaKV meta = flash_meta();
        int rc = RAD_E_INVAL;
        const std::string err = stderr_of([&] { rc = deepseek41::declare(&b, meta.meta(), &ctx); });
        if (rc != RAD_OK) fprintf(stderr, "%s", err.c_str());
        REQUIRE_EQ(rc, RAD_OK);
        CHECK(err.find("the Engram table is") == std::string::npos);

        const deepseek41::Model& m = *deepseek41::models[rank];
        CHECK_EQ(m.ple_group, (int64_t)32);
        CHECK(m.ple_table == 0);
        CHECK(m.ple_tab == 0);
        REQUIRE(m.ple_codes != 0);
        REQUIRE(m.ple_scale != 0);
        const RadWeightDecl& cd = b.weights[(size_t)m.ple_codes - 1].d;
        const RadWeightDecl& sd = b.weights[(size_t)m.ple_scale - 1].d;
        CHECK_EQ(cd.dtype, (uint32_t)RAD_F8E4M3);
        CHECK_EQ(cd.shape[0], kRows);
        CHECK_EQ(cd.shape[1], (int64_t)256);
        CHECK_EQ(sd.dtype, (uint32_t)RAD_E8M0);
        CHECK_EQ(sd.shape[0], kRows);
        CHECK_EQ(sd.shape[1], (int64_t)8);
        CHECK_EQ(cd.access, (int)RAD_ACCESS_VOCAB);
        CHECK_EQ(sd.access, (int)RAD_ACCESS_VOCAB);

        int gathers = 0;
        for (const deepseek41::Layer& L : m.layers) {
            if (!L.e_emb) continue;
            ++gathers;
            const RecOp& o = *b.op_of(L.e_emb);
            CHECK_EQ(o.op, std::string("embed_lookup_planes"));
            const RecParam* g = param(o, "group");
            const RecParam* v = param(o, "n_vocab");
            const RecParam* e = param(o, "n_embd");
            REQUIRE(g && v && e);
            CHECK_EQ(g->ival, 32ll);
            CHECK_EQ(v->ival, (long long)kRows);
            CHECK_EQ(e->ival, 256ll);
            REQUIRE_EQ(o.w.size(), (size_t)2);   /* codes and scale; no codebook */
            CHECK_EQ(o.w[0], m.ple_codes);
            CHECK_EQ(o.w[1], m.ple_scale);
        }
        CHECK_EQ(gathers, 2);
    }
}

RAD_TEST_MAIN()
