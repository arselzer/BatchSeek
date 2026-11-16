/* ds41_engram_file_arch_test.cpp -- deepseek41 on the native container's metadata (the release's
 * config keys, deepseek41.engram_source=ista_iq4_nl_v1: Engram tables per layer) when its two
 * tables are answered as the release stores them (RADIANCE_EXPERIMENTAL_DS41_ENGRAM_FILE,
 * core/format/ds41_engram_file.h), on the recording builder (arch_fakes.h) at TP4, no card or model:
 *   - each Engram layer declares its own table as fp8_e4m3 codes [rows, 256] and e8m0 scale
 *     [rows, 8] views read in place (the vocab access class), no codebook, at the layer's rows;
 *   - each gathers through embed_lookup_planes with group 32 and those two weights only;
 *   - the early read-ahead (RADIANCE_EXPERIMENTAL_DS41_ENGRAM_EARLY=1) asks for layer 14's rows
 *     through the same gather, codes and scale, no codebook operand;
 *   - the same metadata with the container's own IQ4_NL tables declares the codebook as before. */
#define RAD_ARCH_NO_EXPORTS 1

#include "arch_fakes.h"

#include "../arch/deepseek41/deepseek41.cpp"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int kWorld = 4;
const int64_t kRows[2] = { 384006168, 384016682 };

/* The native container's metadata as rad-info --meta prints it (the keys deepseek41 reads). */
MetaKV native_meta() {
    RadModelMeta base{};
    base.arch_id = "deepseek41";
    base.name = "ds41-native-geometry";
    base.n_layers = 40;
    base.n_embd = 5120;
    base.n_head = 64;
    base.n_head_kv = 1;
    base.head_dim = 512;
    base.n_ff = 2304;
    base.n_vocab = 129280;
    base.n_expert = 384;
    base.n_expert_used = 6;
    base.n_expert_shared = 1;
    base.rms_eps = 1e-6f;
    base.rope_theta = 10000.f;
    base.rope_scale = 1.f;
    MetaKV k(base);
    k.set("num_hidden_layers", "40")
        .set("candidate_block_size", "8")
        .set("candidate_source_layer_id", "20")
        .set("candidate_topk_blocks", "2048")
        .set("compress_ratios", "0 0 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0")
        .set("compress_rope_theta", "160000")
        .set("engram_head_dim", "256")
        .set("engram_layer_ids", "1 14")
        .set("engram_max_ngram_size", "4")
        .set("engram_n_heads", "8")
        .set("engram_num_embeddings", "384006168 384016682")
        .set("engram_pad_token_id", "2")
        .set("engram_vocab_size", "16000000")
        .set("hc_eps", "9.9999999999999995e-07")
        .set("hc_mult", "4")
        .set("hc_sinkhorn_iters", "20")
        .set("index_head_dim", "128")
        .set("index_n_heads", "32")
        .set("index_source_layer_ids", "2 8 14 20 24 28 32 36")
        .set("index_topk", "512")
        .set("kv_source_layer_ids", "2 8 14 20")
        .set("moe_intermediate_size", "2304")
        .set("norm_topk_prob", "1")
        .set("o_groups", "8")
        .set("o_lora_rank", "1024")
        .set("q_lora_rank", "1280")
        .set("qk_rope_head_dim", "64")
        .set("routed_scaling_factor", "1.5")
        .set("sliding_window", "128")
        .set("swiglu_limit", "10")
        .set("deepseek41.engram_source", "ista_iq4_nl_v1");
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

RadEncoding iq4nl() {
    RadEncoding e = rad_enc_affine(RAD_U4, RAD_F16, 1, 32);
    rad_enc_add_table(&e, "table", RAD_I8, 1, 16);
    return e;
}

struct EnvEarly {
    EnvEarly() { setenv("RADIANCE_EXPERIMENTAL_DS41_ENGRAM_EARLY", "1", 1); }
    ~EnvEarly() { unsetenv("RADIANCE_EXPERIMENTAL_DS41_ENGRAM_EARLY"); }
};

}  // namespace

TEST(ds41_native_container_takes_the_release_fp8_tables_per_layer) {
    EnvEarly early;
    for (int rank = 0; rank < kWorld; ++rank) {
        RadBuilder b;
        b.kv_block_of = block_of;
        b.encs.push_back({ "per_layer_token_embd", rad_enc_affine(RAD_F8E4M3, RAD_E8M0, 1, 32) });
        RadBuildCtx ctx = build_ctx(rank, kWorld);
        ctx.max_tok = 64;
        ctx.max_ctx = 4096;
        ctx.max_seqs = 4;
        MetaKV meta = native_meta();
        int rc = RAD_E_INVAL;
        const std::string err = stderr_of([&] { rc = deepseek41::declare(&b, meta.meta(), &ctx); });
        if (rc != RAD_OK) fprintf(stderr, "%s", err.c_str());
        REQUIRE_EQ(rc, RAD_OK);
        CHECK(err.find("the Engram table is") == std::string::npos);

        const deepseek41::Model& m = *deepseek41::models[rank];
        REQUIRE(m.local_engram);
        CHECK(m.engram_early);
        for (int t = 0; t < 2; ++t) {
            const auto& et = m.ista_tables[t];
            CHECK_EQ(et.group, (int64_t)32);
            CHECK(et.plain == 0);
            CHECK(et.tab == 0);
            REQUIRE(et.codes != 0);
            REQUIRE(et.scale != 0);
            const RadWeightDecl& cd = b.weights[(size_t)et.codes - 1].d;
            const RadWeightDecl& sd = b.weights[(size_t)et.scale - 1].d;
            CHECK_EQ(cd.dtype, (uint32_t)RAD_F8E4M3);
            CHECK_EQ(cd.shape[0], kRows[t]);
            CHECK_EQ(cd.shape[1], (int64_t)256);
            CHECK_EQ(sd.dtype, (uint32_t)RAD_E8M0);
            CHECK_EQ(sd.shape[0], kRows[t]);
            CHECK_EQ(sd.shape[1], (int64_t)8);
            CHECK_EQ(cd.access, (int)RAD_ACCESS_VOCAB);
            CHECK_EQ(sd.access, (int)RAD_ACCESS_VOCAB);
            const std::string l = "per_layer_token_embd.l" + std::to_string(t ? 14 : 1);
            CHECK_EQ(b.weights[(size_t)et.codes - 1].name, l + ".weight");
            CHECK_EQ(b.weights[(size_t)et.scale - 1].name, l + ".scale");
        }

        int gathers = 0;
        for (const deepseek41::Layer& L : m.layers) {
            if (!L.e_emb) continue;
            const int t = gathers++;
            REQUIRE(t < 2);
            const RecOp& o = *b.op_of(L.e_emb);
            CHECK_EQ(o.op, std::string("embed_lookup_planes"));
            const RecParam* g = param(o, "group");
            const RecParam* v = param(o, "n_vocab");
            REQUIRE(g && v);
            CHECK_EQ(g->ival, 32ll);
            CHECK_EQ(v->ival, (long long)kRows[t]);
            REQUIRE_EQ(o.w.size(), (size_t)2);   /* codes and scale; no codebook */
            CHECK_EQ(o.w[0], m.ista_tables[t].codes);
            CHECK_EQ(o.w[1], m.ista_tables[t].scale);
        }
        CHECK_EQ(gathers, 2);

        /* A decode step: layer 14's rows are asked for early, over the FP8 planes. */
        auto k = kit(b, { 1, 1, 1, 1 }, {});
        REQUIRE(k);
        auto r = run_passes(b, deepseek41::step, { k.get() }, rank, kWorld);
        REQUIRE(steps_ok(r));
        REQUIRE_EQ(r[0]->read_aheads.size(), (size_t)1);
        const RecIssue& ra = r[0]->read_aheads[0].second;
        CHECK_EQ(ra.op, m.layers[14].e_emb);
        REQUIRE_EQ(ra.opd.size(), (size_t)5);
        CHECK_EQ((int)ra.opd[1].kind, (int)RAD_OPK_WEIGHT);
        CHECK_EQ(ra.opd[1].handle, (uint32_t)m.ista_tables[1].codes);
        CHECK_EQ((int)ra.opd[2].kind, (int)RAD_OPK_WEIGHT);
        CHECK_EQ(ra.opd[2].handle, (uint32_t)m.ista_tables[1].scale);
        CHECK_EQ((int)ra.opd[3].kind, (int)RAD_OPK_NONE);
    }
}

TEST(ds41_native_container_keeps_its_iq4nl_tables_without_the_sidecar) {
    RadBuilder b;
    b.kv_block_of = block_of;
    b.encs.push_back({ "per_layer_token_embd", iq4nl() });
    RadBuildCtx ctx = build_ctx(0, kWorld);
    ctx.max_tok = 64;
    ctx.max_ctx = 4096;
    ctx.max_seqs = 4;
    MetaKV meta = native_meta();
    int rc = RAD_E_INVAL;
    const std::string err = stderr_of([&] { rc = deepseek41::declare(&b, meta.meta(), &ctx); });
    if (rc != RAD_OK) fprintf(stderr, "%s", err.c_str());
    REQUIRE_EQ(rc, RAD_OK);
    const deepseek41::Model& m = *deepseek41::models[0];
    REQUIRE(m.local_engram);
    for (int t = 0; t < 2; ++t) {
        const auto& et = m.ista_tables[t];
        REQUIRE(et.codes != 0 && et.scale != 0 && et.tab != 0);
        CHECK_EQ(b.weights[(size_t)et.codes - 1].d.dtype, (uint32_t)RAD_U4);
        CHECK_EQ(b.weights[(size_t)et.scale - 1].d.dtype, (uint32_t)RAD_F16);
    }
    int gathers = 0;
    for (const deepseek41::Layer& L : m.layers) {
        if (!L.e_emb) continue;
        ++gathers;
        CHECK_EQ(b.op_of(L.e_emb)->w.size(), (size_t)3);
    }
    CHECK_EQ(gathers, 2);
}

RAD_TEST_MAIN()
