/* ds41_engram_early_arch_test.cpp -- deepseek41's early Engram read-ahead
 * (RADIANCE_EXPERIMENTAL_DS41_ENGRAM_EARLY=1, rad_row_read_ahead) on the recording builder
 * (arch_fakes.h) with DeepSeek-V4.1-Flash's own geometry at TP4 (Engram layers 1 and 14), no card or
 * model file:
 *   - off, nothing is asked early and no line is printed; on, every rank takes it and rank 0 says so;
 *   - on every kind of pass -- a prefill chunk, a pure decode step, a step mixing decode rows with a
 *     prompt's, a cut pass (consumer skip), a per-row cut chunk, a split (two-batch overlap) decode
 *     step -- the switch changes no issue: the same ops in the same order with the same operands
 *     (so the gathers read the same ids from the same tables into the same rows, and every cast is
 *     the same), and adds exactly one read-ahead, right behind layer 1's gather, of layer 14's gather
 *     over this rank's 1/4 of the step's rows: the gather's own handle, ids buffer and tables, and the
 *     four ranks' slices tile the rows exactly;
 *   - a read-ahead the core declines (0) or refuses (< 0) is counted and fails nothing.
 * The core side (runtime_test: rad_row_read_ahead) proves the hints change no gathered byte. */
#define RAD_ARCH_NO_EXPORTS 1

#include "arch_fakes.h"

#include "../arch/deepseek41/deepseek41.cpp"
#include "../core/sched/consumer_skip.h"

#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

constexpr int kWorld = 4;

const char* const kVars[] = { "RADIANCE_EXPERIMENTAL_DS41_ENGRAM_EARLY",
                              "RADIANCE_EXPERIMENTAL_DS41_CONSUMER_SKIP",
                              "RADIANCE_EXPERIMENTAL_SCHED_TBO",
                              "RADIANCE_EXPERIMENTAL_DS41_PREFETCH_ATTN",
                              "RADIANCE_EXPERIMENTAL_DS41_ENGRAM_SPLIT",
                              "RADIANCE_EXPERIMENTAL_DS41_DENSE_FP8",
                              "RADIANCE_EXPERIMENTAL_DS41_KV_REPLICATED" };
struct Env {
    explicit Env(std::initializer_list<std::pair<const char*, const char*>> kv) {
        for (const char* n : kVars) unsetenv(n);
        for (const auto& p : kv) setenv(p.first, p.second, 1);
    }
    ~Env() {
        for (const char* n : kVars) unsetenv(n);
    }
};

/* The release's geometry (ds41_consumer_skip_arch_test's). */
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

struct Decl {
    RadBuilder b;
    int rc = RAD_E_INVAL;
    std::string err;
    deepseek41::Model* m = nullptr;
};
std::unique_ptr<Decl> declare(int rank, int64_t max_tok) {
    auto d = std::make_unique<Decl>();
    d->b.kv_block_of = block_of;
    RadBuildCtx ctx = build_ctx(rank, kWorld);
    ctx.max_tok = max_tok;
    ctx.max_ctx = 8192;
    MetaKV meta = flash_meta();
    d->err = stderr_of([&] { d->rc = deepseek41::declare(&d->b, meta.meta(), &ctx); });
    d->m = deepseek41::models[rank].get();
    return d;
}

bool same_operand(const RadOperand& a, const RadOperand& b) {
    return a.kind == b.kind && a.dtype == b.dtype && a.handle == b.handle && a.raw == b.raw &&
           a.offset == b.offset && a.rows == b.rows && a.cols == b.cols;
}
bool same_issue(const RecIssue& a, const RecIssue& b) {
    bool same = a.op == b.op && a.n == b.n && a.lane == b.lane && a.opd.size() == b.opd.size();
    for (size_t j = 0; same && j < a.opd.size(); ++j) same = same_operand(a.opd[j], b.opd[j]);
    return same;
}

/* A pass the test runs off and on: its kit, the environment beyond the switch, and its height. */
struct Pass {
    const char* name;
    bool skip = false, tbo = false;
    int64_t max_tok = 16;
};

/* One kit for both runs, so raw operands (pointers into its batch) compare. */
std::unique_ptr<Kit> make(const RadBuilder& b, const std::string& name, std::vector<int32_t>* row0) {
    if (name == "prefill") {
        auto k = kit(b, {}, { 1024 }, 0);
        return k;
    }
    if (name == "decode") return kit(b, { 1, 1, 1 }, {});
    if (name == "mixed") return kit(b, { 1, 1 }, { 5, 3 }, 64);   /* mixed lengths */
    if (name == "cut" || name == "rowcut") {
        auto k = kit(b, {}, { 1024 }, 0);
        k->out.clear();
        k->b.n_out = 0;
        k->b.out_ids = nullptr;
        if (name == "cut") {
            k->b.consumer_full = 25;
            k->b.consumer_end = 3086;
            return k;
        }
        row0->assign(40, 0);
        const rad::ConsumerShape c16{ 40, 20, 128, 16, 16 };
        const std::vector<int64_t> f = rad::consumer_thresholds(c16, 3086);
        rad::consumer_row_starts(c16, f.data(), 0, 1024, row0->data());
        int32_t F = 20;
        while (F < 40 && (*row0)[(size_t)F] < 1024) ++F;
        k->b.consumer_full = F < 40 ? F : 0;
        k->b.consumer_end = 3086;
        k->b.consumer_row0 = row0->data();
        return k;
    }
    if (name == "split") return kit(b, { 1, 1, 1, 1, 1, 1 }, {});
    return nullptr;
}

/* The pass off, then on, at `rank`: the issues match one for one, and the on pass asked once (or
 * not at all for an empty slice) for layer 14's rows behind layer 1's gather. `slice`: the rows
 * [r0, r1) it asked for, or {0, 0}. */
void check_pass(const Pass& p, int rank, int* failures, std::pair<int64_t, int64_t>* slice) {
    *slice = { 0, 0 };
    std::vector<std::pair<const char*, const char*>> base;
    if (p.skip) base.push_back({ "RADIANCE_EXPERIMENTAL_DS41_CONSUMER_SKIP", "1" });
    if (p.tbo) base.push_back({ "RADIANCE_EXPERIMENTAL_SCHED_TBO", "1" });
    std::vector<RecIssue> off;
    std::unique_ptr<Decl> d0;
    std::vector<int32_t> row0;
    std::unique_ptr<Kit> k;
    {
        for (const char* n : kVars) unsetenv(n);
        for (const auto& kv : base) setenv(kv.first, kv.second, 1);
        d0 = declare(rank, p.max_tok);
        REQUIRE_EQ(d0->rc, (int)RAD_OK);
        CHECK(!d0->m->engram_early);
        k = make(d0->b, p.name, &row0);
        REQUIRE(k);
        auto r = run_passes(d0->b, deepseek41::step, { k.get() }, rank, kWorld);
        REQUIRE(steps_ok(r));
        CHECK(r[0]->read_aheads.empty());
        off = r[0]->issues;
        if (p.tbo) CHECK_EQ(d0->m->tbo_steps, (uint64_t)1);
        if (p.skip) CHECK(d0->m->cut_on);
    }
    setenv("RADIANCE_EXPERIMENTAL_DS41_ENGRAM_EARLY", "1", 1);
    auto d = declare(rank, p.max_tok);
    REQUIRE_EQ(d->rc, (int)RAD_OK);
    REQUIRE(d->m->engram_early);
    auto r = run_passes(d->b, deepseek41::step, { k.get() }, rank, kWorld);
    for (const char* n : kVars) unsetenv(n);
    REQUIRE(steps_ok(r));
    const RadCtx& c = *r[0];
    /* The issues: the off pass's, one for one (both declares hand out the same handles). */
    int differ = 0;
    if (c.issues.size() != off.size()) ++differ;
    for (size_t i = 0; i < c.issues.size() && i < off.size(); ++i)
        if (!same_issue(c.issues[i], off[i]) && differ++ < 4)
            fprintf(stderr, "    %s rank %d: issue %zu (%s) differs\n", p.name, rank, i,
                    d->b.op_of(c.issues[i].op) ? d->b.op_of(c.issues[i].op)->op.c_str() : "?");
    CHECK_EQ(differ, 0);
    *failures += differ != 0;
    const deepseek41::Layer& L1 = d->m->layers[1];
    const deepseek41::Layer& L14 = d->m->layers[14];
    const int64_t T = k->b.n_tok, H = d->m->c.ple_heads;
    const int64_t r0 = T * rank / kWorld, r1 = T * (rank + 1) / kWorld;
    if (r1 <= r0) {
        CHECK(c.read_aheads.empty());
        CHECK_EQ(d->m->eas.none, (uint64_t)1);
        return;
    }
    REQUIRE_EQ(c.read_aheads.size(), (size_t)1);
    CHECK_EQ(d->m->eas.started, (uint64_t)1);
    const size_t at = c.read_aheads[0].first;
    const RecIssue& ra = c.read_aheads[0].second;
    /* Right behind layer 1's (first) gather. */
    size_t g1 = c.issues.size();
    for (size_t i = 0; i < c.issues.size(); ++i)
        if (c.issues[i].op == L1.e_emb) { g1 = i; break; }
    CHECK_EQ(at, g1 + 1);
    /* Layer 14's gather: its handle, its ids at this rank's rows, its tables, no output. */
    const RecIssue* g14 = nullptr;
    for (const RecIssue& is : c.issues)
        if (is.op == L14.e_emb) { g14 = &is; break; }
    REQUIRE(g14 != nullptr);
    CHECK_EQ(ra.op, L14.e_emb);
    CHECK_EQ(ra.n, (r1 - r0) * H);
    REQUIRE_EQ(ra.opd.size(), g14->opd.size());
    CHECK(ra.opd[0].kind == RAD_OPK_BUF && ra.opd[0].handle == L14.ids);
    CHECK_EQ(ra.opd[0].offset, r0 * H);
    CHECK_EQ(ra.opd[0].rows, r1 - r0);
    CHECK_EQ(g14->opd[0].handle, L14.ids);
    for (size_t j = 1; j + 1 < ra.opd.size(); ++j) CHECK(same_operand(ra.opd[j], g14->opd[j]));
    CHECK(ra.opd.back().kind == RAD_OPK_NONE);
    *slice = { r0, r1 };
}

}  // namespace

TEST(engram_early_off_takes_nothing_and_on_says_so_once) {
    {
        Env env({});
        auto d = declare(0, 16);
        REQUIRE_EQ(d->rc, (int)RAD_OK);
        CHECK(!d->m->engram_early);
        CHECK(d->err.find("Engram rows early") == std::string::npos);
    }
    Env env({ { "RADIANCE_EXPERIMENTAL_DS41_ENGRAM_EARLY", "1" } });
    for (int rank = 0; rank < kWorld; ++rank) {
        auto d = declare(rank, 16);
        REQUIRE_EQ(d->rc, (int)RAD_OK);
        CHECK(d->m->engram_early);
        const bool line = d->err.find("Engram rows early (RADIANCE_EXPERIMENTAL_DS41_ENGRAM_EARLY=1)") !=
                          std::string::npos;
        CHECK_EQ(line, rank == 0);
    }
}

TEST(engram_early_changes_no_issue_and_asks_once_for_the_next_layer_on_every_pass) {
    const Pass passes[] = {
        { "prefill", false, false, 1024 },
        { "decode", false, false, 16 },
        { "mixed", false, false, 16 },
        { "cut", true, false, 1024 },
        { "rowcut", true, false, 1024 },
        { "split", false, true, 16 },
    };
    for (const Pass& p : passes) {
        int failures = 0;
        std::vector<std::pair<int64_t, int64_t>> slices;
        for (int rank = 0; rank < kWorld; ++rank) {
            std::pair<int64_t, int64_t> s{ 0, 0 };
            check_pass(p, rank, &failures, &s);
            slices.push_back(s);
        }
        /* The ranks' slices tile the step's rows. */
        int64_t next = 0;
        for (const auto& s : slices) {
            if (s.second == 0) continue;
            CHECK_EQ(s.first, next);
            next = s.second;
        }
        const int64_t T = p.max_tok == 1024 ? 1024 : std::string(p.name) == "decode" ? 3
                          : std::string(p.name) == "mixed" ? 10 : 6;
        CHECK_EQ(next, T);
        if (failures) fprintf(stderr, "    pass %s: %d rank(s) differ\n", p.name, failures);
    }
}

TEST(engram_early_counts_a_declined_or_refused_read_ahead_and_fails_nothing) {
    Env env({ { "RADIANCE_EXPERIMENTAL_DS41_ENGRAM_EARLY", "1" } });
    auto d = declare(0, 16);
    REQUIRE_EQ(d->rc, (int)RAD_OK);
    auto k = kit(d->b, {}, { 12 }, 0);
    for (int rc : { 0, (int)RAD_E_INVAL }) {
        auto c = std::make_unique<RadCtx>();
        c->b = &d->b;
        c->batch = &k->b;
        c->rank = 0;
        c->world = kWorld;
        c->read_ahead_rc = rc;
        deepseek41::step(c.get(), &k->b);
        CHECK(c->step_fail.empty());
        CHECK_EQ(c->read_aheads.size(), (size_t)1);
    }
    CHECK_EQ(d->m->eas.started, (uint64_t)0);
    CHECK_EQ(d->m->eas.none, (uint64_t)1);
    CHECK_EQ(d->m->eas.refused, (uint64_t)1);
}

RAD_TEST_MAIN()
