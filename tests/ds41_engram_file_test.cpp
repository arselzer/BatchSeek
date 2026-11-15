/* ds41_engram_file_test.cpp -- RADIANCE_EXPERIMENTAL_DS41_ENGRAM_FILE's sidecar
 * (core/format/ds41_engram_file.h): a small container in the native DeepSeek-V4.1 layout (Engram
 * tables per layer, IQ4_NL-like affine planes, the hash constants as their own entries) and the
 * release's form of the same tables in safetensors files, as shards 47 and 48 hold them.
 *
 *   - accepted, each table is answered from the sidecar: fp8_e4m3 codes and an e8m0 scale per 32
 *     as two planes pointing at the sidecar's own bytes, at the container's shape; the source
 *     function answers them so and every other weight from the container; a table's two tensors
 *     may sit in different files, and the files' other tensors are ignored;
 *   - refused, with the reason: a container whose Engram tables are not per layer, a hash
 *     constant that is not the expected one (one byte off), a constant or table the container
 *     lacks, a table of other rows, a sidecar tensor of another dtype or shape, a tensor held
 *     twice, one held by no file, a file named twice or unreadable, an empty list;
 *   - the release spec is the native container's geometry (ista_dialect.h's rows) and names its
 *     seven constants. */
#include "rad_test.h"
#include "format/radfile.h"
#include "format/ds41_engram_file.h"
#include "build/startup.h"
#include "place/expert_profile_sha256.h"
#include "../arch/deepseek41/ista_dialect.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

using namespace rad;

namespace {

constexpr int64_t kDim = 256, kGroup = 32;
const int64_t kRows[2] = { 5, 7 };
const int64_t kLayer[2] = { 1, 14 };

std::vector<uint8_t> ramp(int64_t n, uint8_t seed) {
    std::vector<uint8_t> v((size_t)n);
    for (int64_t i = 0; i < n; ++i) v[(size_t)i] = (uint8_t)(seed + i * 7 + (i >> 8));
    return v;
}

std::string sha(const std::vector<uint8_t>& b) {
    ds41_integrity::Sha256 h;
    h.update(b.data(), b.size());
    return h.finish();
}

/* The container's controls, by entry name. */
std::map<std::string, std::vector<uint8_t>> controls() {
    std::map<std::string, std::vector<uint8_t>> c;
    c["deepseek41.engram.token_map"] = ramp(4 * 64, 3);
    for (int i = 0; i < 2; ++i) {
        const std::string b = "blk." + std::to_string(kLayer[i]);
        c[b + ".engram_mult"] = ramp(8 * 4, (uint8_t)(10 + i));
        c[b + ".engram_vsz"] = ramp(8 * 24, (uint8_t)(20 + i));
        c[b + ".engram_off"] = ramp(8 * 24, (uint8_t)(30 + i));
    }
    return c;
}

Ds41EngramSpec spec() {
    Ds41EngramSpec s;
    s.layers = { { kLayer[0], kRows[0] }, { kLayer[1], kRows[1] } };
    s.dim = kDim;
    s.group = kGroup;
    for (const auto& [n, b] : controls()) s.controls.push_back({ n, sha(b) });
    return s;
}

struct ContainerOpts {
    const char* source = "ista_iq4_nl_v1";
    std::string flip_control;      /* this control's first byte changed */
    std::string drop;              /* this entry left out */
    int64_t     rows14 = kRows[1];
};

std::string write_container(const char* stem, const ContainerOpts& o = {}) {
    const std::string p = radtest::tmp_path(stem);
    RadWriter w;
    CHECK_OK(w.begin(p.c_str()));
    w.set_arch("deepseek41");
    w.set_model_name("engram-file-test");
    w.set_created_by("ds41_engram_file_test");
    if (o.source) w.meta_s("deepseek41.engram_source", o.source);
    auto put = [&](const std::string& name, const RadEncoding& enc, uint32_t rank, int64_t r,
                   int64_t c, uint8_t seed, const std::vector<uint8_t>* bytes) {
        if (name == o.drop) return;
        RadWriter::Weight x;
        x.name = name;
        x.enc = enc;
        x.rank = rank;
        x.shape[0] = r;
        x.shape[1] = rank > 1 ? c : 0;
        int64_t rows = 0, cols = 0;
        rad_enc_view(x.rank, x.shape, &rows, &cols);
        std::vector<std::vector<uint8_t>> planes;
        std::vector<const void*> ptr;
        for (int k = 0; k < enc.n_planes; ++k) {
            planes.push_back(bytes && k == 0 ? *bytes
                                             : ramp(rad_enc_plane_bytes(&enc.plane[k], rows, cols),
                                                    (uint8_t)(seed + k)));
            REQUIRE_EQ((int64_t)planes.back().size(), rad_enc_plane_bytes(&enc.plane[k], rows, cols));
        }
        for (const auto& b : planes) ptr.push_back(b.data());
        CHECK(w.add_weight(x, ptr.data()) >= 0);
    };
    for (auto [n, b] : controls()) {
        if (n == o.flip_control) b[0] ^= 1;
        if (n == "deepseek41.engram.token_map") put(n, rad_enc_plain(RAD_I32), 1, (int64_t)b.size() / 4, 0, 0, &b);
        else put(n, rad_enc_plain(RAD_I64), 1, (int64_t)b.size() / 8, 0, 0, &b);
    }
    RadEncoding iq4 = rad_enc_affine(RAD_U4, RAD_F16, 1, 32);
    rad_enc_add_table(&iq4, "table", RAD_I8, 1, 16);
    for (int i = 0; i < 2; ++i)
        put("per_layer_token_embd.l" + std::to_string(kLayer[i]) + ".weight", iq4, 2,
            i ? o.rows14 : kRows[i], kDim, (uint8_t)(40 + i), nullptr);
    put("blk.1.engram_kv.weight", rad_enc_plain(RAD_BF16), 2, 4, 64, 50, nullptr);
    CHECK_OK(w.finish());
    return p;
}

/* A safetensors file of `t`: name -> {dtype, shape, bytes}, in order. */
struct St {
    std::string name, dtype;
    std::vector<int64_t> shape;
    std::vector<uint8_t> bytes;
};

std::string write_st(const char* stem, const std::vector<St>& ts) {
    std::string h = "{\"__metadata__\":{\"format\":\"pt\"}";
    uint64_t at = 0;
    for (const St& t : ts) {
        h += ",\"" + t.name + "\":{\"dtype\":\"" + t.dtype + "\",\"shape\":[";
        for (size_t i = 0; i < t.shape.size(); ++i) h += (i ? "," : "") + std::to_string(t.shape[i]);
        h += "],\"data_offsets\":[" + std::to_string(at) + "," + std::to_string(at + t.bytes.size()) + "]}";
        at += t.bytes.size();
    }
    h += "}";
    const std::string p = radtest::tmp_path(stem);
    FILE* f = std::fopen(p.c_str(), "wb");
    CHECK(f != nullptr);
    if (!f) return std::string();
    const uint64_t n = h.size();
    std::fwrite(&n, 8, 1, f);
    std::fwrite(h.data(), 1, h.size(), f);
    for (const St& t : ts) std::fwrite(t.bytes.data(), 1, t.bytes.size(), f);
    std::fclose(f);
    return p;
}

St codes(int i, const char* dtype = "F8_E4M3", int64_t rows = -1) {
    rows = rows < 0 ? kRows[i] : rows;
    return { "layers." + std::to_string(kLayer[i]) + ".engram.embed.weight", dtype, { rows, kDim },
             ramp(rows * kDim, (uint8_t)(60 + i)) };
}
St scale(int i, const char* dtype = "F8_E8M0", int64_t cols = kDim / kGroup) {
    return { "layers." + std::to_string(kLayer[i]) + ".engram.embed.scale", dtype, { kRows[i], cols },
             ramp(kRows[i] * cols, (uint8_t)(70 + i)) };
}
St wkv(int i) {
    return { "layers." + std::to_string(kLayer[i]) + ".engram.wkv.weight", "F8_E4M3", { 4, 64 },
             ramp(4 * 64, 80) };
}

/* The two shards as the release lays them out: each layer's table, its scale and a projection. */
std::string shards() {
    const std::string a = write_st("ef_s47.safetensors", { codes(0), scale(0), wkv(0) });
    const std::string b = write_st("ef_s48.safetensors", { codes(1), scale(1), wkv(1) });
    return a + "," + b;
}

int open_with(const std::string& files, const std::string& container, std::string* why) {
    RadFile f;
    if (f.open(container.c_str()) != RAD_OK) {
        CHECK(!"the test container opens");
        return RAD_E_IO;
    }
    Ds41EngramFile e;
    const int st = e.open(files, f, spec(), why);
    if (st == RAD_OK) CHECK_EQ((int)e.tables().size(), 2);
    else CHECK(e.tables().empty());
    return st;
}

bool says(const std::string& s, const char* sub) {
    if (s.find(sub) != std::string::npos) return true;
    std::fprintf(stderr, "  '%s' not in: %s\n", sub, s.c_str());
    return false;
}

}  // namespace

TEST(the_release_tables_are_answered_from_the_sidecar_as_stored) {
    const std::string c = write_container("ef_ok.rad");
    RadFile f;
    REQUIRE_EQ(f.open(c.c_str()), RAD_OK);
    const std::string files = shards();
    Ds41EngramFile e;
    std::string why;
    REQUIRE_EQ(e.open(files, f, spec(), &why), RAD_OK);
    REQUIRE_EQ((int)e.tables().size(), 2);
    int64_t total = 0;
    for (int i = 0; i < 2; ++i) {
        const std::string n = "per_layer_token_embd.l" + std::to_string(kLayer[i]) + ".weight";
        const Ds41EngramTable* t = e.find(n);
        REQUIRE(t != nullptr);
        CHECK_EQ(t->layer, kLayer[i]);
        CHECK(rad_enc_is(&t->enc, "affine"));
        REQUIRE_EQ(t->enc.n_planes, 2);
        const RadEncPlane* cp = rad_enc_plane(&t->enc, "codes");
        const RadEncPlane* sp = rad_enc_plane(&t->enc, "scale");
        REQUIRE(cp && sp);
        CHECK_EQ(cp->dtype, (uint32_t)RAD_F8E4M3);
        CHECK_EQ(sp->dtype, (uint32_t)RAD_E8M0);
        CHECK_EQ(sp->block[0], (int64_t)1);
        CHECK_EQ(sp->block[1], kGroup);
        CHECK(!rad_enc_plane(&t->enc, "table"));
        CHECK_EQ(t->rank, (uint32_t)2);
        CHECK_EQ(t->shape[0], kRows[i]);
        CHECK_EQ(t->shape[1], kDim);
        /* The planes are the sidecar's bytes, in place. */
        const St cs = codes(i), ss = scale(i);
        REQUIRE_EQ(t->plane_bytes[0], (int64_t)cs.bytes.size());
        REQUIRE_EQ(t->plane_bytes[1], (int64_t)ss.bytes.size());
        CHECK_EQ(std::memcmp(t->plane[0], cs.bytes.data(), cs.bytes.size()), 0);
        CHECK_EQ(std::memcmp(t->plane[1], ss.bytes.data(), ss.bytes.size()), 0);
        CHECK_EQ(rad_enc_plane_bytes(cp, kRows[i], kDim), t->plane_bytes[0]);
        CHECK_EQ(rad_enc_plane_bytes(sp, kRows[i], kDim), t->plane_bytes[1]);
        CHECK(says(t->container_form, "u4"));
        total += t->plane_bytes[0] + t->plane_bytes[1];
    }
    CHECK_EQ(e.bytes(), total);
    CHECK(!e.find("blk.1.engram_kv.weight"));
    CHECK(!e.find("layers.1.engram.embed.weight"));
    const std::string d = e.describe();
    CHECK(says(d, "per_layer_token_embd.l1.weight"));
    CHECK(says(d, "per_layer_token_embd.l14.weight"));
    CHECK(says(d, "fp8_e4m3"));
    CHECK(says(d, "ef_s48.safetensors"));

    /* The source function: the tables from here, the rest from the container. */
    const WeightSourceFn src = e.wrap(rad_container_sources(f));
    Program prog;
    WeightSource ws;
    REQUIRE_EQ(src("per_layer_token_embd.l14.weight", prog, &ws), RAD_OK);
    CHECK(rad_enc_equal(&ws.enc, &e.find("per_layer_token_embd.l14.weight")->enc));
    CHECK_EQ(ws.rank, (uint32_t)2);
    CHECK_EQ(ws.shape[0], kRows[1]);
    CHECK_EQ(ws.shape[1], kDim);
    WeightSource kv;
    REQUIRE_EQ(src("blk.1.engram_kv.weight", prog, &kv), RAD_OK);
    CHECK(rad_enc_equal(&kv.enc, &f.encoding(*f.find("blk.1.engram_kv.weight"))));
    CHECK_EQ(src("nothing.weight", prog, &kv), RAD_E_NOTFOUND);
    CHECK_EQ(e.wrap({})("nothing.weight", prog, &kv), RAD_E_NOTFOUND);
}

TEST(a_tables_tensors_may_sit_in_any_named_file) {
    const std::string c = write_container("ef_split.rad");
    const std::string a = write_st("ef_split_a.safetensors", { codes(0), scale(1) });
    const std::string b = write_st("ef_split_b.safetensors", { wkv(0), codes(1) });
    const std::string s = write_st("ef_split_c.safetensors", { scale(0) });
    std::string why;
    CHECK_EQ(open_with(a + "," + b + ",," + s, c, &why), RAD_OK);
}

TEST(the_container_must_be_the_one_the_tables_index) {
    const std::string files = shards();
    std::string why;
    {
        ContainerOpts o;
        o.source = nullptr;
        CHECK_EQ(open_with(files, write_container("ef_nosrc.rad", o), &why), RAD_E_FORMAT);
        CHECK(says(why, "per layer"));
    }
    {
        ContainerOpts o;
        o.source = "something_else";
        CHECK_EQ(open_with(files, write_container("ef_othersrc.rad", o), &why), RAD_E_FORMAT);
        CHECK(says(why, "something_else"));
    }
    for (const char* k : { "deepseek41.engram.token_map", "blk.14.engram_off", "blk.1.engram_mult" }) {
        ContainerOpts o;
        o.flip_control = k;
        CHECK_EQ(open_with(files, write_container("ef_flip.rad", o), &why), RAD_E_FORMAT);
        CHECK(says(why, k));
        CHECK(says(why, "not the release's"));
    }
    {
        ContainerOpts o;
        o.drop = "blk.14.engram_vsz";
        CHECK_EQ(open_with(files, write_container("ef_dropc.rad", o), &why), RAD_E_FORMAT);
        CHECK(says(why, "blk.14.engram_vsz"));
    }
    {
        ContainerOpts o;
        o.drop = "per_layer_token_embd.l14.weight";
        CHECK_EQ(open_with(files, write_container("ef_dropt.rad", o), &why), RAD_E_FORMAT);
        CHECK(says(why, "no Engram table 'per_layer_token_embd.l14.weight'"));
    }
    {
        ContainerOpts o;
        o.rows14 = kRows[1] + 1;
        CHECK_EQ(open_with(files, write_container("ef_rows.rad", o), &why), RAD_E_FORMAT);
        CHECK(says(why, "per_layer_token_embd.l14.weight"));
    }
}

TEST(the_sidecar_must_hold_each_table_once_in_the_release_form) {
    const std::string c = write_container("ef_side.rad");
    std::string why;
    const std::string b = write_st("ef_side_b.safetensors", { codes(1), scale(1) });
    auto with_a = [&](const char* stem, const std::vector<St>& ts) {
        return open_with(write_st(stem, ts) + "," + b, c, &why);
    };
    CHECK_EQ(with_a("ef_e5.safetensors", { codes(0, "F8_E5M2"), scale(0) }), RAD_E_FORMAT);
    CHECK(says(why, "F8_E5M2"));
    CHECK_EQ(with_a("ef_bf.safetensors", { codes(0), scale(0, "F8_E5M2") }), RAD_E_FORMAT);
    CHECK(says(why, "layers.1.engram.embed.scale"));
    CHECK_EQ(with_a("ef_g16.safetensors", { codes(0), scale(0, "F8_E8M0", kDim / 16) }), RAD_E_FORMAT);
    CHECK(says(why, "[5, 16]"));
    CHECK_EQ(with_a("ef_rows.safetensors", { codes(0, "F8_E4M3", kRows[0] + 1), scale(0) }), RAD_E_FORMAT);
    CHECK(says(why, "[6, 256]"));
    CHECK_EQ(with_a("ef_noscale.safetensors", { codes(0) }), RAD_E_FORMAT);
    CHECK(says(why, "no file named holds 'layers.1.engram.embed.scale'"));
    CHECK_EQ(with_a("ef_dup.safetensors", { codes(0), scale(0), codes(1) }), RAD_E_FORMAT);
    CHECK(says(why, "is in both"));
    const std::string a = write_st("ef_a.safetensors", { codes(0), scale(0) });
    CHECK_EQ(open_with(a + "," + b + "," + a, c, &why), RAD_E_FORMAT);
    CHECK(says(why, "named twice"));
    CHECK_EQ(open_with(a + "," + radtest::tmp_path("ef_missing.safetensors"), c, &why), RAD_E_FORMAT);
    CHECK(says(why, "ef_missing.safetensors"));
    CHECK_EQ(open_with(",", c, &why), RAD_E_FORMAT);
    CHECK(says(why, "no file"));
    CHECK_EQ(open_with(a + "," + b, c, &why), RAD_OK);
}

TEST(the_release_spec_is_the_native_containers_geometry) {
    const Ds41EngramSpec& s = ds41_engram_release_spec();
    REQUIRE_EQ(s.layers.size(), (size_t)2);
    int64_t rows = 0;
    for (size_t i = 0; i < 2; ++i) {
        CHECK_EQ(s.layers[i].layer, (int64_t)deepseek41::ista::layers[i]);
        CHECK_EQ(s.layers[i].rows, (int64_t)deepseek41::ista::layer_rows[i]);
        rows += s.layers[i].rows;
    }
    CHECK_EQ(rows, (int64_t)deepseek41::ista::table_rows);
    CHECK_EQ(s.dim, (int64_t)256);
    CHECK_EQ(s.group, (int64_t)32);
    REQUIRE_EQ(s.controls.size(), (size_t)7);
    for (const auto& [n, d] : s.controls) CHECK_EQ(d.size(), (size_t)64);
    CHECK_EQ(s.controls[0].first, std::string("deepseek41.engram.token_map"));
}

/* The real files, when named: RAD_TEST_DS41_CONTAINER (the native container) and RAD_TEST_DS41_ENGRAM_FILE (the release's
 * shards 47 and 48, comma-separated). Reads their headers and the container's seven constants only. */
TEST(the_release_shards_fit_the_native_container) {
    const char* c = std::getenv("RAD_TEST_DS41_CONTAINER");
    const char* s = std::getenv("RAD_TEST_DS41_ENGRAM_FILE");
    if (!c || !s) SKIP("RAD_TEST_DS41_CONTAINER / RAD_TEST_DS41_ENGRAM_FILE not set");
    RadFile f;
    REQUIRE_EQ(f.open(c), RAD_OK);
    Ds41EngramFile e;
    std::string why;
    const int st = e.open(s, f, ds41_engram_release_spec(), &why);
    if (st != RAD_OK) std::fprintf(stderr, "  refused: %s\n", why.c_str());
    REQUIRE_EQ(st, RAD_OK);
    CHECK_EQ((int)e.tables().size(), 2);
    CHECK_EQ(e.bytes(), (int64_t)(384006168LL + 384016682LL) * (256 + 8));
    std::fprintf(stderr, "  %s\n", e.describe().c_str());
}

RAD_TEST_MAIN()
