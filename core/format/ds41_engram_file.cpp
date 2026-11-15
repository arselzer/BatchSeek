/* ds41_engram_file.cpp -- see ds41_engram_file.h. */
#include "ds41_engram_file.h"

#include "../place/expert_profile_sha256.h"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <set>

namespace rad {

const Ds41EngramSpec& ds41_engram_release_spec() {
    static const Ds41EngramSpec s = [] {
        Ds41EngramSpec r;
        r.layers = { { 1, 384006168 }, { 14, 384016682 } };
        r.dim = 256;
        r.group = 32;
        r.controls = {
            { "deepseek41.engram.token_map", "c60a86322ec17b4142bfef3c57a8d81fb428550cdf487f88a4320cb59fe46481" },
            { "blk.1.engram_mult",  "c576ed83e3e05f5ade50d40bacedb202ad3dd0709b69187d5ec520e566eb090e" },
            { "blk.1.engram_vsz",   "b81f50991dac834241319073c20a50273631ca1ab42c49ff7ad971a69cde9eff" },
            { "blk.1.engram_off",   "7ff307552d9fbe7c49709ea85501f002047a04a8ce16d483909616732e14e718" },
            { "blk.14.engram_mult", "003d26532f2ef2adeb8244c9beb1cade6f1c630f39b843282cda208146afb51f" },
            { "blk.14.engram_vsz",  "4fa8d50c22b130ed3e568a6164b1f1f56337c335d859131936dd08d68c71f9e2" },
            { "blk.14.engram_off",  "083de2f2b2c5390069e7d45ccca57144361042034c39d493f8a815e1f44f1596" },
        };
        return r;
    }();
    return s;
}

namespace {

std::string fmtv(const char* f, ...) __attribute__((format(printf, 1, 2)));
std::string fmtv(const char* f, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(buf, sizeof buf, f, ap);
    va_end(ap);
    return buf;
}

std::string shape_str(const std::vector<int64_t>& s) {
    std::string o = "[";
    for (size_t i = 0; i < s.size(); ++i) o += (i ? ", " : "") + std::to_string(s[i]);
    return o + "]";
}

}  /* namespace */

int Ds41EngramFile::open(const std::string& paths, const RadFile& container,
                         const Ds41EngramSpec& spec, std::string* why) {
    std::string dummy;
    if (!why) why = &dummy;
    files_.clear();
    tables_.clear();
    auto refuse = [&](const std::string& w) {
        *why = w;
        files_.clear();
        tables_.clear();
        return RAD_E_FORMAT;
    };
    if (spec.layers.empty() || spec.dim <= 0 || spec.group <= 0 || spec.dim % spec.group)
        return refuse("no Engram layout to match");

    /* The container: per-layer tables, the spec's rows, the release's hash constants. */
    if (!container.is_open()) return refuse("no container is open");
    const char* src = container.get_s("deepseek41.engram_source");
    if (!src || std::strcmp(src, "ista_iq4_nl_v1"))
        return refuse(fmtv("the container states deepseek41.engram_source=%s; only a container "
                           "that keeps its Engram tables per layer (ista_iq4_nl_v1) takes them "
                           "from a sidecar", src ? src : "(nothing)"));
    for (const auto& [name, want] : spec.controls) {
        const RadFileEntry* e = container.find(name);
        if (!e || e->n_planes != 1)
            return refuse(fmtv("the container has no Engram constant '%s' to check", name.c_str()));
        const RadFilePlane& p = container.plane(*e, 0);
        ds41_integrity::Sha256 h;
        h.update(container.data(p), (size_t)p.bytes);
        const std::string got = h.finish();
        if (got != want)
            return refuse(fmtv("the container's Engram constant '%s' (sha256 %s) is not the "
                               "release's (%s), so its ids would not index the release's tables",
                               name.c_str(), got.c_str(), want.c_str()));
    }

    /* The sidecar's files, each once. */
    std::set<std::string> seen;
    for (size_t at = 0; at <= paths.size();) {
        const size_t comma = std::min(paths.find(',', at), paths.size());
        const std::string p = paths.substr(at, comma - at);
        at = comma + 1;
        if (p.empty()) continue;
        if (!seen.insert(p).second) return refuse(fmtv("'%s' is named twice", p.c_str()));
        auto f = std::make_unique<SafeTensorsFile>();
        if (f->open(p) != RAD_OK) return refuse(fmtv("'%s' is not a readable safetensors file", p.c_str()));
        files_.push_back(std::move(f));
    }
    if (files_.empty()) return refuse("no file is named");

    for (const Ds41EngramSpec::Layer& L : spec.layers) {
        const std::string cname = fmtv("per_layer_token_embd.l%lld.weight", (long long)L.layer);
        const RadFileEntry* ce = container.find(cname);
        if (!ce)
            return refuse(fmtv("the container has no Engram table '%s'", cname.c_str()));
        if (ce->rank != 2 || ce->shape[0] != L.rows || ce->shape[1] != spec.dim)
            return refuse(fmtv("the container's '%s' is %lldx%lld, the model's Engram layer %lld "
                               "is %lldx%lld", cname.c_str(), (long long)ce->shape[0],
                               (long long)(ce->rank > 1 ? ce->shape[1] : 0), (long long)L.layer,
                               (long long)L.rows, (long long)spec.dim));
        const std::string wn = fmtv("layers.%lld.engram.embed.weight", (long long)L.layer);
        const std::string sn = fmtv("layers.%lld.engram.embed.scale", (long long)L.layer);
        const StTensor* w = nullptr;
        const StTensor* s = nullptr;
        const SafeTensorsFile* wf = nullptr;
        const SafeTensorsFile* sf = nullptr;
        for (const auto& f : files_) {
            if (const StTensor* t = f->find(wn)) {
                if (w) return refuse(fmtv("'%s' is in both %s and %s", wn.c_str(),
                                          wf->path().c_str(), f->path().c_str()));
                w = t, wf = f.get();
            }
            if (const StTensor* t = f->find(sn)) {
                if (s) return refuse(fmtv("'%s' is in both %s and %s", sn.c_str(),
                                          sf->path().c_str(), f->path().c_str()));
                s = t, sf = f.get();
            }
        }
        if (!w || !s)
            return refuse(fmtv("no file named holds '%s'", (!w ? wn : sn).c_str()));
        const std::vector<int64_t> ws = { L.rows, spec.dim }, ss = { L.rows, spec.dim / spec.group };
        if (w->dtype != RAD_F8E4M3 || w->shape != ws)
            return refuse(fmtv("%s holds '%s' as %s %s; the model's is F8_E4M3 %s",
                               wf->path().c_str(), wn.c_str(), w->dtype_str.c_str(),
                               shape_str(w->shape).c_str(), shape_str(ws).c_str()));
        if (s->dtype != RAD_E8M0 || s->shape != ss)
            return refuse(fmtv("%s holds '%s' as %s %s; the model's is F8_E8M0 %s",
                               sf->path().c_str(), sn.c_str(), s->dtype_str.c_str(),
                               shape_str(s->shape).c_str(), shape_str(ss).c_str()));
        if ((int64_t)w->bytes != L.rows * spec.dim || (int64_t)s->bytes != L.rows * (spec.dim / spec.group))
            return refuse(fmtv("'%s' or '%s' does not hold its rows' bytes", wn.c_str(), sn.c_str()));

        Ds41EngramTable t;
        t.name = cname;
        t.layer = L.layer;
        t.enc = rad_enc_affine(RAD_F8E4M3, RAD_E8M0, 1, spec.group);
        t.rank = 2;
        t.shape[0] = L.rows;
        t.shape[1] = spec.dim;
        t.plane[0] = w->data;
        t.plane[1] = s->data;
        t.plane_bytes[0] = (int64_t)w->bytes;
        t.plane_bytes[1] = (int64_t)s->bytes;
        t.file[0] = wf->path();
        t.file[1] = sf->path();
        char en[256];
        rad_enc_format(&container.encoding(*ce), en, sizeof en);
        t.container_form = en;
        tables_.push_back(std::move(t));
    }
    return RAD_OK;
}

const Ds41EngramTable* Ds41EngramFile::find(const std::string& name) const {
    for (const Ds41EngramTable& t : tables_)
        if (t.name == name) return &t;
    return nullptr;
}

int64_t Ds41EngramFile::bytes() const {
    int64_t n = 0;
    for (const Ds41EngramTable& t : tables_) n += t.plane_bytes[0] + t.plane_bytes[1];
    return n;
}

WeightSourceFn Ds41EngramFile::wrap(WeightSourceFn inner) const {
    return [this, inner](const std::string& name, const Program& prog, WeightSource* out) {
        if (const Ds41EngramTable* t = find(name)) {
            out->enc = t->enc;
            out->rank = t->rank;
            for (int i = 0; i < RAD_MAX_RANK; ++i) out->shape[i] = t->shape[i];
            return (int)RAD_OK;
        }
        return inner ? inner(name, prog, out) : (int)RAD_E_NOTFOUND;
    };
}

std::string Ds41EngramFile::describe() const {
    std::string o;
    std::set<std::string> files;
    for (const Ds41EngramTable& t : tables_) {
        char en[256];
        rad_enc_format(&t.enc, en, sizeof en);
        o += fmtv("%s%s %lldx%lld as %s (%s; the container's %s is not read)", o.empty() ? "" : "; ",
                  t.name.c_str(), (long long)t.shape[0], (long long)t.shape[1], en,
                  humanb(t.plane_bytes[0] + t.plane_bytes[1]).c_str(), t.container_form.c_str());
        files.insert(t.file[0]);
        files.insert(t.file[1]);
    }
    o += " from";
    for (const std::string& f : files) o += " " + f;
    return o;
}

}  /* namespace rad */
