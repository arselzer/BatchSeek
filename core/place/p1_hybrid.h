/* p1_hybrid.h -- hybrid card-path decode (RADIANCE_EXPERIMENTAL_DS41_P1_HYBRID=<pct>, ds41_p1.h kHybridEnv):
 * in a decode pass armed for the card path (..._P1_DECODE_ROWS), each routed layer gives some of its
 * RAM-cache units to the CPU pool, which computes them while the card streams the others through its ring.
 *
 * ---- why it is exact ----------------------------------------------------------------------------------
 *
 * A routed layer's down projection writes one row a sorted slot (expert e's slots are [off[e], off[e+1]),
 * the scatter's order); the gather reads those rows by slot and weighs them in its own fixed order. Which
 * side wrote a slot's row never reaches the gather: it reads bytes. The card path's rows are the P1 twin's,
 * held byte for byte to the decode band's at every M up to 4096 (device check `p1`); the CPU pool's are
 * ds41hx's, bit-exact with the decode band (host_experts_test; the CPU path of every pass below the
 * threshold). So every slot's bytes are the same whichever side computes its expert, and the gather's
 * sum is the base's. In a hybrid layer:
 *   - a CPU unit is published to the plugin's ops with kMarkSkip, as every non-resident unit of an armed
 *     layer is, and is in no wave, so no card kernel writes its slots (the twin's skip: no rows, no bytes;
 *     its zero-fill is [off[96], T), the other ranks' slots, never a local expert's);
 *   - its rows are the CPU path's own: hx_jobs_into takes expert e's live slots of [off[e], off[e+1]) from
 *     the sorted order fetched with the counts, refuses a layer whose count disagrees with the scatter's,
 *     and hx_capture refuses a gate_up that reads other rows than the ones the pool was given;
 *   - hx_finish copies them over the down projection's rows at the gather's operands, on the compute
 *     stream behind the layer's waves (after_op runs them right behind the down projection, before any
 *     later op), exactly as the CPU path copies its rows before the gather today.
 * Residency: a CPU unit is borrowed (Mover::borrow_host_unit) with the same LRU touch the ring's add_ram
 * gives it, in the same order (the layer's selected units in order), and a borrow that fails sends the
 * unit to the ring as before; so which units the RAM cache holds evolves as without the switch, except
 * that background inserts that wait for an idle pool (HostExpertPool::wait_idle) may land later.
 *
 * ---- prefill (RADIANCE_EXPERIMENTAL_DS41_P1_HYBRID_PREFILL=<pct>) ------------------------------------------------
 *
 * The same split in an armed prefill pass's card-path layers, with a share and counts of its own. Nothing above
 * depends on the pass's height: the CPU path's rows are the decode band's bits at any height (the CPU path takes
 * whole prefill chunks under LCP), the skip mark keeps the twin (and the register-tiled twin) off a unit's slots at
 * any M, and hx_finish copies the rows over the owning rank's down projection before that rank's gather. So each
 * rank's bf16 partial, and the all-reduce's ascending-rank fold of the four, are the base's. Why: a prefill card is
 * link-bound on the units it stages, and which rank stages most changes layer by layer (EPBAL-20261115.md); the
 * pool is one for all ranks, and each rank's budget is its own remaining card bytes, so the CPU takes more from a
 * rank with more to stage.
 *
 * ---- the split ----------------------------------------------------------------------------------------
 *
 * choose(): of a layer's off-card selected units (its ring units without the switch), only RAM-cache hits
 * may go to the CPU (a drive miss stays on the card: its planes are read and packed for the ring). They
 * are taken fewest rows first (the CPU's cost grows with rows, the card's link time does not), while
 *     (pool backlog + CPU cost of the taken) / cpu rate  <=  share * (bytes left to the card) / link rate,
 * with the pool's own cost model (HostExpertPool::cost_rows; backlog over every rank, reservations
 * included, so ranks deciding at once see each other). The first unit that does not fit stops the scan:
 * later ones have as many rows or more.
 *
 * Share: the switch's percent, adapted per rank by Share::layer() from each hybrid layer's outcome at the
 * gather: the CPU done before the issuing thread got there (early: grow by 5% up to the switch's value),
 * the thread waited but the card was still on the layer's waves (hold), or the card had finished them and
 * waited for the CPU's rows (late: shrink by 20%, down to 1%). ..._P1_HYBRID_ADAPT=0 holds it fixed.
 *
 * ---- calibration (RADIANCE_EXPERIMENTAL_DS41_P1_HYBRID_CAL=<mask>, ds41_p1.h kHybridCalEnv) -------------------------
 *
 * Why (H55, ab-20261116T052043Z N512): the four ranks staged the same units in total (CPU + card 258.7k / 259.9k /
 * 258.1k / 256.4k prefill units), but ranks 2 and 3 gave the CPU 15% fewer (113.0k / 112.8k against 95.6k / 96.3k)
 * and so staged ~10% more over their links. Their predicted CPU over predicted card time, about the share each ran
 * with, was 0.33 / 0.32 / 0.24 / 0.24, while the cost the choice saw ahead of it was about the same on every rank:
 * the shares had drifted apart, not the backlog. The share is a multiplicative walk that sits on a knife edge (16-17%
 * late layers at x0.8 against 77-79% early ones at x1.05 nets +-0.2% a layer), and its late signal is the shared
 * pool's: workers take the oldest job first, so the rank whose jobs sit last waits longest when the pool runs behind
 * (wait a late layer 16.5 / 16.8 / 18.3 / 20.0 ms) and backs off alone, while the others' takes caused it. And the
 * choice reads the backlog before the borrows reserve the taken units, so ranks deciding at once both fit into it.
 * The pool also runs ~2.6x below its model rate beside four cards' DMA, which the share alone absorbs.
 *
 *   kHybridCalRate: the CPU rate in choose() is RateCal's, learnt from each split layer's completion: the pool's
 *     submit-to-finish time (Ticket::last_ns) over the cost the layer had to drain (the backlog its choice saw plus
 *     its own reserved cost), an EWMA a pass kind (decode, prefill), process-wide (one pool), started at the model
 *     rate. The share stays the safety loop; its ceiling is lifted to 4P (at most 400%) because P was chosen against
 *     the model rate, which overstates the pool by about that much.
 *   kHybridCalCouple: one share a pass kind for all ranks (CoupledShare), adapted from every rank's outcomes, so a
 *     late pool backs every rank off alike and the split between ranks follows their card bytes; and the backlog
 *     read, the choice and the reservation of its whole cost happen under the pool's decide lock (the borrows then
 *     draw on that reservation, and what falls back is given back after the layer's borrows).
 * Both only change which side computes a unit, never its rows' bytes.
 */
#pragma once
#include "host_expert_pool.h"

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <numeric>
#include <vector>

namespace rad {
namespace p1hyb {

/* One off-card selected unit of the layer: its rows this pass, its bytes over the link, and whether the
 * RAM cache holds it (only then may the CPU take it). */
struct Cand {
    int     rows = 0;
    int64_t bytes = 0;
    bool    cached = false;
};

struct Split {
    int     n_cpu = 0;
    int64_t cpu_cost = 0;     /* the pool's cost of the taken units (HostExpertPool::cost_rows) */
    int64_t card_bytes = 0;   /* bytes left to the card's ring */
    double  cpu_ns = 0;       /* predicted: (backlog + cpu_cost) / cpu rate */
    double  card_ns = 0;      /* predicted: card_bytes / link rate */
};

/* to_cpu[i] (n entries) set for the units the CPU takes. cpu_bpns / link_bpns: bytes a ns
 * (HostExpertPool::cpu_bytes_per_ns, link_bytes_per_ns); share > 0. `order` is scratch.
 *
 * `cost`: a unit's cost of `rows` rows (null: HostExpertPool::cost_rows). `by_cost`
 * (..._HOST_EXPERTS_KERNEL_COST): the scan takes units cheapest first (stable: rows, then the layer's order, on
 * ties) instead of fewest rows first. Every unit stands for the same card bytes here, so cheapest first is the
 * order in which the first misfit bounds every unit after it and the break stays right; fewest rows first is the
 * same order only while the cost grows with the rows, which the default's costs do not past four rows a job. */
inline Split choose(const Cand* c, int n, int64_t backlog, double cpu_bpns, double link_bpns, double share,
                    uint8_t* to_cpu, std::vector<int>* order, bool by_cost = false,
                    int64_t (*cost_fn)(int) = nullptr) {
    int64_t (*const cost_of)(int) = cost_fn ? cost_fn : &HostExpertPool::cost_rows;
    Split s;
    for (int i = 0; i < n; ++i) {
        to_cpu[i] = 0;
        s.card_bytes += c[i].bytes;
    }
    s.card_ns = link_bpns > 0 ? (double)s.card_bytes / link_bpns : 0;
    s.cpu_ns = cpu_bpns > 0 ? (double)std::max<int64_t>(backlog, 0) / cpu_bpns : 0;
    if (n <= 0 || !(share > 0) || !(cpu_bpns > 0) || !(link_bpns > 0)) return s;
    order->clear();
    for (int i = 0; i < n; ++i)
        if (c[i].cached && c[i].rows > 0 && c[i].bytes > 0) order->push_back(i);
    std::stable_sort(order->begin(), order->end(), [&](int a, int b) { return c[a].rows < c[b].rows; });
    if (by_cost)
        std::stable_sort(order->begin(), order->end(), [&](int a, int b) { return cost_of(c[a].rows) < cost_of(c[b].rows); });
    const double base = (double)std::max<int64_t>(backlog, 0);
    for (int i : *order) {
        const int64_t cost = s.cpu_cost + cost_of(c[i].rows);
        const int64_t left = s.card_bytes - c[i].bytes;
        const double cpu = (base + (double)cost) / cpu_bpns, card = (double)left / link_bpns;
        if (cpu > share * card) break;
        to_cpu[i] = 1;
        ++s.n_cpu;
        s.cpu_cost = cost;
        s.card_bytes = left;
        s.cpu_ns = cpu;
        s.card_ns = card;
    }
    return s;
}

/* The share a rank runs with (see the header). */
class Share {
public:
    static constexpr double kMin = 0.01, kGrow = 1.05, kShrink = 0.8;
    /* Starts at pct; early layers grow it up to max(pct, ceil_pct) (0: pct, the switch's value). */
    void init(int pct, bool adapt, int ceil_pct = 0) {
        max_ = std::max(pct, ceil_pct) / 100.0;
        cur_ = pct / 100.0;
        min_seen_ = cur_;
        adapt_ = adapt;
    }
    double share() const { return cur_; }
    double min_seen() const { return min_seen_; }
    double ceiling() const { return max_; }
    /* A hybrid layer's outcome at the gather: `early` the CPU was done before the issuing thread came,
     * `late` the card had run the layer's waves and waited for the CPU. */
    void layer(bool early, bool late) {
        if (!adapt_) return;
        if (late) cur_ = std::max(kMin, cur_ * kShrink);
        else if (early) cur_ = std::min(max_, cur_ * kGrow);
        min_seen_ = std::min(min_seen_, cur_);
    }

private:
    double max_ = 0, cur_ = 0, min_seen_ = 0;
    bool   adapt_ = true;
};

/* kHybridCalCouple: one Share for every rank of a pass kind; each rank's layer outcome moves it. */
class CoupledShare {
public:
    void init(int pct, bool adapt, int ceil_pct = 0) {
        std::lock_guard<std::mutex> lk(mu_);
        s_.init(pct, adapt, ceil_pct);
    }
    double share() const {
        std::lock_guard<std::mutex> lk(mu_);
        return s_.share();
    }
    double min_seen() const {
        std::lock_guard<std::mutex> lk(mu_);
        return s_.min_seen();
    }
    void layer(bool early, bool late) {
        std::lock_guard<std::mutex> lk(mu_);
        s_.layer(early, late);
    }

private:
    mutable std::mutex mu_;
    Share s_;
};

/* kHybridCalRate: the pool's rate as the split sees it, in cost bytes a ns, learnt from completions. A sample is a
 * split layer's: `bytes` the cost it had to drain (the backlog its choice saw plus its own reserved cost), `ns` the
 * pool's submit-to-finish time. Each kind (0 decode, 1 prefill) keeps an EWMA of ns a cost byte, weight kAlpha a
 * sample, started at the model rate; a sample's ns a byte is clamped to [kLo, kHi] times the model's, so one stalled
 * layer (a preempted worker, a page fault) cannot swing it. Thread-safe. */
class RateCal {
public:
    static constexpr double kAlpha = 1.0 / 16, kLo = 0.25, kHi = 16.0;
    void init(double model_bpns) {
        std::lock_guard<std::mutex> lk(mu_);
        model_npb_ = model_bpns > 0 ? 1.0 / model_bpns : 0;
        for (int k = 0; k < 2; ++k) {
            npb_[k] = model_npb_;
            n_[k] = 0;
        }
    }
    double bytes_per_ns(int kind) const {
        std::lock_guard<std::mutex> lk(mu_);
        const double v = npb_[kind & 1];
        return v > 0 ? 1.0 / v : 0;
    }
    uint64_t samples(int kind) const {
        std::lock_guard<std::mutex> lk(mu_);
        return n_[kind & 1];
    }
    void sample(int kind, int64_t bytes, int64_t ns) {
        if (bytes <= 0 || ns <= 0) return;
        std::lock_guard<std::mutex> lk(mu_);
        if (!(model_npb_ > 0)) return;
        const double v = std::clamp((double)ns / (double)bytes, kLo * model_npb_, kHi * model_npb_);
        double& e = npb_[kind & 1];
        e += kAlpha * (v - e);
        ++n_[kind & 1];
    }

private:
    mutable std::mutex mu_;
    double   model_npb_ = 0, npb_[2] = { 0, 0 };
    uint64_t n_[2] = { 0, 0 };
};

/* The process's calibration state: the pool is one for every rank, and so are these. */
inline RateCal& rate_cal() {
    static RateCal r;
    return r;
}
inline CoupledShare& coupled_share(int kind) {
    static CoupledShare s[2];
    return s[kind & 1];
}
/* The share's ceiling under kHybridCalRate: 4P, at most kMaxPct (ds41p1::kMaxHybridPct). */
inline int cal_ceiling(int pct, int max_pct) { return std::min(4 * pct, std::max(pct, max_pct)); }

}  /* namespace p1hyb */
}  /* namespace rad */
