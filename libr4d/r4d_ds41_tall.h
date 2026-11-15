/* r4d_ds41_tall.h -- RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL and RADIANCE_EXPERIMENTAL_DS41_LCP on
 * libr4d's side (host code only).
 *
 * DeepSeek-V4.1's prefill runs in passes of at most 16 tokens today. A taller pass changes the
 * arithmetic of the ops whose kernel follows the pass height:
 *
 *   - the routed MXFP4 GEMM: moe_gemm_ggml_mxfp4_gemv serves up to 16 tokens, the tiled row above
 *     sums each output in another order (r4d_ds41_mxfp4.hip);
 *   - the sparse attention and the indexer score: their exact scalar forms up to 48 rows, WMMA
 *     forms above (r4d_ds41_sparse.hip, kWmRows);
 *   - every norm whose width comes from r4d_act_threads (r4d_common.h): 1024 threads up to 32
 *     rows, 512 or 256 above, which regroups the sum of squares. In ds41's graph: rmsnorm, and the
 *     fused mhc_collapse_rmsnorm, mhc_post_collapse_rmsnorm and rmsnorm_rope_roundtrip.
 *
 * With EXACT_TALL at "1" libr4d keeps the decode band's arithmetic at every height up to
 * r4d_ds41_tall_rows(): the routed GEMM's twin row moe_gemm_ggml_mxfp4_gemv_tall (r4d_rows.cpp), the
 * gemv at any number of tokens, takes the bands between 17 and that bound from the tiled row, the
 * sparse ops keep their scalar forms at every height, and r4d_act_threads gives every height the
 * 1024 threads rows up to 32 get, so every norm sums a row's squares alike. The twin's bound on M is
 * the tallest pass at which every libr4d kernel ds41 runs gives a 16-token pass's bits, so the core
 * reads the limit off the bands it resolved (FileStager::tall_init) rather than assuming it: 48, the
 * CPU path's limit (core/place/ep_route_counts.h) and the sparse ops' scalar band.
 *
 * Large-chunk prefill, RADIANCE_EXPERIMENTAL_DS41_LCP=<C> (C a multiple of 16 in 64 .. 4096): the
 * same pins at every height, with the twin's bound raised to kR4dDs41LcpMax. Every other ds41 kernel
 * that picks a form by the pass height keeps a row's bits by construction at any height (the
 * attention's head groups, the indexer's head split, mhc_mix's split, the f32 router's tile forms,
 * moe_scatter's launches, the exact all-reduce's one- and two-shot), and the gemm_nt tiled row is
 * byte-identical to the skinny one by contract; the core admits those pairs by name and refuses any
 * other op that resolves two kernels (FileStager::lcp_init), and ds41_mxfp4_device_check `lcp` holds
 * each to its rows launched alone and 16 at a time up to the requested chunk. A valid C turns
 * EXACT_TALL's pins on here whatever EXACT_TALL says (the core refuses EXACT_TALL other than 1 or
 * unset beside it).
 *
 * Read once, the first time a launcher or the row table asks. Anything but "1" (EXACT_TALL), or a C
 * outside the range (LCP), is off here; the core refuses such values (core/place/ep_route_counts.h). */
#pragma once
#include <cstdlib>

/* The twin's bound under EXACT_TALL alone: the tallest pass that keeps a 16-token pass's bits (32
 * without the norms' pin). */
constexpr int kR4dDs41TallRows = 48;
/* Under LCP: the largest chunk the switch takes, which is the twin's bound, and the smallest. */
constexpr int kR4dDs41LcpMax = 4096;
constexpr int kR4dDs41LcpMin = 64;

/* RADIANCE_EXPERIMENTAL_DS41_LCP: C, or 0 (unset, or not a multiple of 16 in [64, 4096]). */
inline int r4d_ds41_lcp() {
    static const int c = [] {
        const char* e = std::getenv("RADIANCE_EXPERIMENTAL_DS41_LCP");
        if (!e || !*e) return 0;
        char* end = nullptr;
        const long v = std::strtol(e, &end, 10);
        return end != e && !*end && v >= kR4dDs41LcpMin && v <= kR4dDs41LcpMax && v % 16 == 0 ? (int)v : 0;
    }();
    return c;
}

inline bool r4d_ds41_exact_tall() {
    static const bool on = [] {
        const char* e = std::getenv("RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL");
        return (e && e[0] == '1' && e[1] == '\0') || r4d_ds41_lcp() > 0;
    }();
    return on;
}

/* Prefill Phase 1, RADIANCE_EXPERIMENTAL_DS41_P1 (tools/ds41-host-experts/ds41_p1.h, the contract between the
 * card kernel side and the core's card expert stager; restated here because a plugin includes nothing of the
 * core or the tools, and ds41_mxfp4_device_check `p1` static_asserts that the two agree):
 *
 *   "1" beside a valid LCP: the routed GEMM's twin is registered as kR4dDs41P1Row (in place of
 *   moe_gemm_ggml_mxfp4_gemv_tall) with its bound on M raised to kR4dDs41P1MaxRows, and its launcher
 *   (r4d_rad_moe_gemm_mxfp4_gemv_tall_p1, r4d_ds41_mxfp4.hip) serves a weight table through gemv_vp's kernel at
 *   every height: entries marked kR4dDs41MarkPacked are packed-scale weights, entries marked kR4dDs41MarkSkip
 *   are not computed (none of their rows written, none of their bytes read), every expert owns the live-tail
 *   range [off[e], off[e + 1]), the slots past off[n_expert] are written as zeros by every call, and the work is
 *   spread by row passes. Anything else, or "1" without a valid LCP, is off here (the core refuses an invalid
 *   value and P1 without LCP). Read once. */
constexpr const char* kR4dDs41P1Env = "RADIANCE_EXPERIMENTAL_DS41_P1";
constexpr const char* kR4dDs41P1Row = "moe_gemm_ggml_mxfp4_gemv_tall_p1";
constexpr int kR4dDs41P1MaxRows = 4096;
constexpr unsigned long long kR4dDs41MarkPacked = 1, kR4dDs41MarkSkip = 2;

inline bool r4d_ds41_p1() {
    static const bool on = [] {
        const char* e = std::getenv(kR4dDs41P1Env);
        return e && e[0] == '1' && e[1] == '\0' && r4d_ds41_lcp() > 0;
    }();
    return on;
}

/* RADIANCE_EXPERIMENTAL_DS41_P1_TILED (tools/ds41-host-experts/ds41_p1.h kTiledEnv): "1" beside P1 puts the P1 twin's
 * calls on the register-tiled kernel (r4d_ds41_mxfp4.hip, mx_p1t_kernel: an expert's weights decoded once for 8
 * activation rows, K in chunks), whose every output is the same sum, operation for operation, as the twin's and
 * the decode band's. The row keeps its name; its `computes` names the form (kR4dDs41P1TiledMark), which the core
 * requires when it is asked for, and the row table says so once on stderr. Anything else, or "1" without P1, is
 * off here (the core refuses an invalid value and the switch without P1). Read once. */
constexpr const char* kR4dDs41P1TiledEnv  = "RADIANCE_EXPERIMENTAL_DS41_P1_TILED";
constexpr const char* kR4dDs41P1TiledMark = "the register-tiled P1 twin";

inline bool r4d_ds41_p1_tiled() {
    static const bool on = [] {
        const char* e = std::getenv(kR4dDs41P1TiledEnv);
        return e && e[0] == '1' && e[1] == '\0' && r4d_ds41_p1();
    }();
    return on;
}

/* RADIANCE_EXPERIMENTAL_DS41_P1_WMMA (tools/ds41-host-experts/ds41_p1.h kWmmaEnv): "1" beside P1 puts the P1 twin's
 * calls on the matrix-core kernel (r4d_ds41_mxfp4.hip, mx_p1w_kernel; ahead of the register-tiled switch). OPT-IN AND
 * NOT EXACT: every weight value (the e2m1 code, exact in bf16) and every bf16 activation is the exact twin's, and each
 * 32-value block's products are summed by v_wmma_f32_16x16x16_bf16 in f32, then scaled by the block's e8m0 and added
 * to the running f32 sum -- the same values and products, another f32 summation order, so an output may differ from
 * the decode band's by f32 rounding (mostly identical bf16, rare 1-ulp). The order is fixed per output (16-row tiles,
 * K blocks in increasing order, no split-K), so a row's bits depend neither on M, the grid, nor the other rows of the
 * pass. The row keeps its name; its `computes` names the form (kR4dDs41P1WmmaMark), which the core requires when it
 * is asked for. Anything else, or "1" without P1, is off here (the core refuses an invalid value, the switch without
 * P1, and a device that is not gfx12). Read once. */
constexpr const char* kR4dDs41P1WmmaEnv  = "RADIANCE_EXPERIMENTAL_DS41_P1_WMMA";
constexpr const char* kR4dDs41P1WmmaMark = "the WMMA P1 twin (inexact: f32 summation order)";

inline bool r4d_ds41_p1_wmma() {
    static const bool on = [] {
        const char* e = std::getenv(kR4dDs41P1WmmaEnv);
        return e && e[0] == '1' && e[1] == '\0' && r4d_ds41_p1();
    }();
    return on;
}

/* RADIANCE_EXPERIMENTAL_DS41_DENSE_TILED (tools/ds41-host-experts/ds41_p1.h kDenseTiledEnv): "1" puts gemm_nt_bf16f8
 * (the FP8-stored dense linears) above 64 rows on the LDS-tiled form (r4d_gemm_bf16f8_nt_m64.hip,
 * r4d_gemm_bf16f8_nt_tiled_kernel: gemm_bf16_nt_tiled's 128 x 128 / 64 x 64 blocks with the weight's E4M3 packets
 * staged in LDS and expanded to bf16 per fragment). EXACT: every WMMA operand is the m64 kernel's (the same fp8
 * expansion), each slice is one chain from zero over its k steps in ascending order and the slices are summed from
 * 0.f in ascending order with SK from the same law, so every output is gemm_bf16f8_nt_m64's bits (as
 * gemm_bf16_nt_tiled is gemm_bf16_nt_m64's). Taller passes only, N >= 128, K % 64, 16-byte activation rows, never
 * the chained launch. The row keeps its name; its `computes` names the form (kR4dDs41DenseTiledMark), which the core
 * requires when the switch is set. Anything else is off. Read once. */
constexpr const char* kR4dDs41DenseTiledEnv  = "RADIANCE_EXPERIMENTAL_DS41_DENSE_TILED";
constexpr const char* kR4dDs41DenseTiledMark = "the tiled bf16f8 GEMM above 64 rows (exact: the m64 kernel's bits)";

inline bool r4d_ds41_dense_tiled() {
    static const bool on = [] {
        const char* e = std::getenv(kR4dDs41DenseTiledEnv);
        return e && e[0] == '1' && e[1] == '\0';
    }();
    return on;
}

/* RADIANCE_EXPERIMENTAL_DS41_ATTN_WMMA (tools/ds41-host-experts/ds41_p1.h kAttnWmmaEnv): "1" puts attn_sparse_sink at
 * EVERY height on the matrix-core form (r4d_ds41_sparse.hip, ds41_attn_fast_kernel). OPT-IN AND NOT EXACT: the
 * scores' q.k products (bf16 x bf16, exact in f32) are summed by v_wmma_f32_16x16x16_bf16 in f32 (another order),
 * then everything the exact kernel does after the dot is done the same way: the scale multiply, the max floored at
 * -1e30, p = glibc's expf(s - max) (the same table and polynomial), sum p in key order from 0, exp(sink - max), the
 * division and one bf16 rounding. sum p k takes each f32 p as three bf16 pieces (hi + mid + lo == p exactly) whose
 * products with the bf16 key are exact, summed by WMMA in f32: the exact kernel rounds each p k product to f32
 * first, so that sum differs by summation order and the products' roundings (each within 2^-24 |p k|). A row's
 * bits depend only on the row (a workgroup a 16-head tile and row, no split over keys): batch-invariant at the
 * kernel and, since every height takes it, at the engine. Passes the form cannot take (no 16-byte q / out rows, or
 * a row's keys beyond its LDS) stay on the exact kernels. The row keeps its name; its `computes` names the form
 * (kR4dDs41AttnWmmaMark), which the core requires when the switch is set. Anything else is off. Read once. */
constexpr const char* kR4dDs41AttnWmmaEnv  = "RADIANCE_EXPERIMENTAL_DS41_ATTN_WMMA";
constexpr const char* kR4dDs41AttnWmmaMark = "the WMMA sparse attention at every height (inexact: f32 summation order)";

inline bool r4d_ds41_attn_wmma() {
    static const bool on = [] {
        const char* e = std::getenv(kR4dDs41AttnWmmaEnv);
        return e && e[0] == '1' && e[1] == '\0';
    }();
    return on;
}

/* The twin's bound on M: kR4dDs41P1MaxRows under P1, kR4dDs41LcpMax under LCP, kR4dDs41TallRows under EXACT_TALL
 * alone. */
inline int r4d_ds41_tall_rows() {
    return r4d_ds41_p1() ? kR4dDs41P1MaxRows : r4d_ds41_lcp() > 0 ? kR4dDs41LcpMax : kR4dDs41TallRows;
}
