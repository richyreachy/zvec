# Partial RaBitQ scoring for Symphony QG

2026-10-27. This phase investigates the user's proposal to read part of a code and linearly expand its contribution. It does not run a new retrieval benchmark. The previous phase81 Hinge gate remains failed; the separate fresh-query M2 proposal has not run.

**The proposal is feasible, with additional error and implementation conditions.** The basic idea is existing work: [Split-RaBitQ, October 2026, Eq. (2) and Theorem 3.1](https://arxiv.org/html/2610.02652v1#S3) scales a prefix dot product by D/m, preserves the full-code denominator, and derives a prefix-specific probabilistic bound. Its proof uses random-rotation symmetry; a fixed structured rotation in our code needs a separate justification. The partial-prefix formula alone is therefore not a new paper contribution.

## What can be estimated

Write an ideal full-code score as

\[
f=C-\alpha\sum_{i=1}^{D}v_i s_i,\qquad s_i\in\{-1,+1\}.
\]

For a uniformly random subset S of m coordinates, independent of the fixed query and code,

\[
\widehat f_m=C-\alpha\frac Dm\sum_{i\in S}v_i s_i.
\]

This is conditionally unbiased for **the full-code score**. It does not establish unbiasedness for the true cosine distance. Keep C and the full-code scale/denominator unchanged; scaling the whole distance or re-estimating a denominator from the subset changes the estimator.

Let t_i=v_i s_i and V=Σ(t_i−mean(t))²/(D−1). Direct finite-population calculation gives

\[
\operatorname{Var}(\widehat f_m)=\alpha^2\frac{D^2}{m}
\left(1-\frac mD\right)V.
\]

If we sample groups rather than individual coordinates, this formula applies to the **group totals**. A 16-dimensional SIMD tile is one sampling unit; substituting 768 individual-coordinate variance terms would be incorrect. Fixed prefixes of a fixed index, optional stopping, and candidates selected through previous sampled scores require further arguments. The [theory review](rabitq_partial_score_theory_review.md) gives finite-population bounds, counterexamples, and confidence-budget conditions.

Quantization error, partial-computation error, and finite-precision execution error are different quantities. A bound on the first does not remove the latter two. When valid bounds exist, triangle inequality and a union bound combine them; independence is not required. A small per-comparison failure probability also does not automatically protect an entire graph traversal.

## How this maps to the current codec

The current native path rotates all 768 dimensions with a fixed signed Paley-Hadamard transform. It is not a Haar rotation and has no random output permutation. Its packed code for 32 neighbors is 3,072 bytes, followed by 256 bytes of full-code factors.

With J_m the integer sum of the first m/4 LUT entries for one lane, the existing full score is

\[
f=g+A+B(\Delta J_D+V_D+K_D),\qquad
K_D=-\tfrac12\sum_{i<D}q_i.
\]

For the minimal partial-query-dot variant, retain the full LUT, its scale Δ and full-code A/B. Using ρ=768/m and K_m=−½Σ_{i<m}q_i, evaluate

\[
\widehat f_m^{q}=g+A+B(\rho\Delta J_m+V_D+\rho K_m).
\]

The global LUT zero point makes ρ times its prefix offset equal V_D in ideal arithmetic. The precise FP32 operation order is part of the implementation and is tested separately. The m=768 branch must use the original path without rebuilding constants.

An initial mechanical run exposed a relevant implementation detail: the compiler fuses the written SIMD multiply/add intrinsics. Disassembly of the actual executable shows t=FMA(Δ,J,V_D), then t=round(t+K_D), base=round(A+g), and score=FMA(B,t,base). An oracle using separately rounded multiply/add operations therefore failed the bit gate. The first failure is preserved in `partial82_mechanics_v1`; the correction must model these existing FMA instructions, without changing the native kernel or relaxing exact equality.

The benchmark's recorded build command explains this behavior: its manual kernel build uses `-O3 -march=icelake-server` and does not inherit the CMake source setting `-ffp-contract=off`. The evidence therefore applies to this frozen benchmark binary. A separately built kernel must have its own rounding path verified.

Call the lower-level scan with short m but map factors at the original 768-dimensional offsets. Calling the ordinary batch wrapper with m would incorrectly read factors from the middle of the full code. Existing `QG_PROJECT_DIM` experiments re-encoded short residuals and changed factors; they did not test this unchanged-native-code variant.

## A graph-center correction worth testing separately

The ideal full score can also be written

\[
f=C-\alpha(q-c)^Ts=A_{full}-\alpha q^Ts,
\qquad A_{full}=C+\alpha c^Ts.
\]

Keeping A_full while sampling q adds noise before cancellation with the center. To sample the residual instead, use

\[
A_m=A_{full}+\alpha\left(\rho\sum_{i\in S}c_i s_i-c^Ts\right),
\quad
\widehat f_m^{res}=A_m-\alpha\rho\sum_{i\in S}q_i s_i.
\]

This equals C−αρΣ_S(q−c)_i s_i exactly in the ideal model. For fixed checkpoints, center-prefix corrections can be precomputed per edge. Four additional FP32 factors per edge would add 512 bytes per 32-neighbor block, before alignment; this is a measurable storage/bandwidth cost. Random subsets require corrections for their actual coordinates.

For uniform individual-coordinate sampling, the conditional variance uses q−c instead of q. With P_s=I−ssᵀ/D, it is no larger precisely when ||P_s(q−c)||²≤||P_s q||². For G uniformly sampled groups, instead define Q_g=Σ_group q_i s_i and C_g=Σ_group c_i s_i; compare ||P_G(Q−C)||² with ||P_G Q||², where P_G=I−11ᵀ/G. A small residual can help, but there is no universal gain. This is a control-variate adaptation, not a novelty claim, and it has not been integrated into retrieval.

## A deterministic alternative to probabilistic pruning

For each unread LUT group, precompute its minimum and maximum table entries. The known partial integer sum plus the unread minima/maxima brackets the full integer sum. Passing both endpoints through the same monotone FP32 score conversion brackets the native score; negative scales reverse the endpoints. No random-rotation assumption is needed. The integer accumulator must not overflow, and all finite-value and rounding conditions must be checked.

If both score endpoints map to the same actual FP32 bucket, that bucket is known without reading the tail. Otherwise continue scanning. Every queue comparison, LIFO tie, visited decision, and stopping condition that uses the score must retain its original semantics. A distance lower bound alone does not justify deleting a graph node, which might lead to useful neighbors.

These deterministic bounds may remain too wide until most dimensions have been read. A 32-lane SIMD block also may need to continue when only some lanes remain unresolved. This route trades extrapolation risk for additional interval work; its throughput benefit is unmeasured.

## Verification and next experiment

The exact-arithmetic fixture `partial82_math_v1/report.json` checks 64 synthetic populations, 16,320 exhaustive subsets, and 1,024 fixed-size mean/variance comparisons. It verifies both direct and center-residual estimators, the factor-correction identity, the full-dimension endpoint, and counterexamples for a fixed prefix, data-dependent stopping, and a control variate that increases variance. These checks verify algebra on finite cases, not a universal probabilistic theorem or a retrieval benefit.

The corrected native-kernel run passed in `partial82_mechanics_v2/proof.json` (SHA256 `63374a9a17d750ca7ac4c5caeb02d01a9d3756efeb717276b482f0c36181901c`). It covers all 6,625 training states from 3,072 queries and 20 centers, at m∈{128,256,384,512,768}, for a physical prefix and one fixed shuffled set of four-dimensional LUT groups.

| Mechanical check | Passing comparisons |
|---|---:|
| Full native score versus saved actual score bits | 212,000 |
| Actual shortened production scan versus scalar FMA reference | 2,120,000 |
| Packed-code nibble versus independent compact-code decoder | 217,088,000 |
| Two partial formulas versus scalar FP32 references | 4,240,000 |
| Integer-tail and actual-FP32 score interval containment | 2,120,000 |
| Full-dimension identity and interval collapse, both modes | 424,000 |

The shuffled group table is shared and frozen; this is a mechanics test, not repeated independent sampling or an empirical unbiasedness demonstration. Its groups are four dimensions, not the 16-dimensional SIMD sampling units discussed above. The two partial formulas are the centered signed-query contribution and a separate uncentered integer-LUT expansion; only the former is compared to the shortened production scan. Neither applies the proposed center-residual correction. The run uses only the existing phase81 training export. It does not consume validation labels, compute retrieval quality, or use reserved additional queries.

Source and dependency hashes, the first FMA mismatch, executable disassembly, original failure, and successful compile/run commands are retained. The archived successful C++/Python sources are byte-identical to the executed sources and already pass repository formatting/lint checks. There was no production-source change or altered baseline build.

The next retrieval experiment should freeze a small set of prefix lengths and include the full 768-dimensional control, then compare at matched Recall@10≥90%. Any candidate that survives screening still needs repeated paired timings and additional-query validation. First keep the existing prefetch policy fixed; then test prefix-aware prefetch separately. The current policy prefetches the entire code block, so less arithmetic alone does not imply proportional memory traffic savings. Full rotation, exact center distances, and other traversal work also remain.

There is no new QPS or Recall result in this phase. The 25% breakthrough criterion remains unmet. Potential research value would have to come from a demonstrably effective graph-specific estimator or decision certificate and its cost-aware implementation, beyond the already published prefix estimator.
