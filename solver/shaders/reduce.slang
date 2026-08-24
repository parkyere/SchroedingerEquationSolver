// Preprocessor include, NOT an import module (macros do not cross `import`).
// 256-lane groupshared tree fold, fixed pair order => bit-exact partials.
// Contract: caller stores its lane accumulator to slot tid, then invokes;
// FOLD_OP(a, b) folds slot b into slot a. Barriers included on both sides.
#ifndef SES_REDUCE_SLANG
#define SES_REDUCE_SLANG

#define REDUCE_FOLD_256(FOLD_OP, tid)                                \
    GroupMemoryBarrierWithGroupSync();                               \
    for (uint half_len = 128u; half_len > 0u; half_len >>= 1u) {     \
        if (tid < half_len) {                                        \
            FOLD_OP(tid, tid + half_len);                            \
        }                                                            \
        GroupMemoryBarrierWithGroupSync();                           \
    }

#endif
