# ---------------------------------------------------------------------------
# sesolver_core : the pure numerical/physics core.
#
# HARD RULE: this target must NEVER depend on Qt, OpenGL, or any windowing /
# rendering library. It contains only hand-written math, grid, field, FFT,
# operators, time propagation, potentials, and the pure-data side of
# visualization (e.g. marching-cubes geometry, transfer-function math).
# Everything here is reachable by unit tests with analytic oracles.
#
# Sources are added test-first through the TDD loop (see docs/TDD_RULES.md).
# `placeholder.cpp` only exists so the static library is linkable while the
# library is empty; delete it the moment the first real translation unit lands.
# ---------------------------------------------------------------------------
add_library(sesolver_core STATIC
    src/placeholder.cpp
)

target_include_directories(sesolver_core PUBLIC include)
target_compile_features(sesolver_core PUBLIC cxx_std_20)
target_link_libraries(sesolver_core PRIVATE ses_warnings)
