# Code-Review Backlog

Open work only. This git-tracked file is the **cross-machine handoff** (office ↔
home ↔ P5000) -- per-machine Claude memory does NOT travel, so anything needed to
continue on another machine lives here, self-contained. Completed work is not
recorded here; it's in git history + the per-machine memory files.

Every fix must clear the gates in [`TDD_RULES.md`](TDD_RULES.md): `ctest` (all
pass) + `sesolver_vkcheck` (all PASS) + the `--selftest-*` arcs. Prefer RED-first
for any testable logic.

## Open items

- **[driver bug, 5090/Linux] Vulkan 1.4 hangs the GPU; running at 1.3 as the
  workaround.** At `apiVersion = VK_API_VERSION_1_4` the app device-losts on NVIDIA
  580.139.03 / Blackwell / Linux; at 1.3 it works (confirmed by 1-line A/B). Root-
  caused (3 independent audits): the `source_location` diagnostic pins the FIRST
  fault to the COMPUTE-side `normalize_buffer` (norm reduction, vk_engine.hpp:2659)
  during the startup atlas build -- **NOT the render path** (the earlier "render/
  present" hypothesis is REFUTED). ONLY the instance apiVersion enum differs
  (identical enabled features + SPIR-V, no 1.4-exclusive feature; VMA/ImGui pinned
  to the device's REAL version); the spec lets a driver steer behavior on that
  integer, so 1.4 selects a broken NVIDIA driver path (GSP CTX_SWITCH_TIMEOUT / Xid
  109 class). No app-side hazard can produce the hang. **1.3 loses NOTHING** -- no
  1.4-exclusive feature is used, so this is a non-blocking workaround, not a
  regression. To pursue 1.4 later: (1) UPDATE the 580 driver -- the branch has a
  cluster of Blackwell Vulkan device-lost fixes across point releases
  (580.65.06 / .82.07 / .105.08+) -- and retest; (2) rebuild `sesolver_vkcheck` at
  1.4 (one-line flip, shared Boot path) -- HANG ⇒ minimal compute-only repro to file
  with NVIDIA; PASS ⇒ trigger needs the presenting device + graphics/async-compute
  concurrency, mitigate in-app (skip present during atlas build, or run the startup
  reduction on the graphics queue). Adjunct: run the app at 1.4 with
  `SES_VK_VALIDATION=1` -- validator silent on the normalize submit ⇒ driver-bug
  confirmed. 1.4 lives isolated in commit `649826b` for easy A/B.

- **[correctness] Missing compute→compute barrier between atlas synth and norm.**
  `synthesize_state` records the synth dispatch (vk_engine.hpp:2628), then in a
  SEPARATE fenced submit `normalize_buffer` reads the same buffer (2657) with no
  device-side RAW `barrier_compute_to_compute`; likewise the norm-read → scale-write
  WAR (2689). Visibility rides on the host fence, whereas the codebase's OWN idiom
  carries such an edge in-band (`barrier_transfer_to_compute`, vk_engine.hpp:2476).
  Benign at 1.3 (a WRONG-DATA risk, not the hang), but a real spec gap -- fix it to
  match the idiom; it also doubles as a driver-exoneration test at 1.4.

- **[verify] GPU marching-cubes oracle on 5090/Linux.** The cyclic-hue colour
  metric + valid sort key (a discontinuous-wheel abs-RGB compare false-failed on
  the RTX 5090) is fixed but *unconfirmed on that hardware* -- could not reproduce
  on the RTX 4060. Needs a Linux/5090 `sesolver_vkcheck` re-run to close.

- **[low, deferred] Extract a `MeasurementEngine`** from `HydrogenDirector`
  (`run_partial_measure` / `rebuild_psi_from` / `project_manifold_out`, ~180
  lines). Deferred: the shared `cpu_is_truth_`/display-bridge/engine coupling
  makes the extraction low-cohesion -- it would need a fat back-reference into
  `HydrogenDirector`, so the churn on the (refactored, manually-verified) shell is
  not worth the negligible cohesion gain. Revisit only if that coupling is broken
  first.

- **[physics] No quantitative `⟨L_z⟩` / probability-current diagnostic, and the
  m-sign handedness is untested.** The ±m ring states are preparable (via the L_z
  partial measurement) and the flow streaklines *visualize* the current, but there
  is no `⟨L_z⟩` number and the m-sign's absolute handedness vs Larmor rotation
  under B is unverified. A B-on ring-rotation check would pin the handedness.

## See also

- [`TDD_RULES.md`](TDD_RULES.md) — the verification gates every fix must clear.
