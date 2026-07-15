# Code-Review Backlog

Open work only. This git-tracked file is the **cross-machine handoff** (office ↔
home ↔ P5000) -- per-machine Claude memory does NOT travel, so anything needed to
continue on another machine lives here, self-contained. Completed work is not
recorded here; it's in git history + the per-machine memory files.

Every fix must clear the gates in [`TDD_RULES.md`](TDD_RULES.md): `ctest` (all
pass) + `sesolver_vkcheck` (all PASS) + the `--selftest-*` arcs. Prefer RED-first
for any testable logic.

## Open items

- **[bug, 5090/Linux] Windowed-app GPU hang (10 s fence timeout → device lost).**
  On the RTX 5090 / Linux (driver 580.139.03, which DOES support Vulkan 1.4) the
  app hangs during the startup atlas-build frame; `submit_and_wait`'s 10 s
  `vkWaitForFences` times out and the device is gone. NOT reproducible on
  Windows/RTX 4060. **Synth is exonerated:** `sesolver_vkcheck` runs the EXACT
  `synthesize_state` op (compute, same `OneShot` on `ctx.queue`, vkcheck_main.cpp
  ~2075/2159/2400) and passes 100% on that same 5090 -- the synth kernel + submit
  are fine. So the hang is in the one thing vkcheck can't reach: the RENDER path
  (dynamic rendering, offscreen scene pass, present blit, demote-to-helper), which
  had **never** run validation layers -- vkcheck is compute-only and the app
  hardcoded `create(false)`. FIXED: the app now honors `SES_VK_VALIDATION=1`
  (windowed + headless, prints `[validation ON]`). Robustness already handled
  (`DeviceContext::device_lost` → CPU fallback, no crash/spam). ROOT hang OPEN.
  Next, on the 5090: (1) run `SES_VK_VALIDATION=1 ./sesolver_app` -- a
  barrier/sync2/dynamic-rendering VUID here (the `0c45c36` marching-cubes-barrier
  class, invisible on the tolerant 4060/Windows driver) is the lead; (2) report
  the FIRST `vk: ... VK_TIMEOUT in <method> (file:line)` line -- it names the exact
  hanging submit (atlas synth vs render vs present); (3) if it points at the
  raymarch, rebake volume.frag/slice.frag with `discard` disabled to isolate
  OpDemoteToHelperInvocation.

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
