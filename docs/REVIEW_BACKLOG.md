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

## 미결 설계 결단 (2026-09-26, 사용자 판단 대기)

2026-09 사이클(대역제한 Coulomb → 회전 배치 → 해석적 Ehrenfest 힘 → 광자 방출 보존
감사)에서 리뷰 워크플로가 확정했으나 **설계 결단이 필요한** 항목. 확정 버그(스트릭
나선 반전, 기울기 오라클, 주석)는 처리 완료. A(회전자 총에너지 계약 = dt 수렴 +
2 mHa)·B(EigenstateFlush 0.98)·C(h/2 에너지 2 mHa)는 추천대로 확정(2026-09-26).
각 항목: 현상/증거 → 물리적 의미 → 옵션과 트레이드오프 → 추천 → 비용/계약.

### D. MCWF no-jump 감쇠의 8상태 절단
- **위치**: `scenario/src/hydrogen_director.ixx` `apply_mcwf_damping`
  (`kMcwfMaxStates = 8`), `solver/src/vk_engine.ixx` `kMcwfSlots = 8`
  (`mcwf_axpy.comp`의 슬롯 배열).
- **현상**: no-jump H_eff 감쇠 exp(−Γ_s dt/2)를 인구 ≥1e-3인 상태 중 **상위 8개**에만
  적용. 광자 점프 후 목적 상태가 9개 이상인 그룹(n=6→5: n=5의 25개 상태 대부분이
  한 진동수 그룹)에서 나머지 상태는 감쇠되지 않음 → 조건부 중첩의 상대 진폭
  (l-혼합·방향)이 정확한 H_eff 진화에서 벗어남. 누락 상태는 감쇠된 상태 대비
  e^{+Γ_applied t/2}만큼 상대 성장(수명 1회당 O(50%)); 3d→2p(목적 3개)는 무영향.
- **물리적 의미**: 점프 사이 상태의 "정직한 조건부 진화"가 큰 껍질에서만 근사.
  점프 확률(rates)은 전 상태 인구로 계산되므로 통계는 맞고 **상태 모양**만 왜곡.
- **옵션**: (a) 후보 전부를 8개씩 청크로 융합 디스패치 반복 — 코드 ~20줄, 8개 초과일 때만
  디스패치 추가; (b) `kMcwfSlots`를 32로 확장 — UBO·셰이더 배열 변경, 항상 큰 UBO;
  (c) 유지 + 문서화(현 상태).
- **추천**: (a). **계약(RED)**: vkcheck `mcwf_axpy` 25항 청크 결과 == CPU
  `ses::nojump_damped_amplitudes`(이미 단위 테스트됨) 전 상태 적용값; 디렉터 심은
  기존 `apply_mcwf_damping` 경로에 후보 수 25인 인구 벡터로 순회 검증.

### E. Larmor 방사 판독의 중심차분 스텐실 (대역제한 V에서 9% 저평가)
- **위치**: `solver/shaders/mean_force.comp`(셰이더 내 주기 중심차분),
  `vk_engine.ixx` `set_potential_gradient`/`mean_force`, 소비자 `base_director.ixx`
  `run_real_time_batch`(`radiated_power_`), `hydrogen_director.ixx` 동일; 타이틀
  "emit P".
- **증거(리뷰 재현)**: 1s+p_z 중첩·단일 대역제한 핵·h=0.3125: 해석 ⟨∇V⟩_z 0.25942,
  스텐실 0.24768(비 0.955 → P 비 **0.912**), 격자 정확 d⟨p⟩/dt 0.25955(1.0005).
  h=0.156: 스텐실 비 0.9865. 원인: sin(kh)/(kh) 전달함수가 Nyquist까지 내용이 있는
  대역제한 V를 깎음(회전자 힘에서 24% 저평가와 같은 기전; 회전자는 (b3)에서 해석
  힘으로 해결됨).
- **물리적 의미**: 표시 전용(P 값)이며 점프·상태에 무영향. 하지만 "Ehrenfest 가속"
  숫자가 격자 진실과 9% 다름.
- **옵션**: (a) 중심 목록(≤16, 전하 Z 포함) UBO를 받는 **해석 힘 커널**
  (`two_center_force` 일반화; `coulomb_mean_force` 합) → 수소(1)·H2+(2, 회전 시
  갱신)·벤젠(12)·러더퍼드(1)에 적용, 비Coulomb 씬(HO·우물·2D)은 스텐실 유지;
  (b) 스펙트럴 기울기(FFT·ik) 커널 — 임의 V에 격자 정확(1.0005), VkFFT 스크래치 3회/판독
  (판독은 타이틀 주기당 1회라 감당 가능); (c) 유지 + 라벨("stencil, −9% at h=0.31").
- **추천**: (a) — 단순하고 Coulomb 씬에 정확; (b)는 일반성이 장점이나 스크래치 관리
  필요. **계약(RED)**: vkcheck `mean_force(centers)` vs CPU `coulomb_mean_force` 합
  1e-5; 수소 아크에 "P 판독 = 해석값 ±1%".

### F. 점프 라벨이 조건부 방향을 지움
- **위치**: `hydrogen_director.ixx` `on_decay_jump` →
  `kStateSpec[dominant_to].name`; `core/src/emission.ixx`
  `collective_decay_interval` `dominant_to = argmax|c|`.
- **현상**: σ 광자(ẑ 방향) 후 상태 (2p_x + i·2p_y)/√2 가 "→ 2p_x"(|c| 동률은
  정렬 순서로 결정)로 표시. 스펙트로미터 라벨·stderr 로그도 동일. 상태 자체는 정확.
- **물리적 의미**: 표시가 "실수 tesseral 붕괴"처럼 읽혀 QED 조건부 상태(광자 방향
  기준 원자 각운동량)를 숨김.
- **옵션**: (a) "n=2 l=1, ⟨L·n̂⟩ = −1.00 ħ" 형식 — 광자 방향 n̂ 기준 원자 각운동량
  기대값(축상 검출 시 정수, 비축상 시 분수) 표시; 필요 코어: tesseral 기저 L 행렬로
  ⟨L·n̂⟩ 계산(단위 테스트: 축상 3d_z²→2p, λ=+1 → −1.000); (b) 성분 나열
  "0.71·2p_x + 0.71i·2p_y"; (c) 유지.
- **추천**: (a). 비용: 코어 함수 1개 + 단위 계약 2개 + 라벨 문자열.

### G. 선운동량(광자 반동) 미모델
- **현상**: 점프 경로에 반동(e^{−ik·r}) 없음, 핵은 고정 퍼텐셜 중심 → 총 선운동량이
  **정의되지 않음**(위반이 아니라 부재). 방출 패턴이 n̂→−n̂ 대칭이라 평균 반동 0;
  사건별 요동만 누락. 수치(Lyman-α): ħk = 2.7e-3 au(전자 운동량 폭 1 au의 0.3%),
  핵 반동 에너지 2.0e-9 Ha(선폭 대비 무시), 이동 1.5e-6 bohr/au(1셀 이동에 광자
  수천 개).
- **옵션**: (a) 문서화만(모델 한계; 핵 자유도가 있는 회전자만 J 부기);
  (b) 질량중심 P 장부: 광자당 −ħk n̂ 누적을 패널에 표시(퍼텐셜은 그대로 — 이동이
  표시 불가 크기); (c) 전자에 e^{−ik·r} 킥 — **기각**(반동은 질량중심 몫, 전자에
  주면 물리 오류).
- **추천**: (a)+(b). 비용: Vec3 누산 + 패널 한 줄 + 단위 계약(광자 1개 후 P = −ħk n̂).

### H. 아틀라스 합성 상태의 연속체 후광 — **완료 2026-09-26** (하위 결정 H2 미결)
- **결과**: 프로브가 크기를 정정 — 합성 1sσg는 격자 바닥보다 **145 mHa** 위(Var 1.8 Ha², 연속체 3.7%), σu* 30 mHa. GREEN: `prepare(k)` 후 GPU ITP τ=2(`ses::kH2plusAtlasFlushSteps` 40 × `kH2plusAtlasFlushDtau` 0.05), 하위 멤버(현재 축으로 합성) 딜플레이션, 테이블은 `current_potential()`; 겹침 0.9998, 격자 바닥 0.2 mHa; h2p 아크 격자 ⟨H_el⟩(σu*) −0.6555(Δ 2.9 mHa), 회전자 아크 ⟨H_el⟩ 드리프트 2.7e-4 → **2.0e-5 Ha**.
- **H2(미결)**: S 랜덤 중첩은 미플러시(타이틀 "(sampled, unflushed)"). 옵션: (a) 멤버별 플러시 상태 캐시(134 MB/멤버, 축 회전 시 무효화) 후 중첩 — VRAM 정책 확인 필요; (b) 짧은 τ=0.3 플러시(Var 96% 제거, 가중 ≤13% 변화, 라벨 필요); (c) 유지. 추천 (a).
- (원안 기록) - **증거(계측)**: H2+ 아틀라스 σg 합성 직후 전체 ⟨r²⟩ 3 → 53 bohr²(100 au 사이),
  P(r<4 bohr) 0.977 → 0.960; 대역제한 격자 바닥과 겹침 0.989(EigenstateFlush).
  코어 96%는 핵을 따라 정확히 회전(회전 가시성 문제와 무관).
- **물리적 의미**: 첨점 표본 상태의 격자 비고유 성분이 상자로 퍼져 흡수체 없는 주기
  상자를 돌며 리플을 만듦(비물리 표시 잡음). 아틀라스 표시 에너지(정확 −1.1350)와
  격자 ⟨H⟩(≈−1.115, +20 mHa 이산화)의 불일치는 별개.
- **옵션**: (a) 합성 직후 짧은 ITP 플러시(수소 post-collapse flush 패턴; 6스텝
  dτ 0.05 = τ 0.3은 첨점 교정용이라 후광 제거엔 τ≈1~2 필요 — 측정 후 결정);
  (b) 유지(iso 25% peak 표시라 후광은 거의 안 보임); (c) 아틀라스 대신 격자 relax
  상태 표시(에너지 표시도 격자값으로).
- **추천**: (a), τ는 RED 계약으로 결정: "플러시 후 P(r<4) > 0.99, 격자 E 변화 < 5 mHa".

### I. 러더퍼드 backscatter 정의
- **위치**: `scenario/src/rutherford3d_director.ixx` `after_step_batch`:
  `back_ = max_t Σ_{x<−30}ρ / Σρ`(남은 norm 대비 왼쪽 층 비율의 시간 최대).
- **문제**: "반사 확률"이 아님 — 흡수체(폭 10)가 반사 패킷을 먹는 동안 비율이 오르고,
  흡수 후반에는 잔류에 의존. 아크 계약 `back > 0.1`(측정 0.15)는 이 정의에 묶임.
- **옵션**: (a) 벽별 흡수 손실 tally(absorb 커널이 벽별 부분합 출력) → 반사 =
  −x 벽 흡수 누적, 물리량("θ>90° 산란 확률")로 승격 + 고전 Rutherford
  P(θ>90°)와 비교하는 아크 계약; (b) 유지 + 라벨 정정("left-layer share");
  (c) 삭제.
- **추천**: (a). 비용: absorb 커널 부분합(6벽) + 엔진 판독 + 디렉터 tally + 아크 계약.

### J. "전자 구름이 핵과 함께 돌지 않는다" 관찰 (정보 대기)
- **계측 결과**: 실제 씬 경로(--selftest-rotor)에서 코어(r<4) 2차 모멘트 주축이 핵 축과
  함께 z → −y로 90° 회전(⟨yz⟩ 부호·크기 일치), ⟨H_el⟩ 드리프트 2e-4 Ha(구름이 90°
  어긋나면 ~0.5 Ha 상승해야 함) → **물리 ψ는 따라 돎**. 브리지(psi→볼륨)·플립·
  Surface `mc_dirty_` 경로도 매 배치 작동.
- **확인 필요(4가지)**: ① 표시 궤도가 1sσg였는지(종횡비 1.26의 거의 구형이라 회전이
  잘 안 보임; π 궤도/S 랜덤은 뚜렷); ② 시간 배율·킥 크기(1 ħ: ω≈3e-4 → 한 바퀴 14분;
  J=35·배율 1: 1.5°/s); ③ 실행 exe가 b2·b3 이후 빌드본인지; ④ 뷰 모드와 "정지"
  vs "퍼짐"(후자는 H의 후광).
- **가능한 개선(지시 시)**: 축 방향 표시선, 기본 궤도를 π로 안내, H의 플러시.
