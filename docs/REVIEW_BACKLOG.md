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
  fault to the COMPUTE-side `normalize_buffer` (norm reduction, `vk_engine.ixx` `normalize_buffer`)
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
  `synthesize_state` (`vk_engine.ixx`) records the synth dispatch, then in a
  SEPARATE fenced submit `normalize_buffer` reads the same buffer with no
  device-side RAW `barrier_compute_to_compute`; likewise the norm-read → scale-write
  WAR. Visibility rides on the host fence, whereas the codebase's OWN idiom
  carries such an edge in-band (`barrier_transfer_to_compute` after the staging upload in `vk_engine.ixx`).
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
- **커버리지 보완(2026-09-28)**: 노출 멤버 k=0..4·기울어진 축까지 계약 확장. k≥2는 τ=2로 수렴 격자 상태에 0.99 겹침·E 3 mHa(2pπu 0.9936/2.4 mHa, 2sσg 0.9951/1.4, 2pσu 0.9959/0.7) — 결손은 표본화된 궤도의 **같은 대칭 상위 속박 상태 혼입**(2pπu 안의 3pπu, ΔE 0.23 Ha → 플러시당 e^{−0.47}); 하위 멤버 사전 플러시(6~40스텝)는 무효(σ 하위는 대칭상 직교). 완전 수렴엔 τ≈15(≈300스텝, 256³에서 ~5 s/멤버; 최상위 멤버는 ΔE→0이라 불가) → **H3(미결)**: 상위 멤버 τ 연장 vs 현 τ=2 유지(추천: 유지 + 잔차 문서화).
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

## 2026-10-09 전수 검수 — 미해결 항목 (수정분은 git log 참조)

전 소스(core/solver/viz/scenario/app/tests/shaders/build)를 검수한 결과 중 **이번
사이클에서 고치지 않은** 항목만 적는다. 확정 수정(코어 풀 예외 안전성·1D twiddle
캐시·UBO 슬롯 stride·비동기 배치 드레인·전역 변수 제거·core-only 빌드 복구 등)은
커밋 메시지에 있다. 우선순위순.

### A. 이식성·정합성 버그 (NVIDIA 밖에서 드러남)
- **[solver] 1D 디스패치 수가 Vulkan 보장 한계(65 535)를 넘는다.** `mul_groups_ =
  group_count(cells_)`는 256³에서 65 536, `fft_lines_[0]`도 65 536. MC 경로는
  `mc_nblocks_ > 65535`를 거르지만 핫패스는 안 거른다(NVIDIA 2³¹−1이 숨김).
  `maxComputeWorkGroupCount`를 조회하고 2D 디스패치 또는 grid-stride로.
- **[solver] 제출 간 가시성 규칙이 두 가지.** `Recorder{cb, true}`(배치 첫
  디스패치 무배리어)는 "이전 제출은 펜스 완료"를 전제하나 `norm_and_peak`·
  `project_psi`·`scale` 등은 선행 배리어를 둔다. `upload_raw`는 compute→transfer
  WAR 배리어가 없고 `relax_deflated_step` 체인(inner→subtract→renorm)도 무배리어.
  규칙 하나: psi/V를 만지는 모든 OneShot은 적절한 선행 배리어로 시작(스핀 엔진
  `measure_exact`·`chebyshev_step` 동일).
- **[solver] 실패 경로 정합성.** `synthesize_state_half`·`normalize_buffer`는
  제출 실패 후에도 성공을 보고; `step/driven_step/magnetic_step/write_psi_to_volume`은
  실패한 배치에도 `flip_volume()`; `transition_volume`은 기록 시점에 레이아웃
  메모를 갱신해 미제출 배치가 메모를 어긋나게 함. `vkWaitForFences` TIMEOUT을
  device-lost와 동일 취급(느린 GPU에서 10 s 배치는 손실이 아님).
- **[viz] 엔진→렌더러 교차 큐 읽기에 가시성 연산이 없다**(컴퓨트 전용 큐는
  FRAGMENT/VERTEX_INPUT 스테이지를 이름할 수 없음) → `render()` cb 첫머리에 전역
  `VkMemoryBarrier2` 하나. `dump_scene_bmp`는 transfer→host 배리어·invalidate 없이
  매핑 메모리를 읽음. `currentExtent == 0xFFFFFFFF`(Wayland)·`compositeAlpha`
  미확인; 스왑체인 재생성 시 `oldSwapchain` 미전달.
- **[solver] `project_deposit` groupshared 46 KB·`mc_scan` 1024 레인·subgroup
  arithmetic 요구**를 부팅 시 확인하지 않음(보장 최소 16 KB/128 레인).
  `check_device_features`에 바닥값 단언 추가.

### B. vkcheck 오라클 강도
- 스핀 검사 허용 오차가 **절대값**(2e-3, 2e-4, ~6e-5)인데 정규화 2¹⁶ 상태의
  진폭은 ≤5.4e-3 → 피크의 37 %/4 %/1 % 오차가 PASS. `ErrStats::tol(0, rel)`로.
- 마칭큐브 knife-edge 가드 `1e-9·peak`는 fp32 1 ulp(≈1.5e-8·peak)보다 작아
  정확-개수 단언이 잠재적 flaky → ≥1e-7·peak.
- fp16 왕복 허용 오차 5e-3(실제 반 ulp 2.4e-4); 파일에 이미 있는 `f16_quantize`
  비트 정확 오라클을 쓰면 됨. `flow_velocity` 커널은 오라클이 전혀 없음.
- `spin_fused_gate`/`spin_permute`는 베이크·검증되지만 소비자가 없고 주석은
  존재하지 않는 "Stage 2/3"을 설명. `check_lattice2d_size_guard`는 거부를 단언할
  수 없음(`set_lattice`가 void) → `[[nodiscard]] bool`.
- 13개 raw-kernel 검사가 같은 ~40줄 픽스처를 반복(~600줄) → `KernelFixture`.

### C. 성능 (측정 근거 있음)
- **[viz] 마커 VBO 매 프레임 재구축 + `vkDeviceWaitIdle`** — 회전자 씬은 핵
  마커가 매 프레임 움직이므로 구 메시 테셀레이션·버퍼 재생성·디바이스 idle이
  매 프레임. 인스턴스드 단위 구로. 나머지 5곳의 `vkDeviceWaitIdle`(resize/
  overlay/staging/mesh 성장)은 정확성에 불필요하며 엔진 비동기 큐를 세움.
- **[viz] 프레임당 직렬 펜스 대기 3회**(render submit_and_wait → presenter 펜스
  → 엔진 배치). scene+post+blit을 한 제출로, 펜스 대기는 다음 프레임 첫머리로.
- **[scenario] 스텝마다 전체 상태 왕복**: doubleslit은 화면 열 512개를 읽으려고
  매 스텝 4 MB 다운로드(x16에서 ~1 GB/프레임); qpc는 스텝마다 readback→CPU CAP→
  업로드. GPU 쪽 열 판독/마스크 reduction으로.
- **[core] 스레드 핫패스 안의 직렬 16.7M 패스**: `norm_sq/normalize/inner_product`
  (relax_deflated 스텝마다 디플레이션 벡터당 1회, `multi_quantum_jump` 채널당 1회),
  3D 관측량, `project_radial_angular`(36 Y_lm × 셀, vkcheck 오라클), `synthesize_h2plus`,
  `marching_cubes`, `apply_dipole_halfkick`(분리 가능·인접 킥 병합 가능),
  `build_half_potential_table/build_kinetic_table`. 전부 슬랩 순서 `parallel_for/
  parallel_sum`으로 비트 동일 가능.
- **[core] `rotation.ixx axis_shear`**: 라인마다 twiddle 재할당 + 셀마다 sincos
  (256³·3 shear·2회/스텝 ≈ 39만 회 twiddle 재구축). `magnetic.ixx`의 인접 반회전도
  병합 안 함(주석은 이번에 정정). `kick()`은 호출마다 268 MB 테이블 생성.
- **[solver] relax 테이블을 complex로 저장**(`damp_mul`로 R32 가능 → VRAM·대역폭
  절반); `relax_deflated_step`은 스텝당 (2L+3)회 펜스 제출; Chebyshev 루프는
  반복마다 버퍼 2개 복사(디스크립터 순환으로 0회).
- **[core] `ho1d_spectrum` O(n²·N)**(`ladder_fock`의 단일 체인 패턴 재사용);
  풀은 n이 작아도 모든 워커를 깨움(직렬 임계값 없음), 영역마다 `std::function` 할당.

### D. 설계·구조 (결단 필요)
- **[scenario] spins 정확 모드 틱당 8스텝 vs 평균장 20스텝** — "다이얼이 유일한
  배율" 계약 위반. baseline은 이번에 정직하게 고쳤으나 틱 자체는 그대로: 스텝 수를
  통일하고 CPU 정확 경로는 틱을 드롭할지 결정.
- **[scenario] corral `10.1734681`은 j₁,₃(J0 영점 아님)** — 펜스가 antinode에
  놓임; 주석 4곳이 각각 다른 말(j0_10, 3rd zero of J0 …). 의도한 k_F R을 정하고
  상수·주석 정렬.
- **[scenario] 셀프테스트**: 벽시계 데드라인 15곳 이상(`after(2000/30000/180000)`)
  이 "never wall-clock" 계약과 모순; `--selftest-qdot` 무제한 폴링(행 가능);
  `kArcSpecs`/`kArcs` 두 테이블이 강제되지 않음 → 한 테이블 + `static_assert`.
  프롤로그/에필로그 반복(~20/~38회).
- **[scenario] HydrogenDirector**: 40개 가상함수 API, `stepping_ = RealTime` 직접
  대입 13곳(NVI "모든 복귀 경로가 다이얼을 지운다"는 문서 주장과 불일치),
  base 본문 5개 중복. `FieldState/LaserDrive/AtlasBuilder` 추출 + `enter_real_time()`.
  `AtomModel`의 resident-cache API ~70줄과 hydrogen의 fp16 "atlas precision"
  로그는 orbital-free 리팩터 이후 데드 코드.
- **[scenario] 7개 디렉터 패밀리가 pacing/time_scale/stub 블록을 복제**(~50줄×6);
  `steps_per_tick` vs `steps_per_tick_x1` 이름 두 벌; `ses::Rung`/`ses_shell::Rung`
  중복; `ho_ladder(bool)`/`ladder(bool)`/`set_solenoid(bool cut_up)`/`Recorder{cb,true}`
  bool-trap; 2D 디렉터 3곳의 `{Engine, ok, dirty, rb}` 번들과 seed→normalize→
  dirty→peak 관용구 → `PlanarEngineSync`/`seed_plane`.
- **[solver] Engine 3.9k줄 god class**(OneShot 블록 34회, 디스크립터 쓰기 115회,
  수동 UBO flush 18회; `ses_vk::write_ubo`와 이름만 같은 멤버 `write_ubo`) →
  `submit_compute/update_ubo/wire` 헬퍼, 서브시스템 분리, `Kernel/DescriptorArena`
  RAII 래퍼(create/destroy 2단계 + 36줄 수동 teardown).
- **[core] 삼중대각 고유해법 3벌**(`radial`/`spheroidal`/`bloch`, 피벗 nudge·
  반복 횟수·시프트가 이미 드리프트) → `tridiag` 네임스페이스 하나.
- **[core/arch] `soft_coulomb_potential`이 export되어 vkcheck 픽스처 14곳·bench·
  단위 테스트에서 쓰임** — "No soft-Coulomb anywhere" 규칙과 코드가 모순.
  테스트 전용 모듈로 이동하거나 주석으로 한정.
- **[app] 패널 슬라이더 13개가 디렉터 진실을 읽지 않음**(+ 부팅 상수 12개 수동
  미러); 2-글자 capability 포워더 20개; `refresh_status()` 매 프레임(타이틀
  dirty 플래그 무력화); CLI 파싱 2벌(Boost + raw `starts_with`); 창 제목 수소 고정.
- **[build]** `ses_slang.cmake`의 slangc 다운로드에 `URL_HASH` 없음; `.clang-format`
  (ColumnLimit 100, BinPack false)이 실제 코드(80열 bin-packed) 174/190 파일과
  불일치 — 설정을 코드에 맞추거나 삭제; 기본 `CMAKE_BUILD_TYPE=Debug`는 README의
  "Debug 피하라"와 충돌; clang-scan-deps가 여전히 `-fcx-limited-range` 경고 출력.

### E. 작은 정확성 가드
- `absorbing_mask`: 주기 격자에서 `d_hi = xmax − x`라 셀 0과 셀 n−1이 한 셀
  비대칭. `quadratic_cap_mask`: k=0 평면만 채움(전제조건이 주석뿐). `vec normalized()`
  /`look_at` 퇴화 입력에서 NaN. `gen_h2plus_atlas` n==1 clamp UB. `sphere_mesh`
  rings<2 0으로 나눔. `photon_flight_frames` 0 → `progress` NaN.
- 벤치/테스트의 `soft_coulomb`, `bench_main`의 "per-frame hot path" 주석(CPU 경로).

### F. 테스트 스위트 (툴링 검수분, 미해결)
- **`ladder_test.cpp` `LadderCap.MatchesTheIndependentlyMeasuredCleanCap`은 동어반복**:
  `measure_clean_cap`이 `ladder_cap`과 같은 시드·같은 `ladder_raise`·같은 1e-6
  기준을 다시 쓴 것이라 `|diff| <= 1`이 실패할 수 없음. `ladder_cap`에는 물리
  오라클이 없음(측정 곡선 regression lock만) → 에너지 분산 기반 등 독립 오라클.
- **느린 테스트 10개가 직렬 191 s 중 140 s**(RotorEhrenfest 50 s, AtlasFlush 26 s,
  EggBox 16 s, RotorJMax 14 s …)인데 ctest `LABELS`/`TIMEOUT`이 없고, 같은 64³
  relax를 테스트마다 반복(`SetUpTestSuite` 없음). 기본 `CMAKE_BUILD_TYPE=Debug`
  강제는 README의 "Debug는 기어간다"와 충돌(`RelWithDebInfo` 또는 강제 해제).
- **중복 헬퍼**: `cube()` 8벌(시그니처 3종), `max_abs_diff` 3벌, `node_count`/
  `overlap_sq` 2~4벌, `expect_watertight` 2벌, 3D `energy_variance` 2벌(코어에는 1D만),
  손으로 만든 2D CAP 3벌·2D 패킷 필러 4벌·corral 링 2벌, `axpy` 루프 14곳 →
  `test_util.h`(매크로 게이트 대신 분리 헤더)로.
- **stdout 잡음**: 10개 파일 22곳의 `printf`가 녹색 실행에서 33줄 출력 → `SCOPED_TRACE`/
  `RecordProperty`. `spheroidal_test`/`projection_test`는 `<cstdio>` 없이 `printf`,
  `magnetic_test`는 `<vector>/<algorithm>` 없이 사용(전이 include 의존, MSVC 위험).
- **다중 관심사 테스트**(`potential_test` 412-458, `lattice2d_test` 208-262·376-446,
  `molecule_test` 32-83 …)와 근거 없는 허용 오차(`lattice2d_test` 0.4/0.15/0.2,
  `corral2d_test` 0.6×/2×, `rotor_test` 30..42)는 항목별 분리·유도.
- **빠진 음성 테스트**: `Grid1D{n<=0}`, 빈 벡터 `fft`, `absorbing_mask(width > box)`,
  `photon_flight_frames(de<=0)`, `bound_states_1d(count > n)`, `Scheduler::cancel(unknown)`,
  `every(period<=0)`, `ho_eigenstate(n<0)`, `h2plus_atlas_baked(R 범위 밖)`.
- `TEST_P` 후보: 디렉터 레지스트리 전체에 대한 pacing/`set_real_time`/`handle_key`
  계약(현재 23개 중 2개만), 해석적 우물 스펙트럼, 구면조화함수 점값 표.
- `radial_test.cpp` `SoftCoreAtomMatchesThe3DSolver`의 참조값은 이 프로젝트의 다른
  실행에서 읽은 숫자(외부 유도 없음) → regression lock으로 명시.
- `// RED:` 머리말 54개 파일, "bitwise"라 쓰고 `EXPECT_DOUBLE_EQ`(4 ULP)를 쓰는
  `propagator_tables_test`/`imaginary_tables_test` 주석, `complex_test`의 플래그
  이름 주석(계약이 아닌 플래그를 적음).
