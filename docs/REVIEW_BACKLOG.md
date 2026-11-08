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
  (identical enabled features + SPIR-V, no 1.4-exclusive feature; VMA and ImGui
  are given the instance's 1.3); the spec lets a driver steer behavior on that
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

- **[verify] GPU marching-cubes oracle on 5090/Linux.** The cyclic-hue colour
  metric + valid sort key (a discontinuous-wheel abs-RGB compare false-failed on
  the RTX 5090) is fixed but *unconfirmed on that hardware* -- could not reproduce
  on the RTX 4060. Needs a Linux/5090 `sesolver_vkcheck` re-run to close.

- **[low, deferred] Extract a `MeasurementEngine`** from `HydrogenDirector`
  (`run_partial_measure` / `rebuild_psi_from` / `sin_partner`, ~140
  lines; `project_manifold_out` already lives in `BaseDirector`). Deferred: the shared `cpu_is_truth_`/display-bridge/engine coupling
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
  `vk_engine.ixx` `enable_mean_force`/`mean_force`(이제 `v_buf_`를 직접 읽음), 소비자 `base_director.ixx`
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
- **정정(2026-11-07)**: 아래의 145 mHa·Var 1.8 Ha²·연속체 3.7 %와 k≥2의 겹침 0.99·
  E 3 mHa는 대부분 baked 아틀라스 **로더의 좌표 규약 버그**를 잰 값이다(생성기와 로더가
  ξ/η 표본 위치를 다르게 해석해 프로파일이 ~0.43 ξ 밀리고 첨점에 0.4 bohr 평탄부;
  `14702f8`에서 수정, `BakedGroundIsTheLiveGroundOnTheGrid`가 고정). 수정 후 합성
  1sσg는 격자 바닥보다 **~11 mHa** 위(Var **~0.7 Ha²**), 플러시 후 겹침 > 0.9999·
  |E − E_grid| < 1 mHa; k = 2..4 플러시 잔차(1 − 겹침)는 1e-5 / 1.3e-4 / 1.4e-3.
  따라서 **H3(상위 멤버 τ 연장)은 사실상 불필요**하고, H2의 크기 근거도 이 값으로 다시 잴 것.
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
  group_count(cells_)`는 256³에서 65 536, `fft_lines_[0..2]` 모두 65 536. MC 경로의 블록 가드(`mc_nblocks_ > 65535`)는
  classify/emit만 보호하고 `mc_density`·`bridge_store`도 `mul_groups_`로 디스패치(NVIDIA 2³¹−1이 숨김).
  `maxComputeWorkGroupCount`를 조회하고 2D 디스패치 또는 grid-stride로.
- **[solver] 제출 간 가시성 규칙이 두 가지.** `Recorder{cb, true}`(배치 첫
  디스패치 무배리어)는 "이전 제출은 펜스 완료"를 전제하나 `norm_and_peak`·
  `project_psi`·`scale` 등은 선행 배리어를 둔다. `upload_raw`는 compute→transfer
  WAR 배리어가 없고 `relax_deflated_step` 체인(inner→subtract→renorm)도 무배리어.
  규칙 하나: psi/V를 만지는 모든 OneShot은 적절한 선행 배리어로 시작(스핀 엔진
  `measure_exact`·`chebyshev_step` 동일).
- **[solver] 실패 경로 정합성.** `transition_volume`은 기록 시점에 레이아웃
  메모를 갱신해 미제출 배치가 메모를 어긋나게 함(실패한 배치 뒤의 표시 flip과
  synth/normalize의 성공 오보고는 이번에 수정). `vkWaitForFences` TIMEOUT을
  device-lost와 동일 취급(느린 GPU에서 10 s 배치는 손실이 아님).
- **[viz] 엔진→렌더러 교차 큐 읽기에 가시성 연산이 없다**(컴퓨트 전용 큐는
  FRAGMENT/VERTEX_INPUT 스테이지를 이름할 수 없음) → `render()` cb 첫머리에 전역
  `VkMemoryBarrier2` 하나. `dump_scene_bmp`는 transfer→host 배리어·invalidate 없이
  매핑 메모리를 읽음. `currentExtent == 0xFFFFFFFF`(Wayland)·`compositeAlpha`
  미확인; 스왑체인 재생성 시 `oldSwapchain` 미전달.
- **[solver] `project_deposit` groupshared 46 KB·`mc_scan` 1024 레인·subgroup
  arithmetic 요구**를 부팅 시 확인하지 않음(보장 최소 16 KB/128 레인).
  `DeviceContext::create_device`의 1.3/sync2 부팅 게이트 옆에서 `maxComputeSharedMemorySize`·`maxComputeWorkGroupInvocations`·`subgroupSupportedOperations`를 읽어 거부(또는 폴백)하고, vkcheck `check_device_features`에도 같은 바닥값 단언.

### B. vkcheck 오라클 강도
- 스핀 검사 허용 오차가 **절대값**(2e-3, 2e-4, ~6e-5): 피크 성분 대비 chebyshev 2e-3/4.1e-2 ≈ 5 %(느슨), engine-step 2e-4/8.6e-2 ≈ 0.2 %(적정; 둘 다
  Néel 곱상태 출발), gates 6.1e-5/4.6e-3 ≈ 1.3 % → chebyshev·gates부터 `ErrStats::tol(0, rel)`로.
- 마칭큐브 knife-edge 가드 `1e-9·peak`는 fp32 1 ulp(≈1.5e-8·peak)보다 작아
  정확-개수 단언이 잠재적 flaky → ≥1e-7·peak.
- fp16 왕복 허용 오차 5e-3(|x|<0.5의 법정 오차는 RTE 1.2e-4·RTZ 2.4e-4); `check_engine_bridge`처럼
  `f16_quantize`(RTE) 또는 RTZ 양자화와의 비트 일치를 요구할 것(RTE 단독은 RTZ 구현에서 거짓 FAIL). `flow_velocity` 커널은 오라클이 전혀 없음.
- `spin_fused_gate`/`spin_permute`는 베이크·검증되지만 런타임 소비자가 없음("Stage 2/3"
  주석은 정정됨). `check_lattice2d_size_guard`는 거부를 동작으로 이미
  단언(가드 제거 시 비트 비교 FAIL); 남은 문제는 `set_lattice`가 void라 디렉터가 거부를 모름 →
  `[[nodiscard]] bool`(2026-11-07 A와 같은 계약).
- 13개 raw-kernel 검사가 같은 ~40줄 픽스처를 반복(~600줄) → `KernelFixture`.

### C. 성능 (측정 근거 있음)
- **[viz] 마커 VBO 매 프레임 재구축 + `vkDeviceWaitIdle`** — 회전자 씬은 핵
  마커가 매 프레임 움직이므로 구 메시 테셀레이션·버퍼 재생성·디바이스 idle이
  매 프레임. 인스턴스드 단위 구로. 나머지 5곳의 `vkDeviceWaitIdle`(resize/
  overlay/staging/mesh 성장)은 정확성에 불필요하며 엔진 비동기 큐를 세움.
- **[viz] 프레임당 직렬 펜스 대기 3회**(render submit_and_wait → presenter 펜스
  → 엔진 배치). scene+post+blit을 한 제출로, 펜스 대기는 다음 프레임 첫머리로.
- **[scenario] 스텝마다 전체 상태 왕복**: doubleslit은 화면 열 512개를 읽으려고
  매 스텝 2 MiB fp32 다운로드 + 4 MiB double 언팩, 스텝마다 펜스 제출(x16의 256스텝/프레임에서 ~512 MiB/프레임); qpc는 스텝마다 readback→CPU CAP→
  업로드. GPU 쪽 열 판독/마스크 reduction으로.
- **[core] 스레드 핫패스 안의 직렬 16.7M 패스**: `norm_sq/normalize/inner_product`
  (relax_deflated 스텝마다 디플레이션 벡터당 1회, `multi_quantum_jump` 채널당 1회),
  3D 관측량, `project_radial_angular`(36 Y_lm × 셀, vkcheck 오라클), `synthesize_h2plus`,
  `marching_cubes`, `apply_dipole_halfkick`(분리 가능·인접 킥 병합 가능),
  `build_half_potential_table/build_kinetic_table`. 전부 슬랩 순서 `parallel_for/
  parallel_sum`으로 비트 동일 가능.
- **[core] `rotation.ixx axis_shear`**: 라인마다 twiddle 재할당 + 셀마다 sincos
  (256³·3 shear·2회/스텝 ≈ 79만 회 twiddle 재구축(라인당 fft+ifft 2회)). `magnetic.ixx`의 인접 반회전도
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
  불일치 — 설정을 코드에 맞추거나 삭제;
  clang-scan-deps가 여전히 `-fcx-limited-range` 경고 출력.

### E. 작은 정확성 가드
- `absorbing_mask`: 주기 격자에서 `d_hi = xmax − x`라 셀 0과 셀 n−1이 한 셀
  비대칭. `quadratic_cap_mask`: k=0 평면만 채움(전제조건이 주석뿐). `vec normalized()`
  /`look_at` 퇴화 입력에서 NaN. `gen_h2plus_atlas` n==1 clamp UB. `sphere_mesh`
  rings<2 0으로 나눔. `photon_flight_frames`가 극소 양수 dE에서 `lround`→`int`로 넘침(spawn은 < 1만 1로 clamp하므로 NaN은 불가, 거대값은 그대로).
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
- **중복 헬퍼**: `cube()` 8벌(시그니처 3종) + 이름만 다른 같은 헬퍼 3벌(`imaginary_tables_test`/`propagator_tables_test`의 `cube4()`, `marching_cubes_test`의 `cube_grid()`), `max_abs_diff` 3벌, `node_count` 3벌
  (`radial_test`의 무임계 `count_interior_sign_changes` 포함), |⟨a|b⟩|² 6벌(`overlap_sq`·
  `overlap2`·3D `population`), `expect_watertight` 2벌, 3D `energy_variance` 2벌(코어에는 1D만),
  손으로 만든 2D CAP 3벌·2D 패킷 필러 4벌·corral 링 2벌, `axpy` 루프 14곳 →
  `test_util.h`(매크로 게이트 대신 분리 헤더)로.
- **stdout 잡음**: 9개 파일 21곳의 `printf`가 녹색 실행에서 33줄 출력 → `SCOPED_TRACE`/
  `RecordProperty`.
- **다중 관심사 테스트**(`potential_test` 412-458, `lattice2d_test` 208-262·376-446,
  `molecule_test` 32-83 …)와 근거 없는 허용 오차(`lattice2d_test` 0.4/0.15/0.2,
  `corral2d_test` 0.6×/2×, `rotor_test` 30..42)는 항목별 분리·유도.
- **빠진 음성 테스트**: `Grid1D{n<=0}`, 빈 벡터 `fft`, `absorbing_mask(width > box)`,
  `photon_flight_frames(de<=0)`, `bound_states_1d(count > n)`, `Scheduler::cancel(unknown)`,
  `every(period<=0)`, `ho_eigenstate(n<0)`.
- `TEST_P` 후보: 디렉터 레지스트리 전체에 대한 pacing/`set_real_time`/`handle_key`
  계약(현재 23개 중 2개만), 해석적 우물 스펙트럼, 구면조화함수 점값 표.
- `radial_test.cpp` `SoftCoreAtomMatchesThe3DSolver`의 참조값은 이 프로젝트의 다른
  실행에서 읽은 숫자(외부 유도 없음) → regression lock으로 명시.
- `// RED:` 머리말 54개 파일, "bitwise"라 쓰고 `EXPECT_DOUBLE_EQ`(4 ULP)를 쓰는
  `propagator_tables_test`/`imaginary_tables_test` 주석, `complex_test`의 플래그
  이름 주석(계약이 아닌 플래그를 적음).

## 2026-11-07 재검수 — 미해결 항목 (수정분은 git log 참조)

이번 사이클은 GPU 계층(solver/viz/scenario + vkcheck + 디렉터 테스트)을 **clang-18로
컴파일·테스트**할 수 있었다(시스템 Vulkan 헤더/volk/VMA/VkFFT + SPIR-V 헤더용 slangc
스텁; SDL3 app과 Slang 셰이더 컴파일·vkcheck 실행은 제외). 아래는 확인했지만 고치지
않은 항목이다. 우선순위순.

### A. 정확성·견고성
- **[scenario/solver] device-lost 뒤 2D·spins 디렉터가 동결 스냅샷을 재생.**
  `Lattice2DEngine`/`SpinEngine`의 void API가 제출·복사 실패를 버리고(`submit_and_wait`,
  `copy_sync` 결과 무시), 디렉터들은 `device_lost()`를 보지 않는다 → 매 프레임 마지막
  다운로드가 psi_/tally에 다시 쓰이고 sim_time만 진행. 이번에는 "set_lattice 할당 실패 →
  null `state()` 역참조" 경로만 막았다(`ready()` 게이트). 계약: hydrogen처럼 첫 실패에서
  CPU로 강등, 엔진 API는 `[[nodiscard]] bool`.
- **[scenario] `selftest_scene_wait_running`이 교체 전 디렉터의 시계로 앵커.**
  `request_scene`은 다음 루프 머리에서야 교체하므로 0번째 폴링은 나가는 장면의
  `sim_time`을 앵커로 잡는다(--selftest-scene에서 앵커가 홉마다 누적, 느린 GPU 장면이
  STALLED 오판 가능, 교체가 지연되면 이전 장면의 진행으로 PASS). 셸에 "요청 장면이
  적용됨" 접근자가 필요(app 수정) — 호출부 중 `"switch-target"`처럼 장면 이름이 아닌
  레이블도 있으니 인자 의미부터 정리.
- **[scenario] H2+ 아틀라스 플러시 중 key 1 / M이 `abort_flush` 없이 플러시를 버림**:
  "(flushing...)" 태그와 `prepared()==false`가 남고 deflation 버퍼·134 MB `flush_v_`를
  다음 선택까지 보유(물리 영향 없음).
- **[scenario] Surface 진입 시 `mc_prepare` 실패(VRAM)면 메시지 없이 빈 화면**:
  `stage_active_view`가 `mc_ready()` 대신 `gpu_ok_`를 본다. 강제 Cloud 전환 6곳
  (hydrogen D/5/K/L/3-4, molecule `start_relax_for`)은 `release_mc`를 건너뛰어 ~175 MB가
  Tab-Tab/장면 전환까지 잔류(가시 영향 없음).
- **[core] `radial_eigenstate`의 부호 규칙이 u[0]을 본다** — 금지 영역이 깊은 상자에서
  u[0]은 ~e^-270 깊이라 홀수 상태의 부호가 반올림 오차로 정해진다. doublewell은 이번에
  소비자 쪽에서 방향을 고정했다(`DoubleWell1DDirector.PreparesTheLeftWellForEveryBarrier`).
  부호에 의존하는 다른 소비자가 생기면 "최댓값의 1e-6을 처음 넘는 표본" 기준으로 바꿀 것.
- **[app] Release의 WIN32 서브시스템이 `--help`·스펙트럼 표·selftest PASS/FAIL·종료
  코드를 숨김**(cmd/PowerShell에서 실행 시 콘솔 없음, 대기 안 함). `_WIN32`에서
  `AttachConsole(ATTACH_PARENT_PROCESS)` + stdout/stderr 재연결, 또는 headless/selftest용
  콘솔 실행 파일 분리.
- **[app] `init_compute(ctx, device_ok, ...)`의 `device_ok`는 유일한 호출부에서 리터럴
  `true`**(bool trap, 흔적 매개변수) — 제거하거나 실제 값을 전달.

### B. 성능
- **[scenario/core/viz] 2D 표면 장면이 매 배치/프레임 전체를 다시 만든다**: carpet은
  1024² heightfield 전체 재구축 + 직렬 1M 셀 max 스캔(행은 프레임당 ~6줄만 바뀜; 쓰이지
  않던 `track_peak` 스캔은 이번에 제거); `heightfield_surface`는 셀당
  6정점 비인덱스 soup(~113 MB double)을 매번 새로 할당; `upload_mesh`는 스텝 프레임마다
  ~56 MB 임시 interleave 벡터. → 변경 행만 갱신, 인덱스 메시, 버퍼 재사용.
- **[scenario] billiard/qdot/landau(와 corral relax)도 청크마다 512² 전체 GPU↔호스트
  왕복** — 2026-10-09 C의 "스텝마다 전체 상태 왕복"(doubleslit/qpc)에 추가되는 위치.
- **[core] `axis_shear`는 직렬** gather-FFT-phase-scatter로 `fft_detail::strided_line_pass`의
  스레드 패턴을 재구현하지 않음 — 실사용 경로는 CPU fallback의 1회성 `rotate_state`(256³
  ~1 s)뿐.

### C. 테스트·오라클
- **[tests] `mcwf1d_test`가 ω = 1에서만 돈다**: ω 곱셈/나눗셈 위치 오류가 전부 통과하고,
  n-가중 점프율은 고정되지 않으며(u = 0/1 극단만), 코드가 EXACT라 부르는 감쇠 비율에
  ±30 % 허용. 장면 값 ω = 0.25(또는 {0.25, 2})로, 문턱 양쪽 u로 점프 결정 반전을 단언,
  허용 오차 ~1e-3 상대.
- **[solver] ctest `VkCheck`는 validation layer를 켜지 않는다**(문서는 이번에 정정):
  vcpkg `validation` feature가 있으면 `VkCheck.validation` 항목에
  `SES_VK_VALIDATION=1`·`VK_LOADER_LAYERS_ENABLE=*validation*`·`VK_ADD_LAYER_PATH` 설정.
- **[viz] 렌더 커널(accum/bloom/compose/occupancy/occ_dilate/shadow/particles)에 GPU
  오라클이 없고 `--dump-frame*` 아크는 아무 이미지나 통과**(문서는 이번에 정정):
  vkcheck에 viz 절, `dump_and_exit`에 장면별 이미지 통계(점등 비율·평균 휘도) 단언.
- **[vkcheck] 회전 배치 리그가 주석보다 ~100배 느리게 돈다**(관성 1/dth → |ω|·dt =
  2e-4 rad/step; lag-by-one 등 주석의 수치가 다른 리그를 설명); **MC 오라클의 1e-3 반올림
  순위 키가 fp32/fp64 정점 오차에 불안정**("both sides round identically" 주석은 거짓).
- **[tests] 계약 문구 불일치**: anderson1d 테스트 머리말·이름("SubEnergy")이 실제 지형과
  다름; `test_util.h`가 존재하지 않는 디렉터 테스트(kTrapBox 고정)를 언급; `rotation_test`는
  y축 회전 방향을 고정하지 않음(유일한 axis=1 테스트가 y 대칭 장).

### D. 설계·작은 가드 (LOW)
- **[core]** `measure(u, sigma_m)` ↔ `sample_povm_index(psi, sigma_m, u)` 인접 계층에서
  같은 타입 double 두 개의 순서가 반대; `obs_ratio`가 NaN 분모를 정확히 0으로 바꿔 NaN psi가
  E = 0 eV로 보임; `chunk_count`/`parallel_for`가 n ≈ INT_MAX에서 int 오버플로(int64 가드
  불완전); `nearest_index`/`cell_and_t`의 double→int 캐스트가 clamp보다 먼저(NaN·거대값 UB);
  `rotate_axis`/`MagneticPropagator3D`/`tilted_potential`/`diamagnetic_potential`의 axis
  0..2 전제가 주석뿐; `fft`/`itp`/`rotation`/`sampling`/`measurement`의 `*_detail`이
  export됨; `ifft`가 크기 검사 전에 conj(던지면 버퍼가 conj된 채 남음); nz = 1 평면 FFT의
  무의미한 z 패스; `ImaginaryTimePropagator1D::relax`의 twiddle 재구축;
  `collapse_wavepacket` 직렬 256³ exp 패스(Measure 클릭마다 ~0.3 s);
  `gaussian_wavepacket`의 셀당 pow/exp(분리 가능한 축 테이블로).
- **[solver]** `EngineKernels`의 32개 (ptr, size) 수동 쌍 + 62개 대입 + fft switch.
- **[viz]** `SceneRenderer` 2.2k줄·~90 멤버, `destroy()`가 29개 메모를 손으로 리셋;
  `create_aux_image`가 `create_storage_image_3d`를 실패 정리 없이 재구현;
  `resize`/`ensure_marker_vbuf`/`ensure_staging`이 성공 전에 메모를 커밋;
  `SwapchainPresenter`에 복사 금지·소멸자 release 없음; CPU fallback에서 볼륨 재업로드 시
  occupancy/shadow 볼륨 미재구축; converged early-out이 flow 토글을 무시(정지·수렴 상태에서
  F가 무반응); `kMaxMarkers`↔`SES_MAX_MARKERS`가 주석으로만 연결.

- **[viz] 프레젠터**: 자체 펜스 타임아웃 뒤에도 `acquire`/`present`가 아직 대기 중인
  cb·acquire 세마포어를 재사용한다(대기 중 cb에 `vkResetCommandPool`). 획득만 하고 제시하지
  못한 이미지까지 다루는 프레젠터 전용 stall 가드가 필요(공유 `device_lost`로 조기 반환하면
  엔진 쪽 래치에도 UI가 멈춤). 표면 caps 조회 결과도 무시(표면 손실이 영원한 '최소화'로 보임).
- **[viz/shader]** `bloom_down`의 모서리 탭이 SOURCE 텍셀 반 칸이라 모든 레벨이 2×2 박스;
  `accum.comp`의 fp16 누적 평균은 `kAccumCap = 512`보다 훨씬 일찍 갱신을 못 담아 수렴
  early-out이 수 초 늦게 걸림(전력·GPU 시간); `volume.frag` 자기그림자 투과율을 8스텝 블록
  시작점에서 읽음; R32/R32G32 LINEAR 샘플링이 선택 기능 `FILTER_LINEAR`를 확인하지 않음.
- **[solver/shader]** `spin_born_sample`의 대체값(청크 내 `base+255`, 전역 기본 65535)이
  확률 0인 기저 상태를 고를 수 있음(측정당 ~1e-7..1e-5: 재누적 반올림 차이와 double→float
  `u` 캐스트) → thr을 gtotal 바로 아래로 clamp하고 마지막 비영 진폭으로 대체.
  `project_deposit`은 subgroupSize ≥ 4와 LocalInvocationIndex 연속 subgroup을 가정(미확인);
  가중치는 셰이더가 i0/frac을 다시 계산하고 bin은 CPU 키에서 오므로 `projection.ixx`의
  "fp32 IDENTICAL" 주석은 Vulkan에서 거짓(~1e-4 셀이 한 노드 이동, 허용 오차 안).
- **[solver]** `SpinEngine`: Chebyshev 차수 여유 `ceil(alpha)+24`가 alpha 적응형이 아님
  (잠재), 장(field)이 있을 때 `step(0)`이 짝 없는 반 스윕을 적용. `Lattice2DEngine`: 크기
  변경마다 디스크립터 셋 14개 누수(아레나는 해제 안 함), `allocate_sets`가 null 셋 미확인.
- **[scenario]** Anderson 스페클이 6 bohr CAP 경사와 발사 패킷 아래까지 깔림;
  `collapse_onto`가 `synthesize_into_psi`의 bool을 버림; corral/qdot relax 수렴 판정이 틱마다라
  포획 정밀도(와 corral 원판 절단 주기)가 시간 배율 다이얼에 의존; harmonic1d 고유상태 상향
  cap 판정이 측정 ⟨N⟩과 정수 cap을 비교(짝수 cap에서 cap+1 허용); hydrogen Reset이 광자·점프·
  측정 라벨을 타이틀에 남김; `collapse_onto` 경로 뒤 `peak_` 미갱신(최대 10틱 밝기 오정규화).

### E. 스타일·설계·주석 검증분 (2026-11-08)

스타일·주석·modern-C++·보일러플레이트 위주의 LOW/NIT 후보 434건을 파일 단위
배치로 적대적 검증했다: 244건 반박, 190건 유지. 유지분을 HEAD와 다시 대조한 결과
130건은 이번 사이클에서 고쳤다(git log). 남은 60건(위 항목과 겹치는 것 없음)을 같은
결함끼리 묶어 범주별로 적는다(57줄).

**정확성·견고성**
- **[core/shader]** `marching_cubes.ixx` `mc_detail::gradient`와 `mc_emit.comp` `gradient_at`이 벽에서 인덱스만 clamp하고 2h로 나눔 — 벽 한 칸 띠의 법선 성분이 한쪽 차분의 절반(표시 전용) → `heightfield.ixx`처럼 clamp된 축은 h로 나누고 CPU·GPU 동시 수정(vkcheck가 CPU↔GPU 법선 비교).
- **[scenario/app]** 버튼과 표기 핫키가 갈라짐: `Rutherford3DDirector`는 `handle_key`가 없어 키 2가 `BaseDirector::set_relaxing()` 무동작인데 'Fire alpha (2)'는 `refire()`; hydrogen 'Face z (Z)'는 카메라 스냅인데 Z 키는 psi 회전 → Rutherford에 `case '2': refire()`, hydrogen 라벨에서 '(Z)' 제거.
- **[scenario]** `DoubleSlit2DDirector::transmitted_fraction`은 벽 오른쪽의 순간 norm이라 로브가 CAP(x > 50)에 들어가면 패널 'Transmitted'·타이틀 T가 0으로 떨어짐 → qpc처럼 오른쪽 CAP 흡수 누적 tally로(틀린 `init_compute` 주석은 이번에 정정).
- **[scenario]** `landau2d_director.ixx` `Landau2DDirector`의 `peak_`는 `fire()`에서만 설정되고 `ladder()`·`run_frame()`은 갱신하지 않아 사다리 점프·패킷 호흡 뒤 구름 정규화(`peak()`)가 낡는다 → `measure()` 루프에서 max|psi|²를 함께 구해 `max(pk, kPeakDecay*peak_)`로 추적.
- **[scenario]** `spin_director.ixx` `advect_one`: 컬 반경 √1.4·`kSpFluxR`(≈18.9)인데 재생성 위치 `uni*0.8 − d·kSpFluxR`는 반경 최대 ~34라 재생성의 51-57 %가 구 밖에 놓여 다음 프레임에 다시 무작위 재배치돼 스트릭이 한 프레임 튐(표시 전용) → 재생성을 −d 쪽, d에 수직한 원판(구 안)에서 뽑기.
- **[scenario]** `DoubleSlit2DDirector::rebuild_props_overlays` 솔레노이드 화살표: Φ = 0(부팅값)에서 `std::copysign(0.6f, +0.0f)`로 +z 화살촉을 그리고, |Φ| < ~0.9 rad에서는 촉 오프셋 0.6이 축보다 김 → Φ ≈ 0이면 촉 생략, 촉 길이를 축 길이에 비례해 clamp.
- **[solver]** `Engine::bridge_roundtrip`이 `scratch_bridge_` 존재만으로 lazy 블록을 건너뜀: `load_set_` 할당 실패 뒤 재호출은 null `load_set_[vol_display_]`을 바인딩(vkcheck 전용 경로) → `ensure_two_center`처럼 조각별 생성, 준비 플래그는 마지막에.
- **[solver/scenario]** `Engine`의 오류 반환 API 중 `initialize`만 `[[nodiscard]]`: `molecule_director`의 `after_step_batch`·`after_reset`이 `set_two_center_potential` 실패를 버려 장치 V가 옛 축에 조용히 남음(`bridge_cpu_state`도 `upload_state` 결과를 버림) → `[[nodiscard]]`, 실패 시 CPU 강등.

**성능**
- **[app]** `imgui_ui.ixx` `draw_rutherford_panel`의 E/Z와 hydrogen 패널의 E/B 슬라이더가 바뀐 드래그 프레임마다 `set_energy`/`set_z`(256³ 재구축+퍼텐셜 업로드)·`set_efield_e0`/`set_bfield_b`(67 MB `upload_field_tables`)를 호출 → ω 슬라이더처럼 `IsItemDeactivatedAfterEdit`에서 적용(드래그 중엔 UI 값만 유지).
- **[core]** photon loss(κ > 0)에서 `Harmonic1DDirector::step_batch`가 배치마다 `classify()`로 스트립을 무효화해 O(n²·N) `ho1d_spectrum`(2026-10-09 C)이 매 프레임 재계산(w = 0.25에서 ~2.7e7 체인 스텝) → C의 단일 체인 `ho1d_spectrum`(점프 때만 무효화는 `StripTracksMcwfDampingBetweenJumps`가 금지).
- **[scenario]** `hydrogen_director.ixx` `relax_to_excited`/`relax_to_2s`가 `start_excited_relax`의 `!gpu_ok_ || solving()`·`!manifold_ready()` 거부보다 먼저 256³ 시드(268 MB, 직렬 셀당 exp)를 만들어 거부될 키 3/4도 멈칫한다 → `start_excited_relax`가 시드 생성 람다를 받아 가드 뒤에 호출.
- **[scenario]** `molecule_director.ixx` H2+ `after_step_batch`가 실시간 배치마다 `engine_.wait_async()`와 블로킹 `two_center_forces`/`set_two_center_potential`을 돌려 정지 회전자(부팅)에서도 compute/render 겹침을 잃는다 → 힘 판독·회전자 전진을 다음 `run_frame`의 `wait_async` 뒤로(`RotorEhrenfest`로 재검증).
- **[scenario]** `base_director.ixx` `superpose_into_psi`·`project_manifold_out`이 멤버마다 `TransientState`(134 MB 새 버퍼 + `release_state`의 `vkDeviceWaitIdle`)를 써 91상태 시드·부분측정 재구성·집단 점프에서 할당/idle 최대 91회 → `create_scratch_state` 버퍼 하나에 `synth_over`로 재합성(hydrogen MCWF처럼).
- **[scenario]** `molecule_director.ixx` H2+ `prepare(k)`→`start_flush(k)`가 선택마다 하위 멤버 k개를 직렬 256³ `synth_member`로 다시 합성·업로드하고(k=23이면 ~3 GB 보유) 할당 실패는 `if (h >= 0)`로 조용히 건너뛴다 → 같은 축의 멤버 버퍼를 캐시, 불완전 세트면 플러시 중단.
- **[scenario]** `Bouncer1DDirector::relax_ground`: Reset·'2'마다 ITP 전파자 2개를 새로 만들고 2048점 2×3000스텝 relax를 다시 돌림(퍼텐셜·시드가 고정이라 결과는 매번 비트 동일, 누를 때마다 UI ~0.5 s 정지) → 첫 relax 결과와 `e1_soft_`를 캐시해 `set_state`로 복원.
- **[scenario]** 2D 패널 판독(corral `confinement`, billiard `avg_center_fraction`, doubleslit `transmitted_fraction`, qdot `energy_meas`)이 일시정지 중에도 매 프레임 512²를 직렬 스캔(셀당 `std::hypot`/`stadium_sdf`, 프레임당 수 ms) → psi_가 바뀐 배치 뒤 한 번 계산해 캐시하거나 `parallel_sum`으로.
- **[scenario]** `Corral2DDirector::do_steps` 실시간 GPU 경로가 배치를 `eng_.step(min(left, 8), {.absorb = true})` 청크로 쪼갬 — 청크 사이에 읽는 것이 없어 다이얼 S에서 펜스 동기 제출만 S−1회 늘어남(`record_step_batch`가 이미 매 스텝 감쇠·킥 융합) → `eng_.step(n, {.absorb = true})` 한 번으로.
- **[shader]** `fft_line.comp`이 스테이지마다 스레드당 fp32 `cis` twiddle을 새로 계산(256점 라인당 1024회, 드라이버 sin/cos 정밀도에 기대지만 주석 없음)하고 `record_shear`의 역FFT가 conj→FFT→conj라 `magnetic_step`이 스텝당 전격자 conj 패스 12회를 더함 → 측정 후 twiddle 테이블 + 부호 매개변수 역FFT로 conj 패스 제거, 정밀도 주석 추가.
- **[tests]** `rotor_test` `quarter_turn`이 kick마다 `regularized_coulomb_potential`(셀·중심마다 `sine_integral`, 32³에서 65,536회)을 다시 만들고 `kick()`이 새 위상 테이블을 할당 — ~11.6k회 재구축이 F의 RotorEhrenfest 50 s의 주원인 → `band_limited_coulomb`의 1D 반경 테이블(보간)로 퍼텐셜을 만들고 위상 버퍼 재사용.

**테스트·오라클**
- **[scenario]** `selftest_arcs.ixx` `--selftest-kepler`: `phi_acc`는 마지막 1.5 s 프로브에서 멈추는데 `dt`는 90 s 데드라인 콜백의 `sim_time()`까지 재서(재무장 지연으로 마지막 프로브가 밀림) 회전율이 ~1-2 % 낮음([0.6, 2]× 밴드 안) → 프로브마다 `t_last`를 기록해 `phi_acc / (t_last − t0)`.
- **[tests]** `ladder_rung_stable`의 오분류 입력 분기(오라클 겹침 < 1/2 → raw 출력 유지)는 분류기가 틀릴 때 `harmonic1d_director`가 타는 경로인데 `ladder_test`의 호출이 모두 올바른 라벨의 고유상태라 미검증 → 비고유 중첩(예: (|3>+|4>)/√2, n_from = 3)에서 psi·norm²가 `ladder_raise` 결과와 같음을 단언.
- **[tests]** `hydrogen3_test` 세 테스트가 `relax()`를 지워도 통과: σ = 1.3 시드가 이미 E ≈ −0.255(a=1)/−0.200(a=1.4)로 창 (−0.5, −0.1)·간격 > 0.01을 만족하고 구대칭·중심이라 이완을 검증하지 못함 → 시드 에너지를 변분 상한으로 삼아 이완 후 E0(≈ −0.275)가 충분히 낮은지 단언.
- **[tests]** `cross_section_test`의 `SliceShade` 테스트가 전부 `inv_peak = 1.0`이라 `slice_shade`의 `dens = norm·inv_peak`(선형)·`amp = sqrt(inv_peak)`(제곱근)가 항등식 — 선형/제곱근 혼동이 통과하고 GPU 오라클 없는 `slice.frag`의 피크 스케일링이 무방비 → `inv_peak = 4` 등으로 세 모드 값을 단언.
- **[tests]** `dipole_angular_test` `FactorizedEinsteinAReproducesTextbook2pLifetime`이 무한질량 bare −1/r 해를 환산질량 교과서 값 1.5155e-8과 비교 — 정확값 (2/3)^8 α³ = 1.51623e-8이 7.3e-12(허용 2e-11의 37%) 떨어져 예산을 모델 밖 물리가 잠식 → 기준을 (2/3)^8 α³로, 허용 오차는 격자 오차로 유도.
- **[tests]** `drive_test` `ResonantCoherentLadderAndSelectionRule`이 스스로 exact라 부르는 Ehrenfest 폐형식(|z| ~ 1.41)에 0.02, p0 = p1 ~ e^-1에 절대 0.06(~16%)을 오차 예산 없이 건다(반회전항 보정은 p0에 ~1e-3) → 측정 편차로 허용 오차를 좁히고 근거를 주석으로.
- **[tests]** `--selftest-bouncer`의 `drop_ok`는 `drop()` 직후 스텝 없이 방금 만든 가우시안의 ⟨H⟩≈80을 다시 읽어 드롭 위치 오류만 잡고 `bouncer1d_test.cpp`는 relax 사다리만 검사해 바운서를 진화시키는 검사가 없음(`kBo1dDt`의 Trotter 가열 위험 무고정) → 드롭·바닥 상태를 N 스텝 진화시켜 ⟨H⟩ 보존을 단언.
- **[tests]** `spin_measure`/`exact_measure_z`의 Born 확률을 아무 테스트도 고정 안 함: `spinexact_test`는 p=1/2만, `spin_test`는 고유상태와 u=0/0.999 극단만이라 `spin.ixx`의 (1+n·s)/2를 빼도 통과(vkcheck GPU 측정의 오라클이기도 함) → p≠1/2 상태에서 u=p±ε로 결과 반전 단언.
- **[tests]** `photon_display_test` 스트릭 테스트가 모두 n = ẑ라 `photon_streak_vertices`의 극 분기(phi = 0)만 타서 실제 방출 방향의 (θ̂, φ̂) 틀과 n_z < 0 감김 방향이 미검증; `OverflowEvictsTheMostSpentFlight`는 total이 전부 100이라 age/total 순위가 최고령 축출과 구별 안 됨 → 기울어진 n·서로 다른 total로 단언.
- **[tests]** `molecular_seed_test`의 `SigmaGIsEvenAndNodeless`는 노름·x=0 평면 가중치만, `PiUyIsOddUnderYReflection`은 j=16 마디 평면만 보아 이름의 짝/홀 패리티를 검사하지 않음(y-짝이며 y=0에서 0인 함수도 통과) → `SigmaUIsOddUnderXReflection`의 거울 루프로 패리티 단언(`molecular_orbital_seed` 삭제 시 불필요).
- **[vkcheck]** corral의 평면 GPU relax 경로(512²×1, m* = 0.38 테이블의 `relax_step`/`relax_deflated_step`)에 오라클 없음 — `check_engine_relax`/`_deflation`은 8³ 입방·m = 1만 검사(디렉터 주석은 정정됨) → `check_engine_planar_mass` 격자로 relax·deflation vs `ImaginaryTimePropagator3D` 검사 추가.

**설계·API 위생**
- **[app]** `imgui_ui.ixx` `UiState`의 `efield`/`bfield`/`time_scale`/`ru_e`/`ru_z`는 모든 읽기 직전에 디렉터 진실로 덮어써져 초기값(0/0/1/25/79)이 한 번도 쓰이지 않는데 부팅 미러처럼 보임 → 각 draw 함수의 지역 변수로 옮기고 `UiState`에서 제거.
- **[core]** `bench/gen_h2plus_atlas.cpp` 템플릿이 생성하는 `ses.h2plus_atlas_data`에 안 쓰는 `<cmath>/<cstddef>/<vector>`와 `export import ses.spheroidal`이 남고, `h2plus_atlas_loader`는 `H2plusOrbital`을 이 재export에 암묵 의존 → 템플릿에서 빼고 재생성, 로더에 `export import ses.spheroidal;`.
- **[core/scenario]** `spins_director.ixx` `measure_now`의 CPU 정확 분기가 회전·`exact_measure_z`×16·역회전을 디렉터에서 조립(주석은 "축과 u만 고름", 축 계산은 `SpinEngine::measure_exact` 복사)하고 GPU 결합 Born 샘플러는 CPU 쌍둥이 없이 멱등성만 검사 → `ses.spinexact`에 결합 추출 함수 추가·디렉터가 호출, vkcheck는 같은 u로 비트열 비교.
- **[scenario]** `hydrogen_director.ixx` `upload_field_tables`의 메모 앞 `engine_.wait_async()`는 중복(`set_potential`→`upload_raw`가 먼저 드레인)이고 주석이 말하는 'upload_raw 순서 없음' 위험은 이미 닫혔다; 메모 적중(Reset)에도 드레인 → 호출과 주석 삭제.
- **[scenario]** `molecule_director.ixx` 두 씬이 `set_geometry`/`set_parameter`를 no-op으로 덮고 호출자도 없어 base 본문·`apply_geometry`·`geometry_parameter`/`clamp_parameter`/`geometry_changed` 훅·H2+ 'stretched 6.0' 프리셋과 `scenario.ixx` `MoleculeApi` 노브 4개가 죽은 코드 → 삭제.
- **[scenario]** `molecular_seed.ixx`의 `molecular_orbital_seed`/`MolOrbital`은 테스트 밖 사용처가 없는 데드 코드(`molecule_director`는 `random_molecular_seed`만 씀)이고 결합축 x를 가정하나 H2+ 씬은 핵을 (0,0,±d)·회전자 축 z에 둔다; 그런데도 `molecular_seed_test` 4개(32³×800스텝 relax 3회 포함)가 고정 → 함수·열거형과 해당 테스트 삭제(쓸 계획이면 결합축 인자).
- **[scenario]** `ScenarioDirector::mark_display_dirty()`는 순수 가상 훅인데 app 호출부가 없고(7개 패밀리 구현) 유일한 호출은 harmonic1d가 상속한 Line1D 구현을 부르는 것이며 BaseDirector 판은 `mc_dirty_`도 세우지 않음 → 인터페이스에서 빼고 쓰는 디렉터의 비가상 헬퍼로.
- **[scenario]** `carpet1d_director.ixx` `Carpet1DDirector`가 `init_compute`를 덮지 않아 `Lattice2DDirectorBase::init_compute`가 쓰지 않는 `Lattice2DEngine`(파이프라인 7개+24셋 아레나)을 만들고 `gpu_step`/`gpu_relax`는 호출자의 `gpu_active()` 확인에 기댄다 → carpet은 엔진 없는 `init_compute`, 두 헬퍼에 가드.
- **[scenario]** 2026-10-09 D의 데드 fp16 'atlas precision'은 로그만이 아님: `gpu_synthesize`는 fp32 고정 상태(kS1·kP2X/Y/Z)로만 불려 `choose_state_precision`(`ses.vram_budget`+테스트 4개)·`vram_probe`·`synthesize_state_half`·`free_vram_bytes` 인자(hydrogen만 읽음)가 로그 한 줄만 구동 → 체인째 제거.
- **[scenario]** corral `do_steps`(`int left = 4 * n`)와 qdot `do_steps`(`gpu_relax(4 * n)`)가 relax에 숨은 4× 스텝 배율을 둠 — ARCHITECTURE의 "다이얼이 유일한 배율" 계약 위반(2026-10-09 D spins 8/20과 같은 결단) → 배율을 `steps_per_tick`으로 옮기거나 계약에 relax 예외를 명시.
- **[shader]** `flow_velocity.comp`은 `jc / max(rho, 1e-12)` 뒤 `vmax` clamp, `ses::bohmian_velocity`는 rho ≤ 1e-12에서 정확히 0·clamp 없음(`BohmianVelocityGuardsNodes` 고정) — 향후 오라클(2026-10-09 B)이 가드를 특례 처리해야 함 → rho ≤ 1e-12에서 0을 쓰고 vmax clamp는 표시 전용이라고 머리말에 명시.
- **[shader]** `volume.frag.slang` empty-space skip의 `/ 32.0`이 호스트 `kOccSize`(`vk_render.ixx`)를 베이크 define 없이 복제(상호 참조 주석은 이번에 추가) — skip은 한 occupancy 셀(occ_dilate halo) 이하일 때만 보수적이라 `kOccSize`만 바꾸면 광선이 구름을 건너뜀 → `occ_tex.GetDimensions`로 셀 크기 유도(또는 `-DSES_OCC_SIZE`).
- **[solver]** `Engine::set_relax_tables`의 흔적 매개변수 `cell_volume`이 모든 판독이 쓰는 `cell_volume_`를 덮어씀(fused finalize의 `renorm_ubo_`는 `initialize`의 grid dV 유지)이라 다른 값이 오면 GPU 재정규화와 호스트 통계의 dV가 갈림; 호출부는 모두 grid dV를 넘김 → 매개변수 제거.
- **[solver]** `vk_engine.ixx` `run_step_body`가 `record_kin_body`의 FFT·kin·IFFT 체인을 VkFFT/수작업 두 분기 모두 복사(kin 커널·셋만 다름), 한쪽만 고치면 `step`과 driven/magnetic/relax 경로가 갈라짐 → `record_kin_body(r, kin_k, kin_set)`로 매개변수화해 두 반 킥 사이에서 호출(디스패치·배리어 순서 동일).
- **[viz/app]** `SceneRenderer::initialize/resize/render`·`SwapchainPresenter::present`에 `[[nodiscard]]`가 없고 `main.cpp` `render_scene_offscreen`이 `render(in)` 결과를 버려 overlay/marker 할당 실패 시 이전 프레임이 무언으로 남음 → `[[nodiscard]]`(멱등 `acquire()` 제외) + 실패 시 stderr 보고.
- **[viz]** blit SPIR-V만 `render_blobs()` 밖: `app/src/main.cpp`가 `blit_{vert,frag}_spv.h`를 텍스트 include해 raw (ptr,size)로 `SwapchainPresenter::init`에 넘기고, 이 때문에 ses_viz가 생성 헤더 디렉터리를 PUBLIC으로 노출 → blit을 `render_blobs`로 옮기고 `init`은 `ses_vk::Blob`, 디렉터리는 PRIVATE.
- **[viz/shader]** `vk_render.ixx` `VolumeUbo`와 `solver/shaders/viz_ubo.slang` `Ubo`의 `proton_center`/`proton_color`/`proton_radius`는 어느 스테이지도 읽지 않는 죽은 필드로 바이트 계약만 흐림(주석은 정정됨) → 두 미러에서 함께 삭제하되 std140 패킹용으로 `proton_radius` 자리에 pad float를 두고 셰이더 재베이크.

**빌드·문서·주석**
- **[build]** `vk_render.ixx` `kAccumCap = 512`가 `viz/CMakeLists.txt` `_ses_viz_accum_cap`의 손 복제(어긋나면 early-out 임계≠셰이더 cap)이고 `-DSES_MAX_MARKERS`는 `volume.frag`에만 전달 → `target_compile_definitions(ses_viz …)`로 호스트 상수를 같은 CMake 변수에서 받고 모든 `viz_ubo` importer에 -D 전달.
- **[build]** `app/CMakeLists.txt` `SES_ARC_TESTS`가 36개 아크를 `kArcSpecs`와 따로 손으로 나열(대조 없음; 개명 시 po 오류 모달이 ctest를 600 s TIMEOUT까지 막음)하고 `enable_testing()`이 `SES_BUILD_TESTS`에 묶여 OFF면 0개 등록에도 'registered' 출력 → `file(STRINGS)`로 `kArcSpecs`에서 생성, OFF면 `FATAL_ERROR`.
- **[build]** `.vscode/launch.json` F5 구성이 `linux-clang22`(`linux-release` 상속, `CMAKE_BUILD_TYPE=Release`) 바이너리를 gdb로 띄우나 어떤 CMake 파일·프리셋도 `-g`를 넣지 않아 DWARF가 없음 — 소스 중단점·지역 변수를 볼 수 없음 → `RelWithDebInfo` 프리셋(예: `linux-clang22-dbg`)을 추가해 launch/tasks를 그쪽으로.
- **[build]** `CMakePresets.json` `linux-clang22` description은 AOCC(clang-17 기반)가 constexpr `std::complex` 산술을 못 해 `complex_test`가 실패한다고 하나, `complex_test.cpp`는 clang+libstdc++에서 그 고정값을 `const`(런타임)로 돌려 현재 트리에선 그 실패가 불가 → 실제 사유로 고치거나 그 문장 삭제.
- **[build]** `vcpkg.json`의 `vulkan-loader`는 어떤 타깃도 링크하지 않음(volk가 런타임에 시스템 로더를 열고 `Vulkan::Vulkan`은 주석으로 금지) → 삭제하거나, 포트가 전이적으로 끌어온다면 남기되 `$comment`로 이유를 적을 것.
- **[build]** viz 전용 `viz_ubo`·`viz_util`이 `solver/shaders`에 있고 `ses_bake_shader`는 모든 베이크를 모든 모듈에 DEPENDS → viz 모듈 수정이 solver 커널 전부를(`complex` 수정은 viz 전부를) 재베이크하고 `bin2h.cmake`의 무조건 `file(WRITE)`로 `engine_blobs.ixx` importer까지 재컴파일 → 모듈을 계층별로 나누고, 내용이 같으면 헤더 미갱신.
- **[docs]** `ARCHITECTURE.md`는 2D 격자 장면 표시를 'face-on plane'/'thin volume slab'로만 설명하나 corral·carpet은 heightfield 메시(`cloud() == false`)로 그리고 billiard·qdot·qpc도 `rebuild_surface` 경로를 가짐 → 표시 방식 서술 갱신(패밀리 수·1D 격자 크기는 이번에 정정).
- **[docs]** `README.md` validation layer 절은 Windows cmd `set` 줄과 `x64-windows-static\bin` 경로뿐(Verification 절도 'as above'로 참조)이라 Linux 레시피가 없음 → vcpkg 포트의 Linux 매니페스트 경로를 확인해 `SES_VK_VALIDATION`·`VK_ADD_LAYER_PATH`·`VK_LOADER_LAYERS_ENABLE` `export` 블록 추가.
- **[tests]** `emission_test` PhotonSampling 머리말과 `fft_test` `deterministic_signal` 주석이 문서에 없는 "repo no-RNG rule"을 인용하고 PhotonSampling을 유일한 예외라 하나 `mcwf_test`·`spin_test`·`spinlattice_test`도 고정 시드 `std::mt19937`을 쓴다 → 규칙(고정 시드 허용)을 `TDD_RULES.md`에 명문화하거나 두 주석의 인용 삭제.
- **[viz]** `vk_render.ixx` `Marker`·`ensure_marker_vbuf`, `scenario.ixx` `SceneMarker` 주석은 Cloud 마커를 raymarcher만 그린다고 하나 glass-shell `marker_pipe_` 드로우는 `in.cloud` 게이트 없이 Cloud에도 그려지고 depth를 쓰는 패스가 없어 안개 없이 위에 얹힘 → 의도면 세 주석 정정, 아니면 `!in.cloud`로 게이트.

