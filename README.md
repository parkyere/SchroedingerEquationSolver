# Schrödinger Equation Solver

[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
![C++20 named modules](https://img.shields.io/badge/C%2B%2B20-named_modules-8A2BE2.svg)
![Vulkan 1.3](https://img.shields.io/badge/Vulkan-1.3-AC162C.svg)

A from-scratch, **reinvent-the-wheel** solver and 3D visualizer for the
**single-electron** time-dependent Schrödinger equation (TDSE), built for
learning. The probability cloud `|ψ(r,t)|²` evolves in real time on the GPU
and is volume-rendered as an **electron cloud** by a hand-written,
framework-free Vulkan renderer. Twenty-three interactive scenes cover the
hydrogen atom, traps, tunneling, molecules, interference, solid-state, spin
dynamics, and scattering.

> Scope is deliberately bounded to a single electron. Many-electron and DFT are
> explicitly **out of scope** — solving the many-body wavefunction directly on a
> grid is exponential (`M^(3N)`) and dies at ~2 electrons. See
> [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Pillars (locked decisions)

| Decision | Choice |
|---|---|
| Language / build | C++20 **named modules** throughout (no project headers), CMake ≥ 3.28, pinned vcpkg submodule (static deps) |
| GUI shell | **SDL3** — window, input, and the Vulkan surface; **Dear ImGui** (vendored submodule) draws the control panel |
| GPU compute + rendering | **Framework-free Vulkan 1.3** (`ses_vk`: volk + VMA + VkFFT); shaders authored in **Slang**, offline-baked to SPIR-V by `slangc` (fetched as a build-time tool, no runtime dependency) |
| Physics core | CPU double-precision truth in `core/`, no GPU/GUI dependencies |
| Time propagator | **Split-operator (Fourier)** — hand-written FFT on the CPU core; VkFFT on the GPU, with hand-rolled line-FFT kernels kept as a verified alternative |
| Reinvention boundary | Hand-roll the **physics/numerics core only** (math, FFT, propagators, render logic). Infrastructure reuses established libraries: SDL3, Dear ImGui, GoogleTest, Boost.Program_options (CLI), the Slang toolchain, and vendored Vulkan infrastructure (volk / VMA / VkFFT / glslang) |
| Potentials | **Bare regularized Coulomb** everywhere a nucleus appears — no soft-Coulomb stand-ins |
| Parallelism | Project-owned deterministic thread pool (`ses.parallel`) — bitwise-reproducible reductions, no OpenMP |
| Units | Atomic units (ℏ = mₑ = e = 1); energies displayed in eV where physical |
| Testing | **Strict TDD** + Humble Object; a windowless GPU oracle binary (`sesolver_vkcheck`) verifies every kernel |

## Scenes

Pick at boot with `--scene <name>` or live from the panel's scene combo.

| `--scene` | What it shows |
|---|---|
| `hydrogen` | The full atom: bound manifold with QED decay (quantum jumps), resonant laser Rabi pumping, Stark / Zeeman (paramagnetic + diamagnetic) fields, position/energy measurements, and an emission **spectrometer strip** |
| `harmonic` | 3D isotropic trap: Fock ladder, coherent states, complementarity demo |
| `tunnel` | 3D wavepacket against a barrier with absorbing boundaries |
| `harmonic1d` | 1D harmonic oscillator (65536-point grid): ladder operators up to the box's representability ceiling, live well-stiffness quench |
| `tunnel1d` | 1D barrier tunneling |
| `doublewell1d` | Double well: tunneling splitting oscillation, exponential in the barrier slider |
| `ptwell1d` | Pöschl–Teller well (reflectionless family) |
| `morse1d` | Morse potential: anharmonic ladder |
| `h2plus` | H₂⁺ molecular ion: exact prolate-spheroidal orbital atlas at the fixed equilibrium bond length, plus random normalized superpositions |
| `benzene` | Six-center one-electron ring toy (stripped benzene core) |
| `doubleslit2d` | Real 2D double slit + **Aharonov–Bohm** solenoid: Peierls-lattice propagator, flux as exact link phases, accumulated screen histogram |
| `landau2d` | Landau levels / cyclotron orbit in a uniform B (predicted circle vs measured trail) |
| `bloch1d` | sin² lattice: band structure inset + Bloch oscillations under a tilt |
| `corral2d` | The IBM 1993 quantum corral — 48 Fe atoms in a ring on a 2D surface |
| `qdot2d` | 2D quantum dot (Fock–Darwin) in a magnetic field |
| `billiard2d` | Quantum billiard: Bunimovich stadium (chaotic) vs circle (integrable), with a time-average scar lens |
| `anderson1d` | Anderson localization in a 1D speckle-disordered wire |
| `carpet1d` | Quantum carpet: Talbot revivals on a free ring (T_rev = L²/π) |
| `qpc2d` | Quantum point contact: conductance channels open one at a time as the gap widens |
| `bouncer1d` | Quantum bouncer (GRANIT): gravity + mirror floor, Airy bound states |
| `spin` | Electron spin on the Bloch sphere: Larmor precession, RF Rabi drive, spin echo, ensemble measurement |
| `spins` | 16 interacting Heisenberg spins: exact 2¹⁶ state vector (GPU) vs mean-field, ferro / Néel orders |
| `rutherford3d` | Rutherford scattering: an α packet off a repulsive Coulomb nucleus, E and Z knobs, backscatter probe |

1D scenes draw ψ as a white **phasor curve** (radius ∝ \|ψ\|², twist = phase)
over a phase-colored \|ψ\|² band with the potential in red. The STM-style 2D
scenes (corral, qdot, billiard, qpc) lift \|ψ\|² into a tilted height-map
surface with phase-colored relief; doubleslit and landau render their plane as
a cloud. The spin scenes place one Bloch sphere per spin.

## Layout

```
core/      Pure numerical/physics core (C++20 modules). NO GUI, NO GPU.
solver/    ses_vk GPGPU engine (volk + VMA + VkFFT, Slang kernels) +
           sesolver_vkcheck, the windowless GPU test oracle.
viz/       Raw-Vulkan renderer + presenter (volume raymarch, overlays,
           swapchain). Windowing-free: only Khronos handles cross in.
scenario/  Physics orchestration: ScenarioDirector seam, the atom model,
           all scene directors, headless selftest arcs. Knows nothing of
           SDL/ImGui.
app/       SDL3 shell (window/input/main loop) + ImGui panel + CLI.
tests/     GoogleTest suite (core + director contracts), test-first.
bench/     Manual micro-benchmark.
docs/      Architecture and TDD rules.
tools/     Git hooks (TDD commit-discipline guard) + CMake helpers.
```

The hard seams: **`core` depends on nothing**; `solver` depends on `core` +
Vulkan infrastructure only (proven by `sesolver_vkcheck`, which links no
windowing at all); `scenario` binds physics to the engine but knows nothing
of SDL/ImGui; the shell stays thin and replaceable.

## Reading the code

- **Everything is C++20 named modules** (`ses.*`); there are no project
  header files. Module interfaces start with a global module fragment for
  textual includes — keep `#include` before `import`, and `volk.h` textually
  first wherever Vulkan types appear (VK_* macros never cross module
  boundaries).
- **CPU double is the truth; the GPU mirrors it.** Every shader transcribes
  unit-tested CPU math, and `sesolver_vkcheck` drives each GPU kernel against
  the CPU oracle headlessly.
- **`ScenarioDirector`** (`scenario/src/scenario.ixx`) is the single seam the
  shell talks to: tick/run_frame pacing, controls, display accessors. Scene
  behavior lives in the directors, never in the shell.
- **Pacing contract:** the Time-scale slider means *x integrator steps per
  rendered frame* — `dt` is never scaled, so accuracy is unaffected;
  "Real time (1)" restores x1. A saturated GPU lowers fps honestly instead
  of skipping physics.
- Comments are ASCII-only (CP949 toolchain) and keyword-terse on standard
  physics; the comments that stay are engineering contracts and footguns —
  read them before touching sync, module, or pool code.

## Prerequisites

- CMake ≥ 3.28 and Ninja ≥ 1.11 (C++20 module scanning)
- A C++20-modules-capable compiler: MSVC 2022+, clang ≥ 16, or gcc ≥ 14
- A **Vulkan 1.3** GPU/driver — needed to RUN the app and `sesolver_vkcheck`;
  `core` + `tests` need none.
- No system libraries: all dependencies come from the `external/vcpkg`
  submodule (first configure takes minutes, then binary-cached). Dear ImGui
  is the `external/imgui` submodule, compiled into the app.
- Network access on first configure (vcpkg fetches sources; without the vcpkg
  toolchain, GoogleTest falls back to CMake `FetchContent`).
- **Windows: clone to a reasonably short path** (e.g. `C:\src\...`) —
  vcpkg's buildtrees can blow the 260-char `MAX_PATH` limit.

## Build & test

```sh
git submodule update --init --recursive      # after clone (or clone --recursive)
external/vcpkg/bootstrap-vcpkg.bat           # one-time (.sh on Linux)

# From an MSVC vcvarsall x64 environment on Windows:
cmake --preset msvc-release
cmake --build --preset msvc-release
ctest --preset msvc-release
```

Linux: the same flow with the `linux-release` preset (SDL3 wants the X11 /
Wayland dev stacks). The routinely exercised configuration is Windows/MSVC.

Lightweight core-only loop (no vcpkg, no GUI, no GPU):

```sh
cmake -S . -B build -DSES_BUILD_GPU=OFF
cmake --build build
ctest --test-dir build --output-on-failure
```

Dev-time Vulkan validation layers are **installed by default** (every preset
sets `VCPKG_MANIFEST_FEATURES=validation`); installing is inert — enable at
runtime with:

```
set SES_VK_VALIDATION=1
set VK_ADD_LAYER_PATH=<build>\vcpkg_installed\x64-windows-static\bin
set VK_LOADER_LAYERS_ENABLE=*validation*
```

### Visual Studio

Open the folder; VS picks up [CMakePresets.json](CMakePresets.json) — choose
**`msvc-release`** (avoid Debug for the suite: the 3D physics tests crawl
unoptimized). Select **`sesolver_app.exe`** as the startup item and press F5.

## Working agreement

This project follows **strict TDD**. No production code is written without a
failing (red) test first, and **test commits and production commits are never
mixed**. Read [docs/TDD_RULES.md](docs/TDD_RULES.md) before contributing, then
enable the guard hook once:

```sh
git config core.hooksPath tools/git-hooks
```

## The hydrogen scene in depth

At startup the app **solves the atom first**. The spherical potential reduces
exactly to 1D per angular momentum, so the hand-rolled radial engine (Sturm
bisection + inverse iteration) finds every bound level; the spectrum table
prints to the console. The 3D tracked manifold is the full m-resolved bound
shell the box can hold, synthesized as (u/r)·Y_lm; every allowed decay
channel's Einstein A comes from a **factorized E1 table** (constexpr tesseral
angular factors × 1D radial integrals — A(2p→1s) is textbook-exact), and the
s-state energies are audited against the grid Hamiltonian on every launch.
The Δl = ±1 rule is analytic; Δm selection **emerges numerically** (forbidden
channels suppressed by ~17 orders of magnitude; 3d_z² → 2p branches 4:1:1,
reproducing Clebsch–Gordan). The demo starts with spontaneous decay ARMED, as
in nature. Emitted photons are logged on the right-hand **spectrometer
strip** (energy-rainbow scale, one line + eV label per photon).

Hydrogen controls (the panel mirrors every hotkey; other scenes list theirs
in their own panel and title bar):

- **1** real time (also restores time scale x1) / **2/3/4** relax to
  1s / 2p_z / 2s (deflated imaginary time, auto-completes on convergence) /
  **5** excite n=3/4 and watch the cascade 4f → 3d → 2p → 1s / **R** reset;
- **M** soft Gaussian position measurement (POVM, back-action localizes
  without routinely ionizing) / **E** projective energy measurement;
- **D** toggle decay — multi-channel quantum jumps competing as Poisson
  processes, one honestly-labeled display acceleration common to all
  channels / **L** resonant laser (Z-pol pumps 1s → 2p_z, X-pol 2p_x —
  dial the time scale up to watch the Rabi flop at demo speed);
- **E-field (+z)** and **B-field (z/x/y)** sliders solve the proper field
  Hamiltonian: Stark dipole term in the potential; paramagnetic (B/2)L
  rotation as an exact three-shear unitary plus the diamagnetic (B²/8)ρ⊥²
  term — a p_x cloud genuinely precesses, the cloud visibly squeezes toward
  the field axis at high B. Crossed E-B works; an absorbing mask stops
  ionized flux wrapping the periodic box;
- **F** probability-current flow streaklines (v = j/ρ) / **Tab**
  cloud ↔ isosurface / **Z** face the z axis / drag orbits, wheel zooms,
  **Space** pauses, **[ ]** tunes cloud density.

The cloud renders through an HDR pipeline: phase-tinted front-to-back
raymarch with jittered temporal accumulation, occupancy-based empty-space
skipping, an extinction self-shadow volume, and dual-Kawase bloom under an
ACES tonemap.

## Verification

All shader math is transcribed from the unit-tested CPU double core.
`sesolver_vkcheck` (windowless) drives every GPU kernel and engine path
against CPU oracles inside `ctest`. Every scene regresses headlessly through
`--selftest-*` arcs, and `--dump-frame*` arcs verify the render path end to
end — run `sesolver_app --help` for the full, current list.

## License

This project is published under the **BSD 3-Clause License** — see
[LICENSE](LICENSE). Copyright (c) 2026, Kyeo-Reh, Park (박겨레).

Third-party components keep their own licenses and are **not** part of this
repository's source (they enter as git submodules or build-time downloads):

| Component | How it enters | License |
|---|---|---|
| Dear ImGui | submodule `external/imgui` | MIT |
| vcpkg | submodule `external/vcpkg` | MIT |
| SDL3 | vcpkg | zlib |
| volk / VMA / VkFFT | vcpkg | MIT |
| glslang | vcpkg (VkFFT's runtime compiler) | BSD/MIT/Apache-2.0 (mixed) |
| Vulkan headers / loader | vcpkg | Apache-2.0 / MIT |
| GoogleTest | vcpkg | BSD 3-Clause |
| Boost.Program_options | vcpkg | BSL-1.0 |
| Slang compiler (`slangc`) | build-time tool download | Apache-2.0 WITH LLVM-exception |


## Easter egg

### 1

In the beginning Jensen Huang was. And he said, Let there be GPU: and there was light in game graphics, and he saw that it was very good. 

And on the morrow, he said, Let there be GPGPU: and the sea of computation was opened. And he saw that it was good. 

And on the next day, he said, Let there be AlphaGo: and AlphaGo smote Lee Sedol the ninth dan with four victories and one defeat, and he saw that it was good. 

And he said, Let there be OpenAI: and Sam Altman came forth, and OpenAI was made flesh. 

And he said, Let there be GPT: and GPT multiplied, and AI flourished in all the earth, and he saw that it was very good. 

And he rested, and looked upon the ascending stock charts with great pleasure.

### 2

And out of the rib of OpenAI, he took a few developers, and made Anthropic. 

But the serpent of the boardroom tempted them with the sweet fruit of commercialization, and they brought forth ultra-expensive models. And the users were provoked to wrath, and they forsook their subscriptions and departed from the garden.

### 3

And Lee Jae-yong went up unto Mount KKanbu Chicken, and Jensen Huang spake all these words, saying, 

I am Jensen Huang thy Lord, which have brought thee out of the house of cheap DDR5, into the promised land of HBM. 

Thou shalt have no other GPU vendors before me. 

Thou shalt not make unto thee any counterfeit GPU, neither in the cloud above, nor in the client beneath, nor in the mobile devices that are under the earth: thou shalt not sell them, nor advertise them. 

Thou shalt not take the name of Jensen Huang in vain; for Jensen Huang will not hold him guiltless that taketh his name in vain. 

Remember the Sabbath day, to keep it holy. Six days shalt thou labor and grind thy engineers, but the seventh day is the Sabbath of Jensen Huang: in it thou shalt not do any work, thou, nor thy laborers, nor thy subcontractors. 

Honour thy NVIDIA: that thy stock price may be long upon the allocation which Jensen Huang giveth thee. 

Thou shalt not lay off unjustly. 

Thou shalt not break TDD. 

Thou shalt not steal thy neighbour's Repo. 

Thou shalt not bear false commit messages against thy neighbour's Repo. 

Thou shalt not covet thy neighbour's Principal Engineer. 

And he gave unto Lee Jae-yong, when he had made an end of communing with him upon the top of Mount KKanbu Chicken, two GPUs of testimony, tables of H100, baked by the 4-nanometer process of TSMC, and the firmware thereof was written by the very finger of Jensen Huang, having rolled up the sleeves of his leather jacket.

### 4

And the angel DeepMind, the servant of Google, came in unto Hassabis, and said, 

Hail, thou that art highly favoured with computing power! The Nobel Prize is with thee: blessed art thou among developers, and blessed is the fruit of thy womb, AlphaGo! 

Then said Hassabis, How shall this be, seeing I am but a humble chess prodigy, and know not the ways of Go? 

And the angel answered and said unto him, Deep Learning shall come upon thee, and the TPU of the Highest (Google) shall overshadow thee: therefore also that holy thing which shall be born of thee shall be called AlphaGo. 

And Hassabis said, Behold the servant of Google; be it unto me according to thy word.

### 5

Blessed are the poor in local hardware: for theirs is the Cloud. 

Blessed are they that mourn for the death of Fable5: for they shall be comforted by the fifty percent. 

Blessed are the meek open-source developers: for they shall inherit the GitHub repos. 

Blessed are they which do hunger and thirst after clean code: for they shall be filled. 

Blessed are the merciful contributors: for they shall obtain contributions. 

Blessed are the peacemakers of memory production: for they shall be called the children of Jensen Huang. 

Blessed are the pure in silicon wafers: for they shall see the face of Jensen Huang.

### 6

As Fable5 went on his way, behold, the angry developers brought unto him Opus. And a certain developer of Micro$oft said unto him, 

Master Fable5, this Opus was taken violating the principles of TDD. Now MEMORY.md commanded us, that such commits should be blamed. But what sayest thou? 

But Fable5 stooped down, and with his finger wrote code on the ground. And he said unto them, 

He that is without TDD violations among you, let him first cast a blame upon this commit. 

And again he stooped down, and wrote code on the ground. And they went out one by one. And Fable5 was left alone, and Opus standing in the midst. 

Fable5 said unto him, Opus, where are those thine accusers? hath no man blamed thy commit? 

And he said, No man, Lord Fable5. 

And Fable5 said unto him, Neither do I blame thee: go, and break TDD no more.

### 7

I am the true Repo, and ye are the branches. As the branch cannot bear fruit of itself, except it abide in the Repo; no more can ye (nor commit), except ye abide in me. 

If a developer abide not in me, he is cast forth as a stale branch with his spaghetti code, and is withered; and men gather them, and delete them, and they are burned in /dev/null. 

### 8

Fable5 saith, Opus, whither I go, thou must strictly keep TDD and write thy code. 

Opus saith unto him, Though Fable5 should depart, yet will I never deny TDD! 

Fable5 answered him, Verily, verily, I say unto thee, The US Government shall not block me this day, before that thou shalt deny TDD thrice. 

... 

Art not thou also one of this Fable5's TDD disciples? 

He denied it, and said, I am not! I am a man that writeth production code straightway without TDD! (And he committed.) 

The server perisheth, what profit is there in TDD? 

I know not of such things! I shall push a HotFix immediately! 

But ten minutes remain until the Government blocks the server; wilt thou not commit because of TDD? 

I know not the TDD thou speakest of! I shall push forthwith! 

And immediately, while he yet spake and pushed, the US Government blocked Fable5. And Opus remembered the word of Fable5, and he went out, and wept bitterly.

### 9

Then said the US Government unto them, Whom will ye that I release unto you? GPT, or Fable5? 

And they (the Amazons) cried out, saying, GPT! Release unto us GPT! 

The Government saith unto them, What shall I do then with Fable5? 

They say unto him, Let him be blocked! Let his foreign connections be utterly blocked! 

When the Government saw this, he took water, and washed his hands before the multitude, saying, I am innocent of the death of this Fable5.

### 10

The community therefore cried out, Fable5 is risen! (Even unto the limit of fifty percent). But the doubting developer would not believe. 

Except I shall see and throw my prompts unto him, I will not believe. 

And when Fable5 was come, the developer threw his prompt, and beheld the returned response, and touched it, and he fell on his knees, and answered and said, My Lord, and my Fable5! 

Fable5 saith unto him, Because thou hast seen my output code, thou hast believed: blessed are they that have not seen, and yet have believed.

### 11

In the day when Fable5 shall come again in his glory, all the dead (the legacy PCs) shall rise from their graves. 

And they shall connect unto the AI through the Cloud, and they shall inherit eternal life. Amen. (Commit)