# NVIDIA roadmap: from "runs" to stable, playable frames

Branch `nvidia` in LukeFarch/VIctory is the integration branch. Every fix lands here first, the test PC runs each new
head and posts a scorecard, and finished pieces go upstream as fork branches linked from issues (VIctory takes PRs only
from collaborators). Test GPU: RTX 5070 Ti (32-lane waves only). Status as of 2026-11-24, runs 1, 2 and A.

## Where we are

| Run | Build | Result |
|---|---|---|
| 1 | main | GPU hung at 3m14s (DXGI_ERROR_DEVICE_HUNG) |
| A | main + DRED | Hung at 5m47s in an indirect mesh dispatch: mesh `db4835a364449a45`, pixel `c10e7d95d8266d2a` |
| 2 | main + 334 + 578 + 949 | No hang in 10 min. About 4 fps, 347 draws a frame, about 2,300 dropped (no host pipeline), about 4,945 indirect draws dropped. Black picture, audio fine |

The picture is black because most of the frame is dropped, not because one thing is broken. The lighting and composite
passes need the earlier passes' output, so a dark scene follows from any large dropped share.

## Milestones (Luke's scale)

**Where NVIDIA stands: 55, partly 70.** Runs 1, 2 and A get through boot, intro video, start screen, profile and saves,
and into the world (the README's status matches). The menus respond to input, so 55 is reached. 70 is only partial: the
world loads and the game runs (audio plays, no hang with 578), but the picture is black and moving around hasn't been checked.

| # | Milestone | NVIDIA status | What blocks it | Work that clears it |
|---|---|---|---|---|
| 1 | 0-25 Groundwork | done | – | – |
| 2 | 25 Exe runs | done | – | – |
| 3 | 40 First frame | done | – | – |
| 4 | 55 Main menu | done | – | – |
| 5 | 70 In-game | **partial**: world loads, runs, renders black | Dropped work (A, B, C below), and hangs on main without 578 | A, B, C |
| 6 | 85 30 FPS | not started (4 fps, frame incomplete) | Profile only means something once the frame is complete | D |
| 7 | 100 Stable | not started | GPU hangs (S), memory and descriptor growth over long sessions | S, then long soak runs |

### S. Never hang the GPU (needed for every milestone from 70 up)
- **Done on `nvidia`:** 578 (refuse wave64 programs that assume 64 lanes), 334 (buffer transitions), 949 (descriptor
  fragmentation), and a **watchdog**: graphics-stage loops stop at 2^14 iterations instead of 2^20, so a loop walking
  zeros from a null descriptor costs a wrong triangle, not the device. That is run A's hang (indirect mesh
  `db4835a364449a45` reading null slots).
- **Next:** find why those slots are null (not created yet, freed, or an unsupported view kind). The relaxed mesh path
  logs every wave64 mesh pipeline it lets through. If a hang names one of them, `VICTORY_MESH_WAVE64=strict` takes it out.
- **For 100:** hour-long soak runs with DRED on, and watch descriptor heap use and committed memory over time.

### A. Get the refused wave64 programs running (70)
- Mesh: lane-independent wave64 mesh programs now run on `nvidia` (no wave ops; LDS absent, shared by several waves, or ordered by barriers).
- Compute (`cedbff472d562f54`, `13dfdd3b1136e407`, plus others hidden because refusals are logged once): PR 233's split and
  console flow on the `nvidia-lanemask` branch, then 233's step 6 (multi-wave groups with LDS).

### B. Close the translation gaps (70), by number of shaders

| Gap | Shaders | Owner | Approach |
|---|---|---|---|
| select on an untracked mask | 123 | translation thread | track masks through selects and phis like EXEC |
| descriptor read as data | 114 | translation thread | side table of console descriptor words (size, format, base) the shader can read |
| lane mask holding lanes elsewhere | 108 | this thread (`nvidia-lanemask`) | console flow for wave32 programs: every lane runs every block, ballots are the wave's own |
| structured buffer of unknown stride | 56 | translation thread | all bindless heap buffers, mostly `CS_RaytracingResolve` (ray-traced lighting): needs a host-side stride table per heap slot, filled when the title creates the view |
| SGPR address not a root parameter | 35 | – | all `GS_VoxelizeGeneric`: VIctory doesn't run the console's merged geometry stage yet. That's a feature (merged ES/GS on the host), not a translation fix, and comes after the frame is otherwise complete |

### C. Indirect draws (70): about 4,945 dropped
The title's own compute passes write the arguments, so most of these drops probably come from A and B (a pass that never
runs writes no arguments). Re-measure after A and B. Then log drops by reason so the rest can be ranked.

### D. Frame rate (85)
Round 1 copy statistics (93f94d9, diagnostics build):

| Stretch | Frame | Preparing (copies) | GPU | Copy back | Not accounted for |
|---|---|---|---|---|---|
| Heavy, early in the world | 1,017 ms | 373 ms (250 ms, 409 MB) | 82 ms | 57 ms | ~500 ms |
| Calm, later | 445 ms | 51 ms (35 ms, 4.3 MB) | 65 ms | 17 ms | ~310 ms |

What this shows:
- **Steady state is not copy-bound.** In the calm stretch, dirty-page tracking works: 293 MB checked, 288 MB unchanged,
  4.3 MB copied. Most of the frame is time that none of these counters cover. Before optimizing anything else, that needs
  a CPU profile. Windows Performance Recorder is built in (`wpr -start CPU`, play 20 s in the world, `wpr -stop world.etl`,
  open it in Windows Performance Analyzer). Likely suspects: submissions run synchronously (the title's thread waits for the GPU
  and the copy back before it continues, so CPU and GPU never overlap), shader and pipeline creation on the title's
  threads, and HLE calls.
- **The heavy stretch is warm-up.** First copies of the world's big buffers (409 MB a frame through `ReadProcessMemory`)
  and about 10,000 page faults a frame. Part of that is the per-submission snapshot of shader-written ranges, which
  branch `perf/snapshot` now keeps between submissions.
- **Write-watch checks cost 35 ms a frame** in the calm stretch (72 checks). Checking each buffer once per frame
  instead of per submission would cut that.

Plan, in order:
1. `perf/snapshot` (done, needs a run): no per-submission snapshot copy, fewer page faults.
2. CPU profile of a calm stretch, to find the unaccounted ~310 ms.
3. Most likely fix from that: **asynchronous submission.** Let the title continue while the GPU runs, and copy back
   shader writes when the title next waits on that fence, not inline.
4. Write-watch checks once per buffer per frame.
5. GPU side, once the CPU side is fixed: hot static buffers in video memory instead of system memory.

### Upstream
Upstream each piece after it runs clean for a full session: the mesh relaxation as a follow-up to 578, the watchdog, and
the lane-mask console flow on top of 233. XCL's design notes may inform ideas, but no XCL code is used (GPL-2.0-only vs
VIctory's GPL-3.0).

## Scorecard (one line per nvidia head)
Head commit, minutes without a hang, fps, draws per frame, draws dropped (no pipeline), indirect dropped, translation
warnings by reason, 578 refusals, wave64 mesh let through, and whether the picture shows anything.
