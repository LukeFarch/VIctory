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
| structured buffer of unknown stride | 56 | translation thread | stride from the view the title creates at bind time, or fall back to raw |
| SGPR address | 35 | translation thread | follow addresses through scalar registers |

### C. Indirect draws (70): about 4,945 dropped
The title's own compute passes write the arguments, so most of these drops probably come from A and B (a pass that never
runs writes no arguments). Re-measure after A and B. Then log drops by reason so the rest can be ranked.

### D. Frame rate (85): the per-frame copy of title memory is the blocker
Round 1 (93f94d9): menus run at 30 fps, but world frames are CPU-bound. Preparing submissions spends 150-230 ms a frame
copying title memory (400-900 ms with diagnostics on). The GPU is not the limit yet.

What already exists: copies are by 64 KiB page and skip pages the title hasn't written (write watching, or guard faults
on physical-page views), and shader writes are copied back by piece.

Where the time most likely goes, in order of suspicion (to confirm from the per-frame `copies per frame` line and the top
`copied from` ranges, logged by diagnostics builds every 5 s):
1. **Snapshot and compare around shader writes.** Each submission that lets shaders write the world's 174 MB heap first
   snapshots the whole declared range (`Shadow::writing`), then afterwards compares all of it (`Shadow::copy_back`). That is
   several hundred MB of memory traffic per submission, times the submissions per frame.
2. **Copies of pages the title rewrites every frame** (constants, command memory sharing the big heap).
3. **Diagnostics overhead** (per-range stats), which explains the 400-900 ms.

Candidate wins, cheapest first:
- **Keep the snapshot between submissions.** Update it where the copy changes (refreshes, copy-back) instead of re-copying the
  whole range at every submission. This removes one full pass over the range.
- **Find changes on the GPU.** Copy the declared range on the GPU at submission start (cheap there), have a small compute pass
  compare it after the submission, and read back a bitmap of changed 64 KiB pieces. The CPU then touches only those pieces.
  This removes both passes and is the real fix.
- **Fewer submissions per frame:** batch the title's command lists into one host submission where their fences allow.
- **Put hot buffers in GPU memory** (constants rewritten every frame stay in system memory, the static world heap moves
  to video memory). This is the GPU side, for once the CPU side is fixed.

### Upstream
Upstream each piece after it runs clean for a full session: the mesh relaxation as a follow-up to 578, the watchdog, and
the lane-mask console flow on top of 233. XCL's design notes may inform ideas, but no XCL code is used (GPL-2.0-only vs
VIctory's GPL-3.0).

## Scorecard (one line per nvidia head)
Head commit, minutes without a hang, fps, draws per frame, draws dropped (no pipeline), indirect dropped, translation
warnings by reason, 578 refusals, wave64 mesh let through, and whether the picture shows anything.
