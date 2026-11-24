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

## The ranked list

Each item is ranked by how many frames' worth of work it unblocks and by how stable the result is.

### 1. Never hang the GPU (stability floor)
A hang kills the session, so everything else depends on this.
- **Done:** 578 refuses wave64 programs that assume 64 lanes; 334 adds buffer transitions; 949 stops dynamic descriptor fragmentation.
- **Next, the DRED-named hang:** mesh `db4835a364449a45` reads several null descriptor slots. A null view reads zeros, and
  a console loop that walks data until a terminator or a count it never reaches then spins until `LOOP_LIMIT` (2^20 per lane,
  per loop, so nested loops take seconds) and trips the 2 s TDR. Two fixes:
  a. Find why those slots are null (the descriptor isn't created yet, was freed, or its kind isn't supported) and fix the cause.
  b. A watchdog. Lower the loop limit for mesh and pixel stages (2^14 or so), and cap total iterations per lane across all
     loops, so a bad input costs a wrong triangle rather than the device. This costs nothing when inputs are right.
- **Check:** the mesh relaxation (`VICTORY_MESH_WAVE64`, below) now logs every wave64 mesh pipeline it lets through. If
  `db4835a364449a45` is in that list and hangs, it goes back to refused until a/b land.

### 2. Get the dropped pipelines running (biggest visual win)
About 2,300 draws a frame have no host pipeline. Two causes:
- **578 refusals (wave64).** Mesh: lane-independent wave64 mesh programs now run (no wave ops, LDS ordered by barriers;
  `VICTORY_MESH_WAVE64=strict` reverts). Compute (`cedbff472d562f54`, `13dfdd3b1136e407`, and others hidden by
  "logged once"): PR 233's split and console flow, being merged into a `nvidia-lanemask` branch, then step 6 (multi-wave groups with LDS).
- **Translation gaps**, by count of shaders (each shader can drop many draws):

| Gap | Shaders | Owner | Approach |
|---|---|---|---|
| select on an untracked mask | 123 | translation thread | track masks through selects and phis like EXEC |
| descriptor read as data | 114 | translation thread | side table of console descriptor words (size, format, base) readable by the shader |
| lane mask holding lanes elsewhere | 108 | this thread (`nvidia-lanemask`) | console flow for wave32 programs: every lane runs every block, ballots are the wave's own (no exchange needed) |
| structured buffer of unknown stride | 56 | translation thread | take the stride from the view the title creates at bind time, or fall back to raw |
| SGPR address | 35 | translation thread | follow addresses through scalar registers |

Order: lane mask and untracked mask first, since they are the same masking model and fix the most shaders.

### 3. Indirect draws (about 4,945 dropped)
Indirect execution needs the argument clamp, and the argument buffers are written by the title's own compute passes. Most
drops are probably downstream of item 2: a compute pass that is refused or untranslated never writes its arguments.
Re-measure after item 2. Then log drops by reason (no clamp pipeline, no command signature, missing pipeline) so the rest
can be ranked.

### 4. Frame rate (4 fps, target 30)
Do this only once the frame is complete, since the profile changes completely as passes come back. Known costs today:
copying changed title memory back after each submission (`COPY_BACK`), loop limits on slow passes, and pipelines built
on first use. `VICTORY_GPU_TIMING` already logs slow submissions. The pipeline cache on disk means second runs are
the ones to measure.

### 5. Upstream
Upstream each piece once it has run clean for a full session: the mesh relaxation as a follow-up to 578, the watchdog,
and the lane-mask console flow on top of 233. XCL's design notes may inform ideas, but no XCL code is used (GPL-2.0-only
vs VIctory's GPL-3.0).

## Scorecard (one line per nvidia head)
Head commit, minutes without a hang, fps, draws per frame, draws dropped (no pipeline), indirect dropped, translation
warnings by reason, 578 refusals, wave64 mesh let through, and whether the picture shows anything.
