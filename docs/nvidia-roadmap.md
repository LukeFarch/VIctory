# Ray tracing and DLSS on an NVIDIA card: plan

Target machine: RTX 5070 Ti 16 GB, Ryzen 7 7700X, branch `nvidia-fx`.
Written 2026-11-24 from the code at `208f4f6`, `docs/fork-ray-tracing.md`, the
static translation list of the whole shader pack (`translate-list-5.txt`) and
the run log `run8a.log`.

## Where things stand

**Ray tracing.** The title does not use a ray-tracing API. Its traversal is
ordinary compute code, its static structures ship pre-built in AMD's node
format, its dynamic structures are built every frame by 52 compute programs
from `XboxBvhBuilder.dll`, and only the per-node test is hardware
(`image_bvh_intersect_ray` / `image_bvh64_intersect_ray`). The fork already
emulates that instruction in DXIL (checked bit for bit against
`crates/victory-shader/src/bvh.rs` on the GPU) and reads title memory by raw
address from shaders.

In the latest static translation of the pack:

| Part | State |
|------|-------|
| Builder programs (`XboxBvhBuilder.dll`) | 52 of 52 translate and validate |
| `sr_tracing` (reflections, GI), `rtdirectionalshadows` | all translate and validate |
| `shadowmask:CS_TraceShadowRays_{Full,FullAdaptive,Half,Quarter,Eighth}` | fail: `emit: a descriptor read only as data` |
| `sr_tracing:CS_Tiling_Generate_Opaque` (#2cc4c197ee29e291) | translates, but the host refuses the pipeline (`WaveSize(64)`, which NVIDIA does not run) |

At run time the builder is still off (`VICTORY_BVH_BUILDER` unset), so
nothing traced sees a structure, and interiors stay black.

**Frame rate.** `run8a.log`, in the world: about 170 ms per frame, of which
about 165 ms the card sits idle while the submitting thread works; the card
is about 56 % busy. The wall is on the CPU side (recording and submitting,
about 95 MB of uploads per frame), not shading. Rendering fewer pixels
(DLSS, a lower dynamic resolution) will not raise the frame rate much until
that is fixed; it is owned by the "Fix NVIDIA GPU crash" thread. DLSS is an
image-quality step for now, and a frame-rate step later.

## Decision (a): emulate the AMD structures, do not map them onto DXR

Recommended: keep the present approach (the title's own builder and
traversal run as translated compute; the node test is emulated).

Mapping onto DXR 1.1 `RayQuery` was considered and rejected for now:

- There are no build calls to intercept. Dynamic structures are built by the
  title's own compute programs into plain title memory; static ones are
  loaded from data. D3D12 acceleration structures need vertex and index
  buffers (or AABBs) and instance descriptions, which the title never hands
  over. They could be recovered by walking the AMD nodes (triangle nodes hold
  their vertices), but that is a per-frame re-build of every structure on top
  of the title's own.
- The traversal loops are the title's code, in 64 programs. A `RayQuery`
  replaces a whole loop, not one instruction: each loop would have to be
  recognised and replaced, and its results (node addresses, primitive and
  instance indices in AMD's layout) fed back to shading code that expects
  them. Brittle across 64 programs and title updates.
- The emulated node test is a few slab tests and a sort per call; on a card
  this size the cost is probably acceptable. It is unmeasured; if it turns
  out not to be, stage RT6 below is the place to revisit DXR.

## Decision (b): DLSS through Streamline, at the title's temporal upscale

**What the title exposes.** The pack has a temporal anti-aliasing and
upscaling chain that DLSS can stand in for:

| Pass | Hash (`#`) |
|------|------------|
| `TAA2U:CS_TAA2_Resolve` (the upscaling form) | `22f4971293b8fdcc` |
| `TAA2:CS_TAA2_Resolve` | `df11dcd012cd7382` |
| `TemporalAA:CS_ApplyTemporalAA` | `ac8367cf999810a6` |
| `FSRCreateMasks:CS_CreateMasks` (reactive masks for an FSR path) | `2a961927bdc8230d` |

plus motion-blur passes that read a velocity buffer, `postfx:CS_DepthUpscale`,
and dynamic resolution (the present plane names the sub-rectangle drawn into;
`fix/present-source-rect` shows that part).

DLSS Super Resolution needs, at render resolution: jittered colour, depth,
motion vectors, and the jitter offset; and an output at display resolution.
These are exactly the inputs of a TAAU resolve. Which of the passes above runs
in a frame, and which bindings are colour, depth, velocity and history, is not
known yet: the `VICTORY_PROBE` switch (branch `rt/probe`) logs it.

**Licences.** Streamline's source is MIT (its `license.txt`), compatible with
GPL-3.0. The DLSS runtime DLLs (`sl.dlss.dll`, `nvngx_dlss.dll`, and
`sl.dlss_g.dll` for Frame Generation) are prebuilt NVIDIA binaries under the
NVIDIA RTX SDK licence, not GPL-compatible to ship together with this code.
So: nothing of NVIDIA's is committed to this repository; the feature is off
by default and loads the DLLs at run time from a folder the user points to.
A personal build that is not distributed has no GPL distribution question.
(Not legal advice.)

**Integration.** VIctory makes its own D3D12 device, so Streamline goes in
"manual hooking" mode (`slInit` with `eUseManualHooking`, then
`slSetD3DDevice`): no proxy device, no swap-chain interception. At a dispatch
of the chosen resolve pipeline, the layer tags the resources it already has
bound (`slSetTag`: colour, depth, motion vectors, output), sets the constants
(jitter from the pass's constants, matrices if needed, `mvecScale`, reset on
camera cuts), evaluates DLSS into the pass's output UAV, and drops the title's
dispatch. Quality mode and DLAA are chosen by switch.

**Frame Generation** is not worth doing yet: it needs a base frame rate of
roughly 40 fps or more to feel right, needs Reflex markers and the present
path, and adds latency. Revisit after the frame rate work.

**Fallback** if the inputs turn out not to be usable: a spatial upscale at
present (NVIDIA Image Scaling or FSR 1, both MIT) of the dynamic-resolution
sub-rectangle, in place of the window system's bilinear stretch. Cheap, no
inputs needed, works on any card.

## Stages

Each stage is a branch off `nvidia-fx`; runs go through the "Get GTA 6
running on PC" thread.

| Stage | What | Needs |
|-------|------|-------|
| P0 | `rt/probe`: `VICTORY_PROBE=<hash>,...` logs the bindings, user data, root constants and constant-buffer words of chosen passes (diagnostics builds) | done, needs a run |
| RT1 | Builder on, first in quarantine: `VICTORY_BVH_BUILDER=1 VICTORY_BVH_QUARANTINE=1`; check every builder pipeline is made on the 5070 Ti and nothing crashes | a run |
| RT2 | The five `CS_TraceShadowRays` programs: the descriptor whose dwords are read as data (the structure's address taken from a descriptor) | the containers (`VICTORY_DUMP_MATCH`), then a translator change |
| RT3 | `sr_tracing:CS_Tiling_Generate_Opaque` without `WaveSize(64)` | shared with the wave64 work in shader translation |
| RT4 | Builder on, quarantine off; interiors lit? GPU cost of tracing measured | RT1 to RT3 |
| RT5 | Title pages with no host copy (a read gives zero, a store is dropped) | RT4 shows whether it matters |
| DL1 | From the P0 run: name the resolve pass and its inputs | P0 run |
| DL2 | Streamline in manual-hooking mode, DLSS SR (and DLAA) at that pass, off by default, DLLs supplied by the user | DL1 |
| DL3 | Present-time spatial upscale (NIS or FSR 1) of the sub-rectangle, as fallback and for the UI plane | none |
| RT6 | Only if RT4 shows tracing too slow: recognise the traversal loop and use DXR `RayQuery` over structures rebuilt from the AMD nodes | RT4 numbers |
| DL4 | Frame Generation | base frame rate of about 40 fps |
