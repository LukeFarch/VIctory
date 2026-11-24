# NVIDIA roadmap

Integration branch: `nvidia-fx`. This file collects plans for the larger pieces of work that branch still needs. Each
section says what is missing, how the console does it, the host mapping chosen and why, and what is done.

## Geometry stage (merged ES+GS)

### What is missing

VIctory did not run the console's geometry stage at all. Every pipeline with a geometry shader was refused ("only
vertex, mesh, pixel and compute stages are translated"), and the shader census lists the title's geometry programs
(all `GS_VoxelizeGeneric_*`, about 35 distinct programs) as "an address in an SGPR that is no root parameter". They
voxelize the scene: each triangle is projected along its dominant axis, which by the shape of the pass feeds the
voxel lighting. With them dropped, the lighting they feed is missing. It is not what makes the frame black (that is
the dropped culling passes and draws), but the frame cannot be right without them.

### How the console runs it

RDNA2 has no separate geometry hardware stage. The driver merges the title's vertex shader (compiled as an ES, the
"vertex shader for use with a geometry shader" that the vertex container carries next to the plain VS) and its
geometry shader into one hardware stage, and its outputs leave through the primitive (NGG) path. What the geometry
container holds, as read from the title's containers:

* One native program in the GS slot of the FSSM payload, wave64 here. Its stage data has the VS layout shifted by
  0x88: outputs of 6 bytes at `E+0x150` (usage, signature row, parameter, row again, slot, flags), their count at
  `E+0x210`; `E+0x27c` dwords per lane of output (60), `E+0x284` the maximum vertex count (3), `E+0x28c` dwords per
  output vertex (20). Inputs are not listed: they are the container's `ISG1`.
* The program starts like every merged stage: `s3` is the merged wave info (bits 8-15 the GS lanes of this wave,
  16-23 the wave's index), and EXEC is built from it (`s_bfm_b64 exec`). User data starts at `s8`; the title's root
  parameters start at user-data dword 8, so `s[8:9]` (dwords 0-1) is a driver table the root signature does not
  describe. The voxelizer loads one dword from it, at 0xb8: the start of the output area in LDS, in 256-byte units.
* **Inputs** arrive in LDS, written by the ES half: the GS gets each input vertex's LDS offset in dwords (16 bits
  each: v0 vertices 0 and 1, v1 vertices 2 and 3, v4 vertices 4 and 5), and reads input register `r` at dwords
  `4r..4r+3` from there. v2 is the primitive id, v3 the GS instance.
* **Outputs** are written to LDS too, at the table's output base plus `s2`, plus the lane times the per-lane output
  size: vertex `k` at `20k` dwords, slot 0 the position, slot `1 + p` the output exported as parameter `p`. After
  writing a vertex the program sends `s_sendmsg GS_EMIT` (stream in bits 8-9), ends a strip with `GS_CUT`, and ends
  with `GS_DONE`. The driver's own epilogue then exports the vertices from LDS.

### Host mapping: a DXIL geometry shader

Three ways were weighed:

| Mapping | For | Against |
|---|---|---|
| **DXIL geometry shader** (chosen) | The host does vertex fetch, primitive assembly, strips and cuts as the console's API defines them; the translated VS is reused as it is; one shader per stage, no extra passes; the branch already emits geometry shaders (the rectangle stage) | GS performance on some hosts; streams other than 0 and very large outputs need care |
| Mesh shader emulation (NGG-like: ES and GS in one group, LDS as group-shared) | Closest to how the console runs it | The mesh shader must fetch indices and vertices itself (input assembler emulation, formats, restart), and the ES variant must be translated as well; much larger |
| Compute emulation (ES pass, GS pass to a buffer, indirect draw) | Most general | Extra passes, buffers and synchronisation per draw, and the same input assembler emulation |

The geometry shader is chosen because everything the console's merged stage does maps onto it once the LDS traffic
is turned back into stage I/O, and it needs no new host machinery:

1. **Vertex stage**: the plain VS program of the vertex container, translated as today. Its outputs are the
   container's `OSG1`, the same registers the geometry container's `ISG1` reads.
2. **Geometry stage**: the GS program, translated with:
   * LDS as per-invocation private memory (a static array in the DXIL), not group-shared memory: one host invocation
     is one console lane.
   * The console's lane forced to 0, and the merged wave info set to one GS lane, so EXEC is that lane and every
     address the program computes from its lane is the one for lane 0.
   * Input vertex `i` placed at LDS dword `128 i` (the offsets in v0, v1 and v4 say so); at entry every element of the
     input signature is copied in, `loadInput` of vertex `i` into dwords `128 i + 4 r + c`. Input reads then work for
     any address arithmetic.
   * The driver table read as constants: dword 0xb8 gives an output area that starts past the input vertices; other
     table dwords are refused.
   * `GS_EMIT`: the next output vertex (a counter in private memory) is read back from the output area and stored to
     the outputs, then `EmitStream`; `GS_CUT` is `CutStream`; `GS_DONE` does nothing. Stream 0 only.
   * The declared maximum vertex count from the stage data, input primitive and output topology from the container's
     `PSV0`.
3. **Pipeline**: a graphics pipeline with VS, GS and PS; the GS gets its own root constants for its user data, its
   own static samplers and root views, like the other stages.

### Assumptions to check on a real run

* Output slot `1 + p` is the output with parameter `p` (in the voxelizer parameters and signature rows coincide).
* `s2` is 0 for the first GS wave of a group. The voxelizer adds it to the output base, which is all that matters
  when the host invocation is lane 0.
* Only triangle input and triangle-strip output are handled; other kinds are refused with a reason.

### Status

* Done on `gs/merged-stage`: stage data, driver table, the GS lowering (private LDS, inputs, emit and cut), and
  geometry stages in host pipelines.
* Next: a run on the test PC. Expect `GS_VoxelizeGeneric_*` to translate; then watch for host pipeline refusals that
  name signature linkage (VS outputs to GS inputs, GS outputs to the pixel shader).
