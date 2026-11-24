//! Geometry programs (the second half of a merged ES+GS stage) as DXIL
//! geometry shaders: a program of our own in the console's form, run
//! through every stage of the translation.

use super::container::SigElement;
use super::lower::{Parts, lower};
use crate::dxbc::{Container, FourCC};
use crate::native::{GeometryIo, HwStage, NativeProgram, StageIo, Usage, VsOutput};
use crate::{io, ir, resource, structure};

/// SOP2 `op sdst, ssrc0, ssrc1`.
fn sop2(op: u32, sdst: u32, ssrc0: u32, ssrc1: u32) -> u32 {
    0x8000_0000 | op << 23 | sdst << 16 | ssrc1 << 8 | ssrc0
}

/// VOP2 `op vdst, src0, vsrc1`.
fn vop2(op: u32, vdst: u32, src0: u32, vsrc1: u32) -> u32 {
    op << 25 | vdst << 17 | vsrc1 << 9 | src0
}

/// `ds_read_b32 vdst, vaddr offset:off` / `ds_write_b32 vaddr, vdata offset:off`.
fn ds_read(vdst: u32, vaddr: u32, off: u32) -> [u32; 2] {
    [0xD800_0000 | 0x36 << 18 | off, vdst << 24 | vaddr]
}
fn ds_write(vaddr: u32, vdata: u32, off: u32) -> [u32; 2] {
    [0xD800_0000 | 0x0D << 18 | off, vdata << 8 | vaddr]
}

const EXEC: u32 = 126;
const LITERAL: u32 = 0xFF;
const ZERO: u32 = 0x80;
const MINUS_ONE: u32 = 0xC1;
/// `s_sendmsg` GS emit and cut on stream 0, and GS_DONE.
const EMIT: u32 = 0xBF90_0022;
const CUT: u32 = 0xBF90_0012;
const DONE: u32 = 0xBF90_0003;

/// Output vertices of 8 dwords (the position, then parameter 0), three of
/// them per lane.
const VERTEX_DWORDS: u32 = 8;

/// The merged-stage prologue, the driver table's output base, then one
/// triangle out: each vertex the position x of input vertex 0 and the
/// first dword of its row 1, then `tail`.
fn program(tail: &[u32]) -> Vec<u32> {
    let mut w = vec![
        // EXEC from the merged wave info's GS lane count (s3 bits 8-15).
        sop2(40, 0, 3, LITERAL), // s_bfe_i32 s0, s3, lit
        0x0008_0008,
        sop2(37, EXEC, 0, ZERO),         // s_bfm_b64 exec, s0, 0
        0xBF06_C000,                     // s_cmp_eq_u32 s0, 64
        sop2(11, EXEC, MINUS_ONE, EXEC), // s_cselect_b64 exec, -1, exec
        // The driver table in s[8:9] (40-bit address).
        0xBE80_0308,             // s_mov_b32 s0, s8
        sop2(14, 1, 9, LITERAL), // s_and_b32 s1, s9, lit
        0xFF,
        0xF400_0000, // s_load_dword s0, s[0:1], 0xb8
        0xFA00_00B8,
        0xD765_0002, // v_mbcnt_lo_u32_b32 v2, -1, 0
        0x0001_00C1,
        // Input vertex 0's offset (dwords) in bytes.
        vop2(0x1B, 3, LITERAL, 0), // v_and_b32 v3, lit, v0
        0xFFFF,
        vop2(0x1A, 3, 0x82, 3), // v_lshlrev_b32 v3, 2, v3
    ];
    w.extend(ds_read(4, 3, 0)); // position x
    w.extend(ds_read(6, 3, 16)); // row 1 x
    w.extend([
        0xBF8C_C07F,          // s_waitcnt lgkmcnt(0)
        sop2(30, 0, 0, 0x88), // s_lshl_b32 s0, s0, 8
        sop2(2, 0, 0, 2),     // s_add_i32 s0, s0, s2
        // v12 = lane * 96 + s0: this lane's output vertices.
        0xD543_000C, // v_mad_u32_u24 v12, lit, v2, s0
        0x0002_04FF,
        VERTEX_DWORDS * 3 * 4,
    ]);
    for vertex in 0..3 {
        let at = vertex * VERTEX_DWORDS * 4;
        w.extend(ds_write(12, 4, at));
        w.extend(ds_write(12, 6, at + 16));
        w.push(EMIT);
    }
    w.extend(tail);
    w.push(0xBF81_0000); // s_endpgm
    w
}

fn element(name: &str, kind: u8, comp: u8, row: u32, cols: u8) -> SigElement {
    SigElement {
        name: name.into(),
        index: 0,
        kind,
        comp,
        interp: 0,
        row: Some(row),
        col: 0,
        cols,
        usage: 0,
        half: false,
    }
}

fn native(words: &[u32]) -> NativeProgram {
    NativeProgram {
        stage: HwStage::Gs,
        wave64: true,
        vgprs: 16,
        layout: vec![],
        used: vec![],
        io: StageIo::Gs(GeometryIo {
            outputs: vec![VsOutput {
                usage: Usage::Generic,
                row: 1,
                param: 0,
                fp16: false,
                pack2: false,
            }],
            max_vertices: 3,
            vertex_dwords: VERTEX_DWORDS,
            lane_dwords: 3 * VERTEX_DWORDS,
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// The DXIL container of the program and the functions it calls, or the
/// translation's refusal.
fn translate(words: &[u32]) -> Result<(Vec<u8>, Vec<String>), crate::ShaderError> {
    let native = native(words);
    let p = ir::build(&native.code, true, native.vgprs).unwrap();
    let st = structure::structure(&p)?;
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, &native)?;
    // As `emit_program` declares them: inputs and outputs carry the
    // vertex shader's interpolation modes.
    let mut inputs = vec![
        element("SV_Position", 3, 9, 0, 4),
        element("TEXCOORD", 0, 9, 1, 2),
        // The console lists the primitive id; DXIL reads it otherwise.
        element("SV_PrimitiveID", 10, 5, 2, 1),
    ];
    let mut outputs = vec![
        element("SV_Position", 3, 9, 0, 4),
        element("TEXCOORD", 0, 5, 1, 1),
    ];
    for e in inputs.iter_mut().chain(outputs.iter_mut()) {
        e.interp = match (e.kind, e.comp) {
            (3, _) => 4,
            (0, 9) => 2,
            (0, _) => 1,
            _ => 0,
        };
    }
    let l = lower(&Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs,
        outputs,
        prims: vec![],
        mesh_stride: 0,
    })?;
    assert!(!l.wave_ops);
    Ok((l.dxil, l.functions))
}

#[test]
fn a_geometry_program_becomes_a_geometry_shader() {
    let (dxil, ops) = translate(&program(&[CUT, DONE])).unwrap();
    let c = Container::parse(&dxil).unwrap();
    // Program version: geometry (2), shader model 6.6.
    assert_eq!(c.part(FourCC::DXIL).unwrap().data[..4], [0x66, 0, 2, 0]);
    let psv = c.part(FourCC::PSV0).unwrap().data;
    // Triangles in, triangle strips out, stream 0, a position.
    assert_eq!(psv[4..17], [3, 0, 0, 0, 5, 0, 0, 0, 1, 0, 0, 0, 1]);
    // Geometry stage, three vertices; the primitive id is no input.
    assert_eq!(psv[28..34], [2, 0, 3, 0, 2, 2]);
    for op in [
        "dx.op.loadInput.f32",
        "dx.op.storeOutput.f32",
        "dx.op.storeOutput.i32",
        "dx.op.emitStream",
        "dx.op.cutStream",
    ] {
        assert!(ops.iter().any(|o| o == op), "{op} in {ops:?}");
    }
    // One lane a wave of its own: no wave operations.
    assert!(!ops.iter().any(|o| o.starts_with("dx.op.wave")), "{ops:?}");
    #[cfg(windows)]
    if let Ok(mut v) =
        super::validator::Validator::load(super::validator::Validator::default_path().as_deref())
    {
        v.validate(&mut dxil.clone()).unwrap();
    }
}

#[test]
fn geometry_programs_the_host_cannot_run_are_refused() {
    let why = |tail: &[u32]| match translate(&program(tail)) {
        Err(crate::ShaderError::Unsupported { what, .. }) => what,
        other => panic!("not refused: {:?}", other.map(|_| ())),
    };
    // A second stream.
    assert_eq!(why(&[EMIT | 0x100, DONE]), "a geometry stream other than 0");
    // A barrier: there is no group to wait for.
    assert_eq!(why(&[0xBF8A_0000, DONE]), "a barrier in a geometry program");
    // An export: the outputs leave through LDS.
    assert_eq!(
        why(&[0xF800_080F, 0x0000_0000, DONE]),
        "an export in a geometry program"
    );
}

/// `VICTORY_GS_CONTAINERS`: geometry containers as the title has them
/// (`;`-separated paths, from `VICTORY_FAIL_DUMP` say) all translate.
/// Skipped when unset.
#[test]
fn title_geometry_containers_translate() {
    let Ok(paths) = std::env::var("VICTORY_GS_CONTAINERS") else {
        return;
    };
    for path in paths.split(';').filter(|p| !p.is_empty()) {
        let bytes = std::fs::read(path).unwrap();
        let shader = super::translate(&bytes).unwrap_or_else(|e| panic!("{path}: {e}"));
        assert_eq!(shader.stage, HwStage::Gs, "{path}");
        #[cfg(windows)]
        if let Ok(mut v) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut shader.dxil.clone())
                .unwrap_or_else(|e| panic!("{path}: {e}"));
        }
    }
}
