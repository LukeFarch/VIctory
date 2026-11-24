//! Lowering-only tests of synthetic GFX10 programs (no host GPU needed).

use super::lower;
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{ShaderError, io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

const fn s_mov_b32(sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | 0x03 << 8 | src0 as u32
}
const fn sop2(op: u8, sdst: u8, src0: u8, src1: u8) -> u32 {
    0x8000_0000
        | (op as u32 & 0x7F) << 23
        | (sdst as u32 & 0x7F) << 16
        | (src1 as u32) << 8
        | src0 as u32
}
const fn sopc(op: u8, src0: u8, src1: u8) -> u32 {
    0xBF00_0000 | (op as u32) << 16 | (src1 as u32) << 8 | src0 as u32
}
const fn s_load_dwordx4(dst: u8, base: u8) -> [u32; 2] {
    [
        0x3D << 26 | 0x02 << 18 | (dst as u32) << 6 | (base as u32 >> 1),
        125 << 25,
    ]
}
const fn vop1(op: u16, vdst: u8, src0: u16) -> u32 {
    0x7E00_0000 | (vdst as u32) << 17 | (op as u32) << 9 | (src0 as u32 & 0x1FF)
}
const fn vop3(op: u16, vdst: u8, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
    [
        0x35 << 26 | (op as u32 & 0x3FF) << 16 | vdst as u32,
        (src0 as u32 & 0x1FF) | (src1 as u32 & 0x1FF) << 9 | (src2 as u32 & 0x1FF) << 18,
    ]
}
const fn buffer_store_dword(vdata: u8, srsrc: u8, byte_off: u16) -> [u32; 2] {
    [
        0x38 << 26 | 0x1C << 18 | (byte_off as u32 & 0xFFF),
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8,
    ]
}

/// The usual layout: 4 root constants in s0..s3, a table in s4 whose
/// first descriptor is a raw UAV.
fn usual_layout() -> (Vec<RootParam>, Vec<UsedResource>, u8) {
    (
        vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        5,
    )
}

/// Lowers a 32-thread compute program with the usual layout: the UAV's
/// descriptor is loaded into s[8:11] and the root constants copied to
/// v1..v4; `body` leaves its result in v0, which is stored.
fn lower_cs(body: &[u32]) -> Result<(), ShaderError> {
    let (layout, used, sgprs) = usual_layout();
    lower_cs_with(layout, used, sgprs, body)
}

fn lower_cs_with(
    mut layout: Vec<RootParam>,
    mut used: Vec<UsedResource>,
    user_sgprs: u8,
    body: &[u32],
) -> Result<(), ShaderError> {
    let (l, u, _) = usual_layout();
    for (k, p) in l.into_iter().enumerate() {
        if layout.len() <= k {
            layout.push(p);
        }
    }
    if !used.iter().any(|x| x.kind == ResourceType::Uav) {
        used.extend(u);
    }
    let mut words = vec![s_mov_b32(5, 0x80)];
    words.extend_from_slice(&s_load_dwordx4(8, 4));
    for i in 0..4u8 {
        words.push(vop1(0x001, i + 1, u16::from(i)));
    }
    words.extend_from_slice(body);
    words.extend_from_slice(&buffer_store_dword(0, 8, 0));
    words.push(S_ENDPGM);
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout,
        used,
        io: StageIo::Cs(ComputeIo {
            rsrc2: u32::from(user_sgprs) << 1,
            threads: [32, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs)?;
    let st = structure::structure(&p)?;
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, &native)?;
    lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    })?;
    Ok(())
}

#[test]
fn a_select_on_a_uniform_mask_lowers() {
    // s_cmp_eq_u32 s0, s1; s_cselect_b32 s20, -1, 0; v_cndmask_b32_e64 v0, 0, 1, s20
    let mut body = vec![sopc(0x06, 0, 1), sop2(0x0A, 20, 0xC1, 0x80)];
    body.extend_from_slice(&vop3(0x101, 0, 0x80, 0x81, 20));
    lower_cs(&body).unwrap();
}

#[test]
fn a_select_on_a_constant_mask_lowers() {
    // s_mov_b32 s20, 0x55555555 (literal); v_cndmask_b32_e64 v0, 0, 1, s20
    let mut body = vec![s_mov_b32(20, 0xFF), 0x5555_5555];
    body.extend_from_slice(&vop3(0x101, 0, 0x80, 0x81, 20));
    lower_cs(&body).unwrap();
}

#[test]
fn a_select_on_a_mask_held_as_a_number_lowers() {
    // v_cndmask_b32_e64 v0, 0, 1, s2 (a mask passed in user data)
    let body = vop3(0x101, 0, 0x80, 0x81, 2);
    lower_cs(&body).unwrap();
}

/// SMEM `s_load_dword sdst, s[base:base+1], offset` (soffset = null).
const fn s_load_dword(dst: u8, base: u8, offset: u32) -> [u32; 2] {
    [
        0x3D << 26 | (dst as u32) << 6 | (base as u32 >> 1),
        125 << 25 | (offset & 0x1F_FFFF),
    ]
}

/// A root CBV in s[6:7] (after the usual parameters).
fn root_cbv(kind: ResourceType) -> (Vec<RootParam>, Vec<UsedResource>, u8) {
    let (mut layout, mut used, _) = usual_layout();
    layout.push(RootParam {
        root: RootType::Descriptor,
        table_offset_dwords: 0,
        sgprs: 2,
        base_register: 0,
        user_data: 6,
        slot: 2,
    });
    used.push(UsedResource {
        kind,
        root: RootType::Descriptor,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords: 2,
        user_data: 6,
        slot: 2,
    });
    (layout, used, 8)
}

#[test]
fn a_scalar_load_from_a_root_cbv_reads_the_constants() {
    // s_load_dword s12, s[6:7], 0x14; v_mov_b32 v0, s12
    let mut body = s_load_dword(12, 6, 0x14).to_vec();
    body.push(vop1(0x001, 0, 12));
    let (l, u, n) = root_cbv(ResourceType::Cbv);
    lower_cs_with(l, u, n, &body).unwrap();
}

#[test]
fn a_scalar_load_from_a_root_srv_reads_the_buffer() {
    // s_load_dword s12, s[6:7], 0x14; v_mov_b32 v0, s12
    let mut body = s_load_dword(12, 6, 0x14).to_vec();
    body.push(vop1(0x001, 0, 12));
    let (l, u, n) = root_cbv(ResourceType::Srv);
    lower_cs_with(l, u, n, &body).unwrap();
}

/// A descriptor loaded a second time only to test a field (dword 2, the
/// record count): the view comes from the load that accesses it.
#[test]
fn a_descriptor_reloaded_for_its_record_count_reads_the_views_size() {
    // s_load_dwordx4 s[12:15], s[4:5], 0; v_mov_b32 v0, s14
    let mut body = s_load_dwordx4(12, 4).to_vec();
    body.push(vop1(0x001, 0, 14));
    lower_cs(&body).unwrap();
}

fn lower_raw(
    layout: Vec<RootParam>,
    used: Vec<UsedResource>,
    user_sgprs: u8,
    mut words: Vec<u32>,
) -> Result<(), ShaderError> {
    words.push(S_ENDPGM);
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 12,
        layout,
        used,
        io: StageIo::Cs(ComputeIo {
            rsrc2: u32::from(user_sgprs) << 1 | 1 << 7,
            threads: [64, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs)?;
    let st = structure::structure(&p)?;
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, &native)?;
    lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    })?;
    Ok(())
}

fn table(user_data: u8, offset: u32, slot: u8) -> RootParam {
    RootParam {
        root: RootType::Table,
        table_offset_dwords: offset,
        sgprs: 1,
        base_register: 0,
        user_data,
        slot,
    }
}

#[test]
fn a_structured_store_through_a_table_finds_its_stride() {
    let layout = vec![table(4, 0, 0), table(4, 8, 0), table(5, 0, 1)];
    let used = vec![UsedResource {
        kind: ResourceType::Uav,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 8,
        stride: 12,
        dwords: 4,
        user_data: 5,
        slot: 1,
    }];
    let mut w = vec![s_mov_b32(8, 5), s_mov_b32(9, 0x80)];
    w.extend_from_slice(&s_load_dwordx4(0, 8));
    w.push(vop1(0x001, 3, 256)); // v_mov_b32 v3, v0
    w.push(vop1(0x001, 1, 256));
    w.push(vop1(0x001, 2, 256));
    // buffer_store_dwordx3 v[0:2], v3, s[0:3], 0 idxen
    w.extend_from_slice(&[0xE07C_2000, 0x8000_0003]);
    lower_raw(layout, used, 6, w).unwrap();
}
