//! Immediate samplers the code picks among constants by control flow: the console compiler
//! sets the first word of one sampler on each arm of a uniform branch (or with `s_cselect`)
//! and the others after the join. The candidates become one array of static samplers, and
//! the handle takes the register the program's own choice picks. Runs anywhere (no device,
//! no validator); `SAMPLER_CHOICE_DXIL=<dir>` writes each module there for an outside
//! validator.

use super::lower;
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

/// Clamp to the last texel on every axis (sampler word 0).
const CLAMP: u32 = 0x92;
/// Sampler words 1..3 the code sets after the join.
const REST: [u32; 3] = [0x00C0_0000, 0x1850_0000, 0];

/// A wave64 compute program of 64 threads (the thread index in v0) with 4 root constants in
/// s0..s3, the table of views at s4 (a raw UAV at entry 0, a 2D texture at dword 8) and the
/// table of samplers at s5. The prologue moves the sampler table to s[6:7], loads the UAV
/// into s[8:11] and the texture into s[12:19]; then `s_cmp_eq_u32 s0, 0` and `arms`: the
/// words of the arm taken when s0 is not 0 and of the one taken when it is (each `None`
/// for no branch: then `arms.0` is straight-line code). After the join, `after`; then
/// each lane samples the texture with the sampler in s[24:27] at (lane, lane) and stores
/// the red channel at lane * 4.
fn native(arms: (&[u32], Option<&[u32]>), after: &[u32]) -> NativeProgram {
    let mut words = vec![
        0xBE86_0305, // s_mov_b32 s6, s5
        0xBE87_0380, // s_mov_b32 s7, 0
        0xBE85_0380, // s_mov_b32 s5, 0
        0x3D << 26 | 0x02 << 18 | 8 << 6 | 2,
        125 << 25, // s_load_dwordx4 s[8:11], s[4:5], 0
        0x3D << 26 | 0x03 << 18 | 12 << 6 | 2,
        125 << 25 | 0x20, // s_load_dwordx8 s[12:19], s[4:5], 0x20
        0xBF06_8000,      // s_cmp_eq_u32 s0, 0
    ];
    match arms {
        (a, Some(b)) => {
            words.push(0xBF85_0000 | (a.len() as u32 + 1)); // s_cbranch_scc1 -> b
            words.extend_from_slice(a);
            words.push(0xBF82_0000 | b.len() as u32); // s_branch -> the join
            words.extend_from_slice(b);
        }
        (a, None) => words.extend_from_slice(a),
    }
    words.extend_from_slice(after);
    words.extend_from_slice(&[
        0x7E04_0D00, // v_cvt_f32_u32 v2, v0
        0x7E06_0302, // v_mov_b32 v3, v2
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xF09C_0F08,
        6 << 21 | 3 << 16 | 12 << 8 | 2, // image_sample_lz v[12:15], v[2:3], s[12:19], s[24:27] dmask:0xf 2D
        0xBF8C_3F70,                     // s_waitcnt vmcnt(0)
        0x38 << 26 | 0x1C << 18 | 1 << 12,
        0x80 << 24 | 2 << 16 | 12 << 8 | 16, // buffer_store_dword v12, v16, s[8:11], 0 offen
        0xBF81_0000,                         // s_endpgm
    ]);
    let table = |user_data: u8| RootParam {
        root: RootType::Table,
        table_offset_dwords: 0,
        sgprs: 1,
        base_register: 0,
        user_data,
        slot: user_data,
    };
    let range = |kind, table_offset_dwords, dwords, user_data| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride: 0,
        dwords,
        user_data,
        slot: user_data,
    };
    NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        vgprs: 24,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            table(4),
            table(5),
        ],
        used: vec![
            range(ResourceType::Uav, 0, 4, 4),
            range(ResourceType::Srv, 8, 8, 4),
            range(ResourceType::Sampler, 0, 4, 5),
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 6 << 1, // 6 user SGPRs: s0..s5
            threads: [64, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

fn lower(native: &NativeProgram, name: &str) -> Result<lower::Lowered, &'static str> {
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res =
        resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, native).expect("io::raise");
    let l = lower::lower(&lower::Parts {
        native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    });
    if let (Ok(l), Some(dir)) = (&l, std::env::var_os("SAMPLER_CHOICE_DXIL")) {
        std::fs::write(std::path::Path::new(&dir).join(name), &l.dxil).expect("write");
    }
    l.map_err(|e| match e {
        crate::ShaderError::Unsupported { what, .. } => what,
        _ => "another error",
    })
}

/// The words after the join: sampler words 1..3.
const AFTER: [u32; 6] = [
    0xBE99_03FF,
    REST[0], // s_mov_b32 s25, lit
    0xBE9A_03FF,
    REST[1],     // s_mov_b32 s26, lit
    0xBE9B_0380, // s_mov_b32 s27, 0
    0xBF80_0000, // s_nop 0
];

/// `if (s0 != 0) clamp else wrap`, word 0 set on each arm: the two samplers become one
/// array of two static samplers, picked by a phi of their registers at the join.
#[test]
fn a_sampler_chosen_by_a_uniform_branch_is_an_array_of_static_samplers() {
    let native = native(
        (
            &[0xB018_0000 | CLAMP], // s_movk_i32 s24, 0x92
            Some(&[0xBE98_0380]),   // s_mov_b32 s24, 0
        ),
        &AFTER,
    );
    let l = lower(&native, "branch.dxil").expect("lower");
    let clamp = [CLAMP, REST[0], REST[1], REST[2]];
    let wrap = [0, REST[0], REST[1], REST[2]];
    // In the order of the join's predecessors: the arm that falls through first.
    assert_eq!(l.samplers, [clamp, wrap]);
    assert_eq!(l.sampler_arrays, [(0, 2)]);
    assert_eq!(l.cmp_samplers, 0);
}

/// The same choice made with `s_cselect_b32`: a select of the two registers.
#[test]
fn a_sampler_chosen_by_s_cselect_is_an_array_of_static_samplers() {
    let native = native(
        (
            &[0x8518_80FF, CLAMP], // s_cselect_b32 s24, 0x92, 0
            None,
        ),
        &AFTER,
    );
    let l = lower(&native, "cselect.dxil").expect("lower");
    assert_eq!(
        l.samplers,
        [
            [CLAMP, REST[0], REST[1], REST[2]],
            [0, REST[0], REST[1], REST[2]]
        ]
    );
    assert_eq!(l.sampler_arrays, [(0, 2)]);
}

/// Both arms set the same words: one static sampler of its own, no array.
#[test]
fn a_sampler_the_same_on_both_arms_is_one_static_sampler() {
    let native = native(
        (
            &[0xB018_0000 | CLAMP], // s_movk_i32 s24, 0x92
            Some(&[0xB018_0000 | CLAMP]),
        ),
        &AFTER,
    );
    let l = lower(&native, "same.dxil").expect("lower");
    assert_eq!(l.samplers, [[CLAMP, REST[0], REST[1], REST[2]]]);
    assert!(l.sampler_arrays.is_empty());
}

/// One arm loads the sampler from the sampler table, the other sets immediate words: a
/// heap sampler and a static one have no index in common to pick with a phi (and DXIL
/// joins no handles), so the program is still refused.
#[test]
fn a_sampler_chosen_between_the_heap_and_a_constant_is_refused() {
    let constants = [
        0xB018_0000 | CLAMP, // s_movk_i32 s24, 0x92
        0xBE99_03FF,
        REST[0], // s_mov_b32 s25, lit
        0xBE9A_03FF,
        REST[1],     // s_mov_b32 s26, lit
        0xBE9B_0380, // s_mov_b32 s27, 0
    ];
    let native = native(
        (
            &[
                0x3D << 26 | 0x02 << 18 | 24 << 6 | 3,
                125 << 25, // s_load_dwordx4 s[24:27], s[6:7], 0
            ],
            Some(&constants),
        ),
        &[0xBF80_0000], // s_nop 0
    );
    assert_eq!(
        lower(&native, "mixed.dxil").err(),
        Some("a descriptor chosen by control flow")
    );
}
