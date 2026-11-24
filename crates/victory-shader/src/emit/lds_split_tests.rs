//! Console wave64 compute programs with LDS on a host of 32-lane waves: which wave size the
//! lowering asks of the host. Runs anywhere (no device, no validator).

use super::lower;
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

/// A wave64 compute program of 64 threads (one console wave; the thread index in v0) with 4
/// root constants in s0..s3 and a raw UAV's descriptor in s[8:11], then `body`.
fn native(body: &[u32]) -> NativeProgram {
    let mut words = vec![
        0xBE85_0380, // s_mov_b32 s5, 0
        0x3D << 26 | 0x02 << 18 | 8 << 6 | 2,
        125 << 25, // s_load_dwordx4 s[8:11], s[4:5], 0
    ];
    words.extend_from_slice(body);
    words.push(0xBF81_0000); // s_endpgm
    NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        vgprs: 16,
        layout: vec![
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
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1, // 5 user SGPRs: s0..s4
            threads: [64, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// 64 threads in one console wave with 512 bytes of LDS: `if (t < s0) { lds[t] = t;
/// v2 = lds[t + 1]; store v2 }`, a write and a read in one lane branch's region with no
/// barrier (the console compiler leaves them out in a one-wave group). Two host waves cannot
/// be ordered there, so a 32-lane host keeps `WaveSize(64)`, which NVIDIA refuses at pipeline
/// creation. With the region's accesses taken unordered ([`lower::with_loose_lds`]) the
/// program runs in waves of 32 that meet around the branch.
#[test]
fn lds_written_and_read_in_one_lane_branch_runs_in_waves_of_32_when_loose() {
    let words = [
        0x3402_0082, // v_lshlrev_b32 v1, 2, v0
        0x7D88_0000, // v_cmp_gt_u32 vcc, s0, v0
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
        0xBF88_0007, // s_cbranch_execz to the restore of EXEC
        0xD834_0000,
        0x0000_0001, // ds_write_b32 v1, v0
        0xD8D8_0004,
        0x0200_0001, // ds_read_b32 v2, v1 offset:4
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16 | 2 << 8, // buffer_store_dword v2, off, s[8:11], 0
        0xBEFE_040C, // s_mov_b64 exec, s[12:13]
    ];
    let mut native = native(&words);
    if let StageIo::Cs(cs) = &mut native.io {
        cs.rsrc2 |= 1 << 15; // 512 bytes of LDS
    }
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let lower_it = || {
        let was = lower::TEST_HOST_WAVE.replace(Some(32));
        let l = lower::lower(&lower::Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: Vec::new(),
            outputs: Vec::new(),
            prims: Vec::new(),
            mesh_stride: 0,
        });
        lower::TEST_HOST_WAVE.set(was);
        l.expect("lower")
    };
    assert_eq!(lower_it().wave_size, Some(64));
    let loose = lower::with_loose_lds(lower_it);
    assert_eq!(loose.wave_size, Some(32));
}
