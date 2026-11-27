use super::{Lowered, Numeric, Parts, buffer_format, half_shift, lower};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

#[test]
fn packed_pixel_inputs_find_their_second_row() {
    use super::super::container::SigElement;
    use super::half_input_partners;
    use crate::native::{PsInput, Usage};
    let entry = |row, row2, pack2| PsInput {
        usage: Usage::Generic,
        row,
        mask: 0xF,
        attr: row,
        flat: false,
        fp16: true,
        pack2,
        row2,
    };
    let element = |row| SigElement {
        name: "TEXCOORD".into(),
        index: row,
        kind: 0,
        comp: 9,
        interp: 2,
        row: Some(row),
        col: 0,
        cols: 4,
        usage: 0,
        half: true,
    };
    // Rows 0 and 3 pack and leave their second row out; 5 names its
    // own (6). Unnamed 16-bit rows: 1, 4, 6, 7.
    let io = StageIo::Ps {
        inputs: vec![entry(0, 0, true), entry(3, 3, true), entry(5, 6, true)],
        input_addr: 0,
        db_shader_control: 0,
        patches: vec![],
    };
    let sig: Vec<_> = [0, 1, 3, 4, 5, 6, 7].map(element).into();
    let p = half_input_partners(&io, &sig);
    assert_eq!(p.get(&0), Some(&1));
    assert_eq!(p.get(&3), Some(&4));
    assert_eq!(p.get(&5), None);
}

/// The packed pedestrian layouts of the title: an element
/// behind another one in its register follows in the first parameter
/// with a dword left, and is not a half of its register's row.
#[test]
fn packed_elements_behind_another_take_the_first_free_dword() {
    use super::super::container::SigElement;
    use super::packed_half;
    let element = |index, row, col, cols| SigElement {
        name: "TEXCOORD".into(),
        index,
        kind: 0,
        comp: 9,
        interp: 2,
        row: Some(row),
        col,
        cols,
        usage: 0,
        half: true,
    };
    // MS_VS_PedTransformPreSkinD_Packed 005fe88ec1bdc0c7 and its pixel
    // shader 93667197d3d05e1c: TEXCOORD6 is one value at register 4
    // column 3; parameter 3 (rows 6 and 7) carries it in dword 3.
    let sig = vec![
        element(1, 2, 0, 4),
        element(2, 3, 0, 4),
        element(4, 4, 0, 3),
        element(6, 4, 3, 1),
        element(3, 5, 0, 4),
        element(5, 6, 0, 4),
        element(7, 7, 0, 2),
        element(9, 9, 0, 3),
        element(11, 10, 0, 3),
        element(15, 12, 0, 4),
    ];
    let packs = [
        (2, Some(3)),
        (4, Some(5)),
        (6, Some(7)),
        (9, Some(10)),
        (12, None),
    ];
    let at = |a, h| packed_half(&sig, &packs, a, h);
    // Parameter 2: TEXCOORD4's three values, a zero half, TEXCOORD3.
    assert_eq!(at(4, 2), Some((4, 2)));
    assert_eq!(at(4, 3), None);
    assert_eq!(at(4, 4), Some((5, 0)));
    assert_eq!(at(4, 7), Some((5, 3)));
    // Parameter 3: TEXCOORD5, TEXCOORD7, then TEXCOORD6 alone.
    assert_eq!(at(6, 3), Some((6, 3)));
    assert_eq!(at(6, 5), Some((7, 1)));
    assert_eq!(at(6, 6), Some((4, 3)));
    assert_eq!(at(6, 7), None);
    // Parameters without such an element are as before.
    assert_eq!(at(9, 3), None);
    assert_eq!(at(9, 4), Some((10, 0)));
    assert_eq!(at(12, 3), Some((12, 3)));
    assert_eq!(at(12, 4), None);
    // The common layout (29805d23a495cdf8): TEXCOORD7 is two values at
    // register 1 columns 2-3; parameter 0 is TEXCOORD0, TEXCOORD1,
    // TEXCOORD7.
    let sig = vec![
        element(0, 1, 0, 2),
        element(7, 1, 2, 2),
        element(1, 2, 0, 4),
        element(2, 3, 0, 4),
        element(4, 4, 0, 3),
    ];
    let packs = [(3, Some(4)), (1, Some(2))];
    let at = |a, h| packed_half(&sig, &packs, a, h);
    assert_eq!(at(1, 1), Some((1, 1)));
    assert_eq!(at(1, 2), Some((2, 0)));
    assert_eq!(at(1, 5), Some((2, 3)));
    assert_eq!(at(1, 6), Some((1, 2)));
    assert_eq!(at(1, 7), Some((1, 3)));
    assert_eq!(at(3, 6), Some((4, 2)));
    assert_eq!(at(3, 7), None);
    // A row no element holds gives nothing, as before.
    assert_eq!(packed_half(&sig, &[(8, None)], 8, 0), None);
}

/// Whether a synthetic compute program translates with these input registers.
fn lowers(words: &[u32], rsrc2: u32) -> Result<(), crate::ShaderError> {
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 256,
        layout: vec![],
        used: vec![],
        io: StageIo::Cs(ComputeIo {
            rsrc2,
            threads: [1, 1, 1],
        }),
        code,
    };
    let p = ir::build(&native.code, false, 256).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = io::raise(&p, &native).unwrap();
    lower(&Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs: vec![],
        prims: vec![],
        mesh_stride: 0,
    })
    .map(|_| ())
}

/// Lowers a compute program with the root parameters `layout` (in user
/// SGPRs from s0 on) and the used resources `used`.
fn lower_with(
    words: &[u32],
    layout: Vec<RootParam>,
    used: Vec<UsedResource>,
) -> Result<Lowered, crate::ShaderError> {
    lower_wave(words, layout, used, false)
}

/// [`lower_with`] in wave64 if `wave64`.
fn lower_wave(
    words: &[u32],
    layout: Vec<RootParam>,
    used: Vec<UsedResource>,
    wave64: bool,
) -> Result<Lowered, crate::ShaderError> {
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let sgprs: u32 = layout.iter().map(|p| p.sgprs).sum();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64,
        vgprs: 256,
        layout,
        used,
        io: StageIo::Cs(ComputeIo {
            // The layout's user SGPRs (at least two), 512 bytes of LDS.
            rsrc2: 1 << 15 | sgprs.max(2) << 1,
            threads: [1, 1, 1],
        }),
        code,
    };
    let p = ir::build(&native.code, wave64, 256).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, &native).unwrap();
    lower(&Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs: vec![],
        prims: vec![],
        mesh_stride: 0,
    })
}

/// Root parameter `slot` in user SGPRs from `user_data` on.
fn root(root: RootType, user_data: u8, slot: u8) -> RootParam {
    RootParam {
        root,
        table_offset_dwords: 0,
        sgprs: if root == RootType::Descriptor { 2 } else { 1 },
        base_register: 0,
        user_data,
        slot,
    }
}

#[test]
fn buffer_cache_invalidations_lower_as_no_ops() {
    // MUBUF opcode 0x071/0x072 with no data operand: the title asks the
    // host to invalidate its cached view of a buffer.
    assert!(lowers(&[0xE1C4_0000, 0, S_ENDPGM], 0).is_ok(), "buffer_gl0_inv");
    assert!(lowers(&[0xE1C8_0000, 0, S_ENDPGM], 0).is_ok(), "buffer_gl1_inv");
}

#[test]
fn a_scalar_carry_chain_reaches_the_select() {
    // The high dword of a 64-bit add or subtract, then a select on the
    // carry (borrow) the high half leaves in SCC, and a branch on it
    // (only code with an effect needs its operands).
    let add = [
        0xBE80_03C1, // s_mov_b32 s0, -1
        0xBE81_0380, // s_mov_b32 s1, 0
        0x8000_8100, // s_add_u32 s0, s0, 1
        0x8201_8001, // s_addc_u32 s1, s1, 0
        0x8502_0001, // s_cselect_b32 s2, s1, s0
        0xBF07_8002, // s_cmp_lg_u32 s2, 0
        0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
        0xBE83_0380, // s_mov_b32 s3, 0
        S_ENDPGM,
    ];
    let sub = [
        0xBE80_0380, // s_mov_b32 s0, 0
        0xBE81_0380, // s_mov_b32 s1, 0
        0x8080_8100, // s_sub_u32 s0, s0, 1
        0x8281_8001, // s_subb_u32 s1, s1, 0
        0x8502_0001, // s_cselect_b32 s2, s1, s0
        0xBF07_8002, // s_cmp_lg_u32 s2, 0
        0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
        0xBE83_0380, // s_mov_b32 s3, 0
        S_ENDPGM,
    ];

    for words in [add, sub] {
        lowers(&words, 0).unwrap();
    }
}

fn perm_features(selector: u32) -> u64 {
    // Synthetic compute shader:
    // v0 is the thread ID and can supply a dynamic selector.
    // Store the result in LDS to keep the permutation live.
    let words = [
        0x7E02_02FFu32,
        0x817F_34AB,
        0x7E04_02FF,
        0xFE01_56CD,
        0xD744_0003,
        257 | (258 << 9) | (selector << 18),
        0x3408_0082,
        0xD834_0000,
        0x0000_0304,
        S_ENDPGM,
    ];

    let code: Vec<u8> = words.iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();

    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 256,
        layout: vec![],
        used: vec![],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 15,
            threads: [64, 1, 1],
        }),
        code,
    };

    let p = ir::build(&native.code, false, 256).unwrap();

    assert!(p.blocks.iter()
        .flat_map(|b| &b.insts)
        .any(|i| i.name() == "v_perm_b32"));

    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();

    let io = io::raise(&p, &native).unwrap();

    let emitted = lower(&Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs: vec![],
        prims: vec![],
        mesh_stride: 0,
    }).unwrap();

    let container =
        crate::Container::parse(&emitted.dxil).unwrap();

    let features =
        container.part(crate::FourCC::SFI0).unwrap();

    u64::from_le_bytes(features.data.try_into().unwrap())
}

#[test]
fn perm_constant_selector_needs_no_int64() {
    assert_eq!(perm_features(128) & 0x8000, 0);
}

#[test]
fn perm_dynamic_selector_needs_no_int64() {
    assert_eq!(
        perm_features(256) & 0x8000,
        0,
        "dynamic v_perm_b32 must not require INT64"
    );
}

#[test]
fn a_descriptor_selected_by_scc_picks_its_heap_index() {
    // Two buffer descriptors loaded from the table at s0, one picked by
    // a root constant (s1) with s_cselect_b64 per half, and loaded
    // through by index and stored to LDS so the load is needed.
    let words = [
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
        0xF408_0302,
        0xFA00_0010, // s_load_dwordx4 s[12:15], s[4:5], 0x10
        0xBF06_8001, // s_cmp_eq_u32 s1, 0
        0x8590_0C08, // s_cselect_b64 s[16:17], s[8:9], s[12:13]
        0x8592_0E0A, // s_cselect_b64 s[18:19], s[10:11], s[14:15]
        0xE030_2000,
        0x8004_0100, // buffer_load_dword v1, v0, s[16:19], 0 idxen
        0xD834_0000,
        0x0000_0100, // ds_write_b32 v0, v1
        S_ENDPGM,
    ];
    let layout = || vec![root(RootType::Table, 0, 0), root(RootType::Constants, 1, 1)];
    // Both are structured buffers of dwords in the table's range.
    let used = || {
        vec![UsedResource {
            kind: ResourceType::Srv,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 4,
            dwords: 4,
            user_data: 0,
            slot: 0,
        }]
    };
    lower_with(&words, layout(), used()).unwrap();
}

/// Tables of 2D textures at s0 and samplers at s1.
fn texture_tables() -> (Vec<RootParam>, Vec<UsedResource>) {
    let range = |kind, user_data, slot| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords: if kind == ResourceType::Sampler { 4 } else { 8 },
        user_data,
        slot,
    };
    (
        vec![root(RootType::Table, 0, 0), root(RootType::Table, 1, 1)],
        vec![
            range(ResourceType::Srv, 0, 0),
            range(ResourceType::Sampler, 1, 1),
        ],
    )
}

#[test]
fn a_uniform_branch_can_choose_between_immediate_samplers() {
    // A root constant selects wrap or clamp. Keep the sample live with an LDS store.
    let words = [
        0xBE84_0300, 0xBE85_0380,       // s[4:5] = texture table
        0xF40C_0302, 0xFA00_0000,       // s_load_dwordx8 s[12:19], s[4:5], 0
        0xBE89_0380, 0xBE8A_0380, 0xBE8B_0380, // sampler words 1-3
        0xBF06_8001,                    // s_cmp_eq_u32 s1, 0
        0xBF85_0002,                    // branch to clamp
        0xBE88_0380, 0xBF82_0002,       // wrap, skip clamp
        0xBE88_03FF, 0x0000_0092,       // clamp all axes
        0xF09C_0F08, 0x0043_0C02,       // image_sample_lz v[12:15], ..., s[8:11]
        0xD834_0000, 0x0000_0C00,       // ds_write_b32 v0, v12
        S_ENDPGM,
    ];
    let (mut layout, mut used) = texture_tables();
    layout[1] = root(RootType::Constants, 1, 1);
    used.truncate(1);
    let shader = lower_with(&words, layout, used).unwrap();
    assert_eq!(shader.samplers.len(), 2);
    assert!(shader.samplers.contains(&[0, 0, 0, 0]));
    assert!(shader.samplers.contains(&[0x92, 0, 0, 0]));
}

#[test]
fn waterfall_loops_over_per_lane_descriptors_sample_with_each_lanes_own() {
    use crate::resource::tests::{
        WATERFALL_FETCHED_WAVE32, WATERFALL_FETCHED_WAVE64, WATERFALL_PICKED_WAVE32,
    };
    for (words, wave64) in [
        (WATERFALL_FETCHED_WAVE32, false),
        (WATERFALL_FETCHED_WAVE64, true),
        (WATERFALL_PICKED_WAVE32, false),
    ] {
        let (layout, used) = texture_tables();
        let l = lower_wave(words, layout, used, wave64).unwrap();
        // The loop's reads of the first lane became each lane's own
        // dwords: no wave operation is left.
        assert_eq!(l.nonuniform_handles, 1);
        assert!(!l.wave_ops);
    }
}

/// [`texture_tables`] with a raw buffer UAV at dword 16 of the first.
pub(crate) fn texture_and_uav_tables() -> (Vec<RootParam>, Vec<UsedResource>) {
    let (layout, mut used) = texture_tables();
    used.push(UsedResource {
        kind: ResourceType::Uav,
        root: RootType::Table,
        table_offset_dwords: 16,
        lower_bound: 0,
        stride: 0,
        dwords: 4,
        user_data: 0,
        slot: 0,
    });
    (layout, used)
}

#[test]
fn waterfall_loops_over_descriptors_joined_per_lane_sample_with_each_lanes_own() {
    use crate::resource::tests::{WATERFALL_JOINED_MIXED_WAVE32, WATERFALL_JOINED_WAVE32};
    for words in [WATERFALL_JOINED_WAVE32, WATERFALL_JOINED_MIXED_WAVE32] {
        let (layout, used) = texture_and_uav_tables();
        let l = lower_wave(words, layout, used, false).unwrap();
        // One image handle, from the join's per-lane heap index.
        assert_eq!(l.nonuniform_handles, 1);
        assert!(!l.wave_ops);
    }
}

#[test]
fn scalar_bit_tests_lower() {
    // Each tests one bit of a constant into SCC and branches on it (only
    // code with an effect needs its operands).
    let tail = [
        0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
        0xBE84_0380, // s_mov_b32 s4, 0
        S_ENDPGM,
    ];
    let bit_tests: [&[u32]; 4] = [
        &[0xBE81_0387, 0xBF0C_8301], // s_mov_b32 s1, 7; s_bitcmp0_b32 s1, 3
        &[0xBE81_0387, 0xBF0D_8301], // s_mov_b32 s1, 7; s_bitcmp1_b32 s1, 3
        &[
            0xBE80_0387, // s_mov_b32 s0, 7
            0xBE81_0380, // s_mov_b32 s1, 0
            0xBF0E_8300, // s_bitcmp0_b64 s[0:1], 3
        ],
        &[
            0xBE80_0387, // s_mov_b32 s0, 7
            0xBE81_0380, // s_mov_b32 s1, 0
            0xBF0F_8300, // s_bitcmp1_b64 s[0:1], 3
        ],
    ];
    for head in bit_tests {
        let words: Vec<u32> = head.iter().chain(&tail).copied().collect();
        lowers(&words, 0).unwrap();
    }
}

#[test]
#[ignore = "ours marks joins read through joins only with VICTORY_STRICT_JOINS (40 programs stop otherwise)"]
fn a_missing_value_stays_missing_through_two_live_joins() {
    let mut words = [
        0xF490_0180,
        0,           // s_memtime s[6:7]
        0xBF06_8000, // s_cmp_eq_u32 s0, 0 (s0 = group id X)
        0xBF85_0001, // skip the first overwrite
        0xBE86_0380, // s_mov_b32 s6, 0
        0xBF06_8101, // s_cmp_eq_u32 s1, 1 (s1 = group id Y)
        0xBF85_0001, // skip the second overwrite
        0xBE86_0381, // s_mov_b32 s6, 1
        0xBF07_8006, // s_cmp_lg_u32 s6, 0
        0xBF85_0001, // use the result to branch
        0xBE83_0380,
        S_ENDPGM,
    ];
    let group_ids_xy = 3 << 7;
    // Compare to a direct read so future timestamp support works in both cases.
    let refusal = |code: &[u32]| match lowers(code, group_ids_xy) {
        Ok(()) => None,
        Err(crate::ShaderError::Unsupported { what, .. }) => Some(what),
        Err(error) => panic!("{error}"),
    };
    let direct = [&words[..2], &words[8..]].concat();
    assert_eq!(refusal(&words), refusal(&direct));
    words[8] = 0xBF07_8000; // read the group id instead; the timestamp is dead
    lowers(&words, group_ids_xy).expect("dead timestamp");
}

#[test]
fn buffer_formats_by_number() {
    // The numbers the disassembler names 8_8_8_8_UNORM, 16_16_SNORM,
    // 2_10_10_10_UNORM, 16_FLOAT, 32_32_32_32_FLOAT.
    assert_eq!(buffer_format(56), Some((&[8, 8, 8, 8][..], Numeric::Unorm)));
    assert_eq!(buffer_format(24), Some((&[16, 16][..], Numeric::Snorm)));
    assert_eq!(
        buffer_format(50),
        Some((&[10, 10, 10, 2][..], Numeric::Unorm))
    );
    assert_eq!(buffer_format(13), Some((&[16][..], Numeric::Float)));
    assert_eq!(
        buffer_format(77),
        Some((&[32, 32, 32, 32][..], Numeric::Float))
    );
    // Invalid, past the table.
    assert_eq!(buffer_format(0), None);
    assert_eq!(buffer_format(78), None);
}

#[test]
fn packed_buffer_formats_by_number() {
    // LLVM's `UfmtGFX10`: 10_11_11 is 30-36, 11_11_10 37-43,
    // 10_10_10_2 44-49. Channels from the lowest bits: R11G11B10_FLOAT
    // is 10_11_11 (Mesa), 10_10_10_2 has the 2-bit channel first.
    assert_eq!(buffer_format(30), Some((&[11, 11, 10][..], Numeric::Unorm)));
    assert_eq!(buffer_format(36), Some((&[11, 11, 10][..], Numeric::Float)));
    assert_eq!(buffer_format(37), Some((&[10, 11, 11][..], Numeric::Unorm)));
    assert_eq!(buffer_format(43), Some((&[10, 11, 11][..], Numeric::Float)));
    assert_eq!(
        buffer_format(44),
        Some((&[2, 10, 10, 10][..], Numeric::Unorm))
    );
    assert_eq!(
        buffer_format(45),
        Some((&[2, 10, 10, 10][..], Numeric::Snorm))
    );
    assert_eq!(
        buffer_format(46),
        Some((&[2, 10, 10, 10][..], Numeric::Uscaled))
    );
    assert_eq!(
        buffer_format(47),
        Some((&[2, 10, 10, 10][..], Numeric::Sscaled))
    );
    assert_eq!(
        buffer_format(48),
        Some((&[2, 10, 10, 10][..], Numeric::Uint))
    );
    assert_eq!(
        buffer_format(49),
        Some((&[2, 10, 10, 10][..], Numeric::Sint))
    );
}

/// A half's value (sign, 5-bit exponent biased 15, 10-bit mantissa), as
/// `legacyF16ToF32` widens it.
fn half(h: u32) -> f32 {
    let (s, e, m) = ((h >> 15) & 1, (h >> 10) & 31, h & 1023);
    let v = match e {
        0 => m as f32 * 2f32.powi(-24),
        31 if m == 0 => f32::INFINITY,
        31 => f32::NAN,
        _ => (1024 + m) as f32 * 2f32.powi(e as i32 - 25),
    };
    if s == 1 { -v } else { v }
}

#[test]
fn small_floats_widen_as_halves() {
    // Unsigned 11- and 10-bit floats: a 5-bit exponent biased 15 over a
    // 6- or 5-bit mantissa, no sign; values from the D3D/GL spec.
    let f11 = |x: u32| half(x << half_shift(11));
    let f10 = |x: u32| half(x << half_shift(10));
    assert_eq!(f11(0), 0.0);
    assert_eq!(f11(0x001), 2f32.powi(-20)); // smallest denormal
    assert_eq!(f11(0x03F), 63.0 * 2f32.powi(-20)); // largest denormal
    assert_eq!(f11(0x040), 2f32.powi(-14)); // smallest normal
    assert_eq!(f11(0x3C0), 1.0);
    assert_eq!(f11(0x3E0), 1.5);
    assert_eq!(f11(0x7BF), 65024.0); // largest
    assert_eq!(f11(0x7C0), f32::INFINITY);
    assert!(f11(0x7C1).is_nan());
    assert_eq!(f10(0x001), 2f32.powi(-19));
    assert_eq!(f10(0x020), 2f32.powi(-14));
    assert_eq!(f10(0x1E0), 1.0);
    assert_eq!(f10(0x3DF), 64512.0);
    assert_eq!(f10(0x3E0), f32::INFINITY);
    assert!(f10(0x3FF).is_nan());
    // A 16-bit float is a half already, sign included.
    assert_eq!(half(0xBC00 << half_shift(16)), -1.0);
}

#[test]
fn paired_lowering_scope_restores_on_unwind() {
    assert!(!super::PAIRED.get());
    let failed = std::panic::catch_unwind(|| super::with_paired_lanes(|| panic!("translation panic")));
    assert!(failed.is_err());
    assert!(!super::PAIRED.get());
}

/// The structure builder's group prefix sums (#39321a95223001ed, #77360df21e12ea43,
/// #9fc111d3be79364f: wave64, 256 or 512 threads), words 0x14c..0x210 as the console
/// compiler made them: each wave's last lane parks its total in shared memory, a barrier,
/// then every wave but the first adds up the totals of the waves before it in a loop over a
/// lane test (`v_cmpx_eq_u32 exec, s18, v7`) that carries the running sum in `s17` with a
/// `v_readfirstlane_b32` of `s17 + lds[s18 * 4]`. The `s_load_dword` of a root constant at
/// 0x1a0 is two `s_nop`s here (no root parameters); 0x1cc is the loop's alignment pad.
/// `addr` is the register the loop's `ds_read_b32` takes its address from: v8 (the
/// program's, `v_mov_b32 v8, s16`) or v0 (a per-lane address, to see the refusal stay).
fn prefix_sum_words(addr: u32) -> Vec<u32> {
    vec![
        0x7E0C_0300, // v_mov_b32 v6, v0 (the lane's own partial sum)
        0x7E0E_0300, // v_mov_b32 v7, v0
        0xBE80_0380, // 14c s_mov_b32 s0, 0
        0xBE81_03C1, // 150 s_mov_b32 s1, -1
        0x4A0C_0D07, // 154 v_add_nc_u32 v6, v7, v6
        0xD765_0007, // 158 v_mbcnt_lo_u32_b32 v7, -1, 0
        0x0001_00C1,
        0xD760_0006, // 160 v_readlane_b32 s6, v6, 31
        0x0001_3F06,
        0xD766_0009, // 168 v_mbcnt_hi_u32_b32 v9, -1, v7
        0x0002_0EC1,
        0x2C0E_0086, // 170 v_lshrrev_b32 v7, 6, v0 (the console wave's number in the group)
        0xD501_0008, // 174 v_cndmask_b32 v8, 0, s6, s[0:1]
        0x0000_0C80,
        0xBE80_047E, // 17c s_mov_b64 s[0:1], exec
        0x4A0C_0D08, // 180 v_add_nc_u32 v6, v8, v6
        0x7DA4_12BF, // 184 v_cmpx_eq_u32 exec, 63, v9 (the wave's last lane)
        0x3410_0E82, // 188 v_lshlrev_b32 v8, 2, v7
        0xD834_0000, // 18c ds_write_b32 v8, v6 (the wave's total)
        0x0000_0608,
        0xBEFE_0400, // 194 s_mov_b64 exec, s[0:1]
        0xBF8C_C07F, // 198 s_waitcnt lgkmcnt(0)
        0xBF8A_0000, // 19c s_barrier
        0xBF80_0000, // 1a0 (s_load_dword s5, s[2:3], 0x10)
        0xBF80_0000,
        0x7E10_0280, // 1a8 v_mov_b32 v8, 0
        0xBE80_047E, // 1ac s_mov_b64 s[0:1], exec
        0x7DAA_0E80, // 1b0 v_cmpx_ne_u32 exec, 0, v7 (every wave but the first)
        0xBF88_0014, // 1b4 s_cbranch_execz 0x208
        0xBE82_047E, // 1b8 s_mov_b64 s[2:3], exec
        0xBE86_047E, // 1bc s_mov_b64 s[6:7], exec
        0xBE92_0380, // 1c0 s_mov_b32 s18, 0
        0xBE91_0380, // 1c4 s_mov_b32 s17, 0
        0xBF82_0003, // 1c8 s_branch 0x1d8
        0xBF80_0000, // 1cc (pad)
        0x87FE_0610, // 1d0 s_and_b64 exec, s[16:17], s[6:7]
        0x7E22_0508, // 1d4 v_readfirstlane_b32 s17, v8 (the running sum)
        0x8F10_8212, // 1d8 s_lshl_b32 s16, s18, 2
        0x8112_8112, // 1dc s_add_i32 s18, s18, 1
        0x7E10_0210, // 1e0 v_mov_b32 v8, s16
        0xD8D8_0000, // 1e4 ds_read_b32 v8, v{addr}
        0x0800_0000 | addr,
        0xBF8C_C07F, // 1ec s_waitcnt lgkmcnt(0)
        0x4A10_1011, // 1f0 v_add_nc_u32 v8, s17, v8
        0xBE90_047E, // 1f4 s_mov_b64 s[16:17], exec
        0x7DA4_0E12, // 1f8 v_cmpx_eq_u32 exec, s18, v7
        0xBF80_0000, // 1fc s_nop 0
        0x8A86_7E06, // 200 s_andn2_b64 s[6:7], s[6:7], exec
        0xBF85_FFF2, // 204 s_cbranch_scc1 0x1d0
        0xBEFE_0400, // 208 s_mov_b64 exec, s[0:1]
        0x4A0C_0D08, // 20c v_add_nc_u32 v6, v8, v6
        0x3402_0082, // v_lshlrev_b32 v1, 2, v0
        0xD834_0040, // ds_write_b32 v1, v6 offset:64 (keeps the sum live)
        0x0000_0601,
        S_ENDPGM,
    ]
}

fn lower_wave64_on_32(words: &[u32], threads: u32) -> Result<Lowered, crate::ShaderError> {
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        vgprs: 256,
        layout: vec![],
        used: vec![],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 15, // 2560 bytes: the totals and one dword a thread
            threads: [threads, 1, 1],
        }),
        code,
    };
    let p = ir::build(&native.code, true, 256).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = io::raise(&p, &native).unwrap();
    let prev = super::TEST_HOST_WAVE.replace(Some(32));
    let out = lower(&Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs: vec![],
        prims: vec![],
        mesh_stride: 0,
    });
    super::TEST_HOST_WAVE.set(prev);
    out
}

#[test]
fn the_builders_prefix_sum_lowers_exactly_on_32_lane_waves() {
    // The running sum `s17 + lds[s18 * 4]` is the same in every lane (a scalar plus a
    // shared-memory read at a scalar address, after the barrier), so each host half's
    // first lane holds the console's first lane's value: no exchange, which would need
    // every lane of the group (the loop runs only in waves past the first).
    for threads in [256, 512] {
        if let Err(e) = lower_wave64_on_32(&prefix_sum_words(8), threads) {
            panic!("{threads} threads: the builder's prefix sum must lower on 32-lane waves, got: {e}");
        }
    }
}

#[test]
fn a_first_lane_read_of_a_per_lane_shared_memory_read_is_still_refused() {
    // The same loop with the read at the lane's own address: the first lane's value may
    // be in the other host half, and not every lane runs the loop.
    let err = lower_wave64_on_32(&prefix_sum_words(0), 256)
        .err()
        .expect("a per-lane shared-memory read must not count as the same in every lane")
        .to_string();
    assert!(err.contains("a first-lane read where not every lane runs"), "got: {err}");
}

#[test]
fn a_first_lane_read_of_a_uniform_shared_memory_read_lowers_on_wave32() {
    // PR #36's own case: inside a lane branch (`v0 < 1`) a shared-memory read at a
    // uniform address (`v1 = s16`) plus a scalar, read by its first lane; at the
    // per-lane address v0 it stays refused.
    let make = |addr: u32| {
        vec![
            0xBE90_0380, // s_mov_b32 s16, 0
            0xBE91_0380, // s_mov_b32 s17, 0
            0xD760_0004, // v_readlane_b32 s4, v0, 0 (the halves exchange)
            0x0001_0100,
            0x7D88_0081, // v_cmp_gt_u32_e32 vcc, 1, v0
            0xBE86_246A, // s_and_saveexec_b64 s[6:7], vcc
            0xBF88_0005, // s_cbranch_execz -> restore exec
            0x7E02_0210, // v_mov_b32 v1, s16
            0xD8D8_0000, // ds_read_b32 v2, v{addr}
            0x0200_0000 | addr,
            0x4A10_0411, // v_add_nc_u32 v8, s17, v2
            0x7E22_0508, // v_readfirstlane_b32 s17, v8
            0xBEFE_0406, // s_mov_b64 exec, s[6:7]
            0x8011_0411, // s_add_u32 s17, s17, s4
            0x7E06_0211, // v_mov_b32 v3, s17
            0xD834_0000, // ds_write_b32 v0, v3
            0x0000_0300,
            S_ENDPGM,
        ]
    };
    if let Err(e) = lower_wave64_on_32(&make(1), 128) {
        panic!("a uniform ds_read_b32 plus a scalar must count as the same in every lane, got: {e}");
    }
    let err = lower_wave64_on_32(&make(0), 128).err().expect("per-lane ds_read_b32 must be refused").to_string();
    assert!(err.contains("a first-lane read where not every lane runs"), "got: {err}");
}

#[test]
fn the_builders_prefix_sum_lowers_and_validates_as_a_spilled_dispatch_x_program() {
    // As the builder has it: a spilled user-data table (s[0:1], user data 16), so a
    // `DispatchX` may count threads and the threads past the count stay for the barrier and
    // the halves' exchange (`partial_active`). They skip the loop's shared-memory read, so
    // the running sum's first-lane read must take the first lane within the count.
    let words = prefix_sum_words(8);
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        vgprs: 256,
        layout: vec![RootParam {
            root: RootType::Constants,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 16,
            slot: 0,
        }],
        used: vec![],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 2 << 1 | 5 << 15, // s[0:1]: the spilled table's pointer; 2560 bytes LDS
            threads: [256, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    assert!(crate::spill::is_spilled(&native));
    let p = crate::spill::build(&native).expect("spill::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let prev = super::TEST_HOST_WAVE.replace(Some(32));
    let out = lower(&Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs: vec![],
        prims: vec![],
        mesh_stride: 0,
    });
    super::TEST_HOST_WAVE.set(prev);
    let out = out.unwrap_or_else(|e| panic!("the spilled prefix sum must lower on 32-lane waves, got: {e}"));
    let counts = RootType::Other(super::super::USER_DISPATCH_THREADS);
    assert!(
        out.user_data.iter().any(|u| u.root == counts && u.dword == 0),
        "the program does not take the DispatchX thread count of x"
    );
    if let Ok(mut v) = super::super::validator::Validator::load(super::super::validator::Validator::default_path().as_deref()) {
        let mut copy = out.dxil.clone();
        v.validate(&mut copy).unwrap_or_else(|e| panic!("DXIL validation failed: {e}"));
    }
}
