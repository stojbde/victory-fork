use super::*;

pub(super) fn lds_meetings(p: &crate::ir::Program, st: &Structured, all: &[bool]) -> Option<Vec<(bool, bool)>> {
    let n = st.exits.len();
    let succ = |b: usize| -> Vec<usize> {
        match st.exits[b] {
            LaneExit::Jump(t) => vec![t],
            LaneExit::Branch { taken, not_taken, .. } => vec![taken, not_taken],
            LaneExit::End | LaneExit::Indirect => Vec::new(),
        }
    };
    // regions: blocks not every lane runs, joined by the edges between them
    let mut region: Vec<usize> = (0..n).collect();
    fn find(region: &mut [usize], mut b: usize) -> usize {
        while region[b] != b {
            region[b] = region[region[b]];
            b = region[b];
        }
        b
    }
    for b in 0..n {
        for t in succ(b) {
            if !all[b] && !all[t] {
                let (x, y) = (find(&mut region, b), find(&mut region, t));
                region[x] = y;
            }
        }
    }
    // per region: LDS reads, LDS writes
    let mut reads = vec![false; n];
    let mut writes = vec![false; n];
    for (b, block) in p.blocks.iter().enumerate().take(n) {
        if all[b] {
            continue;
        }
        let r = find(&mut region, b);
        for i in &block.insts {
            let name = i.name();
            if !matches!(i.fields, Fields::Ds { .. }) || name.starts_with("ds_swizzle") || name.contains("permute") {
                continue;
            }
            if name.starts_with("ds_read") {
                reads[r] = true;
            } else {
                writes[r] = true;
            }
        }
    }
    // A region that reads and writes LDS is still taken when it only raises flags
    // (`lds_flag_region`): a read that misses the other host wave's write changes nothing
    // the region leaves behind. (After DrTardis1963 1b5c3a1e; VICTORY_LDS_FLAGS=0: off.)
    let flags_on = std::env::var_os("VICTORY_LDS_FLAGS").is_none_or(|v| v != "0");
    let unflagged: Vec<usize> = (0..n)
        .filter(|&r| reads[r] && writes[r])
        .filter(|&r| {
            let blocks: Vec<usize> = (0..n).filter(|&b| !all[b] && find(&mut region, b) == r).collect();
            !(flags_on && lds_flag_region(p, st, &blocks))
        })
        .collect();
    if !unflagged.is_empty() {
        if std::env::var_os("VICTORY_WAVE_WHY").is_some() {
            for r in (0..n).filter(|&r| reads[r] && writes[r]) {
                let mut list = Vec::new();
                for (b, block) in p.blocks.iter().enumerate().take(n) {
                    if all[b] || find(&mut region, b) != r {
                        continue;
                    }
                    for i in &block.insts {
                        let name = i.name();
                        if !matches!(i.fields, Fields::Ds { .. }) || name.starts_with("ds_swizzle") || name.contains("permute") {
                            continue;
                        }
                        // the address register and the instruction that wrote it
                        let addr = match &i.fields {
                            Fields::Ds { addr: Some(Operand::Vector { reg, .. }), offset0, offset1, .. } => {
                                let made = i.read(Reg::V(*reg)).and_then(|v| {
                                    p.blocks
                                        .iter()
                                        .flat_map(|bl| &bl.insts)
                                        .find(|x| x.writes.iter().any(|w| w.1 == v))
                                        .map(|x| format!("{:#x} {}", x.machine.offset, x.name()))
                                });
                                format!("v{reg} off {offset0}/{offset1} made by {}", made.as_deref().unwrap_or("an input or a phi"))
                            }
                            _ => "no address".to_string(),
                        };
                        list.push(format!("{:#x} {name} [{addr}]", i.machine.offset));
                    }
                }
                eprintln!("WAVE64-LDS region with reads and writes: {}", list.join("; "));
            }
        }
        return None;
    }
    let mut meet = vec![(false, false); n];
    for b in 0..n {
        for t in succ(b) {
            if all[b] && !all[t] {
                let r = find(&mut region, t);
                meet[b].1 |= reads[r] || writes[r];
            } else if !all[b] && all[t] {
                let r = find(&mut region, b);
                meet[t].0 |= reads[r] || writes[r];
            }
        }
    }
    Some(meet)
}

/// Whether a lane-branch region (`blocks`, see [`lds_meetings`]) that both reads and writes
/// LDS only raises flags, so that it may run as two host waves with group barriers at its
/// edges alone. The console runs such a region as one wave, where a lane's read sees every
/// write before it; split in two, a read may miss the other half's write. That changes nothing
/// the region leaves behind when:
/// - every LDS access in it is a `ds_read_b32` or `ds_write_b32` at a constant address, and
///   every write stores the same constant;
/// - one address at most is both read and written in it (the flag), and every write a lane
///   can reach after one of its reads goes to that flag;
/// - it has no other effect (no store, atomic, export, message or barrier);
/// - no value it computes is used after it.
///
/// A missed write only makes a lane go on as if the flag were still down; whatever that lane
/// writes is the flag's own constant, which the write it missed has already put there. So the
/// flag ends up raised exactly when some lane raises it, as on the console, and no other word
/// or register depends on when it was seen.
fn lds_flag_region(p: &Program, st: &Structured, blocks: &[usize]) -> bool {
    let inside: HashSet<usize> = blocks.iter().copied().collect();
    // where each value is written: (block, instruction)
    let mut def: HashMap<Value, (usize, usize)> = HashMap::new();
    for (b, block) in p.blocks.iter().enumerate() {
        for (i, inst) in block.insts.iter().enumerate() {
            for &(_, v) in &inst.writes {
                def.insert(v, (b, i));
            }
        }
    }
    let inline = |op: &Operand| -> Option<u32> {
        match *op {
            Operand::Const(c @ 128..=192) => Some(u32::from(c) - 128),
            Operand::Const(c @ 193..=208) => Some((-(i32::from(c) - 192)) as u32),
            Operand::Literal(x) => Some(x),
            _ => None,
        }
    };
    // a VGPR value a `v_mov_b32` of a constant wrote, unguarded or under the same guard as `at`
    let constant = |v: Option<Value>, at: (usize, usize)| -> Option<u32> {
        let (b, i) = *def.get(&v?)?;
        let inst = &p.blocks[b].insts[i];
        if inst.name() != "v_mov_b32" {
            return None;
        }
        let guard = st.guards.get(b).and_then(|g| g.get(i)).copied().flatten();
        let same = guard.is_none() || (b == at.0 && guard == st.guards[at.0][at.1]);
        let Fields::Valu(valu) = &inst.fields else { return None };
        if !same || valu.dpp.is_some() || valu.sdwa.is_some() {
            return None;
        }
        inline(valu.src.first()?)
    };
    // every LDS access: (block, index, is_read, address)
    let mut access: Vec<(usize, usize, bool, u32)> = Vec::new();
    let mut stored: Option<u32> = None;
    for &b in blocks {
        for (i, inst) in p.blocks[b].insts.iter().enumerate() {
            let name = inst.name();
            if name.contains("store")
                || name.contains("atomic")
                || name.starts_with("exp")
                || name.starts_with("s_sendmsg")
                || name == "s_barrier"
            {
                return false;
            }
            let Fields::Ds { addr, data, offset0, offset1, gds, .. } = &inst.fields else {
                if name.starts_with("ds_") {
                    return false;
                }
                continue;
            };
            if name.starts_with("ds_swizzle") || name.contains("permute") {
                continue;
            }
            let is_read = match name {
                "ds_read_b32" => true,
                "ds_write_b32" => false,
                _ => return false,
            };
            if *gds {
                return false;
            }
            let Some(Operand::Vector { reg, .. }) = *addr else { return false };
            let Some(base) = constant(inst.read(Reg::V(reg)), (b, i)) else { return false };
            let at = base.wrapping_add(u32::from(*offset0) | u32::from(*offset1) << 8);
            if !is_read {
                let Some(&Operand::Vector { reg, .. }) = data.first() else { return false };
                let Some(k) = constant(inst.read(Reg::V(reg)), (b, i)) else { return false };
                if stored.is_some_and(|s| s != k) {
                    return false;
                }
                stored = Some(k);
            }
            access.push((b, i, is_read, at));
        }
    }
    let read_at: HashSet<u32> = access.iter().filter(|a| a.2).map(|a| a.3).collect();
    let written_at: HashSet<u32> = access.iter().filter(|a| !a.2).map(|a| a.3).collect();
    let flags: Vec<u32> = read_at.intersection(&written_at).copied().collect();
    if flags.len() > 1 {
        return false;
    }
    if let Some(&flag) = flags.first() {
        // the blocks of the region a lane can reach from `b` along one edge or more
        let succ = |b: usize| -> Vec<usize> {
            match st.exits[b] {
                LaneExit::Jump(t) => vec![t],
                LaneExit::Branch { taken, not_taken, .. } => vec![taken, not_taken],
                LaneExit::End | LaneExit::Indirect => Vec::new(),
            }
        };
        for &(rb, ri, is_read, _) in &access {
            if !is_read {
                continue;
            }
            let mut seen: HashSet<usize> = HashSet::new();
            let mut work: Vec<usize> = succ(rb).into_iter().filter(|t| inside.contains(t)).collect();
            while let Some(b) = work.pop() {
                if seen.insert(b) {
                    work.extend(succ(b).into_iter().filter(|t| inside.contains(t)));
                }
            }
            let after = access.iter().any(|&(wb, wi, w_read, at)| {
                !w_read && at != flag && ((wb == rb && wi > ri) || seen.contains(&wb))
            });
            if after {
                return false;
            }
        }
    }
    // Nothing the region computes is used after it: no register it writes is live, lane by
    // lane, where a lane leaves it. A lane's own write kills a register for that lane (an
    // unguarded instruction that writes the whole dword); what an instruction reads of a VGPR
    // only for the lanes outside EXEC (the old value of the one it writes) is no use.
    let succ = |b: usize| -> Option<Vec<usize>> {
        match st.exits[b] {
            LaneExit::Jump(t) => Some(vec![t]),
            LaneExit::Branch { taken, not_taken, .. } => Some(vec![taken, not_taken]),
            LaneExit::End => Some(Vec::new()),
            LaneExit::Indirect => None,
        }
    };
    let n = st.exits.len().min(p.blocks.len());
    if (0..n).any(|b| succ(b).is_none()) {
        return false;
    }
    let exec = |r: Reg| matches!(r, Reg::S(c) if c == EXEC || c == EXEC + 1);
    let covers = |ops: &[Operand], r: Reg| {
        let Reg::V(x) = r else { return false };
        ops.iter().any(|op| {
            matches!(*op, Operand::Vector { reg, dwords }
                if (u16::from(reg)..u16::from(reg) + u16::from(dwords.max(1))).contains(&u16::from(x)))
        })
    };
    // per instruction: the registers it reads for its lanes, and those it writes in full
    let effect = |b: usize, i: usize| -> (Vec<Reg>, Vec<Reg>) {
        let inst = &p.blocks[b].insts[i];
        let name = inst.name();
        let partial = name.contains("16") || name.contains("mac") || name.contains("lane")
            || matches!(&inst.fields, Fields::Valu(v) if v.dpp.is_some() || v.sdwa.is_some());
        // the operands it reads explicitly, where its old destination is not one of them
        let sources: Option<Vec<Operand>> = match &inst.fields {
            Fields::Valu(v) if !partial => Some(v.src.clone()),
            Fields::Ds { addr, data, .. } if !partial => Some(addr.iter().copied().chain(data.iter().copied()).collect()),
            Fields::Buffer { addr, data, .. } if !partial && !name.contains("store") && !name.contains("atomic") => {
                let _ = data;
                Some(vec![*addr])
            }
            Fields::Image { addr, .. } if !partial && !name.contains("store") && !name.contains("atomic") => {
                Some(addr.clone())
            }
            _ => None,
        };
        let mut reads = Vec::new();
        for &(r, _) in &inst.reads {
            if exec(r) {
                continue;
            }
            let old_only = matches!(r, Reg::V(_))
                && inst.write(r).is_some()
                && sources.as_ref().is_some_and(|src| !covers(src, r));
            if !old_only {
                reads.push(r);
            }
        }
        let whole = st.guards.get(b).and_then(|g| g.get(i)).copied().flatten().is_none();
        let kills: Vec<Reg> = if whole && (sources.is_some() || !matches!(inst.fields, Fields::Valu(_) | Fields::Ds { .. } | Fields::Buffer { .. } | Fields::Image { .. })) {
            inst.writes.iter().map(|w| w.0).filter(|&r| !exec(r) && !(partial && matches!(r, Reg::V(_)))).collect()
        } else {
            Vec::new()
        };
        (reads, kills)
    };
    let mut live_in: Vec<HashSet<Reg>> = vec![HashSet::new(); n];
    loop {
        let mut changed = false;
        for b in (0..n).rev() {
            let mut live: HashSet<Reg> = HashSet::new();
            for t in succ(b).unwrap_or_default() {
                if let Some(l) = live_in.get(t) {
                    live.extend(l.iter().copied());
                }
            }
            for i in (0..p.blocks[b].insts.len()).rev() {
                let (reads, kills) = effect(b, i);
                for r in kills {
                    live.remove(&r);
                }
                live.extend(reads);
            }
            if live != live_in[b] {
                live_in[b] = live;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let written: HashSet<Reg> = blocks
        .iter()
        .flat_map(|&b| p.blocks[b].insts.iter().flat_map(|i| i.writes.iter().map(|w| w.0)))
        .filter(|&r| !exec(r))
        .collect();
    let mut leaks: Vec<(usize, Reg)> = Vec::new();
    for &b in blocks {
        for t in succ(b).unwrap_or_default() {
            if inside.contains(&t) {
                continue;
            }
            for r in &live_in[t] {
                if written.contains(r) {
                    leaks.push((t, *r));
                }
            }
        }
    }
    if std::env::var_os("VICTORY_WAVE_DUMP").is_some() {
        eprintln!(
            "WAVE64-LDS-FLAGS blocks={blocks:?} flag={flags:?} stored={stored:?} leaks={leaks:?}"
        );
    }
    leaks.is_empty()
}

/// Masks held as numbers or constant lane sets in guards or lane branches: the lane tests its
/// own bit by its console lane.
pub(super) fn lane_constant_masks(st: &Structured) -> bool {
    let whens = st.exits.iter().filter_map(|e| match *e {
        LaneExit::Branch { when, .. } => Some(when),
        _ => None,
    });
    let mut work: Vec<Bool> = st.guards.iter().flatten().flatten().copied().chain(whens).collect();
    // Through lane-mask joins too: an edge that brings a number or a constant lane set needs
    // the lane's own bit of it.
    // shortcut: masks only selects read (`v_cndmask_b32`, carries in) are not followed; such
    // a program still stops with "a constant lane mask" or "an opaque lane mask".
    let (mut seen, mut joins) = (HashSet::new(), HashSet::new());
    while let Some(g) = work.pop() {
        if !seen.insert(g) {
            continue;
        }
        for a in st.bools.support(g) {
            match a {
                Atom::Opaque(_) | Atom::Constant(_) => return true,
                Atom::Phi(id) if joins.insert(id) => {
                    if let Some(edges) = st.phi_masks.get(&id) {
                        work.extend(edges.iter().copied());
                    }
                }
                _ => {}
            }
        }
    }
    false
}

/// Whether accumulator initialisations are also stored at the start of the program
/// ([`lds_init_hoists`]); `VICTORY_LDS_INIT_HOIST=0` turns it off. Part of the translation
/// (the switch scan keys the shader cache on it).
pub(super) fn lds_init_hoist_on() -> bool {
    std::env::var_os("VICTORY_LDS_INIT_HOIST").is_none_or(|v| v != "0")
}

/// The value an LDS dword must start with for the atomic `name` to accumulate into it (`min`
/// of everything with u32::MAX, `max` with 0, ...): `None` for an operation that is not such
/// an accumulation.
fn accumulator_identity(name: &str) -> Option<u32> {
    Some(match name.replace("_rtn", "").as_str() {
        "ds_min_u32" | "ds_and_b32" => u32::MAX,
        "ds_max_u32" | "ds_add_u32" | "ds_or_b32" | "ds_xor_b32" => 0,
        "ds_min_i32" => 0x7FFF_FFFF,
        "ds_max_i32" => 0x8000_0000,
        _ => return None,
    })
}

/// The LDS dwords (below `dwords`, the program's own) that the program initialises for
/// accumulation, with their starting value: a dword every write to which is a `ds_write_b32`
/// or `ds_write2_b32` of one constant at a constant address, and every atomic on which
/// accumulates with that constant as identity (at least one). The title's tile programs have
/// one thread of the group store the starting minimum and maximum (`if (all(tid == 0))`)
/// after the per-pixel early returns, and every thread then `ds_min_u32`/`ds_max_u32`s its
/// key in, with no barrier between (one console wave per group). When that thread has
/// returned early, the atomics combine with whatever LDS held before (on the console another
/// group's values, on the host anything) and the tile comes out wrong: the water's
/// flickering dark patches (5dc3b25986622949 and 6f225b641c5b0708, stores at 0x230 and 0x244).
///
/// Stored by every thread at the program's start as well (then a group barrier), before any
/// LDS access, the dwords hold their starting value whether or not that thread gets there.
/// That is exact: where the program's own store runs it stores the same value again before the
/// dword's first use in that order, and a read that would see the early store instead of a
/// later write of this group would have read memory this group never wrote (undefined).
/// Empty when any LDS write or atomic of the program has an address or (for a write) data that
/// is not a constant of its own block: the dwords it could touch are not known.
pub(in super::super) fn lds_init_hoists(p: &Program, dwords: u32) -> Vec<(u32, u32)> {
    // `x` at instruction `i` of `insts`: a constant set by a plain `v_mov_b32` before it in the
    // block, with EXEC unchanged since (so in every lane that runs `i`)
    let constant = |insts: &[Inst], i: usize, x: Value| -> Option<u32> {
        let d = insts[..i]
            .iter()
            .rposition(|j| j.writes.iter().any(|w| w.1 == x))?;
        let Fields::Valu(f) = &insts[d].fields else {
            return None;
        };
        let plain = insts[d].name() == "v_mov_b32"
            && f.abs == 0
            && f.neg == 0
            && !f.clamp
            && f.omod == 0
            && f.opsel == 0
            && f.sdwa.is_none()
            && f.dpp.is_none()
            && insts[d + 1..i]
                .iter()
                .all(|j| j.write(Reg::S(EXEC)).is_none() && j.write(Reg::S(EXEC + 1)).is_none());
        if !plain {
            return None;
        }
        match f.src.first()? {
            Operand::Const(c) => crate::resource::inline(*c),
            Operand::Literal(k) => Some(*k),
            _ => None,
        }
    };
    let operand = |insts: &[Inst], i: usize, o: Option<&Operand>| -> Option<u32> {
        let &Operand::Vector { reg, .. } = o? else {
            return None;
        };
        constant(insts, i, insts[i].read(Reg::V(reg))?)
    };
    // per dword: the values stored, the identities of the atomics
    let mut stores: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    let mut atomics: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    for block in &p.blocks {
        let insts = &block.insts;
        for (i, inst) in insts.iter().enumerate() {
            let Fields::Ds {
                addr,
                data,
                offset0,
                offset1,
                gds,
                ..
            } = &inst.fields
            else {
                continue;
            };
            let name = inst.name();
            if *gds || !super::accesses_lds(inst) || name.starts_with("ds_read") {
                continue;
            }
            let Some(base) = operand(insts, i, addr.as_ref()) else {
                return Vec::new();
            };
            let single = u32::from(*offset0) | u32::from(*offset1) << 8;
            let dword = |bytes: u32| base.wrapping_add(bytes) / 4;
            match name {
                "ds_write_b32" | "ds_write2_b32" => {
                    let at: &[u32] = if name == "ds_write_b32" {
                        &[single]
                    } else {
                        &[u32::from(*offset0) * 4, u32::from(*offset1) * 4]
                    };
                    for (n, &bytes) in at.iter().enumerate() {
                        let Some(k) = operand(insts, i, data.get(n)) else {
                            return Vec::new();
                        };
                        stores.entry(dword(bytes)).or_default().push(k);
                    }
                }
                _ => {
                    let Some(identity) = accumulator_identity(name) else {
                        return Vec::new();
                    };
                    atomics.entry(dword(single)).or_default().push(identity);
                }
            }
        }
    }
    stores
        .into_iter()
        .filter_map(|(dword, values)| {
            let k = values[0];
            let ids = atomics.get(&dword)?;
            (dword < dwords && values.iter().all(|&v| v == k) && ids.iter().all(|&v| v == k))
                .then_some((dword, k))
        })
        .collect()
}
