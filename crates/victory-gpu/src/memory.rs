//! Title memory by address, for translated shaders that read and write
//! through raw 40-bit addresses (`victory_shader::emit::SLOT_MEMORY`: the
//! console's ray-tracing structure builder and the programs that walk its
//! structures; the structures themselves lie in plain title memory, in no
//! buffer object).
//!
//! The copies are the ones every other access uses ([`crate::shadow`]: a
//! title buffer's copy, or the 4 MiB window of memory outside buffers), so
//! a root view and a raw address reach the same host bytes. This module
//! keeps what a shader needs to find them:
//!
//! - the **directory**: per 64 KiB page of title address space the number
//!   of the copy that holds the page (0: none) and the page's byte offset
//!   in it; one buffer for all lists, entries only ever go from empty to
//!   filled;
//! - per view heap a run of slots with a raw unordered access view of each
//!   numbered copy (slot 0 of the run: a null view, for pages without a
//!   copy);
//! - the **faults**: the page numbers shaders found empty, read back here
//!   once a frame, so the copies exist from the next frame on. A pointer a
//!   shader takes out of memory cannot be known before it runs; the
//!   addresses a dispatch is given as root views are made ready before it.
//!
//! **A page that is not there when a shader touches it is a lost access**:
//! the read gives 0, the store is dropped, and the fault note only helps
//! the next frame. For the structure builder that is the same as a dropped
//! stage, so before a dispatch the host makes ready everything it can see
//! the dispatch name: the whole copy around each root view, around each
//! pair of root constants that is a title address and around each root
//! constant that is one in 256-byte units (the builder passes structures
//! all three ways), and the 4 MiB after each (a structure may cross a
//! window's end).
//! shortcut: what a stage reaches only through pointers it reads out of
//! memory (InitHeaders' table of structure addresses, LoadInstances'
//! instance records) is not made ready unless something else named it;
//! the fault count in the log says whether that happens. The whole answer
//! is the extent of the title's allocation around an address, which the
//! memory module of the layer knows and this one does not ask yet.
//!
//! shortcut: a copy that is numbered is kept for the rest of the run, and
//! there are at most [`COPIES`] of them (4 MiB windows mostly: 4 GiB at
//! most).
//!
//! **What a list brings up to date** (and, if its shader can store,
//! declares written: the bytes are compared on the CPU afterwards, as for
//! any unordered access view): once per list, the copies its own
//! dispatches name, and every copy a shader reached through a pointer
//! (numbered from a fault note: kept for every list). Runs 371/372 brought
//! ALL numbered copies up to date in every list: the set only grows (1023
//! in 6 s of the structure builder), each list then snapshot, copied back
//! and compared up to 4 GiB, the submitting thread never caught up, no
//! frame was presented for 30 s, and the title's hang watchdog
//! (`GPU Loc Sig`, a store to address 0 at title +0x23cf683) ended the run.
//! `VICTORY_MEMORY_SYNC_ALL=1`: as before (all numbered copies).
//! shortcut: a copy named by another list and reached here only through a
//! pointer it reads is neither refreshed nor declared in this list (as a
//! copy numbered by a later list always was): the card's copy keeps what
//! it last had; a dirty-page mask written by the shader would close it.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use victory_shader::emit::{MEMORY_FAULTS, MEMORY_NOTES, MEMORY_PAGES};
use windows::Win32::Graphics::Direct3D12::*;

use crate::descriptor::host::HostHeap;
use crate::host::{HostList, host};
use crate::shadow::Shadow;

/// Copies that can be numbered (the run of heap slots is one longer). A
/// listed frame of the opening cutscene names about 50 windows; the
/// structure builder numbered 1023 in its first 3 s (run 374), after which
/// new pages read zeros.
const COPIES: u32 = 4095;
/// A window of memory outside buffers (`shadow.rs`).
const CELL: u64 = 4 << 20;
/// Frames after a page is seen written before its bytes are pulled into
/// title memory: the lists that wrote it have run by then.
const PULL_AFTER: u64 = 3;
/// A directory page.
const PAGE: u64 = 1 << 16;

struct Memory {
    /// Buffer, its mapping, its GPU address.
    directory: (ID3D12Resource, *mut u32, u64),
    faults: (ID3D12Resource, *mut u32, u64),
    /// Copy `n` is `copies[n - 1]`.
    copies: Vec<Arc<Shadow>>,
    /// Per host view heap (by address): the first slot of its run, and how
    /// many copies have their view there.
    heaps: HashMap<usize, (u32, u32)>,
    /// The frame the faults were last read in.
    frame: u64,
    /// Pages asked for this frame that are not the title's memory.
    refused: std::collections::HashSet<u64>,
    /// Copies (index into `copies`) a shader reached through a pointer: a
    /// fault note numbered or found them. Brought up to date in every list.
    faulted: Vec<usize>,
    /// Per copy (index into `copies`): the pages (title page numbers,
    /// sorted) shaders were seen to store to, from the written flags
    /// (`victory_shader::emit::MEMORY_NOTES`). Only these are declared
    /// written; learned once, kept.
    written: Vec<Vec<u64>>,
    /// Pages seen written (copy, page, frame seen): their bytes are pulled
    /// into title memory once [`PULL_AFTER`] frames have passed
    /// ([`Shadow::pull_back`]); a page seen again is pulled again.
    pulls: Vec<(usize, u64, u64)>,
    /// The meter ([`Meter`]).
    meter: Meter,
}

/// The cost of title memory by address, for the log: names by where they
/// came from, copies numbered, and the bytes lists brought up to date.
#[derive(Default)]
struct Meter {
    /// Names: root views, constant pairs, constants in 256-byte units,
    /// fault notes.
    names: [u64; 4],
    /// Of them, the ones that numbered a new copy.
    new: [u64; 4],
    /// Bytes of all numbered copies, and the largest.
    bytes: u64,
    largest: u64,
    /// Since the last line: lists, copies brought up to date in them, their
    /// bytes, of them declared written, frames.
    lists: u64,
    synced: u64,
    synced_bytes: u64,
    written_bytes: u64,
    frames: u64,
    /// Written pages learned so far, and pulls that wrote title memory.
    learned: u64,
    late: u64,
    /// Stores to pages with no copy (dropped: they read 0 later), and
    /// pulls refused (a copy in the card's memory or in place).
    dropped: u64,
    refused_pulls: u64,
    /// The last line, and the copy count it was written at.
    said: Option<std::time::Instant>,
    said_copies: usize,
}

const VIEW: usize = 0;
const PAIR: usize = 1;
const UNITS: usize = 2;
const FAULT: usize = 3;

/// `VICTORY_MEMORY_SYNC_ALL=1`: every list brings all numbered copies up to
/// date (runs before 373; see the module).
fn sync_all() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_MEMORY_SYNC_ALL").is_some_and(|v| v == "1");
        if on {
            tracing::info!("title memory by address: every list brings all numbered copies up to date (VICTORY_MEMORY_SYNC_ALL=1)");
        }
        on
    })
}

/// `VICTORY_MEMORY_WHOLE_WRITES=1`: a list that stores declares its copies
/// written whole (run 374), not only the pages shaders were seen to write.
fn whole_writes() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_MEMORY_WHOLE_WRITES").is_some_and(|v| v == "1");
        if on {
            tracing::info!("title memory by address: copies are declared written whole (VICTORY_MEMORY_WHOLE_WRITES=1)");
        }
        on
    })
}

/// Runs of consecutive pages in sorted `pages`, as (first page, count).
fn runs(pages: &[u64]) -> Vec<(u64, u64)> {
    let mut out: Vec<(u64, u64)> = Vec::new();
    for &p in pages {
        match out.last_mut() {
            Some((first, n)) if *first + *n == p => *n += 1,
            _ => out.push((p, 1)),
        }
    }
    out
}

// SAFETY: the mappings are of buffers the struct owns; all access is under
// the mutex.
unsafe impl Send for Memory {}

static MEMORY: Mutex<Option<Memory>> = Mutex::new(None);

/// What a draw binds.
pub(crate) struct Ready {
    pub(crate) directory: u64,
    pub(crate) faults: u64,
    /// The view heap slot of the null view; copy `n`'s view is `n` further.
    pub(crate) base: u32,
}

impl Memory {
    fn new() -> Option<Memory> {
        let host = host()?;
        let make = |dwords: u64, what: &str| {
            let (buffer, data) = host.shared_buffer(dwords * 4)?;
            crate::host::name(&buffer, || format!("title memory by address: {what}"));
            // SAFETY: a fresh mapping of at least that many bytes.
            unsafe { std::ptr::write_bytes(data, 0, (dwords * 4) as usize) };
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            Some((buffer, data.cast::<u32>(), gpu))
        };
        tracing::info!("title memory by address: a shader reads or writes through raw addresses");
        Some(Memory {
            directory: make(u64::from(MEMORY_PAGES) * 2, "directory")?,
            faults: make(u64::from(MEMORY_NOTES), "faults and written pages")?,
            copies: Vec::new(),
            heaps: HashMap::new(),
            frame: u64::MAX,
            refused: Default::default(),
            faulted: Vec::new(),
            written: Vec::new(),
            pulls: Vec::new(),
            meter: Meter::default(),
        })
    }

    /// Gives the page of `address` its copy, if it has none and the memory
    /// is the title's; the copy that holds the page (index into `copies`),
    /// if one does. `source`: where the address came from (the meter).
    fn map(&mut self, address: u64, source: usize) -> Option<usize> {
        let page = address / PAGE;
        // The last entry stays empty: addresses past the directory read it.
        if page >= u64::from(MEMORY_PAGES) - 1 {
            return None;
        }
        self.meter.names[source] += 1;
        // SAFETY: inside the directory's mapping.
        let entry = unsafe { self.directory.1.add(page as usize * 2) };
        // SAFETY: as above.
        let numbered = unsafe { entry.read() };
        if numbered != 0 {
            return Some(numbered as usize - 1);
        }
        let start = page * PAGE;
        // A shader's pointer, or a constant taken for an address, may be
        // anything: only memory the title has.
        if self.refused.contains(&page) {
            return None;
        }
        if !crate::texture::committed(start as *const u8, PAGE as usize) {
            self.refused.insert(page);
            return None;
        }
        // The copy a root view at that address would get; a buffer that
        // ends inside the page is passed over for what holds all of it.
        let whole = |s: &Arc<Shadow>| s.offset_in(start).is_some_and(|_| s.left(start) >= PAGE);
        let shadow = crate::shadow::find(start, 1)
            .filter(whole)
            .or_else(|| crate::shadow::find(start, PAGE).filter(whole))?;
        // SAFETY: a live buffer.
        let flags = unsafe { shadow.buffer.GetDesc() }.Flags;
        if flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0 {
            return None;
        }
        let n = match self.copies.iter().position(|c| Arc::ptr_eq(c, &shadow)) {
            Some(n) => n,
            None if self.copies.len() < COPIES as usize => {
                self.copies.push(shadow.clone());
                self.written.push(Vec::new());
                let n = self.copies.len() - 1;
                let len = shadow.span().1;
                if len >= 64 << 20 {
                    tracing::info!(
                        n,
                        start = format_args!("{:#x}", shadow.span().0),
                        mb = len >> 20,
                        source,
                        "title memory by address: a big copy is numbered (source 0 view, 1 pair, 2 units, 3 fault)"
                    );
                }
                self.meter.new[source] += 1;
                self.meter.bytes += len;
                self.meter.largest = self.meter.largest.max(len);
                // Every page the copy holds whole, not only the one asked
                // for: a structure is more than 64 KiB.
                // shortcut: a page of a title buffer inside this one that
                // has a copy of its own is given this copy too.
                let (first, len) = shadow.span();
                let mut at = first.next_multiple_of(PAGE);
                while at + PAGE <= first + len && at / PAGE < u64::from(MEMORY_PAGES) - 1 {
                    // SAFETY: inside the directory's mapping.
                    let other = unsafe { self.directory.1.add((at / PAGE) as usize * 2) };
                    // SAFETY: as above; the offset first, as below.
                    unsafe {
                        if other.read() == 0 {
                            other.add(1).write_volatile((at - first) as u32);
                            other.write_volatile(n as u32 + 1);
                        }
                    }
                    at += PAGE;
                }
                n
            }
            None => {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| {
                    tracing::warn!(COPIES, "title memory by address: no number left for another copy; its pages read zeros")
                });
                return None;
            }
        };
        let offset = shadow.offset_in(start).unwrap_or(0);
        // The offset first: a shader that sees the number sees it too.
        // SAFETY: inside the directory's mapping.
        unsafe {
            entry.add(1).write_volatile(offset as u32);
            entry.write_volatile(n as u32 + 1);
        }
        Some(n)
    }

    /// The pages shaders found empty since the last frame.
    /// The written flags shaders set since the last look, for the pages of
    /// numbered copies (each page counted for the copy the directory gives
    /// it): cleared, new ones learned (`written`), each queued to be pulled
    /// into title memory (`pulls`).
    fn take_written(&mut self) {
        let last = u64::from(MEMORY_PAGES) - 1;
        for n in 0..self.copies.len() {
            let (first, len) = self.copies[n].span();
            if len == 0 {
                continue;
            }
            let (from, to) = (first / PAGE, ((first + len - 1) / PAGE).min(last - 1));
            for page in from..=to {
                // SAFETY: inside the directory's mapping. The owner first:
                // a page inside this copy's span that another copy holds is
                // left for that copy (its flag stays; rt-datapath audit 3).
                let owner = unsafe { self.directory.1.add(page as usize * 2).read() };
                if owner != 0 && owner as usize != n + 1 {
                    continue;
                }
                // SAFETY: inside the notes' mapping (page < MEMORY_PAGES);
                // the GPU may be writing: a flag set again is seen next time.
                let flag = unsafe { self.faults.1.add((MEMORY_FAULTS as u64 + page) as usize) };
                // SAFETY: as above.
                if unsafe { flag.read_volatile() } == 0 {
                    continue;
                }
                // SAFETY: as above.
                unsafe { flag.write_volatile(0) };
                if owner == 0 {
                    // A store to a page without a copy was dropped.
                    self.meter.dropped += 1;
                    continue;
                }
                let pages = &mut self.written[n];
                if let Err(at) = pages.binary_search(&page) {
                    pages.insert(at, page);
                    self.meter.learned += 1;
                }
                self.pulls.push((n, page, self.frame));
            }
        }
        // Pulled once their work has run: what the shaders stored there is
        // in the copy, and (unless a list declared it and copied it back)
        // not in title memory, where the title reads results back.
        let frame = self.frame;
        let mut keep = Vec::new();
        for (n, page, seen) in std::mem::take(&mut self.pulls) {
            if frame < seen + PULL_AFTER {
                keep.push((n, page, seen));
                continue;
            }
            match self.copies[n].pull_back(page * PAGE, PAGE) {
                Some(bytes) => {
                    if bytes > 0 {
                        self.meter.late += 1;
                    }
                }
                // Declared meanwhile: its own copy-back comes; else a copy
                // that cannot be pulled (counted).
                None if self.copies[n].pull_back(0, 0).is_none() => self.meter.refused_pulls += 1,
                None => keep.push((n, page, frame)),
            }
        }
        // Pages seen again collapse to their latest sighting.
        keep.sort_unstable_by_key(|&(n, page, seen)| (n, page, std::cmp::Reverse(seen)));
        keep.dedup_by_key(|&mut (n, page, _)| (n, page));
        self.pulls = keep;
    }

    fn take_faults(&mut self) {
        let mut pages = Vec::new();
        for i in 1..MEMORY_FAULTS as usize {
            // SAFETY: inside the faults' mapping; the GPU may be writing,
            // a number missed now is written again next frame.
            let at = unsafe { self.faults.1.add(i) };
            // SAFETY: as above.
            let page = unsafe { at.read_volatile() };
            if page != 0 {
                pages.push(page);
                // SAFETY: as above.
                unsafe { at.write_volatile(0) };
            }
        }
        if !pages.is_empty() {
            // Each is an access that read 0 or was dropped.
            static SAID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if SAID.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 50 {
                tracing::warn!(
                    pages = pages.len(),
                    first = format_args!("{:#x}", u64::from(pages[0]) * PAGE),
                    "title memory by address: shaders touched pages that had no copy (those accesses read 0 or were dropped); the copies are made now"
                );
            }
        }
        for page in pages {
            // Reached through a pointer: kept up to date in every list.
            if let Some(n) = self.map(u64::from(page) * PAGE, FAULT)
                && !self.faulted.contains(&n)
            {
                self.faulted.push(n);
            }
        }
    }

    /// What a dispatch names: the copy around `address` and the 4 MiB
    /// after it; the copies that hold them go into `named`.
    fn name(&mut self, address: u64, source: usize, named: &mut Vec<usize>) {
        named.extend(self.map(address, source));
        named.extend(self.map((address & !(CELL - 1)) + CELL, source));
    }

    /// A line for the log every 5 s while shaders read title memory by
    /// address, and at each doubling of the copies (from 64).
    fn say(&mut self) {
        let m = &mut self.meter;
        let now = std::time::Instant::now();
        let due = m.said.is_none_or(|at| now.duration_since(at).as_secs() >= 5);
        let doubled = self.copies.len() >= 64 && self.copies.len() >= m.said_copies * 2;
        if !due && !doubled {
            return;
        }
        let per = |v: u64| v as f64 / m.frames.max(1) as f64;
        let mb = |v: u64| v as f64 / (1 << 20) as f64;
        tracing::info!(
            "title memory by address: {} copies numbered ({:.0} MB, largest {:.1} MB), {} reached through pointers; names so far: views {} pairs {} units {} faults {}, of them numbering a copy: {} {} {} {}; per frame over the last {} frames: {:.1} lists brought {:.1} copies up to date, {:.1} MB, {:.1} MB of it declared written; written pages learned {}, pulls that wrote title memory {}, refused {}, stores dropped (no copy) {}{}",
            self.copies.len(),
            mb(m.bytes),
            mb(m.largest),
            self.faulted.len(),
            m.names[VIEW],
            m.names[PAIR],
            m.names[UNITS],
            m.names[FAULT],
            m.new[VIEW],
            m.new[PAIR],
            m.new[UNITS],
            m.new[FAULT],
            m.frames,
            per(m.lists),
            per(m.synced),
            mb(m.synced_bytes) / m.frames.max(1) as f64,
            mb(m.written_bytes) / m.frames.max(1) as f64,
            m.learned,
            m.late,
            m.refused_pulls,
            m.dropped,
            if sync_all() { " (VICTORY_MEMORY_SYNC_ALL=1: all copies)" } else { "" }
        );
        m.said = Some(now);
        m.said_copies = self.copies.len();
        (m.lists, m.synced, m.synced_bytes, m.written_bytes, m.frames) = (0, 0, 0, 0, 0);
    }
}

/// Before a draw whose shader reads title memory by address: what its
/// root arguments name gets its copies (`addresses`: its root views;
/// `constants`: its root constants, see the module), every numbered copy
/// has its view in `heap`'s run and is brought up to date when `list` is
/// submitted (and, with `writes`, copied back after). None if the host
/// cannot hold it; the draw is then dropped.
pub(crate) fn ready(
    list: &HostList,
    heap: &HostHeap,
    addresses: &[u64],
    constants: &[u32],
    writes: bool,
) -> Option<Ready> {
    let mut guard = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        *guard = Memory::new();
    }
    let m = guard.as_mut()?;
    let frame = crate::frame::current();
    if m.frame != frame {
        if m.frame != u64::MAX {
            m.meter.frames += frame.saturating_sub(m.frame).max(1);
        }
        m.frame = frame;
        m.refused.clear();
        m.take_faults();
        m.take_written();
        m.say();
    }
    let numbered = m.copies.len();
    let mut named = Vec::new();
    for &address in addresses {
        m.name(address, VIEW, &mut named);
    }
    // A title address in two constants (low dword, then bits 32 to 39), or
    // in one in 256-byte units. A constant that only looks like one names
    // memory nothing reads; one that is not the title's memory is refused.
    for (i, &c) in constants.iter().enumerate() {
        if let Some(&high) = constants.get(i + 1)
            && (1..=0xFF).contains(&high)
            && c >= 0x1_0000
        {
            m.name(u64::from(c) | u64::from(high) << 32, PAIR, &mut named);
        }
        if c >= 0x0100_0000 {
            m.name(u64::from(c) << 8, UNITS, &mut named);
        }
    }
    named.sort_unstable();
    named.dedup();
    // On time even when no frame is presented (run 373 froze with it silent).
    m.say();
    let key = std::ptr::from_ref(heap) as usize;
    let (base, made) = match m.heaps.get(&key) {
        Some(&run) => run,
        None => {
            // Held for the heap's life: not given back with a list's slots.
            let base = heap.dynamic(COPIES + 1)?;
            heap.null_raw_uav_at(base);
            (base, 0)
        }
    };
    for (n, shadow) in m.copies.iter().enumerate().skip(made as usize) {
        let (_, len) = shadow.span();
        heap.raw_uav_at(base + 1 + n as u32, &shadow.buffer, 0, len);
    }
    m.heaps.insert(key, (base, m.copies.len() as u32));
    // Each copy once per list: the builder alone dispatches hundreds of times
    // a frame. Without the switch only the copies this dispatch names and
    // those reached through pointers (see the module); a copy another list
    // numbers meanwhile is brought up to date by that list's submission,
    // which may run after this one's.
    const READ: usize = usize::MAX - 0x6d;
    const WRITE: usize = usize::MAX - 0x6e;
    let first_read = list.first_use(READ);
    let first_write = writes && list.first_use(WRITE);
    if first_read {
        m.meter.lists += 1;
    }
    // What a list that stores declares written: the whole copy (with either
    // switch), else only the runs of pages shaders were seen to write
    // (`take_written`); a copy no shader wrote is only refreshed.
    let whole = sync_all() || whole_writes();
    let sync = |shadow: &Arc<Shadow>, pages: &[u64], read: bool, write: bool, meter: &mut Meter| {
        let (start, len) = shadow.span();
        if read && whole {
            shadow.refresh(list, start, len);
            meter.synced += 1;
            meter.synced_bytes += len;
        } else if read {
            // Not over pages shaders were seen to write: the copy holds the
            // newest bytes there (title memory gets them by a declared
            // copy-back or a pull), and a refresh of an unwatched copy, or
            // of a page our own copy-back touched, put older title bytes
            // over the structures (rt-datapath audit 2).
            // shortcut: a title write into such a page does not reach the
            // copy any more.
            let mut at = start;
            for (first, count) in runs(pages) {
                let (from, to) = ((first * PAGE).max(start), ((first + count) * PAGE).min(start + len));
                if from > at {
                    shadow.refresh(list, at, from - at);
                }
                at = at.max(to);
            }
            if at < start + len {
                shadow.refresh(list, at, start + len - at);
            }
            meter.synced += 1;
            meter.synced_bytes += len;
        }
        if write && whole {
            shadow.gpu_writes(list, start, len);
            meter.written_bytes += len;
        } else if write {
            for (first, count) in runs(pages) {
                let from = (first * PAGE).max(start);
                let to = ((first + count) * PAGE).min(start + len);
                if from < to {
                    shadow.gpu_writes(list, from, to - from);
                    meter.written_bytes += to - from;
                }
            }
        }
    };
    if sync_all() {
        for (n, shadow) in m.copies.iter().enumerate() {
            let (read, write) = (first_read || n >= numbered, writes && (first_write || n >= numbered));
            sync(shadow, &m.written[n], read, write, &mut m.meter);
        }
    } else {
        // Keys of this list's set of first uses: a copy's address with the
        // low bits 1 (refreshed) and 2 (declared written); an `Arc`'s
        // address is 8-aligned, as are the other keys of the set.
        for &n in named.iter().chain(m.faulted.iter()) {
            let shadow = &m.copies[n];
            let at = Arc::as_ptr(shadow) as usize;
            let read = list.first_use(at | 1);
            let write = writes && list.first_use(at | 2);
            sync(shadow, &m.written[n], read, write, &mut m.meter);
        }
    }
    if first_read {
        list.keep(&m.directory.0);
        list.keep(&m.faults.0);
    }
    Some(Ready {
        directory: m.directory.2,
        faults: m.faults.2,
        base,
    })
}

/// Forgets the run of title-memory views kept for host view heap `heap`
/// (`Memory::heaps`, by address), when the title releases its heap. A heap
/// made later at the same address found the old run: its null view and the
/// copies' views were never written there, and the run's first slot could
/// lie among the new heap's title slots or past its end, so shaders reading
/// through raw addresses indexed views that were not theirs. (Fork pull
/// request #589, AscilCH.)
pub(crate) fn forget_heap(heap: &HostHeap) {
    let key = std::ptr::from_ref(heap) as usize;
    if let Some(m) = MEMORY.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        m.heaps.remove(&key);
    }
}

/// Whether a run of title-memory views is kept for the host view heap at
/// address `key` (tests).
#[cfg(test)]
pub(crate) fn has_run(key: usize) -> bool {
    MEMORY.lock().unwrap_or_else(|e| e.into_inner()).as_ref().is_some_and(|m| m.heaps.contains_key(&key))
}
