//! Indirect execution's arguments, bounded before the GPU reads them. The
//! title's own compute passes write most of them; a pass we drop or
//! translate wrongly leaves whatever the memory held, and a draw of four
//! billion vertices or a dispatch past the host's limits runs until the
//! host resets the display driver. So every record is clamped on the GPU,
//! just before its execution, by a compute shader of ours (HLSL compiled by
//! the system's compiler; a 5.1 shader with its root signature inside).
//!
//! Each list also has a budget (2^27 vertex instances or threads, groups
//! counted as 64 threads, mesh groups as 128): records past it run empty,
//! so a list of thousands of garbage records cannot run for seconds
//! either.
//!
//! shortcut: the bounds (2^24 vertices, 2^26 vertex instances, 65,535
//! groups per dimension and 2^20 per dispatch, one-dimensional mesh
//! dispatches, the budget) cut legitimate work beyond them; raise them if
//! the title needs more.

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::s;

use crate::host::{HostList, host, transition};

#[cfg(test)]
const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=4, b0), UAV(u0), UAV(u1)"
cbuffer Record : register(b0) { uint kind; uint stride; uint records; uint unused; };
RWByteAddressBuffer args : register(u0);
// The list's work so far (vertex instances, threads): past the budget,
// records run empty.
RWByteAddressBuffer budget : register(u1);
static const uint BUDGET = 1u << 27;
// Groups of one dispatch; more run empty.
static const uint GROUPS = 1u << 20;
// Clamps the record at `at` in place; its cost.
uint clamped(uint at) {
    uint cost;
    if (kind <= 1) {
        // Draws: vertex or index count, then instances.
        uint count = min(args.Load(at), 1u << 24);
        uint instances = min(args.Load(at + 4), (1u << 26) / max(count, 1u));
        cost = count * instances;
        args.Store2(at, uint2(count, instances));
    } else if (kind == 2) {
        uint3 g = min(args.Load3(at), uint3(65535, 65535, 65535));
        // x * y fits 32 bits; the whole count may not.
        if (g.z != 0 && g.x * g.y > GROUPS / g.z) {
            g = uint3(0, 0, 0);
        }
        cost = g.x * g.y * g.z * 64;
        args.Store3(at, g);
    } else {
        // Mesh dispatches: the console's are one-dimensional.
        uint3 g = args.Load3(at);
        g = uint3(min(g.x, 65535u), min(g.y, 1u), min(g.z, 1u));
        cost = g.x * g.y * g.z * 128;
        args.Store3(at, g);
    }
    return cost;
}
// A record past the budget (no cost reaches it).
static const uint PAST = 0xffffffffu;
groupshared uint costs[64];
// One group walks the records in tiles of 64: every lane clamps its own
// record, then the first lane charges the tile's records in order. No
// other group shares the budget: a shared add wraps once enough records
// are rejected, and a compare-exchange retries more the more records fit.
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint lane : SV_GroupIndex) {
    uint left = 0;
    if (lane == 0) {
        left = BUDGET - min(budget.Load(0), BUDGET);
    }
    for (uint tile = 0; tile < records / 64 + min(records % 64, 1u); tile++) {
        uint record = tile * 64 + lane;
        uint at = record * stride;
        uint cost = 0;
        if (record < records) {
            cost = clamped(at);
        }
        costs[lane] = cost;
        GroupMemoryBarrierWithGroupSync();
        if (lane == 0) {
            for (uint i = 0; i < 64; i++) {
                if (costs[i] > left) {
                    // The first record that does not fit takes what is
                    // left, so those after it run empty too.
                    costs[i] = PAST;
                    left = 0;
                } else {
                    left -= costs[i];
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
        if (costs[lane] == PAST) {
            // Draws keep their layout with no vertices; groups become zero.
            args.Store(at, 0);
        }
    }
    if (lane == 0) {
        budget.Store(0, BUDGET - left);
    }
}
"#;

// Unlike `SOURCE`, this program never writes the title's argument buffer.
// A title compute pass may have produced those arguments earlier in the same
// command list; replacing them with a CPU-side shadow before ExecuteIndirect
// discards that producer's result.  This copy is therefore the production
// path. It reads its title sources through UAV bindings but never stores to
// them, matching the state used by the legacy path. `SOURCE` remains for the WARP regression tests until they are moved
// to a readback-based harness.
const COPY_SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=6, b0), UAV(u0), UAV(u1), UAV(u2), UAV(u3)"
cbuffer Record : register(b0) {
    uint kind; uint stride; uint records; uint args_offset;
    uint has_count; uint count_offset;
};
// These are UAVs because title buffers can have been written by an earlier
// dispatch in this list.  They are deliberately read-only in this shader.
RWByteAddressBuffer src_args : register(u0);
RWByteAddressBuffer count_buf : register(u1);
RWByteAddressBuffer dst_args : register(u2);
RWByteAddressBuffer budget : register(u3);
static const uint BUDGET = 1u << 27;
static const uint GROUPS = 1u << 20;
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint active = records;
    if ((has_count & 1u) != 0) active = min(active, count_buf.Load(count_offset));
    if (id.x >= active) return;
    uint src_at = args_offset + id.x * stride;
    uint dst_at = id.x * stride;
    uint cost = 0;
    uint2 draw = uint2(0, 0);
    uint3 groups = uint3(0, 0, 0);
    if (kind <= 1) {
        uint2 asked = src_args.Load2(src_at);
        draw.x = min(asked.x, 1u << 24);
        draw.y = min(asked.y, (1u << 26) / max(draw.x, 1u));
        cost = draw.x * draw.y;
    } else if (kind == 2) {
        groups = min(src_args.Load3(src_at), uint3(65535u, 65535u, 65535u));
        if (groups.z != 0 && groups.x * groups.y > GROUPS / groups.z) groups = uint3(0, 0, 0);
        cost = groups.x * groups.y * groups.z * 64u;
    } else {
        groups = src_args.Load3(src_at);
        groups = uint3(min(groups.x, 65535u), min(groups.y, 1u), min(groups.z, 1u));
        cost = groups.x * groups.y * groups.z * 128u;
    }
    bool over = false;
    if (cost > 0) {
        uint before;
        budget.InterlockedOr(0, 0, before);
        [allow_uav_condition]
        while (before < BUDGET) {
            uint next = (BUDGET - before <= cost) ? BUDGET : before + cost;
            uint prior;
            budget.InterlockedCompareExchange(0, before, next, prior);
            if (prior == before) break;
            before = prior;
        }
        over = before >= BUDGET || BUDGET - before < cost;
    }
    if (kind <= 1) {
        if (over) draw = uint2(0, 0);
        dst_args.Store2(dst_at, draw);
        for (uint off = 8; off < stride; off += 4) dst_args.Store(dst_at + off, src_args.Load(src_at + off));
    } else {
        if (over) groups = uint3(0, 0, 0);
        dst_args.Store3(dst_at, groups);
        for (uint off = 12; off < stride; off += 4) dst_args.Store(dst_at + off, src_args.Load(src_at + off));
    }
}
"#;

struct Clamp {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Clamp {}
// SAFETY: as above.
unsafe impl Sync for Clamp {}

#[cfg(test)]
fn clamp() -> Option<&'static Clamp> {
    static CLAMP: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    CLAMP
        .get_or_init(|| match make() {
            Ok(c) => Some(c),
            Err(e) => {
                tracing::warn!(error = %e, "no argument clamp: indirect execution dropped");
                None
            }
        })
        .as_ref()
}

fn copy_clamp() -> Option<&'static Clamp> {
    static CLAMP: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    CLAMP
        .get_or_init(|| match make_source(COPY_SOURCE) {
            Ok(c) => Some(c),
            Err(e) => {
                tracing::warn!(error = %e, "no copy argument clamp: indirect execution dropped");
                None
            }
        })
        .as_ref()
}

#[cfg(test)]
fn make() -> windows::core::Result<Clamp> {
    make_source(SOURCE)
}

/// Builds the small safety compute program from a supplied, embedded HLSL
/// source.  The destructive compatibility clamp and the newer copy clamp use
/// the same D3D12 plumbing but deliberately have independent cached PSOs.
fn make_source(source: &str) -> windows::core::Result<Clamp> {
    let device = &host().ok_or_else(windows::core::Error::empty)?.device;
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            s!("clamp"),
            None,
            None,
            s!("main"),
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            // SAFETY: the blob's own bytes.
            let text = unsafe {
                std::slice::from_raw_parts(errors.GetBufferPointer().cast::<u8>(), errors.GetBufferSize())
            };
            tracing::warn!("argument clamp: {}", String::from_utf8_lossy(text));
        }
        return Err(e);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: the blob's own bytes; the root signature is inside the shader.
    unsafe {
        let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
        let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        })?;
        Ok(Clamp { root, pso })
    }
}

/// Copies and bounds the title's indirect arguments into a per-list scratch
/// buffer.  In particular, this does not overwrite results written by an
/// earlier title compute dispatch in the same command list.
pub(crate) fn copy_bound(
    list: &HostList,
    src: (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    (kind, stride, records): (u32, u32, u32),
) -> Option<ID3D12Resource> {
    let minimum = match kind {
        0 => 16,
        1 => 20,
        2 | 10 => 12,
        _ => return None,
    };
    if stride < minimum || !stride.is_multiple_of(4) || !src.1.is_multiple_of(4)
        || count.is_some_and(|(_, off)| !off.is_multiple_of(4))
    {
        return None;
    }
    let bytes = u64::from(stride).checked_mul(u64::from(records))?;
    if bytes > u64::from(u32::MAX) || src.1.checked_add(bytes)? > u64::from(u32::MAX) {
        return None;
    }
    let (clamp, budget, scratch) = (copy_clamp()?, list.budget()?, list.indirect_scratch(bytes)?);
    // SAFETY: the source/count/scratch are live buffers owned by the list or
    // its shadows. Sources retain the legacy UAV state; scratch starts in
    // COMMON on every use.
    unsafe {
        let src_gpu = src.0.GetGPUVirtualAddress();
        let dst_gpu = scratch.GetGPUVirtualAddress();
        let (count_gpu, has_count, count_offset) = match count {
            Some((buffer, offset)) if offset.checked_add(4)? <= u64::from(u32::MAX) => {
                (buffer.GetGPUVirtualAddress(), 1u32, offset as u32)
            }
            Some(_) => return None,
            None => (src_gpu, 0u32, 0u32),
        };
        if !src_gpu.is_multiple_of(4) || !dst_gpu.is_multiple_of(4)
            || !budget.is_multiple_of(4) || !count_gpu.is_multiple_of(4)
        {
            return None;
        }
        let constants = [kind, stride, records, src.1 as u32, has_count, count_offset];
        list.list.ResourceBarrier(&[
            uav_barrier(),
            transition(&scratch, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
        ]);
        if records != 0 {
            list.list.SetComputeRootSignature(&clamp.root);
            list.list.SetPipelineState(&clamp.pso);
            list.list.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
            list.list.SetComputeRootUnorderedAccessView(1, src_gpu);
            list.list.SetComputeRootUnorderedAccessView(2, count_gpu);
            list.list.SetComputeRootUnorderedAccessView(3, dst_gpu);
            list.list.SetComputeRootUnorderedAccessView(4, budget);
            list.list.Dispatch(records.div_ceil(64), 1, 1);
        }
        list.list.ResourceBarrier(&[
            uav_barrier(),
            transition(&scratch, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT),
        ]);
    }
    Some(scratch)
}

/// Returns a copy-clamp scratch buffer to COMMON so its next use can be
/// implicitly promoted again.
pub(crate) fn copy_done(list: &HostList, scratch: &ID3D12Resource) {
    // SAFETY: the preceding copy clamp put this buffer in INDIRECT_ARGUMENT.
    unsafe {
        list.list.ResourceBarrier(&[transition(
            scratch,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            D3D12_RESOURCE_STATE_COMMON,
        )])
    };
}

/// Records, in `list`, the legacy in-place clamp of `records` records of `kind` (PC
/// argument numbering) `stride` bytes apart at host address `gpu` in
/// `buffer`, and leaves the buffer ready to be read as arguments. Replaces
/// the list's compute root signature and pipeline (callers bind theirs
/// after). False if the clamp is not available.
#[cfg(test)]
pub(crate) fn bound(
    list: &HostList,
    buffer: &ID3D12Resource,
    gpu: u64,
    (kind, stride, records): (u32, u32, u32),
) -> bool {
    let (Some(c), Some(budget)) = (clamp(), list.budget()) else {
        return false;
    };
    if !gpu.is_multiple_of(4) || !stride.is_multiple_of(4) {
        return false;
    }
    let l = &list.list;
    let constants = [kind, stride, records, 0];
    // SAFETY: an open list; our root signature, pipeline and a live buffer
    // the caller's list holds.
    unsafe {
        // Earlier shader writes (the title's argument passes) first.
        l.ResourceBarrier(&[uav_barrier()]);
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        l.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
        l.SetComputeRootUnorderedAccessView(1, gpu);
        l.SetComputeRootUnorderedAccessView(2, budget);
        // One group walks every record.
        l.Dispatch(records.min(1), 1, 1);
        l.ResourceBarrier(&[
            uav_barrier(),
            transition(
                buffer,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            ),
        ]);
    }
    true
}

/// After the execution: the buffer back to unordered access, where the
/// title's passes keep using it.
#[cfg(test)]
pub(crate) fn done(list: &HostList, buffer: &ID3D12Resource) {
    // SAFETY: an open list and a live buffer.
    unsafe {
        list.list.ResourceBarrier(&[transition(
            buffer,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        )])
    };
}

fn uav_barrier() -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: std::mem::ManuallyDrop::new(None) }),
        },
    }
}

#[cfg(test)]
mod tests {
    /// The clamp compiles with the system's compiler and the host takes it.
    #[test]
    fn argument_clamp_builds() {
        if super::host().is_none() {
            return; // no host GPU here
        }
        assert!(super::clamp().is_some());
    }

    /// Clamps `records` records of `kind`, filled by repeating `words`,
    /// `calls` times in one list: each record's first word after (zero when
    /// it runs empty). One record more follows them and must stay as
    /// written. `None` without a host GPU.
    fn clamped(kind: u32, words: &[u32], records: u32, calls: u32) -> Option<Vec<u32>> {
        let host = super::host()?;
        // Words in a record of `kind`.
        let per = match kind {
            0 => 4,
            1 => 5,
            _ => 3,
        };
        let spec = (kind, per as u32 * 4, records);
        let all = per * (records as usize + 1);
        let (buffer, data) = host.shared_buffer(all as u64 * 4).unwrap();
        let data = data.cast::<u32>();
        let list = host.new_list().unwrap();
        // SAFETY: our mapped buffer, written before the list runs and read
        // after it has; a live buffer.
        unsafe {
            for at in 0..all {
                data.add(at).write_volatile(words[at % words.len()]);
            }
            let gpu = buffer.GetGPUVirtualAddress();
            for _ in 0..calls {
                assert!(super::bound(&list, &buffer, gpu, spec));
                super::done(&list, &buffer);
            }
            list.close();
            host.execute(&[&list]);
            for at in all - per..all {
                assert_eq!(data.add(at).read_volatile(), words[at % words.len()]);
            }
            let first = |i: usize| data.add(i * per).read_volatile();
            Some((0..records as usize).map(first).collect())
        }
    }

    /// Rejected records do not wrap the budget: of 128 garbage draws (2^26
    /// vertex instances each once clamped) two run, where a wrapping add
    /// let four; of 600 garbage mesh dispatches 16, not 31.
    #[test]
    fn rejected_records_do_not_wrap_the_budget() {
        let Some(draws) = clamped(0, &[u32::MAX], 128, 1) else {
            return; // no host GPU here
        };
        assert_eq!(draws.iter().filter(|&&count| count != 0).count(), 2);
        let meshes = clamped(10, &[u32::MAX], 600, 1).unwrap();
        assert_eq!(meshes.iter().filter(|&&groups| groups != 0).count(), 16);
    }

    /// The budget goes to the first records, in order: of 200 draws of
    /// 3 * 2^18 vertices the first 170 run (it ends inside the third tile);
    /// small records that all fit all run.
    #[test]
    fn the_budget_goes_to_the_first_records() {
        let Some(draws) = clamped(0, &[3 << 18, 1, 0, 0], 200, 1) else {
            return; // no host GPU here
        };
        assert!(draws[..170].iter().all(|&count| count != 0));
        assert!(draws[170..].iter().all(|&count| count == 0));
        let small = clamped(0, &[1], 1 << 16, 1).unwrap();
        assert!(small.iter().all(|&count| count == 1));
    }

    /// A record that does not fit ends the budget: after two draws of
    /// 3 * 2^24 vertex instances a third takes what is left, and the small
    /// ones after it run empty.
    #[test]
    fn a_record_that_does_not_fit_ends_the_budget() {
        let mut words = [1 << 24, 3, 0, 0].repeat(3);
        words.extend([3, 1, 0, 0].repeat(6));
        let Some(draws) = clamped(0, &words, 8, 1) else {
            return; // no host GPU here
        };
        let ran: Vec<bool> = draws.iter().map(|&count| count != 0).collect();
        assert_eq!(ran, [true, true, false, false, false, false, false, false]);
    }

    /// The budget is the list's: a second execution of the same records
    /// finds it spent, and they all run empty.
    #[test]
    fn a_spent_budget_stays_spent() {
        let Some(twice) = clamped(1, &[u32::MAX], 64, 2) else {
            return; // no host GPU here
        };
        assert!(twice.iter().all(|&count| count == 0));
    }
}
