use super::*;
use crate::{v41_experts::ExpertLayer, v41_memory::LoadStream};

struct Graph<'a> { library: &'a NativeLibrary, exec: *mut c_void, stream: *mut c_void }
impl Drop for Graph<'_> {
    fn drop(&mut self) {
        unsafe {
            let _ = self.library.cuda_stream_synchronize(self.stream);
            let _ = self.library.cuda_graph_exec_destroy(self.exec);
        }
    }
}

#[test]
#[ignore = "requires EXL3 snapshot, TP2 package, native library and two CUDA devices"]
fn shared_capacity_scratch_matches_private_and_graph_replay() -> Result<()> {
    let library = unsafe { NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
    let catalog = ds41rt_loader::read_official_v41_catalog(
        ds41rt_loader::OFFICIAL_V41_MODEL_ID, Path::new(&std::env::var("DS41RT_SNAPSHOT")?))?;
    let root = PathBuf::from(std::env::var("DS41RT_EXL3_AOT")?);
    let capacities = [1usize, 16, 80, 256, 1024, 4096];
    let directories: Vec<_> = capacities.iter().map(|c| root.join(format!("m{c}"))).collect();
    let format = Exl3InputFormat::Fp8K32;
    for device in 0..2 {
        library.cuda_set_device(device)?;
        let weights = Rc::new(vec![Exl3Weights::load(&library, &catalog,
            ExpertLayer::BackboneTp2 { layer: 0, rank: device as usize }, 4_000_000_000)?]);
        let arena = Exl3Workspace::new(&library, &directories)?;
        let mut states = directories.iter().map(|path| unsafe {
            Exl3Execution::with_shared_workspace(&library, weights.clone(), path, format, Some(arena.clone()))
        }).collect::<Result<Vec<_>>>()?;
        let actual_bytes = arena.allocations.values().map(|a| a.buffer.bytes).sum::<usize>()
            + states.iter().map(|s| s.workspace_bytes()).sum::<usize>();
        let planned = Exl3Workspace::plan(&directories, format)?;
        ensure!(actual_bytes == planned, "shared allocation budget mismatch");
        let old = directories.iter().map(|p| Exl3Execution::plan(p, format)).collect::<Result<Vec<_>>>()?
            .into_iter().sum::<usize>();
        ensure!(planned < old, "workspace sharing did not reduce storage");
        let inputs = [DeviceAllocation::new(&library, 4096*5280)?,
            DeviceAllocation::new(&library, 4096*24)?, DeviceAllocation::new(&library, 4096*24)?];
        let mut wire = vec![0;4096*5280];
        for (row, bytes) in wire.chunks_exact_mut(5280).enumerate() {
            for (col, value) in bytes[..5120].iter_mut().enumerate() { *value = 0x30 + ((row+col)%16) as u8; }
            bytes[5120..].fill(127);
        }
        library.copy_h2d(inputs[0].buffer, &wire)?;
        library.copy_h2d(inputs[1].buffer, &(0..4096*6i32).flat_map(|i| (i%6).to_ne_bytes()).collect::<Vec<_>>())?;
        library.copy_h2d(inputs[2].buffer, &(0..4096*6).flat_map(|_| (1f32/6.).to_ne_bytes()).collect::<Vec<_>>())?;
        let stream = LoadStream { library: &library, raw: library.cuda_stream_create()? };
        let read = |buffer: Ds41rtDeviceBuffer| -> Result<Vec<u8>> {
            let mut bytes = vec![0; buffer.bytes]; library.copy_d2h(&mut bytes, buffer)?; Ok(bytes)
        };
        let input_buffers = || std::array::from_fn(|i| inputs[i].buffer);
        let mut graphs = Vec::new();
        for (i, &capacity) in capacities.iter().enumerate() {
            let mut private = unsafe { Exl3Execution::with_input_format(&library, weights.clone(), &directories[i], format)? };
            let output = unsafe { private.launch(input_buffers(), capacity, stream.raw)? };
            unsafe { library.cuda_stream_synchronize(stream.raw)?; }
            let expected = read(output)?;
            ensure!(expected.chunks_exact(4).any(|b| f32::from_ne_bytes(b.try_into().unwrap()) != 0.), "zero reference");
            let actual = unsafe { states[i].launch(input_buffers(), capacity, stream.raw)? };
            unsafe { library.cuda_stream_synchronize(stream.raw)?; }
            ensure!(read(actual)? == expected, "private/shared mismatch at {capacity}");
            unsafe { library.cuda_graph_begin_capture(stream.raw)?; }
            let captured = unsafe { states[i].launch(input_buffers(), capacity, stream.raw) };
            let graph = Graph { library: &library,
                exec: unsafe { library.cuda_graph_end_capture(stream.raw)? }, stream: stream.raw };
            graphs.push((graph, captured?, expected));
        }
        // Replay every capacity after larger and smaller kernels have overwritten
        // the arena, with both zeroed and restored inputs. Counters stay private.
        for i in (0..capacities.len()).rev().chain(0..capacities.len()) {
            let (graph, output, expected) = &graphs[i];
            let mut zero = wire.clone();
            for row in zero.chunks_exact_mut(5280) { row[..5120].fill(0); }
            library.copy_h2d(inputs[0].buffer, &zero)?;
            unsafe { library.cuda_graph_launch(graph.exec, stream.raw)?; library.cuda_stream_synchronize(stream.raw)?; }
            ensure!(read(*output)?.chunks_exact(4).all(|b| f32::from_ne_bytes(b.try_into().unwrap()) == 0.), "stale graph input");
            library.copy_h2d(inputs[0].buffer, &wire)?;
            unsafe { library.cuda_graph_launch(graph.exec, stream.raw)?; library.cuda_stream_synchronize(stream.raw)?; }
            ensure!(read(*output)? == *expected, "shared graph mismatch at {}", capacities[i]);
        }
        eprintln!("GPU {device}: shared={planned} private={old} saved={} bytes; six capacities, changed-input graph replay exact", old-planned);
    }
    Ok(())
}
