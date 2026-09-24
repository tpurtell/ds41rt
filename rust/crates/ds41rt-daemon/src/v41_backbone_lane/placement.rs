//! Place attention/dense weights and reusable lane state beside their caches.
use super::*;
impl<'a> BackboneLaneWeights<'a> {
    /// TP2 shared weights are budgeted and loaded by their rank owners, not here.
    pub fn distributed_device_bytes(
        library: &NativeLibrary,
        catalog: &OfficialV41Catalog,
        placement: CachePlacement,
    ) -> Result<[usize; 2]> {
        Self::distributed_device_bytes_with_split(library,catalog,placement,false,false)
    }
    pub fn distributed_device_bytes_with_split(library:&NativeLibrary,catalog:&OfficialV41Catalog,
        placement:CachePlacement,split_query_b:bool,split_output_b:bool)->Result<[usize;2]> {
        let mut bytes = [0usize; 2];
        for layer in 0..40 {
            let gpu = placement.attention(layer)?;
            let mut groups = Self::layer_bytes_with_split(library, catalog, layer,split_query_b,split_output_b)?;
            groups[3] = 0;
            for group in groups {
                bytes[gpu] = bytes[gpu]
                    .checked_add(group)
                    .context("placed backbone weights overflow")?;
            }
        }
        Ok(bytes)
    }
    pub fn load_distributed(
        library: &'a NativeLibrary,
        catalog: &OfficialV41Catalog,
        placement: CachePlacement,
        budgets: [usize; 2],
        staging: usize,
    ) -> Result<Self> {
        Self::load_distributed_with_split(library,catalog,placement,budgets,staging,false,false)
    }
    pub fn load_distributed_with_split(library:&'a NativeLibrary,catalog:&OfficialV41Catalog,
        placement:CachePlacement,budgets:[usize;2],staging:usize,split_query_b:bool,split_output_b:bool)->Result<Self> {
        ensure!(
            Self::distributed_device_bytes_with_split(library, catalog, placement,split_query_b,split_output_b)?
                .into_iter()
                .zip(budgets)
                .all(|(need, budget)| need <= budget),
            "backbone weights exceed a GPU budget"
        );
        Self::load_placed_with_split(library, catalog, staging, Some(placement),split_query_b,split_output_b)
    }
    pub(super) fn load_placed(
        library: &'a NativeLibrary,
        catalog: &OfficialV41Catalog,
        staging: usize,
        placement: Option<CachePlacement>,
    ) -> Result<Self> {
        Self::load_placed_with_split(library,catalog,staging,placement,false,false)
    }
    fn load_placed_with_split(library:&'a NativeLibrary,catalog:&OfficialV41Catalog,staging:usize,
        placement:Option<CachePlacement>,split_query_b:bool,split_output_b:bool)->Result<Self> {
        let original = library.cuda_get_device()?;
        let layers = (0..40)
            .map(|layer| {
                let id = match placement {
                    Some(map) => map.attention(layer)? as i32,
                    None => original,
                };
                let [hc, query, projection, shared, router] =
                    Self::layer_bytes_with_split(library, catalog, layer,split_query_b,split_output_b)?;
                Device { library, id }.own(|| {
                    Ok(LayerWeights {
                        hc: BackboneHcWeights::load(library, catalog, layer, hc, staging)?,
                        query: AttentionQueryWeights::load_with_split(
                            library, catalog, layer, query, staging,split_query_b,
                        )?,
                        projection: AttentionOutputWeights::load_with_split(
                            library, catalog, layer, projection, staging,split_output_b,
                        )?,
                        shared: if placement.is_some() {
                            None
                        } else {
                            Some(BackboneSharedWeights::load(
                                library, catalog, layer, shared, staging,
                            )?)
                        },
                        router: BackboneRouterWeights::load(
                            library, catalog, layer, router, staging,
                        )?,
                    })
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            library,
            layers,
            placement,
            split_query_b,
            split_output_b,
        })
    }
}
impl<'w, 'a> BackboneLane<'w, 'a> {
    pub fn enable_tp2_query(&mut self,weights:[&'w [crate::v41_projection_tp2::Weights<'a>];2],
        budgets:[usize;2])->Result<()> {
        ensure!(self.weights.split_query_b && self.tp2_query.is_none(),"query split configuration differs");
        let owner=self.query.input().device_id as usize;
        self.tp2_query=Some(crate::v41_projection_tp2::Wave::new(weights,self.capacity as u32,owner,budgets)?);
        Ok(())
    }
    pub fn enable_tp2_output(&mut self,weights:[&'w [crate::v41_projection_tp2::Weights<'a>];2],
        budgets:[usize;2])->Result<()> {
        ensure!(self.weights.split_output_b && self.tp2_output.is_none(),"output split configuration differs");
        let owner=self.projection.input().device_id as usize;
        self.tp2_output=Some(crate::v41_projection_tp2::Wave::new(weights,self.capacity as u32,owner,budgets)?);
        Ok(())
    }
    pub fn placed_workspace_bytes(library: &NativeLibrary, capacity: u32) -> Result<usize> {
        Self::placed_workspace_bytes_with_split(library,capacity,false,false)
    }
    pub fn placed_workspace_bytes_with_split(library:&NativeLibrary,capacity:u32,split_query_b:bool,split_output_b:bool)->Result<usize> {
        let mut groups = Self::workspace_bytes(library, capacity)?;
        groups[1]=AttentionQueryWave::device_bytes_with_split(library,capacity,split_query_b)?;
        groups[2]=AttentionOutputWave::device_bytes_with_split(library,capacity,split_output_b)?;
        groups[3] = 0;
        groups.into_iter().try_fold(0usize, |n, b| {
            n.checked_add(b)
                .context("placed backbone workspace overflow")
        })
    }
    pub fn new_on_device(
        weights: &'w BackboneLaneWeights<'a>,
        capacity: u32,
        budget: usize,
        gpu: usize,
    ) -> Result<DeviceOwner<'a, Self>> {
        ensure!(
            gpu < 2 && weights.placement.is_some(),
            "placed backbone lane requires distributed weights and GPU 0 or 1"
        );
        let first = weights
            .layers
            .iter()
            .position(|layer| layer.device.id == gpu as i32)
            .context("GPU owns no backbone layers")?;
        let device = Device {
            library: weights.library,
            id: gpu as i32,
        };
        device.own(|| Self::new_inner(weights, capacity, budget, first))
    }
    /// Import the preceding layer into this GPU's reusable attention/block state.
    /// # Safety
    /// Previous output is complete. Source and destination remain exclusive and
    /// alive through completion or drained cancellation. No cross-lane join.
    pub async unsafe fn import_previous_cooperative(
        &mut self,
        previous: &BlockOutput<'_>,
        transfer: &mut crate::v41_block::BlockTransfer<'a>,
    ) -> Result<()> {
        self.phase = Phase::Invalid;
        ensure!(previous.layer < 39, "final layer has no successor");
        let layer = previous.layer + 1;
        let weights = &self.weights.layers[layer];
        ensure!(
            weights.device.id == self.block.inputs()[0].device_id,
            "imported layer belongs to another GPU"
        );
        let device = weights.device;
        device.run(|| {
            self.query.rebind(&weights.query)?;
            self.projection.rebind(&weights.projection)?;
            self.router.rebind(&weights.router)?;
            if let Some(shared) = &mut self.shared {
                shared.rebind(
                    weights
                        .shared
                        .as_ref()
                        .context("local shared weights absent")?,
                )?;
            }
            Ok(())
        })?;
        unsafe {
            self.block
                .import_previous_cooperative(&weights.hc, previous, transfer)
                .await?;
        }
        self.layer = layer;
        self.phase = Phase::Prepared;
        device.run(|| { self.record_layer_entry(); Ok(()) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires DS41RT_NATIVE_LIB, DS41RT_SNAPSHOT and two CUDA GPUs"]
    fn placed_backbone_weights_and_router_graphs_match_direct() -> Result<()> {
        fn execute_router(
            lib: &NativeLibrary,
            wave: &mut BackboneRouterWave<'_, '_>,
            hidden: &[u8],
            mask: &[u8],
            rows: u32,
            captured: bool,
        ) -> Result<Vec<Vec<u8>>> {
            let [input, modality] = wave.inputs();
            lib.copy_h2d(input, hidden)?;
            lib.copy_h2d(modality, mask)?;
            let output = unsafe {
                if captured {
                    wave.execute_captured(rows)?
                } else {
                    wave.execute(rows)?
                }
            };
            [
                output.scores,
                output.ids,
                output.routing,
                output.expert_input,
            ]
            .into_iter()
            .map(|b| {
                let mut bytes = vec![0; b.bytes];
                lib.copy_d2h(&mut bytes, b)?;
                Ok(bytes)
            })
            .collect()
        }
        let lib = unsafe { NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
        let catalog = ds41rt_loader::read_official_v41_catalog(
            ds41rt_loader::OFFICIAL_V41_MODEL_ID,
            std::path::Path::new(&std::env::var("DS41RT_SNAPSHOT")?),
        )?;
        lib.cuda_set_device(0)?;
        let placement = CachePlacement::new(std::array::from_fn(|layer| usize::from(layer >= 14)))?;
        let budgets = BackboneLaneWeights::distributed_device_bytes(&lib, &catalog, placement)?;
        let shared = (0..40)
            .map(|l| BackboneSharedWeights::device_bytes(&lib, &catalog, l))
            .collect::<Result<Vec<_>>>()?
            .iter()
            .sum::<usize>();
        assert_eq!(
            budgets.iter().sum::<usize>() + shared,
            BackboneLaneWeights::device_bytes(&lib, &catalog)?
        );
        assert!(
            BackboneLaneWeights::load_distributed(
                &lib,
                &catalog,
                placement,
                [budgets[0] - 1, budgets[1]],
                1024 * 1024
            )
            .is_err()
        );
        let weights =
            BackboneLaneWeights::load_distributed(&lib, &catalog, placement, budgets, 1024 * 1024)?;
        for (layer, w) in weights.layers.iter().enumerate() {
            assert_eq!(w.device.id, placement.attention(layer)? as i32);
            assert!(w.shared.is_none());
        }
        let bytes = BackboneLane::placed_workspace_bytes(&lib, 16)?;
        assert!(BackboneLane::new(&weights, 16, usize::MAX).is_err());
        let mut lanes = [
            BackboneLane::new_on_device(&weights, 16, bytes, 0)?,
            BackboneLane::new_on_device(&weights, 16, bytes, 1)?,
        ];
        assert!(lanes.iter().all(|lane| lane.shared.is_none()));
        assert_eq!([lanes[0].layer, lanes[1].layer], [0, 14]);
        for layer in [0, 14, 20] {
            let reference_weights = BackboneRouterWeights::load(
                &lib,
                &catalog,
                layer,
                BackboneRouterWeights::device_bytes(&catalog, layer)?,
                1024 * 1024,
            )?;
            let mut reference =
                reference_weights.wave(16, BackboneRouterWave::device_bytes(16)?)?;
            let gpu = placement.attention(layer)?;
            let device = weights.layers[layer].device;
            device.run(|| lanes[gpu].router.rebind(&weights.layers[layer].router))?;
            for (rows, seed) in [(16, 0), (1, 7), (16, 13)] {
                let hidden: Vec<u8> = (0..rows * 5120)
                    .flat_map(|i| {
                        let value = ((i + seed) % 31) as f32 / 32.0 - 0.5;
                        ((value.to_bits() >> 16) as u16).to_ne_bytes()
                    })
                    .collect();
                let mask: Vec<u8> = (0..rows).map(|i| (i % 2) as u8).collect();
                let mut results = Vec::new();
                for actual in [false, true] {
                    let d = if actual {
                        device
                    } else {
                        Device {
                            library: &lib,
                            id: 0,
                        }
                    };
                    results.push(d.run(|| {
                        if actual {
                            execute_router(
                                &lib,
                                &mut lanes[gpu].get_mut().router,
                                &hidden,
                                &mask,
                                rows as u32,
                                true,
                            )
                        } else {
                            execute_router(&lib, &mut reference, &hidden, &mask, rows as u32, false)
                        }
                    })?);
                    assert_eq!(lib.cuda_get_device()?, 0);
                }
                assert!(
                    results[0] == results[1],
                    "placed router graph differs at layer {layer}"
                );
            }
        }
        let single_weights = BackboneLaneWeights::load(
            &lib,
            &catalog,
            BackboneLaneWeights::device_bytes(&lib, &catalog)?,
            1024 * 1024,
        )?;
        assert!(
            single_weights
                .layers
                .iter()
                .all(|w| w.device.id == 0 && w.shared.is_some())
        );
        let mut single = BackboneLane::new(
            &single_weights,
            16,
            BackboneLane::workspace_bytes(&lib, 16)?.iter().sum(),
        )?;
        assert!(single.shared.is_some());
        single.restart()?;
        assert_eq!(lib.cuda_get_device()?, 0);
        eprintln!(
            "placed backbone weights={budgets:?}; omitted full shared weights={shared}; lane C16 bytes={bytes}"
        );
        Ok(())
    }
}
