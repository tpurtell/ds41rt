//! Select expert storage and execution from the checkpoint format at startup.
use super::*;
use crate::v41_experts::{
    exl3::{worker::Exl3Worker, Exl3Weights},
    ExpertExecution,
};
use ds41rt_ffi::Ds41rtDeviceBuffer;
use ds41rt_loader::OfficialV41Catalog;
use ds41rt_transport::{ExpertProtocolV2DeviceResponseRef, ExpertProtocolV2ResponseRef};
use std::rc::Rc;

pub(super) enum Weights<'a> {
    Full(Vec<ExpertWeights<'a>>),
    Exl3(Rc<Vec<Exl3Weights<'a>>>),
}

impl<'a> Weights<'a> {
    pub(super) fn len(&self) -> usize {
        match self {
            Self::Full(weights) => weights.len(),
            Self::Exl3(weights) => weights.len(),
        }
    }
    /// Logical intermediate values this worker loaded per expert, reported as
    /// part of the structured startup evidence. `None` when nothing is resident
    /// (an all-remote worker), where the value would be a claim, not a fact.
    pub(super) fn intermediate(&self) -> Option<usize> {
        match self {
            Self::Full(weights) => weights.first().map(|weight| weight.intermediate()),
            // EXL3 packs its own tiered layout; the logical intermediate is the
            // checkpoint's own value, reported by the EXL3 weights themselves.
            Self::Exl3(_) => None,
        }
    }
    pub(super) fn execution(
        &self,
        library: &'a NativeLibrary,
        config: &NativeExpertServiceConfig,
        remaining: usize,
    ) -> Result<Execution<'_, 'a>> {
        Ok(match self {
            Self::Full(weights) => {
                let mut execution = weights[0].execution(config.capacity, remaining)?;
                execution.install_native_group(config.native_group()?)?;
                Execution::Full(execution)
            }
            Self::Exl3(weights) => Execution::Exl3(Exl3Worker::new(
                library,
                weights.clone(),
                &config.exl3_directory_for(
                    weights
                        .first()
                        .map(|weight| weight.layout.tiers.as_slice())
                        .unwrap_or(&[]),
                ),
                config.capacity,
                remaining,
            )?),
        })
    }
}

pub(super) enum Execution<'w, 'a> {
    Full(ExpertExecution<'w, 'a>),
    Exl3(Exl3Worker<'a>),
}
impl<'w, 'a> Execution<'w, 'a> {
    pub(super) fn is_paired(&self) -> bool { matches!(self, Self::Exl3(worker) if worker.is_paired()) }
    pub(super) fn bind_layer(&mut self, weights: &'w Weights<'a>, index: usize) -> Result<()> {
        match (self, weights) {
            (Self::Full(execution), Weights::Full(weights)) => execution.bind_layer(
                weights
                    .get(index)
                    .context("requested expert layer is not resident on this Spark")?,
            ),
            (Self::Exl3(execution), Weights::Exl3(_)) => execution.bind_layer(index),
            _ => anyhow::bail!("expert backend/weight mismatch"),
        }
    }
    pub(super) unsafe fn execute_mapped_request(
        &mut self,
        request: &V41BackboneRequest<'_>,
        executor_id: u64,
        exchange: &mut HostExpertExchange,
        slot: Ds41rtDeviceBuffer,
    ) -> Result<Option<ExpertProtocolV2DeviceResponseRef<'static>>> {
        match self {
            Self::Full(execution) => {
                execution.execute_mapped_request(request, executor_id, exchange, slot)
            }
            Self::Exl3(execution) => {
                execution.execute_mapped_request(request, executor_id, exchange, slot)
            }
        }
    }
    pub(super) fn execute_host_chunks<F>(
        &mut self,
        request: &V41BackboneRequest<'_>,
        executor_id: u64,
        exchange: &mut HostExpertExchange,
        row_indices: &mut [u32],
        max_frame_bytes: usize,
        sink: F,
    ) -> Result<()>
    where
        F: FnMut(ExpertProtocolV2ResponseRef<'_>) -> Result<()>,
    {
        match self {
            Self::Full(execution) => execution.execute_host_chunks(
                request,
                executor_id,
                exchange,
                row_indices,
                max_frame_bytes,
                sink,
            ),
            Self::Exl3(execution) => execution.execute_host_chunks(
                request,
                executor_id,
                exchange,
                row_indices,
                max_frame_bytes,
                sink,
            ),
        }
    }
}

pub(super) fn load_exl3<'a>(
    library: &'a NativeLibrary,
    catalog: &OfficialV41Catalog,
    config: &NativeExpertServiceConfig,
) -> Result<(Weights<'a>, usize)> {
    let exl3_tiers: &[usize] = catalog
        .exl3()
        .map(|manifest| manifest.decoder_tiers())
        .unwrap_or(&[]);
    let exl3_directory = config.exl3_directory_for(exl3_tiers);
    let partition = Exl3Worker::partition(&exl3_directory, config.capacity, config.rank)?;
    ensure!(config.world == 4 || partition == ds41rt_loader::V41Exl3Partition::Disjoint,
        "an implicit Spark TP2/TP3 group cannot use paired TP4 artifacts");
    let workspace = Exl3Worker::plan(&exl3_directory, config.capacity)
        .context("EXL3 checkpoint requires matching native AOT artifacts; set --exl3-aot-dir for a custom export")?;
    let plans = (config.first_layer..40)
        .map(|layer| {
            Exl3Weights::plan_with_layout(
                catalog,
                config.selection(layer)?,
                partition,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let resident = plans.iter().try_fold(0usize, |total, p| {
        total
            .checked_add(p.resident_bytes)
            .context("EXL3 resident budget overflow")
    })?;
    ensure!(
        resident
            .checked_add(workspace)
            .context("EXL3 worker budget overflow")?
            <= config.device_budget,
        "compressed EXL3 weights and workspace exceed device budget"
    );
    tracing::info!(rank=config.rank, world=config.world, first_layer=config.first_layer,
        layer_count=plans.len(), resident_bytes=resident, workspace_bytes=workspace,
        device_budget_bytes=config.device_budget, "EXL3 Spark residency plan");
    let mut weights = Vec::with_capacity(plans.len());
    let mut remaining = config.device_budget;
    for (index, plan) in plans.iter().enumerate() {
        let layer = config.first_layer + index;
        let started = std::time::Instant::now();
        let weight = Exl3Weights::load_with_layout(
            library,
            catalog,
            config.selection(layer)?,
            remaining,
            partition,
        )?;
        remaining = remaining
            .checked_sub(plan.resident_bytes)
            .context("EXL3 resident budget exhausted")?;
        tracing::info!(
            rank = config.rank,
            layer,
            resident_bytes = plan.resident_bytes,
            elapsed_ms = started.elapsed().as_millis() as u64,
            "compressed EXL3 expert layer loaded"
        );
        weights.push(weight);
    }
    Ok((Weights::Exl3(Rc::new(weights)), remaining))
}
