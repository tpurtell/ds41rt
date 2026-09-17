//! RTX-only ownership for every native dSpark tensor and independent stage experts.
mod chain;
pub(crate) use chain::{DsparkChain, DistributedDsparkChain};
mod stage;
pub(crate) use stage::DsparkStage;
mod main_context;
pub(crate) use main_context::{DsparkMainContext, MainProposal, prepare_commit_rows};
mod attention_wave;
pub(crate) use attention_wave::DsparkAttentionWave;
mod attention_output;
pub(crate) use attention_output::DsparkAttentionOutput;
mod confidence;
mod projection;
pub(crate) use projection::{DsparkProjection, ProjectionKind};
mod ffn;
pub(crate) use ffn::DsparkFfn;
mod hc;
pub(crate) use hc::HcSublayer;
mod markov;
mod router;
mod shared;
pub(crate) use shared::DsparkSharedFfn;
mod terminal;
pub(crate) use confidence::DsparkConfidence;
pub(crate) use markov::DsparkMarkov;
pub(crate) use router::DsparkRouter;
pub(crate) use terminal::{DsparkTerminal, DistributedDsparkTerminal};

use super::{ExpertLayer, ExpertWeights};
use crate::v41_dspark_cache::DsparkWindow;
use crate::v41_tensors::NativeRtxTensors;
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{Ds41rtDeviceBuffer, NativeLibrary};
use ds41rt_loader::OfficialV41Catalog;
use std::{path::{Path, PathBuf}, rc::Rc};
use super::exl3::Exl3Weights;
mod expert_backend;
mod tp2;
use expert_backend::{DraftExperts, CompressedDraftExperts};

#[derive(Debug, Clone, Copy)]
pub(crate) struct DsparkBudget {
    pub expert_resident_bytes: usize,
    pub auxiliary_resident_bytes: usize,
    pub shared_packed_scale_bytes: usize,
    pub projection_packed_scale_bytes: usize,
    pub grouped_output_resident_bytes: usize,
    pub projection_bytes_per_wave: usize,
    pub main_context_additional_bytes_per_wave: usize,
    pub draft_token_bytes_per_wave: usize,
    pub attention_output_additional_bytes_per_wave: usize,
    pub attention_wave_additional_bytes_per_wave: usize,
    pub shared_execution_bytes_per_wave: usize,
    pub load_staging_bytes: usize,
    pub window_cache_bytes: usize,
    pub execution_bytes_per_wave: usize,
    pub hc_bytes_per_wave: usize,
    pub router_bytes_per_wave: usize,
    pub confidence_bytes_per_wave: usize,
    pub markov_bytes_per_wave: usize,
    pub terminal_additional_bytes_per_wave: usize,
}
impl DsparkBudget {
    pub fn resident_bytes(self) -> Result<usize> {
        self.expert_resident_bytes
            .checked_add(self.auxiliary_resident_bytes)
            .and_then(|bytes| bytes.checked_add(self.shared_packed_scale_bytes))
            .and_then(|bytes| bytes.checked_add(self.projection_packed_scale_bytes))
            .and_then(|bytes| bytes.checked_add(self.grouped_output_resident_bytes))
            .context("dSpark residency overflow")
    }
    /// Experts load serially, then native auxiliary tensors, then execution waves.
    /// Committed dSpark windows are shared across waves and counted once below.
    /// Remaining draft-attention/norm scratch, shared head,
    /// driver and graph allocations must be budgeted separately by the coordinator.
    pub fn peak_device_bytes(self, waves: usize) -> Result<usize> {
        ensure!((1..=2).contains(&waves), "dSpark needs one or two waves");
        let execution = self
            .execution_bytes_per_wave
            .checked_add(self.draft_token_bytes_per_wave)
            .and_then(|bytes| bytes.checked_add(self.main_context_additional_bytes_per_wave))
            .and_then(|bytes| bytes.checked_add(self.shared_execution_bytes_per_wave))
            .and_then(|bytes| bytes.checked_add(self.projection_bytes_per_wave))
            .and_then(|bytes| bytes.checked_add(self.attention_output_additional_bytes_per_wave))
            .and_then(|bytes| bytes.checked_add(self.attention_wave_additional_bytes_per_wave))
            .and_then(|bytes| bytes.checked_add(self.hc_bytes_per_wave))
            .and_then(|bytes| bytes.checked_add(self.router_bytes_per_wave))
            .and_then(|bytes| bytes.checked_add(self.confidence_bytes_per_wave))
            .and_then(|bytes| bytes.checked_add(self.markov_bytes_per_wave))
            .and_then(|bytes| bytes.checked_add(self.terminal_additional_bytes_per_wave))
            .context("dSpark combined wave budget overflow")?
            .checked_mul(waves)
            .context("dSpark wave budget overflow")?;
        let loading = self
            .expert_resident_bytes
            .checked_add(self.load_staging_bytes)
            .context("dSpark load peak overflow")?;
        let serving = self
            .resident_bytes()?
            .checked_add(execution)
            .and_then(|bytes| bytes.checked_add(self.window_cache_bytes))
            .context("dSpark serving peak overflow")?;
        Ok(loading.max(serving))
    }
}

enum ExpertStages<'a> {
    Tp2([Rc<tp2::Weights<'a>>;2]),
    Full([ExpertWeights<'a>; 3]),
    Exl3 { weights: Rc<Vec<Exl3Weights<'a>>>, directory: PathBuf },
}
pub(crate) struct DsparkWeights<'library> {
    library: &'library NativeLibrary,
    draft_width: usize,
    experts: ExpertStages<'library>,
    auxiliary: NativeRtxTensors<'library>,
    budget: DsparkBudget,
    shared_scales: [crate::v41_memory::DeviceAllocation<'library>; 9],
    grouped_output_scales: [crate::v41_memory::DeviceAllocation<'library>; 3],
    projection_scales: [crate::v41_memory::DeviceAllocation<'library>; 13],
}
impl<'library> DsparkWeights<'library> {
    fn full_expert(&self, stage: usize) -> Option<&ExpertWeights<'library>> {
        match &self.experts { ExpertStages::Full(weights) => weights.get(stage), ExpertStages::Exl3 { .. } | ExpertStages::Tp2(_) => None }
    }
    fn expert_bytes(&self, capacity: u32) -> Result<usize> {
        match &self.experts {
            ExpertStages::Tp2(_) => Ok(tp2::RoutedWave::device_bytes(self.library,capacity)?[1]),
            ExpertStages::Full(weights) => weights[0].execution_budget(capacity)?.total(),
            ExpertStages::Exl3 { directory, .. } => CompressedDraftExperts::device_bytes(directory, capacity),
        }
    }
    fn expert_wave(&self, stage: usize, capacity: u32) -> Result<DraftExperts<'_, 'library>> {
        ensure!(stage < 3, "invalid dSpark expert stage");
        match &self.experts {
            ExpertStages::Tp2(weights) => Ok(DraftExperts::Tp2(tp2::RoutedWave::new(self,weights.clone(),stage,capacity)?)),
            ExpertStages::Full(weights) => Ok(DraftExperts::Full(weights[stage].execution(capacity, self.expert_bytes(capacity)?)?)),
            ExpertStages::Exl3 { weights, directory } => Ok(DraftExperts::Exl3(unsafe {
                CompressedDraftExperts::new(self, weights.clone(), directory, stage, capacity)?
            })),
        }
    }
    pub fn peer_chain_bytes(&self,requests:u32)->Result<usize> {
        if matches!(self.experts,ExpertStages::Tp2(_)) {
            let capacity=DsparkAttentionWave::projection_capacity_with_width(requests,self.draft_width)?;
            tp2::RoutedWave::device_bytes(self.library,capacity)?[0].checked_mul(3).context("draft TP2 peer chain overflow")
        } else {Ok(0)}
    }
    pub fn draft_width(&self) -> usize { self.draft_width }
    fn auxiliary_names(catalog: &OfficialV41Catalog) -> Vec<String> {
        catalog
            .tensors()
            .iter()
            .map(|tensor| &tensor.metadata.name)
            .filter(|name| name.starts_with("mtp.") && !name.contains(".ffn.experts."))
            .cloned()
            .collect()
    }
    pub fn plan(
        library: &NativeLibrary,
        catalog: &OfficialV41Catalog,
        capacity: u32,
    ) -> Result<DsparkBudget> {
        Self::plan_with_width(library, catalog, capacity, 5, None)
    }
    fn plan_with_width(library: &NativeLibrary, catalog: &OfficialV41Catalog,
        capacity: u32, width: usize, exl3_directory: Option<&Path>) -> Result<DsparkBudget> {
        ensure!(matches!(width, 5 | 7), "draft width must be five or seven");
        let mut expert_resident_bytes = 0usize;
        let mut load_staging_bytes = 0usize;
        for stage in 0..3 {
            let budget = if catalog.exl3().is_some() {
                Exl3Weights::plan(catalog, ExpertLayer::Dspark { stage })?
            } else { ExpertWeights::plan(library, catalog, ExpertLayer::Dspark { stage })? };
            expert_resident_bytes = expert_resident_bytes
                .checked_add(budget.resident_bytes)
                .context("dSpark expert residency overflow")?;
            load_staging_bytes = load_staging_bytes.max(budget.device_staging_bytes);
        }
        let auxiliary_resident_bytes =
            NativeRtxTensors::plan(catalog, &Self::auxiliary_names(catalog))?;
        let execution_bytes_per_wave = (if catalog.exl3().is_some() {
            CompressedDraftExperts::device_bytes(exl3_directory.context("dSpark EXL3 AOT directory missing")?, capacity)?
        } else { ExpertWeights::plan_execution(library, capacity)?.total()? })
            .checked_mul(3)
            .context("dSpark stage workspace overflow")?;
        Ok(DsparkBudget {
            expert_resident_bytes,
            auxiliary_resident_bytes,
            shared_packed_scale_bytes: shared::packed_scale_bytes(library)?,
            projection_packed_scale_bytes: projection::packed_bytes(library)?,
            grouped_output_resident_bytes: 3 * 1048576,
            projection_bytes_per_wave: projection::wave_bytes(library, capacity)?,
            draft_token_bytes_per_wave: 64,
            main_context_additional_bytes_per_wave: DsparkMainContext::additional_bytes(library, capacity)?,
            attention_output_additional_bytes_per_wave: DsparkAttentionOutput::additional_bytes(library, capacity)? * 3,
            attention_wave_additional_bytes_per_wave: DsparkAttentionWave::additional_bytes(capacity)? * 3,
            shared_execution_bytes_per_wave: DsparkSharedFfn::device_bytes(library, capacity)? * 3,
            load_staging_bytes,
            window_cache_bytes: DsparkWindow::device_bytes(16, 4096)? * 3,
            execution_bytes_per_wave,
            router_bytes_per_wave: DsparkRouter::device_bytes(capacity as usize)? * 3,
            hc_bytes_per_wave: HcSublayer::device_bytes(capacity as usize)?
                .checked_mul(6)
                .context("mHC wave budget overflow")?,
            confidence_bytes_per_wave: DsparkConfidence::device_bytes((capacity as usize).max(16 * width))?,
            markov_bytes_per_wave: DsparkMarkov::device_bytes(16)?,
            terminal_additional_bytes_per_wave: DsparkTerminal::additional_bytes_with_width(16, width)?,
        })
    }
    /// Serving retains a large target-context projection buffer, but expert
    /// execution only needs the bounded speculative request batch. Reserve the
    /// full context owner in addition to the conservative draft-wave plan before
    /// loading weights; never size expert scratch from prefill capacity.
    pub fn load_serving(
        library: &'library NativeLibrary,
        catalog: &OfficialV41Catalog,
        context_capacity: u32,
        requests: u32,
        device_budget: usize,
        pinned_staging_bytes: usize,
    ) -> Result<Self> {
        Self::load_serving_with_width(library, catalog, context_capacity, requests,
            device_budget, pinned_staging_bytes, 5, None)
    }
    pub fn load_serving_with_width(library: &'library NativeLibrary, catalog: &OfficialV41Catalog,
        context_capacity: u32, requests: u32, device_budget: usize,
        pinned_staging_bytes: usize, width: usize, exl3_directory: Option<&Path>) -> Result<Self> {
        let draft_capacity = DsparkAttentionWave::projection_capacity_with_width(requests, width)?;
        let context_bytes = DsparkMainContext::device_bytes(library, context_capacity)?;
        let draft_budget = device_budget.checked_sub(context_bytes)
            .context("dSpark main context exceeds device budget")?;
        Self::load_with_width(library, catalog, draft_capacity, 1, draft_budget, pinned_staging_bytes, width, exl3_directory)
    }

    pub fn load_serving_tp2(library:&'library NativeLibrary,catalog:&OfficialV41Catalog,
        context_capacity:u32,requests:u32,mut budgets:[usize;2],pinned_staging_bytes:usize,width:usize)->Result<Self> {
        let capacity=DsparkAttentionWave::projection_capacity_with_width(requests,width)?;
        budgets[1]=budgets[1].checked_sub(DsparkMainContext::device_bytes(library,context_capacity)?)
            .context("TP2 draft main context exceeds RTX1 budget")?;
        Self::load_tp2_with_width(library,catalog,capacity,2,budgets,pinned_staging_bytes,width)
    }

    /// Native routed-expert TP2; attention/shared/projections stay on RTX1.
    /// Both ranks are admitted before loading any checkpoint payload.
    pub fn load_tp2_with_width(library:&'library NativeLibrary,catalog:&OfficialV41Catalog,
        capacity:u32,waves:usize,budgets:[usize;2],pinned_staging_bytes:usize,width:usize)->Result<Self> {
        ensure!(catalog.exl3().is_none(),"TP2 dSpark requires native expert weights");
        ensure!(library.cuda_get_device()?==1,"TP2 dSpark transformer belongs on RTX1");
        let devices=[crate::v41_memory::device::Device {library,id:0},crate::v41_memory::device::Device {library,id:1}];
        let mut budget=Self::plan_with_width(library,catalog,capacity,width,None)?;
        let mut resident=[0usize;2];let mut staging=[0usize;2];
        for rank in 0..2 { for stage in 0..3 {
            let plan=devices[rank].run(||ExpertWeights::plan(library,catalog,ExpertLayer::DsparkTp2 {stage,rank}))?;
            resident[rank]=resident[rank].checked_add(plan.resident_bytes).context("draft TP2 residency overflow")?;
            staging[rank]=staging[rank].max(plan.device_staging_bytes);
        }}
        let workspace=tp2::RoutedWave::device_bytes(library,capacity)?;
        budget.expert_resident_bytes=resident[1];
        budget.load_staging_bytes=budget.load_staging_bytes.max(staging[1]);
        budget.execution_bytes_per_wave=workspace[1].checked_mul(3).context("draft TP2 workspace overflow")?;
        let root_peak=budget.peak_device_bytes(waves)?;
        let peer_peak=resident[0].checked_add(staging[0].max(workspace[0].checked_mul(3*waves).context("draft TP2 peer wave overflow")?)).context("draft TP2 peer budget overflow")?;
        ensure!(peer_peak<=budgets[0] && root_peak<=budgets[1],
            "draft TP2 needs GPU0/GPU1 bytes {peer_peak}/{root_peak}, budgets are {}/{}",budgets[0],budgets[1]);
        let weights=tp2::Weights::load_pair(devices,catalog,budgets)?;
        Self::load_with_experts(library,catalog,capacity,waves,budgets[1],pinned_staging_bytes,width,None,Some((weights,budget)))
    }

    /// Admit all three stages and the requested expert wave workspaces before
    /// reading payloads; this does not allocate the wave workspaces themselves.
    pub fn load(
        library: &'library NativeLibrary,
        catalog: &OfficialV41Catalog,
        capacity: u32,
        waves: usize,
        device_budget: usize,
        pinned_staging_bytes: usize,
    ) -> Result<Self> {
        Self::load_with_width(library, catalog, capacity, waves, device_budget, pinned_staging_bytes, 5, None)
    }
    fn load_with_width(library: &'library NativeLibrary, catalog: &OfficialV41Catalog,
        capacity: u32, waves: usize, device_budget: usize, pinned_staging_bytes: usize,
        width: usize, exl3_directory: Option<&Path>) -> Result<Self> {
        Self::load_with_experts(library,catalog,capacity,waves,device_budget,pinned_staging_bytes,width,exl3_directory,None)
    }
    fn load_with_experts(library: &'library NativeLibrary, catalog: &OfficialV41Catalog,
        capacity:u32,waves:usize,device_budget:usize,pinned_staging_bytes:usize,width:usize,
        exl3_directory:Option<&Path>,tp2:Option<([Rc<tp2::Weights<'library>>;2],DsparkBudget)>)->Result<Self> {
        let budget = if let Some((_,budget))=&tp2 {*budget} else {Self::plan_with_width(library, catalog, capacity, width, exl3_directory)?};
        ensure!(
            budget.peak_device_bytes(waves)? <= device_budget,
            "dSpark RTX residency and expert waves exceed device budget"
        );
        ensure!(
            (1..=64 * 1024 * 1024).contains(&pinned_staging_bytes),
            "dSpark auxiliary pinned staging must be 1 byte through 64 MiB"
        );
        let mut resident = 0usize;
        let experts = if let Some((weights,_))=tp2 {
            resident=weights[1].resident_bytes();
            ExpertStages::Tp2(weights)
        } else if catalog.exl3().is_some() {
            let mut weights = Vec::with_capacity(3);
            for stage in 0..3 {
                let weight = Exl3Weights::load(library, catalog, ExpertLayer::Dspark { stage },
                    device_budget.checked_sub(resident).context("dSpark remaining budget underflow")?)?;
                resident = resident.checked_add(weight.budget.resident_bytes).context("dSpark residency overflow")?;
                weights.push(weight);
            }
            ExpertStages::Exl3 { weights: Rc::new(weights), directory: exl3_directory.context("dSpark EXL3 directory missing")?.to_owned() }
        } else {
            let mut weights = Vec::with_capacity(3);
            for stage in 0..3 {
                let weight = ExpertWeights::load(library, catalog, ExpertLayer::Dspark { stage },
                    device_budget.checked_sub(resident).context("dSpark remaining budget underflow")?)?;
                resident = resident.checked_add(weight.budget().resident_bytes).context("dSpark residency overflow")?;
                weights.push(weight);
            }
            ExpertStages::Full(weights.try_into().ok().context("dSpark requires three stages")?)
        };
        let auxiliary = NativeRtxTensors::load(
            library,
            catalog,
            &Self::auxiliary_names(catalog),
            device_budget
                .checked_sub(resident)
                .context("dSpark auxiliary budget underflow")?,
            pinned_staging_bytes,
        )?;
        let shared_scales = shared::pack_scales(library, &auxiliary)?;
        let projection_scales = projection::pack_scales(library, &auxiliary)?;
        let grouped_output_scales = attention_output::pack_grouped_scales(library, &auxiliary)?;
        Ok(Self {
            library,
            draft_width: width,
            grouped_output_scales,
            shared_scales,
            projection_scales,
            experts,
            auxiliary,
            budget,
        })
    }
    /// Three independent committed windows, shared across alternating waves.
    /// Each admits sixteen requests and a total of 4096 source KV rows per batch.
    pub fn windows(&self, budget: usize) -> Result<[DsparkWindow<'library>; 3]> {
        let per_stage = DsparkWindow::device_bytes(16, 4096)?;
        ensure!(
            per_stage * 3 <= budget,
            "dSpark windows exceed device budget"
        );
        let library = self.library;
        Ok([
            DsparkWindow::new(library, 16, 4096, per_stage)?,
            DsparkWindow::new(library, 16, 4096, per_stage)?,
            DsparkWindow::new(library, 16, 4096, per_stage)?,
        ])
    }
    pub fn budget(&self) -> DsparkBudget {
        self.budget
    }

    /// Includes stage-zero target projection, all attention/shared FFN/router/mHC
    /// tensors and the final Markov/confidence heads in native representations.
    pub fn tensor(&self, name: &str) -> Result<Ds41rtDeviceBuffer> {
        self.auxiliary.get(name)
    }

}
