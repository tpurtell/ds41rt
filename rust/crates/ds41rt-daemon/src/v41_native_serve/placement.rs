//! Startup-only handoff between the live GPU memory planner and release launcher.
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Plan {
    version: u32,
    nonce: String,
    rtx_gpus: u32,
    rtx_expert_layers: usize,
    spark_first_layer: usize,
}

pub(super) struct StartupPlacement {
    directory: PathBuf,
    plan: Plan,
}
impl StartupPlacement {
    /// State lives inside one coordinator container. New deployments start empty;
    /// acknowledged boundaries survive stop/start of that same container.
    pub fn publish(directory: &Path, layers: usize) -> Result<Self> {
        ensure!(
            (1..=40).contains(&layers),
            "invalid planned RTX expert boundary"
        );
        fs::create_dir_all(directory)?;
        if let Some(plan) = Self::read_plan(&directory.join("committed.json"))? {
            ensure!(
                plan.rtx_expert_layers == layers,
                "container restart cannot change acknowledged worker boundary"
            );
            return Ok(Self {
                directory: directory.to_owned(),
                plan,
            });
        }
        ensure!(
            !directory.join("plan.json").exists() && !directory.join("ready.json").exists(),
            "placement handoff directory contains stale state"
        );
        let plan = Plan {
            version: 1,
            nonce: format!(
                "{}-{}",
                std::process::id(),
                SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
            ),
            rtx_gpus: 2,
            rtx_expert_layers: layers,
            spark_first_layer: layers.min(39),
        };
        let temporary = directory.join(".plan-pending");
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .context("creating placement handoff")?;
        file.write_all(&serde_json::to_vec(&plan)?)?;
        file.sync_all()?;
        fs::rename(temporary, directory.join("plan.json"))?;
        Ok(Self {
            directory: directory.to_owned(),
            plan,
        })
    }
    fn read_plan(path: &Path) -> Result<Option<Plan>> {
        use std::io::Read;
        let file = match fs::File::open(path) {
            Ok(file) => file,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e).context("opening placement state"),
        };
        let mut bytes = Vec::new();
        file.take(4097).read_to_end(&mut bytes)?;
        ensure!(bytes.len() <= 4096, "placement state exceeds size limit");
        let plan: Plan = serde_json::from_slice(&bytes)?;
        ensure!(
            plan.version == 1
                && plan.rtx_gpus == 2
                && (1..=40).contains(&plan.rtx_expert_layers)
                && plan.spark_first_layer == plan.rtx_expert_layers.min(39)
                && !plan.nonce.is_empty()
                && plan.nonce.len() <= 64,
            "invalid placement state"
        );
        Ok(Some(plan))
    }
    pub fn resumed_layers(directory: &Path) -> Result<Option<usize>> {
        Ok(Self::read_plan(&directory.join("committed.json"))?.map(|p| p.rtx_expert_layers))
    }
    fn acknowledged(&self) -> Result<bool> {
        if let Some(committed) = Self::read_plan(&self.directory.join("committed.json"))? {
            ensure!(committed == self.plan, "committed placement changed");
            return Ok(true);
        }
        let Some(ready) = Self::read_plan(&self.directory.join("ready.json"))? else {
            return Ok(false);
        };
        ensure!(
            ready == self.plan,
            "worker acknowledgement does not match this placement and launch"
        );
        let temporary = self.directory.join(".committed-pending");
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&temporary)?;
        file.write_all(&serde_json::to_vec(&self.plan)?)?;
        file.sync_all()?;
        fs::rename(temporary, self.directory.join("committed.json"))?;
        Ok(true)
    }
    /// Called only during startup, after RTX weight loading and before transport
    /// connection. No serving lane exists in this wait loop.
    pub fn wait_ready(&self, timeout: Duration) -> Result<()> {
        let start = Instant::now();
        loop {
            if self.acknowledged()? {
                return Ok(());
            }
            ensure!(
                start.elapsed() < timeout,
                "timed out waiting for Spark placement readiness"
            );
            std::thread::sleep(
                Duration::from_millis(50).min(timeout.saturating_sub(start.elapsed())),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn placement_publication_and_exact_acknowledgement() -> Result<()> {
        for layers in [1, 17, 20, 40] {
            let directory = tempfile::tempdir()?;
            let handoff = StartupPlacement::publish(directory.path(), layers)?;
            assert!(!directory.path().join(".plan-pending").exists());
            let bytes = fs::read(directory.path().join("plan.json"))?;
            let plan: Plan = serde_json::from_slice(&bytes)?;
            assert_eq!(plan.spark_first_layer, layers.min(39));
            assert!(handoff.wait_ready(Duration::ZERO).is_err());
            fs::write(directory.path().join("ready.json"), bytes)?;
            handoff.wait_ready(Duration::ZERO)?;
            assert_eq!(
                StartupPlacement::resumed_layers(directory.path())?,
                Some(layers)
            );
            fs::remove_file(directory.path().join("ready.json"))?;
            StartupPlacement::publish(directory.path(), layers)?.wait_ready(Duration::ZERO)?;
            assert!(
                StartupPlacement::publish(directory.path(), if layers == 1 { 2 } else { 1 })
                    .is_err()
            );
        }
        Ok(())
    }
    #[test]
    fn placement_rejects_wrong_launch_boundary_and_oversized_ack() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let handoff = StartupPlacement::publish(directory.path(), 17)?;
        for mutation in [0, 1, 2] {
            let mut wrong = handoff.plan.clone();
            match mutation {
                0 => wrong.nonce.push('x'),
                1 => wrong.spark_first_layer = 20,
                _ => wrong.rtx_gpus = 1,
            }
            fs::write(
                directory.path().join("ready.json"),
                serde_json::to_vec(&wrong)?,
            )?;
            assert!(handoff.wait_ready(Duration::ZERO).is_err());
        }
        fs::write(directory.path().join("ready.json"), vec![b' '; 4097])?;
        assert!(handoff.wait_ready(Duration::ZERO).is_err());
        assert!(StartupPlacement::publish(directory.path(), 0).is_err());
        assert!(StartupPlacement::publish(directory.path(), 41).is_err());
        Ok(())
    }
}
