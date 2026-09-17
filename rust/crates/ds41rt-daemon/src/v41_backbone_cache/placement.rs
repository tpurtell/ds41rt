//! Cache/attention placement follows compressed-source consumer groups.
use anyhow::{ensure, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CachePlacement { attention: [usize; 40] }
impl CachePlacement {
    pub fn new(attention: [usize; 40]) -> Result<Self> {
        ensure!(attention.iter().all(|&gpu| gpu < 2), "cache GPU must be zero or one");
        for (source,end) in [(2,8),(8,14),(14,20),(20,40)] {
            ensure!(attention[source..end].iter().all(|&gpu| gpu == attention[source]),
                "compressed source {source} must share its GPU with attention consumers");
        }
        Ok(Self { attention })
    }
    pub fn encoder_decoder() -> Self {
        Self { attention: std::array::from_fn(|layer| usize::from(layer >= 20)) }
    }
    pub fn attention(&self, layer: usize) -> Result<usize> {
        self.attention.get(layer).copied().ok_or_else(|| anyhow::anyhow!("invalid attention layer"))
    }
    pub fn sources(&self) -> [usize; 4] {
        [self.attention[2],self.attention[8],self.attention[14],self.attention[20]]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires DS41RT_NATIVE_LIB and two CUDA GPUs"]
    fn distributed_cache_allocations_and_leases_follow_placement() -> Result<()> {
        cache_allocations(false)
    }
    #[test]
    #[ignore = "requires SM peer copy and two CUDA GPUs"]
    fn replicated_cache_allocations_preserve_capacity_and_reset_both_devices() -> Result<()> {
        cache_allocations(true)
    }
    fn cache_allocations(replicated:bool) -> Result<()> {
        use super::super::BackboneCache;
        use ds41rt_ffi::NativeLibrary;
        let lib = unsafe { NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
        lib.cuda_set_device(0)?;
        for boundary in [20,14] {
            let placement = CachePlacement::new(std::array::from_fn(|layer| usize::from(layer >= boundary)))?;
            let pages = [2,2,2,4];
            let bytes = BackboneCache::distributed_device_bytes(placement,2,pages)?;
            assert_eq!(bytes.iter().sum::<usize>(),BackboneCache::device_bytes(2,pages)?);
            assert!(BackboneCache::new_distributed(&lib,placement,2,pages,[bytes[0]-1,bytes[1]]).is_err());
            let mut bank = if replicated {
                let replicated_bytes=BackboneCache::replicated_device_bytes(placement,2,pages)?;
                assert!(BackboneCache::new_replicated(&lib,placement,2,pages,
                    [replicated_bytes[0]-1,replicated_bytes[1]]).is_err());
                BackboneCache::new_replicated(&lib,placement,2,pages,replicated_bytes)?
            } else { BackboneCache::new_distributed(&lib,placement,2,pages,bytes)? };
            for cycle in 0..2 {
                let lease = bank.begin_request(0,42+cycle)?;
                assert_eq!(bank.committed_end(lease)?,0);
                let request = bank.request(lease)?;
                for layer in 0..40 {
                    let view = bank.windows[layer].view(request.windows[layer])?;
                    let expected = placement.attention(layer)? as i32;
                    assert_eq!(bank.attention_device(layer)?.id,expected);
                    assert_eq!(view.device_end.device_id,expected);
                    let mut end = [255;8];
                    bank.attention_device(layer)?.run(|| lib.copy_d2h(&mut end,view.device_end))?;
                    assert_eq!(end,[0;8]);
                    if replicated {
                        let replica=bank.windows[layer].replica().unwrap();
                        let peer=unsafe { replica.view(&bank.windows[layer],request.windows[layer])? };
                        assert_eq!(peer.device_end.device_id,1-expected);
                        replica.device().run(||lib.copy_d2h(&mut end,peer.device_end))?;
                        assert_eq!(end,[0;8]);
                    }
                }
                for (index,gpu) in placement.sources().into_iter().enumerate() {
                    let kv = bank.sources[index].kv_cache(request.sources[index])?;
                    let keys = bank.sources[index].index_cache(request.sources[index])?;
                    for buffer in [kv.values,kv.scales,kv.device_pages,kv.device_rows,keys.packed,keys.scales] {
                        assert_eq!(buffer.device_id,gpu as i32);
                    }
                    assert_eq!(kv.rows,0);
                    assert_eq!(bank.sources[index].source_cache().capacity,pages[index]*256);
                    if replicated {
                        let replica=bank.sources[index].replica().unwrap();
                        let peer=unsafe { replica.view(bank.sources[index].source_cache(),0,0)? };
                        assert_eq!(peer.values.device_id,1-gpu as i32);
                        let mut end=[255;8];
                        replica.device().run(||lib.copy_d2h(&mut end,peer.device_rows))?;
                        assert_eq!(end,[0;8]);
                    }
                }
                bank.release(&[lease])?;
                assert!(bank.committed_end(lease).is_err());
                assert_eq!(lib.cuda_get_device()?,0);
            }
            drop(bank);
            assert_eq!(lib.cuda_get_device()?,0);
        }
        Ok(())
    }
    #[test]
    fn placement_keeps_sources_with_consumers_and_allows_rebalancing() -> Result<()> {
        let initial = CachePlacement::encoder_decoder();
        assert_eq!(initial.sources(),[0,0,0,1]);
        let mut rebalanced = initial.attention;
        rebalanced[14..20].fill(1);
        assert_eq!(CachePlacement::new(rebalanced)?.sources(),[0,0,1,1]);
        rebalanced[19] = 0;
        assert!(CachePlacement::new(rebalanced).is_err());
        rebalanced[19] = 1;
        rebalanced[0] = 2;
        assert!(CachePlacement::new(rebalanced).is_err());
        assert!(initial.attention(40).is_err());
        Ok(())
    }
}
