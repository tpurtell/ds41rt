//! Sparse image identity for the u32 token radix. Text retains its native keys.
use anyhow::{ensure, Context, Result};
use ds41rt_loader::{V41ImageSpan, V41_IMAGE_TOKEN_ID, V41_MAX_IMAGES};
use std::{
    borrow::Cow,
    collections::BTreeMap,
    rc::{Rc, Weak},
};

const FIRST_IMAGE_KEY: u64 = 1 << 31;
struct Identity {
    key: u32,
}
#[derive(Clone)]
struct Binding {
    start: usize,
    end: usize,
    identity: Rc<Identity>,
}
/// Active requests and saved snapshots own these pins. A key may be recycled
/// only after both have released it and the radix has removed orphaned edges.
#[derive(Clone, Default)]
pub(in crate::v41_native_serve) struct ImageKeys {
    bindings: Vec<Binding>,
}
impl ImageKeys {
    /// The native token slice remains authoritative for model execution and
    /// Engram history. Only cache matching sees the image-specific symbols.
    /// Token IDs must come from the official tokenizer/model vocabulary.
    pub fn encode<'t>(&self, tokens: &'t [u32]) -> Result<Cow<'t, [u32]>> {
        if self.bindings.is_empty() {
            return Ok(Cow::Borrowed(tokens));
        }
        let mut keys = tokens.to_vec();
        for binding in &self.bindings {
            if binding.start >= tokens.len() {
                break;
            }
            let end = binding.end.min(tokens.len());
            ensure!(
                tokens[binding.start..end]
                    .iter()
                    .all(|&t| t == V41_IMAGE_TOKEN_ID),
                "image cache binding differs from native tokens"
            );
            keys[binding.start..end].fill(binding.identity.key);
        }
        Ok(Cow::Owned(keys))
    }
    pub fn through(&self, end: usize) -> Self {
        Self {
            bindings: self
                .bindings
                .iter()
                .take_while(|b| b.start < end)
                .cloned()
                .collect(),
        }
    }
}
/// Weak identities never keep an evicted image alive. Collection happens only
/// on image admission, so text decode and text cache lookup do no interning work.
#[derive(Default)]
pub(super) struct ImageKeySpace {
    identities: BTreeMap<[u8; 32], (u32, Weak<Identity>)>,
    free: Vec<u32>,
    next: u64,
}
impl ImageKeySpace {
    pub fn prepare(&mut self, tokens: &[u32], images: &[V41ImageSpan]) -> Result<ImageKeys> {
        self.prepare_spans(
            tokens,
            &images
                .iter()
                .map(|span| {
                    (
                        span.start,
                        span.image.grid().tokens(),
                        *span.image.identity(),
                    )
                })
                .collect::<Vec<_>>(),
        )
    }
    fn prepare_spans(
        &mut self,
        tokens: &[u32],
        spans: &[(usize, usize, [u8; 32])],
    ) -> Result<ImageKeys> {
        if spans.is_empty() {
            return Ok(ImageKeys::default());
        }
        ensure!(
            spans.len() <= V41_MAX_IMAGES,
            "too many image cache identities"
        );
        ensure!(
            tokens.iter().all(|&t| t < 129280),
            "image prompt outside vocabulary"
        );
        let mut previous_end = 0;
        for &(start, length, _) in spans {
            let end = start
                .checked_add(length)
                .context("image cache span overflow")?;
            ensure!(
                length > 0
                    && length <= 1024
                    && start >= previous_end
                    && end <= tokens.len()
                    && tokens[start..end].iter().all(|&t| t == V41_IMAGE_TOKEN_ID),
                "invalid image cache span"
            );
            previous_end = end;
        }
        self.identities.retain(|_, (key, weak)| {
            if weak.strong_count() == 0 {
                self.free.push(*key);
                false
            } else {
                true
            }
        });
        let mut bindings = Vec::with_capacity(spans.len());
        for &(start, length, hash) in spans {
            let identity = if let Some(live) = self
                .identities
                .get(&hash)
                .and_then(|(_, weak)| weak.upgrade())
            {
                live
            } else {
                let key = if let Some(key) = self.free.pop() {
                    key
                } else {
                    self.next = self.next.max(FIRST_IMAGE_KEY);
                    let key =
                        u32::try_from(self.next).context("image cache key space exhausted")?;
                    self.next += 1;
                    key
                };
                let identity = Rc::new(Identity { key });
                self.identities
                    .insert(hash, (key, Rc::downgrade(&identity)));
                identity
            };
            bindings.push(Binding {
                start,
                end: start + length,
                identity,
            });
        }
        Ok(ImageKeys { bindings })
    }
}

#[cfg(test)]
mod tests {
    use ds41rt_core::prefix::{Radix, Retention, SnapshotKind};
    use super::*;

    fn prompt(text: usize, images: usize) -> Vec<u32> {
        [
            vec![42; text],
            vec![V41_IMAGE_TOKEN_ID; images * 10],
            vec![43; 160],
        ]
        .concat()
    }
    #[test]
    fn text_borrows_keys_and_does_not_collect_or_allocate_identities() -> Result<()> {
        let mut space = ImageKeySpace::default();
        let tokens = vec![42; 1_048_576];
        let keys = space.prepare(&tokens, &[])?;
        let encoded = keys.encode(&tokens)?;
        assert!(matches!(encoded, Cow::Borrowed(_)));
        assert_eq!(encoded.as_ptr(), tokens.as_ptr());
        assert!(space.identities.is_empty() && space.free.is_empty());
        assert_eq!(space.next, 0);
        Ok(())
    }
    #[test]
    fn prepared_image_spans_use_the_loader_content_identity() -> Result<()> {
        use ds41rt_loader::{V41Image, V41VisionPrompt};
        let mut space = ImageKeySpace::default();
        let native = [42, V41_IMAGE_TOKEN_ID, 43];
        let black =
            V41VisionPrompt::expand(&native, vec![V41Image::from_rgb(1, 1, &[0; 3])?], 1024)?;
        let same =
            V41VisionPrompt::expand(&native, vec![V41Image::from_rgb(4, 4, &[0; 48])?], 1024)?;
        let white =
            V41VisionPrompt::expand(&native, vec![V41Image::from_rgb(1, 1, &[255; 3])?], 1024)?;
        assert_eq!(black.tokens, same.tokens);
        assert_eq!(black.tokens, white.tokens);
        let first = space.prepare(&black.tokens, &black.images)?;
        let repeat = space.prepare(&same.tokens, &same.images)?;
        let changed = space.prepare(&white.tokens, &white.images)?;
        assert_eq!(first.encode(&black.tokens)?, repeat.encode(&same.tokens)?);
        assert_ne!(first.encode(&black.tokens)?, changed.encode(&white.tokens)?);
        assert_eq!(
            first.bindings[0].end - first.bindings[0].start,
            black.images[0].image.grid().tokens()
        );
        Ok(())
    }

    #[test]
    fn images_match_content_and_geometry_identity_without_changing_token_positions() -> Result<()> {
        let mut space = ImageKeySpace::default();
        let tokens = prompt(256, 2);
        let first = space.prepare_spans(&tokens, &[(256, 10, [1; 32]), (266, 10, [2; 32])])?;
        let repeat = space.prepare_spans(&tokens, &[(256, 10, [1; 32]), (266, 10, [2; 32])])?;
        let swapped = space.prepare_spans(&tokens, &[(256, 10, [2; 32]), (266, 10, [1; 32])])?;
        let changed_second =
            space.prepare_spans(&tokens, &[(256, 10, [1; 32]), (266, 10, [3; 32])])?;
        let a = first.encode(&tokens)?;
        assert_eq!(a.len(), tokens.len());
        assert_eq!(a, repeat.encode(&tokens)?);
        assert_ne!(a, swapped.encode(&tokens)?);
        assert_eq!(&a[..256], &tokens[..256]);
        assert_eq!(&a[276..], &tokens[276..]);
        assert!(a[256..276].iter().all(|&k| k >= FIRST_IMAGE_KEY as u32));
        let mut radix = Radix::new(4);
        radix.insert(&a, first.clone());
        assert_eq!(
            radix.lookup_reusable(&repeat.encode(&tokens)?).unwrap().0,
            tokens.len()
        );
        assert_eq!(
            radix.lookup_reusable(&swapped.encode(&tokens)?).unwrap().0,
            256
        );
        assert_eq!(
            radix
                .lookup_reusable(&changed_second.encode(&tokens)?)
                .unwrap()
                .0,
            266
        );
        // A partial matching image and a completed-turn suffix preserve positions.
        assert_eq!(
            radix
                .lookup_reusable(&repeat.encode(&tokens[..261])?)
                .unwrap()
                .0,
            261
        );
        let mut generated = tokens.clone();
        generated.extend([9, 10, 11]);
        assert_eq!(&repeat.encode(&generated)?[tokens.len()..], &[9, 10, 11]);
        assert_eq!(tokens[256], V41_IMAGE_TOKEN_ID);
        Ok(())
    }
    #[test]
    fn active_requests_and_both_retention_banks_pin_keys_until_edges_expire() -> Result<()> {
        let mut space = ImageKeySpace::default();
        let tokens = prompt(256, 1);
        let original = space.prepare_spans(&tokens, &[(256, 10, [1; 32])])?;
        let original_key = original.encode(&tokens)?[256];
        let mut retained = Retention::new(1);
        retained
            .bank_mut(SnapshotKind::Prompt)
            .insert(&original.encode(&tokens)?, original.clone());
        retained
            .bank_mut(SnapshotKind::Turn)
            .insert(&original.encode(&tokens)?, original.clone());
        assert!(retained.evict_one());
        drop(original);
        let other = space.prepare_spans(&tokens, &[(256, 10, [2; 32])])?;
        assert_ne!(other.encode(&tokens)?[256], original_key);
        assert!(retained.evict_one());
        let replacement = space.prepare_spans(&tokens, &[(256, 10, [3; 32])])?;
        assert_eq!(replacement.encode(&tokens)?[256], original_key);
        assert!(retained.bank(SnapshotKind::Prompt).is_empty());
        assert!(retained.bank(SnapshotKind::Turn).is_empty());
        // An active owner still pins a key after its retained snapshots disappear.
        let live_key = replacement.encode(&tokens)?[256];
        retained
            .bank_mut(SnapshotKind::Turn)
            .insert(&replacement.encode(&tokens)?, replacement.clone());
        retained.evict_one();
        let fresh = space.prepare_spans(&tokens, &[(256, 10, [4; 32])])?;
        assert_ne!(fresh.encode(&tokens)?[256], live_key);
        assert_eq!(space.identities.len(), 3);
        Ok(())
    }
    #[test]
    fn recycled_symbols_cannot_hit_a_removed_branch_below_a_shared_text_edge() -> Result<()> {
        let mut space = ImageKeySpace::default();
        let tokens = prompt(256, 1);
        let a = space.prepare_spans(&tokens, &[(256, 10, [0; 32])])?;
        let mut second_hash = [0; 32];
        second_hash[31] = 1;
        let b = space.prepare_spans(&tokens, &[(256, 10, second_hash)])?;
        let old_key = a.encode(&tokens)?[256];
        assert_ne!(old_key, b.encode(&tokens)?[256]);
        let mut radix = Radix::new(2);
        radix.insert(&a.encode(&tokens)?, a.clone());
        radix.insert(&b.encode(&tokens)?, b.clone());
        assert!(radix.evict_one());
        drop(a);
        let c = space.prepare_spans(&tokens, &[(256, 10, [3; 32])])?;
        assert_eq!(c.encode(&tokens)?[256], old_key);
        assert_eq!(radix.lookup_reusable(&c.encode(&tokens)?).unwrap().0, 256);
        radix.insert(&c.encode(&tokens)?, c.clone());
        assert_eq!(
            radix.lookup_reusable(&c.encode(&tokens)?).unwrap().0,
            tokens.len()
        );
        let original = space.prepare_spans(&tokens, &[(256, 10, [0; 32])])?;
        assert_eq!(
            radix.lookup_reusable(&original.encode(&tokens)?).unwrap().0,
            256
        );
        assert_eq!(
            radix.lookup_reusable(&b.encode(&tokens)?).unwrap().0,
            tokens.len()
        );
        Ok(())
    }

    #[test]
    fn spans_are_bounded_and_failed_or_expired_admissions_reclaim_keys() -> Result<()> {
        let mut space = ImageKeySpace::default();
        let tokens = prompt(256, 16);
        let spans: Vec<_> = (0..16).map(|i| (256 + i * 10, 10, [i as u8; 32])).collect();
        let keys = space.prepare_spans(&tokens, &spans)?;
        assert_eq!(keys.bindings.len(), 16);
        assert_eq!(keys.through(256).bindings.len(), 0);
        assert_eq!(keys.through(257).bindings.len(), 1);
        assert_eq!(keys.through(266).bindings.len(), 1);
        assert!(space
            .prepare_spans(&tokens, &[(256, 10, [0; 32]); 17])
            .is_err());
        for bad in [
            vec![(usize::MAX, 10, [0; 32])],
            vec![(256, 0, [0; 32])],
            vec![(0, 10, [0; 32])],
            vec![(256, 1025, [0; 32])],
            vec![(256, 10, [0; 32]), (260, 10, [1; 32])],
        ] {
            assert!(space.prepare_spans(&tokens, &bad).is_err());
        }
        let mut corrupt = tokens.clone();
        corrupt[256] = 42;
        assert!(keys.encode(&corrupt).is_err());
        drop(keys);
        for i in 0..1000u32 {
            let mut hash = [0; 32];
            hash[..4].copy_from_slice(&i.to_le_bytes());
            let keys = space.prepare_spans(&tokens, &[(256, 10, hash)])?;
            assert_eq!(space.identities.len(), 1);
            assert_eq!(keys.bindings.len(), 1);
        }
        assert_eq!(space.next, FIRST_IMAGE_KEY + 16);
        let mut exhausted = ImageKeySpace {
            next: u32::MAX as u64,
            ..Default::default()
        };
        let final_key = exhausted.prepare_spans(&tokens, &[(256, 10, [1; 32])])?;
        assert!(exhausted
            .prepare_spans(&tokens, &[(256, 10, [2; 32])])
            .is_err());
        drop(final_key);
        assert!(exhausted
            .prepare_spans(&tokens, &[(256, 10, [2; 32])])
            .is_ok());
        Ok(())
    }
}
