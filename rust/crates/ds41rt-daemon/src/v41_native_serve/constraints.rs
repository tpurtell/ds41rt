use super::scores::{BatchScores, VOCAB};
use anyhow::{ensure, Result};
use ds41rt_api::native_v41::{NativeConstraint, NativeFailure};
use ds41rt_ffi::{NativeLibrary, Ds41rtXGrammarCompiler, Ds41rtXGrammarGrammar,
    Ds41rtXGrammarMatcher, DS41RT_XGRAMMAR_STRUCTURAL_TAG};
use std::{collections::{HashMap, VecDeque}, path::PathBuf, sync::Arc};

pub(super) struct Compiler<'a> {
    library: &'a NativeLibrary,
    tokenizer: PathBuf,
    compiler: Option<Ds41rtXGrammarCompiler<'a>>,
    grammars: HashMap<NativeConstraint, Arc<Ds41rtXGrammarGrammar<'a>>>,
    order: VecDeque<NativeConstraint>,
}
impl<'a> Compiler<'a> {
    pub fn new(library: &'a NativeLibrary, tokenizer: PathBuf) -> Self {
        Self { library, tokenizer, compiler: None, grammars: HashMap::new(), order: VecDeque::new() }
    }
    pub fn matcher(&mut self, spec: &NativeConstraint) -> Result<State<'a>> {
        if self.compiler.is_none() {
            self.compiler = Some(self.library.xgrammar_compiler(&self.tokenizer, VOCAB, &[1])?);
        }
        let grammar = if let Some(grammar) = self.grammars.get(spec) { grammar.clone() } else {
            let grammar = Arc::new(self.compiler.as_ref().unwrap().compile(
                DS41RT_XGRAMMAR_STRUCTURAL_TAG, Some(&spec.0), true)
                .map_err(|error| NativeFailure::BadRequest(format!("{error:#}")))?);
            if self.grammars.len() == 64 {
                if let Some(old) = self.order.pop_front() { self.grammars.remove(&old); }
            }
            self.grammars.insert(spec.clone(), grammar.clone());
            grammar
        };
        self.order.retain(|key| key != spec);
        self.order.push_back(spec.clone());
        Ok(State { matcher: grammar.matcher()?, mask: vec![0; VOCAB.div_ceil(32)] })
    }
}

/// Collect one mask per row, honouring each row's `needs_mask`.
///
/// `fill` reuses a single word buffer across rows, exactly like the XGrammar
/// matcher's own mask, so a row that needs no mask must be recorded as `None`
/// rather than inheriting the previous row's bits. Keeping that decision in one
/// place is what makes it testable without a compiled grammar.
fn collect_row_masks<F>(rows: usize, words: usize,
    mut fill: F,
) -> Result<Vec<Option<Vec<u32>>>>
where
    F: FnMut(usize, &mut [u32]) -> Result<bool>,
{
    let mut buffer = vec![0u32; words];
    let mut collected = Vec::with_capacity(rows);
    for index in 0..rows {
        let needs_mask = fill(index, &mut buffer)?;
        collected.push(needs_mask.then(|| buffer.clone()));
    }
    Ok(collected)
}

pub(super) struct State<'a> {
    matcher: Ds41rtXGrammarMatcher<'a>,
    mask: Vec<u32>,
}
impl State<'_> {
    pub fn mask(&mut self) -> Result<Option<&[u32]>> {
        Ok(if self.matcher.fill_bitmask(&mut self.mask)? { Some(&self.mask) } else { None })
    }
    pub fn accept(&mut self, token: u32) -> Result<()> {
        ensure!(self.matcher.accept_token(token)?, "emitted token violates request grammar");
        Ok(())
    }
    pub fn truncate_proposal(&self, input: &mut Vec<u32>) -> Result<()> {
        ensure!(!input.is_empty(), "grammar proposal has no emitted anchor");
        let mut branch = self.matcher.fork()?;
        if branch.is_completed()? { input.truncate(1); return Ok(()); }
        for index in 1..input.len() {
            if !branch.accept_token(input[index])? { input.truncate(index); break; }
            if branch.is_completed()? { input.truncate(index + 1); break; }
        }
        Ok(())
    }
    /// The packed grammar mask of **one** row of a verification round, or
    /// `None` when the grammar allows every token at that row.
    ///
    /// Rows up to and including `row` are accepted into a private fork. Used to
    /// validate a device-selected masked row at retention time, where only the
    /// accepted frontier's mask is needed.
    pub fn prepare_verification_mask_row(&self, input: &[u32],
        row: usize,
    ) -> Result<Option<Vec<u32>>> {
        ensure!(row < input.len(), "grammar mask row is outside the verification round");
        let mut branch = self.matcher.fork()?;
        let mut mask = vec![0u32; self.mask.len()];
        let mut needs_mask = false;
        for index in 0..=row {
            if index > 0 {
                ensure!(branch.accept_token(input[index])?, "illegal verification draft token");
            }
            needs_mask = branch.fill_bitmask(&mut mask)?;
        }
        Ok(needs_mask.then_some(mask))
    }
    /// The packed grammar mask of every row of a verification round, in row
    /// order, or `None` for a row that needs no mask.
    ///
    /// Row 0 is the mask *before* any draft is accepted; row `index > 0` first
    /// accepts `input[index]` into a private fork. That is exactly the
    /// hypothetical-prefix mask `select_verification` uses, so the device reader
    /// produces the same masked argmax as the CPU.
    ///
    /// A row whose grammar allows every token must report `None`, never the
    /// previous row's bits: the matcher's mask buffer is reused across rows, so
    /// dropping `fill_bitmask`'s return value would silently apply a stale
    /// grammar to that row.
    pub fn prepare_verification_masks(&self, input: &[u32]) -> Result<Vec<Option<Vec<u32>>>> {
        ensure!(!input.is_empty(), "grammar mask request has no emitted anchor");
        let mut branch = self.matcher.fork()?;
        collect_row_masks(input.len(), self.mask.len(), |index, mask| {
            // The authoritative state already contains input[0], the emitted
            // anchor. Each later row follows the preceding legal draft token.
            if index > 0 {
                ensure!(branch.accept_token(input[index])?, "illegal verification draft token");
            }
            branch.fill_bitmask(mask)
        })
    }
    pub fn select_verification(&self, scores: &BatchScores, offset: usize, input: &[u32]) -> Result<Vec<u32>> {
        let mut branch = self.matcher.fork()?;
        let mut mask = vec![0; self.mask.len()];
        let mut next = Vec::with_capacity(input.len());
        for index in 0..input.len() {
            // The authoritative state already contains input[0], the emitted
            // anchor. Each later row follows the preceding legal draft token.
            if index > 0 { ensure!(branch.accept_token(input[index])?, "illegal verification draft token"); }
            let needs_mask = branch.fill_bitmask(&mut mask)?;
            next.push(scores.select(offset + index, needs_mask.then_some(mask.as_slice()))?);
        }
        Ok(next)
    }
    /// Stochastic twin of [`Self::select_verification`]. The grammar mask is
    /// applied first along the hypothetical draft prefix; the exact sampler then
    /// draws from the masked full-vocabulary row at the absolute emitted-token
    /// position `base_position + index`. The resulting tokens feed the same
    /// sample-and-match verifier, so an accepted draft never biases the target
    /// distribution.
    pub fn select_verification_sampled(
        &self,
        scores: &BatchScores,
        offset: usize,
        input: &[u32],
        params: ds41rt_core::TargetSamplingParams,
        base_position: u64,
    ) -> Result<Vec<u32>> {
        let mut branch = self.matcher.fork()?;
        let mut mask = vec![0; self.mask.len()];
        let mut next = Vec::with_capacity(input.len());
        for index in 0..input.len() {
            if index > 0 { ensure!(branch.accept_token(input[index])?, "illegal verification draft token"); }
            let needs_mask = branch.fill_bitmask(&mut mask)?;
            next.push(scores.sample(
                offset + index,
                needs_mask.then_some(mask.as_slice()),
                params,
                base_position + index as u64,
            )?);
        }
        Ok(next)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The XGrammar matcher's mask buffer is reused across rows and
    /// `fill_bitmask` returns whether that row needs a mask at all. A row whose
    /// grammar allows every token must be recorded as `None`; recording the
    /// buffer anyway would apply the **previous** row's grammar to it, which is
    /// a wrong-answer path (the device would either report a spurious
    /// `grammar allows no target token` or silently pick a filtered token).
    ///
    /// The closure below deliberately leaves the reused buffer untouched when
    /// no mask is needed, exactly as the matcher does.
    #[test]
    fn a_row_that_needs_no_mask_does_not_inherit_the_previous_rows_bits() {
        let words = 4;
        // Masked, unmasked, masked, unmasked: reuse is exercised both ways.
        let needs = [true, false, true, false];
        let collected = collect_row_masks(needs.len(), words, |index, mask| {
            if needs[index] {
                for word in mask.iter_mut() {
                    *word = 0xAA00 + index as u32;
                }
            }
            Ok(needs[index])
        })
        .unwrap();
        assert_eq!(collected.len(), 4);
        assert_eq!(collected[0].as_deref().unwrap()[0], 0xAA00);
        assert!(collected[1].is_none(), "row 1 must not inherit row 0's bits");
        assert_eq!(collected[2].as_deref().unwrap()[0], 0xAA02);
        assert!(collected[3].is_none(), "row 3 must not inherit row 2's bits");

        // And the reverse: an unmasked first row, then masked rows.
        let needs = [false, true, true];
        let collected = collect_row_masks(needs.len(), words, |index, mask| {
            if needs[index] {
                for word in mask.iter_mut() {
                    *word = 0xBB00 + index as u32;
                }
            }
            Ok(needs[index])
        })
        .unwrap();
        assert!(collected[0].is_none(), "an unmasked first row stays None");
        assert_eq!(collected[1].as_deref().unwrap()[0], 0xBB01);
        assert_eq!(collected[2].as_deref().unwrap()[0], 0xBB02);
    }

    /// Every row of an all-unmasked or all-masked sequence keeps its own answer.
    #[test]
    fn collect_row_masks_preserves_each_rows_decision() {
        let all_unmasked = collect_row_masks(5, 4, |_, _| Ok(false)).unwrap();
        assert!(all_unmasked.iter().all(Option::is_none));
        let all_masked = collect_row_masks(5, 4, |index, mask| {
            mask[0] = index as u32;
            Ok(true)
        })
        .unwrap();
        assert_eq!(
            all_masked.iter().map(|mask| mask.as_deref().unwrap()[0]).collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4]
        );
    }

    /// Chunk 4b: the **real** xgrammar matcher, forked per speculative prefix row.
    ///
    /// This is the one test that exercises `prepare_verification_masks` and
    /// `prepare_verification_mask_row` against a compiled grammar (the other tests
    /// drive `collect_row_masks` with a synthetic fill). It pins the three chunk-4b
    /// constraints on the mask-preparation half of the flow:
    /// per-row masks follow the hypothetical prefix, the single-row retention
    /// variant agrees with the batch form, and the authoritative matcher is never
    /// advanced (fork/rollback: the branches are dropped, so the committed state
    /// still governs the next authoritative `accept`).
    ///
    /// Skips without `DS41RT_NATIVE_LIB`; uses the committed tiny tokenizer
    /// (vocab 8) so the test does not need a model snapshot.
    #[test]
    fn verification_masks_fork_per_row_without_mutating_the_authoritative_matcher() {
        let Some(path) = std::env::var_os("DS41RT_NATIVE_LIB") else {
            eprintln!("skipping: DS41RT_NATIVE_LIB is not set");
            return;
        };
        let library = unsafe { NativeLibrary::load(path).unwrap() };
        let tokenizer = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../native/tests/fixtures/xgrammar_tiny_tokenizer.json");
        let compiler = library.xgrammar_compiler(&tokenizer, 8, &[6]).unwrap();
        let schema = r#"{"type":"object","properties":{"x":{"type":"string"}},"required":["x"],"additionalProperties":false}"#;
        let grammar = compiler
            .compile(ds41rt_ffi::DS41RT_XGRAMMAR_JSON_SCHEMA, Some(schema), true)
            .unwrap();
        let mut matcher = grammar.matcher().unwrap();
        // The committed anchor `{` (token 1) is already in the authoritative state.
        assert!(matcher.accept_token(1).unwrap(), "the anchor must be an allowed token");
        let mut state = State { matcher, mask: vec![0u32; 1] };
        let before = state.mask().unwrap().map(<[u32]>::to_vec);

        // Hypothetical draft prefix: anchor, space, "x", colon.
        let input = [1u32, 7, 2, 3];
        let masks = state.prepare_verification_masks(&input).unwrap();
        assert_eq!(masks.len(), input.len());
        assert!(masks.iter().all(Option::is_some), "this grammar needs a mask on every row");
        // Per-row masks follow the prefix: after `{ "x"` the allowed set is no
        // longer the anchor's.
        assert_ne!(masks[0], masks[2], "row 2 must follow the x-token prefix, not the anchor");
        assert_ne!(masks[0], masks[3], "row 3 must follow the colon prefix");
        // The retention-time single-row form agrees with the batch form.
        for (row, mask) in masks.iter().enumerate() {
            let single = state.prepare_verification_mask_row(&input, row).unwrap();
            assert_eq!(mask, &single, "row {row}: single-row mask differs from the batch form");
        }
        // Fork/rollback: preparing (and dropping) every branch left the
        // authoritative state exactly where it was.
        let after = state.mask().unwrap().map(<[u32]>::to_vec);
        assert_eq!(before, after, "verification mask preparation mutated the authoritative matcher");

        // An illegal draft token is the same hard error the CPU path reports.
        let illegal = state.prepare_verification_masks(&[1, 5]).unwrap_err();
        assert!(illegal.to_string().contains("illegal verification draft token"),
            "unexpected error: {illegal}");
        // A row outside the round is rejected rather than silently clamped.
        assert!(state.prepare_verification_mask_row(&input, input.len()).is_err());
    }
}
