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
