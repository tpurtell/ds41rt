//! Host ownership for immutable source prefixes. Active request tables own one
//! reference per page; retained prefixes release their references on eviction.
use std::{cell::RefCell, rc::Rc};

pub(super) struct PagePool {
    pub free: Vec<u32>,
    references: Vec<usize>,
    /// Bumped when a page's last reference goes, so a reused index is a new identity.
    generations: Vec<u32>,
}
impl PagePool {
    pub fn new(pages: usize) -> Self {
        Self {
            free: (0..pages as u32).rev().collect(),
            references: vec![0; pages],
            generations: vec![0; pages],
        }
    }
    pub fn capacity(&self) -> usize {
        self.references.len()
    }
    pub fn generation(&self, page: u32) -> u32 {
        self.generations[page as usize]
    }
    /// Take `count` free pages, each with one reference owned by the caller; `None` if fewer are free.
    pub fn allocate(&mut self, count: usize) -> Option<Vec<u32>> {
        if self.free.len() < count {
            return None;
        }
        let pages = self.free.split_off(self.free.len() - count);
        self.retain(&pages);
        Some(pages)
    }
    pub fn shared(&self, page: u32) -> bool {
        self.references(page) > 1
    }
    pub fn references(&self, page: u32) -> usize {
        self.references[page as usize]
    }
    pub fn retain(&mut self, pages: &[u32]) {
        for &page in pages {
            self.references[page as usize] += 1;
        }
    }
    pub fn release(&mut self, pages: &[u32]) {
        for &page in pages {
            let count = &mut self.references[page as usize];
            assert!(*count > 0, "source page released without ownership");
            *count -= 1;
            if *count == 0 {
                self.generations[page as usize] += 1;
                self.free.push(page);
            }
        }
    }
}

pub(crate) struct SourcePrefix {
    pub(super) pool: Rc<RefCell<PagePool>>,
    pub(super) pages: Vec<u32>,
    pub(super) rows: usize,
}
impl SourcePrefix {
    pub fn pages(&self) -> &[u32] {
        &self.pages
    }
    pub fn rows(&self) -> usize {
        self.rows
    }
    /// Retain a shorter initialized frontier after the original request was
    /// released. Future rows in its physical tail remain owned by the original
    /// snapshot; an appending branch must still use copy-on-write.
    pub fn truncate(&self, rows: usize) -> anyhow::Result<Self> {
        anyhow::ensure!(
            rows <= self.rows,
            "source prefix truncation exceeds initialized rows"
        );
        let pages = self.pages[..rows.div_ceil(super::PAGE_ROWS)].to_vec();
        self.pool.borrow_mut().retain(&pages);
        Ok(Self {
            pool: Rc::clone(&self.pool),
            pages,
            rows,
        })
    }
}
impl Drop for SourcePrefix {
    fn drop(&mut self) {
        self.pool.borrow_mut().release(&self.pages);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prefix_eviction_frees_only_pages_without_active_owners() {
        let pool = Rc::new(RefCell::new(PagePool::new(3)));
        let pages = {
            let mut pool = pool.borrow_mut();
            vec![pool.free.pop().unwrap(), pool.free.pop().unwrap()]
        };
        pool.borrow_mut().retain(&pages);
        pool.borrow_mut().retain(&pages);
        let prefix = SourcePrefix {
            pool: Rc::clone(&pool),
            pages: pages.clone(),
            rows: 300,
        };
        assert!(pool.borrow().shared(pages[0]));
        pool.borrow_mut().release(&pages[..1]);
        assert_eq!(pool.borrow().free.len(), 1);
        drop(prefix);
        assert_eq!(pool.borrow().free.len(), 2);
        assert!(!pool.borrow().shared(pages[1]));
        pool.borrow_mut().release(&pages[1..]);
        let mut free = pool.borrow().free.clone();
        free.sort_unstable();
        assert_eq!(free, vec![0, 1, 2]);
    }
}
