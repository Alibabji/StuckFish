use crate::chess::Move;
use std::mem;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Bound {
    Exact,
    Lower,
    Upper,
}

#[derive(Copy, Clone, Debug)]
struct Entry {
    key: u64,
    value: i32,
    depth: u8,
    used: bool,
    bound: Bound,
    best_move: Option<Move>,
}

impl Entry {
    const fn empty() -> Self {
        Self {
            key: 0,
            value: 0,
            depth: 0,
            used: false,
            bound: Bound::Exact,
            best_move: None,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct TableEntry {
    pub value: i32,
    pub depth: u8,
    pub bound: Bound,
    pub best_move: Option<Move>,
}

/// Replacement scheme that keeps the deepest stored position per bucket.
pub struct TranspositionTable {
    entries: Vec<Entry>,
    mask: usize,
}

impl TranspositionTable {
    /// Allocate a table whose size is roughly `size_in_mb`.
    pub fn new(size_in_mb: usize) -> Self {
        let bytes = size_in_mb.max(1).saturating_mul(1024 * 1024);
        let entry_bytes = mem::size_of::<Entry>().max(1);
        let mut entry_count = bytes / entry_bytes;

        if entry_count == 0 {
            entry_count = 1;
        }

        let pow_two = entry_count.next_power_of_two();

        Self {
            entries: vec![Entry::empty(); pow_two],
            mask: pow_two - 1,
        }
    }

    #[inline]
    fn index(&self, key: u64) -> usize {
        (key as usize) & self.mask
    }

    pub fn clear(&mut self) {
        for entry in &mut self.entries {
            *entry = Entry::empty();
        }
    }

    pub fn capacity(&self) -> usize {
        self.entries.len()
    }

    /// Store a search result keyed by the position hash.
    pub fn store(
        &mut self,
        key: u64,
        depth: u8,
        value: i32,
        bound: Bound,
        best_move: Option<Move>,
    ) {
        let idx = self.index(key);
        let slot = &mut self.entries[idx];

        if slot.used && slot.key == key && slot.depth > depth {
            return;
        }

        if slot.used && slot.key != key && slot.depth > depth {
            return;
        }

        *slot = Entry {
            key,
            value,
            depth,
            used: true,
            bound,
            best_move,
        };
    }

    /// Probe the table for an entry.
    pub fn probe(&self, key: u64) -> Option<TableEntry> {
        let entry = &self.entries[self.index(key)];

        if entry.used && entry.key == key {
            Some(TableEntry {
                value: entry.value,
                depth: entry.depth,
                bound: entry.bound,
                best_move: entry.best_move,
            })
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn store_and_probe_round_trip() {
        let mut tt = TranspositionTable::new(1);
        tt.store(0x1234, 5, 42, Bound::Exact, None);
        let entry = tt.probe(0x1234).expect("entry");
        assert_eq!(entry.value, 42);
        assert_eq!(entry.depth, 5);
        assert_eq!(entry.bound, Bound::Exact);
    }

    #[test]
    fn replacement_prefers_deeper_entry() {
        let mut tt = TranspositionTable::new(1);
        tt.store(0x1, 2, 7, Bound::Exact, None);
        tt.store(0x1, 5, 8, Bound::Lower, None);
        let entry = tt.probe(0x1).expect("entry");
        assert_eq!(entry.value, 8);
        assert_eq!(entry.depth, 5);
        assert_eq!(entry.bound, Bound::Lower);

        tt.store(0x1, 3, 9, Bound::Upper, None);
        let entry = tt.probe(0x1).expect("entry");
        assert_eq!(entry.value, 8);
        assert_eq!(entry.depth, 5);
    }

    #[test]
    fn clear_resets_entries() {
        let mut tt = TranspositionTable::new(1);
        tt.store(0x42, 1, 100, Bound::Exact, None);
        assert!(tt.probe(0x42).is_some());
        tt.clear();
        assert!(tt.probe(0x42).is_none());
    }
}
