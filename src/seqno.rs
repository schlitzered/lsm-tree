// Copyright (c) 2024-present, fjall-rs
// This source code is licensed under both the Apache 2.0 and MIT License
// (found in the LICENSE-* files in the repository)

use crate::SeqNo;
use std::sync::{
    atomic::{
        AtomicU64,
        Ordering::{AcqRel, Acquire, Release},
    },
    Arc,
};

/// Thread-safe sequence number generator
///
/// # Examples
///
/// ```
/// # use lsm_tree::{AbstractTree, Config, SequenceNumberCounter};
/// #
/// # let path = tempfile::tempdir()?;
///
/// let seqno = SequenceNumberCounter::default();
/// let visible_seqno = SequenceNumberCounter::default();
///
/// let tree = Config::new(path, seqno.clone(), visible_seqno.clone()).open()?;
///
/// // Do some inserts...
/// for k in [b"a", b"b", b"c"] {
///     let batch_seqno = seqno.next();
///     tree.insert(k, "abc", batch_seqno);
///     visible_seqno.fetch_max(batch_seqno + 1);
/// }
///
/// // Create a batch
/// let batch_seqno = seqno.next();
/// tree.remove("a".as_bytes(), batch_seqno);
/// tree.remove("b".as_bytes(), batch_seqno);
/// tree.remove("c".as_bytes(), batch_seqno);
/// visible_seqno.fetch_max(batch_seqno + 1);
/// #
/// # assert!(tree.is_empty(visible_seqno.get(), None)?);
/// # Ok::<(), lsm_tree::Error>(())
/// ```
#[derive(Clone, Default, Debug)]
pub struct SequenceNumberCounter(Arc<AtomicU64>);

impl SequenceNumberCounter {
    /// Creates a new counter, setting it to some previous value
    #[must_use]
    pub fn new(prev: SeqNo) -> Self {
        Self(Arc::new(AtomicU64::new(prev)))
    }

    /// Gets the would-be-next sequence number, without incrementing the counter.
    ///
    /// This should only be used when creating a snapshot.
    #[must_use]
    pub fn get(&self) -> SeqNo {
        self.0.load(Acquire)
    }

    /// Gets the next sequence number.
    #[must_use]
    #[allow(clippy::missing_panics_doc, reason = "we should never run out of u64s")]
    pub fn next(&self) -> SeqNo {
        let seqno = self.0.fetch_add(1, AcqRel);

        // The MSB is reserved for transactions.
        //
        // This gives us 63-bit sequence numbers technically.
        assert!(seqno < 0x8000_0000_0000_0000, "Ran out of sequence numbers");

        seqno
    }

    /// Sets the sequence number.
    pub fn set(&self, seqno: SeqNo) {
        self.0.store(seqno, Release);
    }

    /// Maximizes the sequence number.
    pub fn fetch_max(&self, seqno: SeqNo) {
        self.0.fetch_max(seqno, AcqRel);
    }
}

/// A lock that writers hold from allocating a sequence number until they have
/// published it into the visible sequence number, for example the journal lock of a database.
///
/// Every version a tree installs (after a flush, a compaction or an ingestion)
/// gets a fresh sequence number, which the tree then publishes into the visible
/// sequence number. If a writer has allocated a lower sequence number, but has not
/// yet inserted all items of its write batch, that publication would make the partially
/// applied batch visible to new snapshots.
///
/// If a [`PublishLock`] is set using [`Config::with_publish_lock`](crate::Config::with_publish_lock),
/// the tree publishes a new version only while holding the lock. An ingestion holds it
/// from flushing the active memtable until it has published the ingested tables.
///
/// # Lock order
///
/// The tree takes this lock while holding its flush lock (flush, ingestion) or its
/// major compaction lock (compaction), but none of its other locks. While holding
/// this lock, the tree takes its compaction state lock and its version history lock
/// (ingestion only).
///
/// So a writer must not wait for a flush or compaction of the same tree while holding this lock.
/// Writes, [`AbstractTree::rotate_memtable`](crate::AbstractTree::rotate_memtable)
/// and [`AbstractTree::clear`](crate::AbstractTree::clear) do not take this lock:
/// they are expected to be called with the lock held (as a writer would do anyway).
pub trait PublishLock: Send + Sync + std::panic::RefUnwindSafe {
    /// Acquires the lock, calls `f` exactly once and releases the lock.
    fn with_lock(&self, f: &mut dyn FnMut());
}

thread_local! {
    /// Set while this thread holds the publish lock through [`with_publish_lock`],
    /// so a nested call (an ingestion that flushes the tree) does not lock it twice.
    static HOLDS_PUBLISH_LOCK: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Runs `f` while holding the publish lock, if one is set.
pub fn with_publish_lock<R>(lock: Option<&dyn PublishLock>, f: impl FnOnce() -> R) -> R {
    struct Reset;

    impl Drop for Reset {
        fn drop(&mut self) {
            HOLDS_PUBLISH_LOCK.with(|x| x.set(false));
        }
    }

    let Some(lock) = lock else {
        return f();
    };

    if HOLDS_PUBLISH_LOCK.with(std::cell::Cell::get) {
        return f();
    }

    let mut f = Some(f);
    let mut result = None;

    lock.with_lock(&mut || {
        if let Some(f) = f.take() {
            HOLDS_PUBLISH_LOCK.with(|x| x.set(true));
            let _reset = Reset;
            result = Some(f());
        }
    });

    #[expect(clippy::expect_used, reason = "PublishLock::with_lock has to call f")]
    result.expect("PublishLock::with_lock did not call the function")
}

#[cfg(test)]
mod tests {
    use test_log::test;

    #[test]
    fn not_max_seqno() {
        let counter = super::SequenceNumberCounter::default();
        counter.set(0x7FFF_FFFF_FFFF_FFFF);
        let _ = counter.next();
    }

    #[test]
    #[should_panic = "Ran out of sequence numbers"]
    fn max_seqno() {
        let counter = super::SequenceNumberCounter::default();
        counter.set(0x8000_0000_0000_0000);
        let _ = counter.next();
    }
}
