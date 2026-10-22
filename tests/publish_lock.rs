use lsm_tree::{
    get_tmp_folder, AbstractTree, AnyTree, Config, KvSeparationOptions, PublishLock, SeqNo,
    SequenceNumberCounter,
};
use std::{
    sync::{
        atomic::{AtomicUsize, Ordering::Relaxed},
        Arc, Mutex,
    },
    time::Duration,
};
use test_log::test;

/// Stands in for a database's journal lock
#[derive(Default)]
struct TestLock {
    mutex: Mutex<()>,
    calls: AtomicUsize,
}

impl PublishLock for TestLock {
    fn with_lock(&self, f: &mut dyn FnMut()) {
        let _guard = self.mutex.lock().expect("lock is poisoned");
        self.calls.fetch_add(1, Relaxed);
        f();
    }
}

struct Setup {
    _folder: tempfile::TempDir,
    tree: AnyTree,
    seqno: SequenceNumberCounter,
    visible_seqno: SequenceNumberCounter,
    lock: Arc<TestLock>,
}

impl Setup {
    fn new(kv_separation: bool) -> lsm_tree::Result<Self> {
        let folder = get_tmp_folder();
        let seqno = SequenceNumberCounter::default();
        let visible_seqno = SequenceNumberCounter::default();
        let lock = Arc::new(TestLock::default());

        let tree = Config::new(folder.path(), seqno.clone(), visible_seqno.clone())
            .with_kv_separation(
                kv_separation.then(|| KvSeparationOptions::default().separation_threshold(1)),
            )
            .with_publish_lock(Some(lock.clone()))
            .open()?;

        Ok(Self {
            _folder: folder,
            tree,
            seqno,
            visible_seqno,
            lock,
        })
    }

    /// Writes like a database would: under the publish lock
    fn write(&self, key: &str) {
        let _guard = self.lock.mutex.lock().expect("lock is poisoned");
        let seqno = self.seqno.next();
        self.tree.insert(key, key, seqno);
        self.visible_seqno.fetch_max(seqno + 1);
    }

    /// Holds the publish lock while `op` runs in another thread, and checks
    /// that `op` cannot make anything visible until the lock is released.
    fn assert_publishes_under_lock(
        &self,
        op: impl FnOnce(&AnyTree) -> lsm_tree::Result<()> + Send,
    ) -> lsm_tree::Result<()> {
        let guard = self.lock.mutex.lock().expect("lock is poisoned");
        let visible_before = self.visible_seqno.get();

        std::thread::scope(|scope| -> lsm_tree::Result<()> {
            let handle = scope.spawn(|| op(&self.tree));

            std::thread::sleep(Duration::from_millis(250));

            assert!(
                !handle.is_finished(),
                "operation should wait for the publish lock",
            );
            assert_eq!(visible_before, self.visible_seqno.get());

            drop(guard);

            handle.join().expect("should join")
        })?;

        assert!(self.visible_seqno.get() > visible_before);
        assert_eq!(
            self.seqno.get(),
            self.visible_seqno.get(),
            "the new version should be visible",
        );

        Ok(())
    }
}

#[test]
fn publish_lock_flush() -> lsm_tree::Result<()> {
    for kv_separation in [false, true] {
        let setup = Setup::new(kv_separation)?;
        setup.write("a");

        setup.assert_publishes_under_lock(|tree| tree.flush_active_memtable(0))?;

        assert_eq!(1, setup.tree.table_count());
        assert_eq!(1, setup.lock.calls.load(Relaxed));
    }

    Ok(())
}

#[test]
fn publish_lock_compaction() -> lsm_tree::Result<()> {
    for kv_separation in [false, true] {
        let setup = Setup::new(kv_separation)?;

        for key in ["a", "b"] {
            setup.write(key);
            setup.tree.flush_active_memtable(0)?;
        }
        assert_eq!(2, setup.tree.table_count());

        let calls = setup.lock.calls.load(Relaxed);

        setup.assert_publishes_under_lock(|tree| tree.major_compact(u64::MAX, SeqNo::MAX))?;

        assert_eq!(1, setup.tree.table_count());
        assert_eq!(calls + 1, setup.lock.calls.load(Relaxed));
    }

    Ok(())
}

#[test]
fn publish_lock_ingestion() -> lsm_tree::Result<()> {
    for kv_separation in [false, true] {
        let setup = Setup::new(kv_separation)?;

        // NOTE: The ingestion flushes the active memtable while it holds the lock,
        // that flush must not take the lock again
        setup.write("a");

        setup.assert_publishes_under_lock(|tree| {
            let mut ingestion = tree.ingestion()?;
            ingestion.write("b", "b")?;
            ingestion.write("c", "c")?;
            ingestion.finish()
        })?;

        assert_eq!(1, setup.lock.calls.load(Relaxed));

        let visible = setup.visible_seqno.get();
        for key in ["a", "b", "c"] {
            assert!(setup.tree.contains_key(key, visible)?);
        }
    }

    Ok(())
}

#[test]
fn publish_lock_not_taken_by_writes_rotate_and_clear() -> lsm_tree::Result<()> {
    for kv_separation in [false, true] {
        let setup = Setup::new(kv_separation)?;

        // NOTE: These are called with the publish lock held, like the writes of a database
        let _guard = setup.lock.mutex.lock().expect("lock is poisoned");

        setup.tree.insert("a", "a", setup.seqno.next());
        setup.tree.remove("b", setup.seqno.next());
        setup.tree.rotate_memtable();

        setup.tree.clear()?;
        assert_eq!(setup.seqno.get(), setup.visible_seqno.get());
        assert!(setup.tree.is_empty(setup.visible_seqno.get(), None)?);

        assert_eq!(0, setup.lock.calls.load(Relaxed));
    }

    Ok(())
}

#[test]
fn publish_lock_no_version_change_does_not_lock() -> lsm_tree::Result<()> {
    let setup = Setup::new(false)?;

    // NOTE: Nothing to flush or compact
    setup.tree.flush_active_memtable(0)?;
    setup
        .tree
        .compact(Arc::new(lsm_tree::compaction::Leveled::default()), 0)?;

    assert_eq!(0, setup.lock.calls.load(Relaxed));

    Ok(())
}
