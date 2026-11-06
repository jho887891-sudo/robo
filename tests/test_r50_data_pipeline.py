import os
import unittest

from training.r50_data_pipeline import R50WorkerDataset, create_loader


class IndexedDataset:
    def __len__(self):
        return 19

    def __getitem__(self, index):
        return index, os.getpid()


class R50DataPipelineTests(unittest.TestCase):
    def test_spawn_loader_preserves_exact_order_and_uses_workers(self):
        loader = create_loader(IndexedDataset(), workers=2, prefetch=2,
                               pin_memory=False)
        try:
            for _ in range(2):
                observed = [(int(i.item()), int(pid.item())) for i, pid in loader]
                self.assertEqual([i for i, _ in observed], list(range(19)))
                self.assertTrue(all(pid != os.getpid() for _, pid in observed))
        finally:
            if loader._iterator is not None:
                loader._iterator._shutdown_workers()

    def test_reader_is_not_serialized_into_worker(self):
        dataset = R50WorkerDataset([], "/tmp/nonexistent", "/tmp/nonexistent",
                                   "/tmp/nonexistent")
        dataset._pid = os.getpid()
        dataset._reader = object()
        dataset._processor = object()
        state = dataset.__getstate__()
        self.assertIsNone(state["_reader"])
        self.assertIsNone(state["_processor"])
        self.assertIsNone(state["_pid"])
        dataset._reader = None

    def test_invalid_memory_or_prefetch_is_rejected(self):
        with self.assertRaises(ValueError):
            R50WorkerDataset([], "a", "b", "c", cache_bytes=-1)
        partially_constructed = R50WorkerDataset.__new__(R50WorkerDataset)
        partially_constructed.close()
        with self.assertRaises(ValueError):
            create_loader(IndexedDataset(), workers=2, prefetch=0,
                          pin_memory=False)


if __name__ == "__main__":
    unittest.main()
