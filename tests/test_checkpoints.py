import json
import os
import tempfile
import sys
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from emotion_hierarchy.checkpoints import prepare_checkpoint, validate_checkpoint
from emotion_hierarchy.scoring import score_with_nnsight


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / "model"
        self.model.mkdir()

    def index(self, name):
        (self.model / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"weight": name}})
        )

    def test_traversal_and_absolute_shards_are_rejected(self):
        (self.root / "outside.bin").write_bytes(b"weights")
        for name in ("../outside.bin", str(self.root / "outside.bin"),
                     "shards/../../outside.bin", "C:\\outside.bin", "..\\outside.bin"):
            with self.subTest(name=name):
                self.index(name)
                with self.assertRaises(ValueError):
                    validate_checkpoint(self.model)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "requires FIFOs")
    def test_fifo_shard_and_index_are_rejected_without_opening(self):
        fifo = self.model / "pipe.bin"
        os.mkfifo(fifo)
        self.index("pipe.bin")
        with self.assertRaises(ValueError):
            validate_checkpoint(self.model)
        fifo.unlink()
        index = self.model / "model.safetensors.index.json"
        index.unlink()
        os.mkfifo(index)
        with self.assertRaises(ValueError):
            validate_checkpoint(self.model)

    def test_nested_shard_is_allowed(self):
        (self.model / "shards").mkdir()
        (self.model / "shards/weights.bin").write_bytes(b"weights")
        self.index("./shards/weights.bin")
        self.assertEqual(prepare_checkpoint(str(self.model), None), str(self.model.resolve()))

    def test_symlink_escape_is_rejected(self):
        outside = self.root / "outside.bin"
        outside.write_bytes(b"weights")
        (self.model / "weights.bin").symlink_to(outside)
        self.index("weights.bin")
        with self.assertRaises(ValueError):
            validate_checkpoint(self.model)

    def test_hub_blob_links_are_allowed_within_model_cache(self):
        snapshot = self.model / "snapshots/revision"
        snapshot.mkdir(parents=True)
        blobs = self.model / "blobs"
        blobs.mkdir()
        (blobs / "hash").write_bytes(b"weights")
        (snapshot / "weights.bin").symlink_to(blobs / "hash")
        download = Mock(return_value=str(snapshot))
        with patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(snapshot_download=download)}):
            self.assertEqual(prepare_checkpoint("owner/model", "revision"), str(snapshot))
        download.assert_called_once_with(repo_id="owner/model", revision="revision")

    def test_missing_and_malformed_shards_fail_closed(self):
        for name in ("missing.bin", None, 42, ""):
            with self.subTest(name=name):
                self.index(name)
                with self.assertRaises((ValueError, FileNotFoundError)):
                    validate_checkpoint(self.model)

    def test_remote_backend_fails_before_loading(self):
        with self.assertRaisesRegex(ValueError, "temporarily disabled"):
            score_with_nnsight(["hello"], "owner/model", remote=True)
