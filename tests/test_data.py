"""Protocol invariants; synthetic examples here are NOT experimental results."""
import tempfile
from pathlib import Path
import unittest

import numpy as np

from c04.common import file_sha256, read_json, write_json, write_jsonl
from c04.noise import align_embeddings, inject
from c04.prepare import nested_sample, prepare, verify_source_checksum


def fixture(classes=3, per_class=20):
    return [{"id": f"train:{y*per_class+i:05d}", "text": f"class {y} text {i}", "label": y}
            for y in range(classes) for i in range(per_class)]


class DataProtocolTests(unittest.TestCase):
    def test_fixed_checksum_rejects_modified_source(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d)/"data_full.json"
            source.write_bytes(b'{"train": []}\n')
            expected = {"sha256": file_sha256(source)}
            self.assertEqual(verify_source_checksum(source, expected), expected["sha256"])
            source.write_bytes(b'{"train": [1]}\n')
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                verify_source_checksum(source, expected)

    def test_json_manifests_use_lf(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"manifest.json"
            write_json(p, {"value": [1, 2]})
            self.assertNotIn(b"\r\n", p.read_bytes())
            self.assertEqual(read_json(p), {"value": [1, 2]})

    def test_local_import_records_source_and_label_mapping(self):
        # Small three-class fixture with the official per-class split sizes.
        raw = {split: [[f"{split} class {y} text {i}", f"label_{y}"]
                       for y in range(3) for i in range(n)]
               for split, n in [("train", 100), ("val", 20), ("test", 30)]}
        raw.update({split: [[f"{split} text {i}", "oos"] for i in range(2)]
                    for split in ["oos_train", "oos_val", "oos_test"]})
        with tempfile.TemporaryDirectory() as d:
            source, output = Path(d)/"external.json", Path(d)/"processed"
            write_json(source, raw)
            protocol = {"dataset": {"classes": 3, "sha256": file_sha256(source),
                                     "split_counts": {s: len(rows) for s, rows in raw.items()}},
                        "seeds": [17], "shots": [5, 10, 20, 100]}
            result = prepare(source, output, protocol)
            self.assertEqual(Path(result["source_path"]), source.resolve())
            self.assertEqual(result["source_sha256"], file_sha256(source))
            self.assertEqual(result["labels_sha256"], file_sha256(output/"labels.json"))
            self.assertEqual(len(result["files"]), 10)

    def test_nested_sampling_and_seed_variation(self):
        rows = fixture()
        ids = lambda xs: {r["id"] for r in xs}
        a, b, c = [nested_sample(rows, k, 17) for k in (5, 10, 20)]
        self.assertTrue(ids(a) < ids(b) < ids(c))
        self.assertNotEqual(ids(a), ids(nested_sample(rows, 5, 42)))
        self.assertEqual(a, nested_sample(list(reversed(rows)), 5, 17))

    def test_same_seed_same_bytes_and_no_truth_in_training(self):
        a, audit, _ = inject(fixture(), 3, .3, 17)
        b, _, _ = inject(fixture(), 3, .3, 17)
        with tempfile.TemporaryDirectory() as d:
            p, q = Path(d)/"a.jsonl", Path(d)/"b.jsonl"
            write_jsonl(p, a)
            write_jsonl(q, b)
            self.assertEqual(p.read_bytes(), q.read_bytes())
        self.assertTrue(all(set(r) == {"id", "text", "label"} for r in a))
        self.assertTrue(any(r["flipped"] for r in audit))

    def test_exact_fraction_no_self_flip(self):
        for rate in (0, .1, .2, .3, 1):
            _, audit, diag = inject(fixture(), 3, rate, 42)
            self.assertEqual(sum(r["clean_label"] != r["observed_label"] for r in audit), round(60*rate))
            self.assertEqual(sum(map(sum, diag["transition_counts"])), 60)

    def test_symmetric_targets_are_not_self_and_cover_other_classes(self):
        _, audit, _ = inject(fixture(per_class=500), 3, 1, 42)
        self.assertTrue(all(r["clean_label"] != r["observed_label"] for r in audit))
        for y in range(3):
            targets = [r["observed_label"] for r in audit if r["clean_label"] == y]
            self.assertEqual(len(set(targets)), 2)
            self.assertLess(abs(targets.count((y+1)%3)-250), 60)

    def test_nested_corruption_masks(self):
        sets = []
        for rate in (.1, .2, .3):
            _, audit, _ = inject(fixture(), 3, rate, 17)
            sets.append({r["id"] for r in audit if r["flipped"]})
        self.assertTrue(sets[0] < sets[1] < sets[2])

    def test_reject_eval_oos_and_bad_input(self):
        for prefix in ("val:", "test:", "oos_train:", "oos_test:"):
            rows = fixture()
            rows[0]["id"] = prefix + "00000"
            with self.assertRaises(ValueError):
                inject(rows, 3, .3, 17)
        with self.assertRaises(ValueError):
            inject(fixture(), 3, 1.1, 17)
        with self.assertRaises(ValueError):
            inject(fixture(), 3, .3, 17, temperature=float("nan"))
        rows = fixture()
        rows[0]["label"] = -1
        with self.assertRaises(ValueError):
            inject(rows, 3, .1, 17)

    def test_instance_noise_requires_real_vectors(self):
        with self.assertRaises(ValueError):
            inject(fixture(), 3, .3, 17, "instance")

    def test_instance_noise_favors_semantic_neighbors(self):
        # Only validates the generator on synthetic geometry, not on CLINC150.
        rows = fixture()
        x = np.repeat(np.array([[1.,0.], [.99,.1], [-1.,0.]]), 20, axis=0)
        a, audit, diag = inject(rows, 3, 1., 17, "instance", x, .05)
        b, _, _ = inject(rows, 3, 1., 17, "instance", x, .05)
        self.assertEqual(a, b)
        close = [r for r in audit if r["clean_label"] in (0,1)]
        self.assertTrue(all(r["observed_label"] == 1-r["clean_label"] for r in close))
        self.assertEqual(diag["changed"], 60)

    def test_embedding_alignment_is_by_id(self):
        rows = fixture(per_class=2)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"vectors.npz"
            ids = [r["id"] for r in reversed(rows)]
            vectors = np.array([[i+1., 1.] for i in range(len(rows))])
            np.savez(p, ids=np.array(ids), embeddings=vectors,
                     encoder_id=np.array("synthetic-test-only"), encoder_revision=np.array("test-only"))
            aligned, provenance = align_embeddings(p, rows)
            np.testing.assert_array_equal(aligned, vectors[::-1])
            self.assertEqual(provenance["encoder_id"], "synthetic-test-only")

    def test_input_is_not_mutated(self):
        rows = fixture()
        before = [dict(r) for r in rows]
        inject(rows, 3, .3, 17)
        self.assertEqual(rows, before)


if __name__ == "__main__":
    unittest.main()
