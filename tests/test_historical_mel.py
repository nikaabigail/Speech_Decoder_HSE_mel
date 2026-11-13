"""Check the published historical MEL metadata, without scientific data I/O."""
from pathlib import Path
import json
import unittest

ROOT = Path(__file__).resolve().parents[1]


class HistoricalMelTests(unittest.TestCase):
    def test_exact_scores_and_single_run_identity(self):
        data = json.loads((ROOT / 'benchmarks/mel_baseline_runs.json').read_text(encoding='utf-8'))
        rows = {row['participant']: row for row in data['runs']}
        self.assertEqual(set(rows), {'P1', 'P2'})
        for participant, seed, score, width in [('P1', 42, .4882435129740519, 81),
                                                ('P2', 3, .6739121756487026, 52)]:
            row = rows[participant]
            self.assertEqual((row['seed'], row['accuracy'], row['max_words_length']), (seed, score, width))
            self.assertAlmostEqual(row['accuracy_percent'], 100 * score)
            self.assertEqual(row['mode'], 'classification_hidden')
            self.assertEqual(row['augmentation'], 'none')
            self.assertFalse(row['multitask'])
            self.assertIn('sampled', row['metric'])
            self.assertEqual(len(row['source_report_sha256']), 64)
            self.assertIn('run_date', row)
            self.assertNotIn('recording_date', row)

    def test_record_splits_and_documented_metric_boundary(self):
        data = json.loads((ROOT / 'benchmarks/mel_baseline_runs.json').read_text(encoding='utf-8'))
        for row in data['runs']:
            for split in (row['head_split'], row['regression_split']):
                groups = [set(split[key]) for key in ('train_files', 'val_files', 'test_files')]
                self.assertTrue(all(groups))
                self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        text = (ROOT / 'MEL_BASELINE.md').read_text(encoding='utf-8')
        self.assertIn('test_accuracy_full', text)
        self.assertIn('not five-seed means', text)
        self.assertIn('train.py --target mel', text)


if __name__ == '__main__':
    unittest.main()
