import unittest
import numpy as np
import pandas as pd
from src.evaluation import classification_discrimination, expected_calibration_error, regression_metrics
from src.audit import fairness_slice

class TestEvaluation(unittest.TestCase):
    def test_discrimination(self):
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.1, 0.9, 0.2, 0.8])
        # Use small n_boot for speed and stability in tests
        res = classification_discrimination(y_true, y_prob, n_boot=10)
        self.assertGreater(res['auc_roc'], 0.8)
        self.assertIn('auc_pr', res)
        self.assertIn('auc_roc_ci_lo', res)

    def test_ece(self):
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.5, 0.5, 0.5, 0.5])
        ece = expected_calibration_error(y_true, y_prob)
        self.assertIsInstance(ece, float)
        self.assertEqual(ece, 0.0) # 0.5 mean prob, 0.5 frac pos -> diff 0

    def test_regression(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.9, 3.2])
        res = regression_metrics(y_true, y_pred, n_boot=10)
        self.assertIn('rmse_log', res)
        self.assertLess(res['rmse_log'], 0.3)

class TestAudit(unittest.TestCase):
    def test_fairness_slice(self):
        # Create a larger dataset to avoid single-class bootstrap samples easily
        df = pd.DataFrame({
            'y_true': [0, 1] * 50,
            'y_prob': [0.1, 0.9] * 50,
            'sex': ['M'] * 50 + ['F'] * 50
        })
        
        def auc_fn(sub_df):
            # Pass n_boot=0 to sub-calls to avoid nested bootstrap in smoke test
            return classification_discrimination(sub_df['y_true'], sub_df['y_prob'], n_boot=0)['auc_roc']
            
        res = fairness_slice(df, 'sex', auc_fn, min_n=10, n_boot=10)
        
        # Verify structure
        self.assertIsInstance(res, pd.DataFrame)
        self.assertIn('sex', res.columns)
        self.assertIn('metric', res.columns)
        self.assertIn('n', res.columns)
        
        # Verify content
        self.assertEqual(len(res), 2)
        m_row = res[res['sex'] == 'M'].iloc[0]
        self.assertEqual(m_row['n'], 50)
        self.assertGreater(m_row['metric'], 0.9)

class TestSensors(unittest.TestCase):
    def _make_splits(self):
        train = pd.DataFrame({"sale_year": [2010, 2011, 2012, 2013]})
        val   = pd.DataFrame({"sale_year": [2014, 2015]})
        oot   = pd.DataFrame({"sale_year": [2016, 2017]})
        return train, val, oot

    def test_temporal_split_ok(self):
        from src.sensors import temporal_split_validator
        train, val, oot = self._make_splits()
        temporal_split_validator(train, val, oot)  # must not raise

    def test_temporal_split_train_val_overlap(self):
        from src.sensors import temporal_split_validator
        train = pd.DataFrame({"sale_year": [2010, 2014]})
        val   = pd.DataFrame({"sale_year": [2014, 2015]})
        oot   = pd.DataFrame({"sale_year": [2016]})
        with self.assertRaises(AssertionError):
            temporal_split_validator(train, val, oot)

    def test_temporal_split_val_oot_overlap(self):
        from src.sensors import temporal_split_validator
        train = pd.DataFrame({"sale_year": [2010, 2013]})
        val   = pd.DataFrame({"sale_year": [2014, 2016]})
        oot   = pd.DataFrame({"sale_year": [2016, 2017]})
        with self.assertRaises(AssertionError):
            temporal_split_validator(train, val, oot)

    def test_encoding_leakage_check_ok(self):
        from src.sensors import encoding_leakage_check
        # Correct case: within each (year, entity) group, encoded value is constant
        df = pd.DataFrame({
            "sale_year": [2010, 2010, 2011, 2011, 2012],
            "sire":      ["A",  "A",  "A",  "B",  "A"],
            "sire_enc":  [5.0,  5.0,  5.2,  4.8,  5.3],
        })
        encoding_leakage_check(df, [("sire", "sire_enc")])  # must not raise

    def test_encoding_leakage_check_within_group_variance(self):
        from src.sensors import encoding_leakage_check
        # Two rows same (year, sire) but different encoded value → leakage
        df = pd.DataFrame({
            "sale_year": [2011, 2011],
            "sire":      ["A",  "A"],
            "sire_enc":  [5.0,  6.0],  # different within same group
        })
        with self.assertRaises(AssertionError):
            encoding_leakage_check(df, [("sire", "sire_enc")])

    def test_encoding_leakage_recomputation_ok(self):
        from src.sensors import encoding_leakage_check
        # Build a dataset where sire_enc is the correct M-estimate (m=10).
        # global_mean is computed over ALL rows by the sensor, so we must match that.
        # log_price values: 7, 8, 9 → global_mean = 8.0
        m = 10.0
        global_mean = 8.0  # (7+8+9)/3
        rows = []
        # year 2010: no prior data → enc = global_mean
        rows.append({"sale_year": 2010, "sire": "A", "log_price": 7.0,
                     "sire_enc": global_mean})
        # year 2011: 1 prior obs (sire A, log_price=7) → enc=(1*7+10*8)/11
        rows.append({"sale_year": 2011, "sire": "A", "log_price": 8.0,
                     "sire_enc": (1 * 7.0 + m * global_mean) / (1 + m)})
        # year 2012: 2 prior obs (sire A, mean=7.5) → enc=(2*7.5+10*8)/12
        rows.append({"sale_year": 2012, "sire": "A", "log_price": 9.0,
                     "sire_enc": (2 * 7.5 + m * global_mean) / (2 + m)})
        df = pd.DataFrame(rows)
        encoding_leakage_check(
            df, [("sire", "sire_enc")],
            target_col="log_price", m=m, sample_n=10, tol=1e-4,
        )

    def test_universe_consistency_ok(self):
        from src.sensors import universe_consistency_check
        univ = pd.DataFrame({"sale_year": [2022, 2022, 2023], "day": [1, 2, 1], "lot": [10, 20, 10]})
        reg  = pd.DataFrame({"sale_year": [2022, 2022],       "day": [1, 2],    "lot": [10, 20]})
        universe_consistency_check(univ, reg)  # must not raise

    def test_universe_consistency_missing_id(self):
        from src.sensors import universe_consistency_check
        univ = pd.DataFrame({"sale_year": [2022], "day": [1], "lot": [10]})
        reg  = pd.DataFrame({"sale_year": [2022, 2022], "day": [1, 2], "lot": [10, 99]})
        with self.assertRaises(AssertionError):
            universe_consistency_check(univ, reg)


if __name__ == '__main__':
    unittest.main()
