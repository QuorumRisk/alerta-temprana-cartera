"""Pruebas sin acceso a red: python -m unittest discover -s tests -v."""

import unittest

import numpy as np
import pandas as pd

from run_demo import (
    DESCRIPTIONS,
    REQUIRED,
    STATUS_COLS,
    TARGET,
    choose_threshold,
    engineer_features,
    gain_table,
    select_cohort,
)


def synthetic_rows() -> pd.DataFrame:
    rows = []
    statuses = [
        [-1, 0, 2, -1, -1, -1],
        [2, 1, 0, -1, -1, -2],
        [0, 0, 0, 0, 0, 0],
    ]
    for idx, sequence in enumerate(statuses):
        row = {key: 0 for key in REQUIRED}
        row.update({"ID": idx + 1, "LIMIT_BAL": 100000, TARGET: int(idx == 0)})
        for key, value in zip(STATUS_COLS, sequence):
            row[key] = value
        for k in range(1, 7):
            row[f"BILL_AMT{k}"] = 14000 + (6 - k) * 1000
            row[f"PAY_AMT{k}"] = 1200 + (6 - k) * 50
        rows.append(row)
    return pd.DataFrame(rows)


class PortfolioDemoTests(unittest.TestCase):
    def test_preventive_cohort_excludes_existing_positive_delay(self):
        cohort, counts = select_cohort(synthetic_rows())
        self.assertEqual(len(cohort), 2)
        self.assertEqual(counts["filas_atraso_positivo_actual_excluidas"], 1)
        self.assertTrue(cohort["PAY_0"].le(0).all())

    def test_all_inputs_are_engineered(self):
        result = engineer_features(synthetic_rows())
        self.assertEqual(set(result.columns), set(DESCRIPTIONS))
        self.assertTrue(set(result.columns).isdisjoint(REQUIRED))
        self.assertTrue(np.isfinite(result.to_numpy()).all())

    def test_recency_means_months_before_september(self):
        result = engineer_features(synthetic_rows())
        self.assertEqual(result.loc[0, "recencia_ultimo_atraso"], 2)
        self.assertEqual(result.loc[2, "recencia_ultimo_atraso"], 7)

    def test_lift_first_decile_and_cumulative_capture(self):
        y = np.array([1] * 10 + [0] * 90)
        p = np.linspace(1.0, 0.0, 100)
        lift = gain_table(y, p, "precision")
        self.assertAlmostEqual(lift.iloc[0]["lift"], 10.0)
        self.assertAlmostEqual(lift.iloc[-1]["captura_acumulada"], 1.0)

    def test_oof_threshold_selection_has_explicit_constraints(self):
        y = np.array([1] * 20 + [0] * 80)
        p = np.array([0.8] * 20 + [0.1] * 80)
        for objective in ("precision", "recall", "f1"):
            chosen = choose_threshold(y, p, objective)
            self.assertGreater(chosen["contact_rate"], 0)
            self.assertLess(chosen["threshold"], 1)


if __name__ == "__main__":
    unittest.main()
