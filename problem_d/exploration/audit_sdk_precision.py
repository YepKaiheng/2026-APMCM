"""Evaluate legal-state ordering after the official Kaiwu 8-bit transform."""
import importlib.metadata
from pathlib import Path

import kaiwu as kw
import numpy as np
import pandas as pd


def qubo(cost, penalty):
    rows, cols = cost.shape
    matrix = np.zeros((rows + cols, rows + cols))
    np.fill_diagonal(matrix, -penalty)
    matrix[:rows, :rows] += np.triu(np.full((rows, rows), 2 * penalty), 1)
    matrix[rows:, rows:] += np.triu(np.full((cols, cols), 2 * penalty), 1)
    matrix[:rows, rows:] = cost
    return matrix


def legal_energies(matrix, rows):
    diag = np.diag(matrix)
    return diag[:rows, None] + diag[None, rows:] + matrix[:rows, rows:] + matrix[rows:, :rows].T


def audit():
    result_dir = Path(__file__).resolve().parents[1] / "results"
    table = pd.read_csv(result_dir / "encoding_seed26.csv")
    rows = []
    print("Kaiwu distribution", importlib.metadata.version("kaiwu"))
    for model in ["direct", "mixture"]:
        cube = table.pivot(index=["A", "B"], columns="X", values=model)
        original = cube.min(axis=1).unstack("B").to_numpy()
        low = original.min()
        for quantile in [1.0, 0.1]:
            high = np.quantile(original, quantile)
            cost = (np.minimum(original, high) - low) / (high - low)
            for penalty in [1.1, 3.0, 10.0]:
                matrix = qubo(cost, penalty)
                ising, _ = kw.core.qubo_matrix_to_ising_matrix(matrix)
                roundtrip, _ = kw.core.ising_matrix_to_qubo_matrix(ising)
                exact = legal_energies(roundtrip, original.shape[0])
                np.testing.assert_allclose(exact - exact[0, 0], cost - cost[0, 0],
                                           rtol=0, atol=1e-10)
                adjusted = kw.preprocess.adjust_ising_matrix_precision(ising, bit_width=8)
                np.testing.assert_array_equal(adjusted, np.rint(adjusted))
                assert adjusted.min() >= -128 and adjusted.max() <= 127
                rounded_qubo, _ = kw.core.ising_matrix_to_qubo_matrix(adjusted)
                energy = legal_energies(rounded_qubo, original.shape[0])
                winners = original[energy == energy.min()]
                result = dict(model=model, cap_quantile=quantile, penalty=penalty,
                              original_variables=matrix.shape[0], ising_variables=ising.shape[0],
                              ties=len(winners), retains_original_min=bool(np.isclose(winners.min(), low, atol=1e-9, rtol=0)),
                              best_original_gap=float(winners.min() - low),
                              worst_original_gap=float(winners.max() - low))
                rows.append(result)
                print(result, flush=True)
    pd.DataFrame(rows).to_csv(result_dir / "sdk_precision_audit.csv", index=False)
    print("Legal-state comparisons only. Feasibility of the unconstrained ground state and actual solver performance remain unmeasured.")


if __name__ == "__main__":
    audit()
