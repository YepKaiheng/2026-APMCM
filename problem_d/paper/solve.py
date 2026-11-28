"""Solve 4-by-4 blocks with Kaiwu and certify every decoded ground state."""
from time import perf_counter

import kaiwu as kw
import numpy as np
import pandas as pd

from screen import OUT, load_file
from audit_sdk_precision import qubo

SIZE = 4
LEVELS = 15
PENALTY = 16


def exact_block(cost, optimizer, ids=None):
    start = perf_counter()
    cap = cost.max()
    bins = np.rint(LEVELS * cost / cap).astype(int) if cap > 0 else np.zeros_like(cost, int)
    matrix = qubo(bins, PENALTY)
    raw, _ = kw.core.qubo_matrix_to_ising_matrix(matrix)
    couplings = 8 * raw
    assert np.array_equal(couplings, np.rint(couplings))
    assert couplings.min() >= -128 and couplings.max() <= 127
    t1 = perf_counter()
    samples = optimizer.solve(couplings)
    t2 = perf_counter()
    # Fix the reference spin to +1. The SDK can return both gauges.
    bits = (samples[:, :-1] * samples[:, -1:] + 1) // 2
    unique = np.unique(bits, axis=0)
    assert (unique[:, :SIZE].sum(axis=1) == 1).all()
    assert (unique[:, SIZE:].sum(axis=1) == 1).all()
    locations = set(zip(unique[:, :SIZE].argmax(axis=1), unique[:, SIZE:].argmax(axis=1)))
    energies = -np.einsum("bi,ij,bj->b", samples, couplings, samples)
    assert (energies == energies[0]).all()

    # Independently enumerate all 256 binary assignments, including illegal ones.
    states = (np.arange(1 << (2 * SIZE))[:, None] >> np.arange(2 * SIZE)) & 1
    spins = np.column_stack([2 * states - 1, np.ones(len(states), int)])
    h = -np.einsum("bi,ij,bj->b", spins, couplings, spins)
    e = np.einsum("bi,ij,bj->b", states, matrix, states)
    np.testing.assert_array_equal(h - h[0], 8 * (e - e[0]))
    assert h.min() == energies[0] and (h == h.min()).sum() == len(locations)
    assert locations == set(zip(*np.where(bins == bins.min())))
    i, j = min(locations, key=lambda ij: (cost[ij], int(ids[ij]) if ids is not None else ij))
    assert cost[i, j] == cost.min()
    end = perf_counter()
    return dict(i=int(i), j=int(j), ties=len(locations), bin_min=int(bins.min()),
                score=cost[i, j], worst_tied=max(cost[k] for k in locations),
                ising_energy=int(energies[0]), j_min=int(couplings.min()),
                j_max=int(couplings.max()), prepare_s=t1-start, solve_s=t2-t1,
                check_decode_s=end-t2)


def run():
    data = pd.read_csv(OUT / "q23_full.csv")
    sources = {n: load_file(n) for n in ["02", "03", "04", "07", "08", "11", "12"]}
    optimizer = kw.classical.BruteForceOptimizer()
    selections, blocks, budgets = [], [], []
    for problem, prefix in [("Q2", "07"), ("Q3", "08")]:
        start = perf_counter()
        pool = data.merge(sources[prefix][["id"]], on="id", validate="one_to_one")
        assert len(pool) == len(sources[prefix])
        # Retain the ID minimizing over X within each pair (ID settles exact ties).
        best = pool.sort_values("id").loc[lambda f: f.groupby(["A", "B"]).V_hat.idxmin()]
        ids = best.pivot(index="A", columns="B", values="id").to_numpy()
        cost = best.pivot(index="A", columns="B", values="V_hat").to_numpy()
        assert cost.shape == (52, 52) and np.isfinite(cost).all() and cost.min() >= 0
        setup = perf_counter() - start
        records = []
        for a in range(0, 52, SIZE):
            for b in range(0, 52, SIZE):
                result = exact_block(cost[a:a+SIZE, b:b+SIZE], optimizer,
                                     ids[a:a+SIZE, b:b+SIZE])
                winner = ids[a + result.pop("i"), b + result.pop("j")]
                result.update(problem=problem, block_a=a//SIZE, block_b=b//SIZE, id=winner)
                records.append(result)
        t1 = perf_counter()
        chosen = min(records, key=lambda r: (r["score"], r["id"]))
        full_values = pool.V_hat.to_numpy()
        idx = np.lexsort((pool.id.to_numpy(), full_values))[0]
        assert chosen["id"] == pool.id.iloc[idx] and chosen["score"] == full_values[idx]
        verify = perf_counter() - t1
        times = []
        for _ in range(1001):
            t0 = perf_counter()
            _index = full_values.argmin()
            times.append(perf_counter() - t0)
        winner = data.loc[data.id.eq(chosen["id"])].iloc[0].to_dict()
        for key in ["02", "03", "04"]:
            entry = sources[key].loc[sources[key].id.eq(winner["id"])]
            if len(entry):
                winner.update(source=key, public_V=entry.V.item() if "V" in entry else None)
        winner.update(problem=problem, pool=prefix, ising_energy=chosen["ising_energy"],
                      bin_min=chosen["bin_min"], tile_ties=chosen["ties"],
                      runner_up=pool.nsmallest(2, "V_hat").V_hat.iloc[1],
                      enum_gap=chosen["score"]-full_values[idx])
        selections.append(winner)
        blocks.extend(records)
        frame = pd.DataFrame(records)
        budgets.append(dict(problem=problem, setup_s=setup, calls=len(records),
                            prepare_s=frame.prepare_s.sum(), solve_s=frame.solve_s.sum(),
                            decode_check_s=frame.check_decode_s.sum()+verify,
                            enum_median_us=np.median(times)*1e6,
                            tied_blocks=(frame.ties>1).sum(), finalists=frame.ties.sum(),
                            worst_single_tie_gap=frame.worst_tied.min()-full_values[idx],
                            j_min=frame.j_min.min(), j_max=frame.j_max.max()))
    pd.DataFrame(selections).to_csv(OUT / "selections.csv", index=False)
    pd.DataFrame(blocks).to_csv(OUT / "block_runs.csv", index=False)
    pd.DataFrame(budgets).to_csv(OUT / "costs.csv", index=False)
    blind = []
    for key in ["11", "12"]:
        valid = data[data.id.isin(sources[key].id)]
        blind.append(dict(pool=key, candidates=len(valid),
                          **valid.nsmallest(1, "V_hat").iloc[0].to_dict()))
    pd.DataFrame(blind).to_csv(OUT / "blind_minima.csv", index=False)


if __name__ == "__main__":
    run()

