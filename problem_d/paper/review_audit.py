"""Verify archived scores and all binary block states without CatBoost or Kaiwu.

This rechecks saved results; it does not retrain a predictor or rerun the SDK.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / 'artifacts'
DATA = HERE.parent / 'date'


def check(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def audit():
    report = {'scope': 'archived predictions and independent binary enumeration',
              'sdk_rerun': False, 'model_retraining': False}
    for row in pd.read_csv(OUT / 'score_runs.csv').itertuples():
        path = OUT / f'{row.run}.csv'
        check(hashlib.sha256(path.read_bytes()).hexdigest() == row.sha256,
              f'{row.run}: SHA-256 differs from frozen score ledger')
        frame = pd.read_csv(path)
        check(len(frame) == row.candidates and frame.id.is_unique, 'score IDs/count')
        check(np.isfinite(frame.V_hat).all() and (frame.V_hat >= 0).all(), 'invalid score')
    report['score_hashes_verified'] = 2
    full = pd.read_csv(OUT / 'q23_full.csv')
    hidden = pd.read_csv(OUT / 'q1_holdout.csv')
    blocks = pd.read_csv(OUT / 'block_runs.csv')
    selections = pd.read_csv(OUT / 'selections.csv')
    report['q1_predictions'] = len(hidden)
    report['q23_predictions'] = len(full)
    # Every hidden pair must contribute all seven families; Q3 contributes three.
    check(hidden.groupby(['A', 'B']).size().eq(7).all(), 'hidden pair grouping')
    check(full.groupby(['A', 'B']).size().eq(3).all(), 'Q3 family grouping')
    report['validation'] = []
    for seed in (26, 42):
        val = pd.read_csv(HERE.parent / 'results' / f'validation_seed{seed}.csv')
        for model in ('direct', 'mixture'):
            for name, pool in [('all', val), ('Q2', val[val.X.eq('NO2')]),
                               ('Q3', val[val.X.isin(['O3', 'NO2', 'FO2'])])]:
                ranked = pool.sort_values([model, 'id'])
                report['validation'].append(dict(seed=seed, model=model, pool=name,
                    hits50=int((ranked.head(50).V < 1).sum()),
                    first_id=int(ranked.iloc[0].id), first_V=float(ranked.iloc[0].V),
                    regret=float(ranked.iloc[0].V - ranked.V.min())))
    report['blocks'] = []
    states = (np.arange(256)[:, None] >> np.arange(8)) & 1
    spins = np.column_stack([2 * states - 1, np.ones(256, int)])
    for task in ('Q2', 'Q3'):
        pool = full[full.X.eq('NO2')] if task == 'Q2' else full
        reduced = pool.sort_values(['V_hat', 'id']).drop_duplicates(['A', 'B'])
        costs = reduced.pivot(index='A', columns='B', values='V_hat').sort_index().sort_index(axis=1).to_numpy()
        ids = reduced.pivot(index='A', columns='B', values='id').sort_index().sort_index(axis=1).to_numpy()
        ledger = blocks[blocks.problem.eq(task)]
        check(len(ledger) == 169 and not ledger.duplicated(['block_a','block_b']).any(), 'block coverage')
        for row in ledger.itertuples():
            a, b = int(row.block_a)*4, int(row.block_b)*4
            c, labels = costs[a:a+4, b:b+4], ids[a:a+4, b:b+4]
            q = np.rint(15*c/c.max()).astype(int) if c.max() else np.zeros((4,4), int)
            matrix = np.zeros((8,8), int)
            np.fill_diagonal(matrix, -16)
            matrix[:4,:4] += np.triu(np.full((4,4), 32), 1)
            matrix[4:,4:] += np.triu(np.full((4,4), 32), 1)
            matrix[:4,4:] = q
            energy = np.einsum('bi,ij,bj->b', states, matrix, states)
            legal = (states[:,:4].sum(axis=1)==1) & (states[:,4:].sum(axis=1)==1)
            ground = states[energy==energy.min()]
            check(legal[energy==energy.min()].all(), 'infeasible ground state')
            # Construct scaled symmetric Ising matrix algebraically, independently of SDK.
            coupling = np.zeros((9,9), int)
            coupling[:8,:8] = -(matrix + matrix.T)
            np.fill_diagonal(coupling, 0)
            ref = -(2*np.diag(matrix) + (matrix+matrix.T).sum(axis=1) - 2*np.diag(matrix))
            coupling[:8,8] = coupling[8,:8] = ref
            ising = -np.einsum('bi,ij,bj->b', spins, coupling, spins)
            check(np.array_equal(ising-ising[0], 8*(energy-energy[0])), 'Ising affine relation')
            check(coupling.min()>=-128 and coupling.max()<=127, '8-bit overflow')
            locations = list(zip(ground[:,:4].argmax(axis=1), ground[:,4:].argmax(axis=1)))
            ij = min(locations, key=lambda k:(c[k],labels[k]))
            check(len(locations)==row.ties and q.min()==row.bin_min, 'tie ledger')
            check(int(labels[ij])==row.id, 'block winner ID')
            np.testing.assert_allclose([c[ij],max(c[k] for k in locations)],
                [row.score,row.worst_tied],rtol=0,atol=1e-10)
            check(ising.min()==row.ising_energy, 'archived Ising minimum')
            check(coupling.min()==row.j_min and coupling.max()==row.j_max,'coupling range')
        picked = pool.sort_values(['V_hat','id']).iloc[0]
        archived = selections[selections.problem.eq(task)].iloc[0]
        check(picked.id==archived.id, 'full-pool selection')
        report['blocks'].append(dict(task=task, blocks=169, states=169*256,
            winner_id=int(picked.id), score=float(picked.V_hat)))
    blind_ids = set(hidden.id)
    for row in pd.read_csv(OUT/'blind_minima.csv').itertuples():
        pool = full[full.id.isin(blind_ids)]
        if str(row.pool) == '11':
            pool = pool[pool.X.eq('NO2')]
        check(len(pool)==row.candidates, 'blind membership count')
        check(pool.sort_values(['V_hat','id']).iloc[0].id==row.id, 'blind winner')
    cross = pd.read_csv(OUT/'q2_blind_comparison.csv')
    check(set(cross.id)==set(full[full.id.isin(blind_ids)&full.X.eq('NO2')].id), 'cross-rank IDs')
    for score, rank in [('property_score','property_rank'),('mixture_score','mixture_rank')]:
        ordered = cross.sort_values([score,'id'])
        check(np.array_equal(ordered[rank],np.arange(1,530)), 'cross ranks')
    d = np.column_stack([cross.pred_e_form-.21,1.4-cross.pred_gap,
                         cross.pred_gap-3.1,5.73-cross.pred_vbm,cross.pred_cbm-4.5])
    np.testing.assert_allclose(((np.maximum(d,0)/.5)**2).sum(axis=1),cross.property_score,atol=1e-10,rtol=0)
    gate = pd.read_csv(OUT/'gate_components.csv')
    replay = gate.p_gap*gate.gapped_score+(1-gate.p_gap)*gate.metal_score
    np.testing.assert_allclose(replay,gate.soft,atol=1e-8,rtol=0)
    for row in pd.read_csv(OUT/'score_margins.csv').itertuples():
        prefix = int(row.pool)
        pool = full[full.X.eq('NO2')] if prefix in (7,11) else full
        if prefix in (11,12):
            pool = pool[pool.id.isin(blind_ids)]
        ranked = pool.sort_values(['V_hat','id'])
        check(ranked.iloc[0].id==row.first_id and ranked.iloc[1].id==row.second_id, 'margin IDs')
        np.testing.assert_allclose(ranked.iloc[1].V_hat-ranked.iloc[0].V_hat,row.margin,atol=1e-10,rtol=0)
    report['gate_replay_rows'] = len(gate)
    raw_files = list(DATA.glob('02_*.csv'))
    report['original_data_available'] = bool(raw_files)
    if raw_files:
        train = pd.read_csv(raw_files[0]); val = pd.read_csv(next(DATA.glob('03_*.csv')))
        raw_hidden = pd.read_csv(next(DATA.glob('04_*.csv')))
        groups = [set(zip(f.A,f.B)) for f in (train,val,raw_hidden)]
        check(all(not groups[i]&groups[j] for i,j in [(0,1),(0,2),(1,2)]), 'pair leakage')
        check(set(hidden.id)==set(raw_hidden.id), 'Q1 official IDs')
        public = pd.concat([train,val])
        dev = np.column_stack([public.e_form-.21,1.4-public.gap,public.gap-3.1,
                               5.73-public.vbm,public.cbm-4.5])
        report['max_loss_rounding_difference'] = float(np.abs(((np.maximum(dev,0)/.5)**2).sum(axis=1)-public.V).max())
        check(report['max_loss_rounding_difference']<1.1e-5, 'loss reconstruction')
        check((public.V==0).sum()==0, 'public exact targets')
    report['status'] = 'PASS'
    (OUT/'review_audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    audit()
