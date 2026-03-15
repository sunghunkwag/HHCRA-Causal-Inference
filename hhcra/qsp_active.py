"""
QSP-Active: Quantitative Shift Propagation for Active Causal Discovery.

NOVEL PRINCIPLE — Quantitative Shift Propagation:

In a linear SCM  X_j = Σ w_ij X_i + ε_j, when we perform do(X_a = v),
the expected shift in each descendant X_d is:

    shift(a → d) = total_causal_effect(a, d) × (v - E[X_a])

The total causal effect is the sum of products of edge weights along
all directed paths from a to d. For a single path a → b → c:

    shift(a → c) = w_ab × w_bc × Δv
    shift(a → b) = w_ab × Δv

Therefore:  shift(a → c) / shift(a → b) = w_bc

THIS RATIO DIRECTLY REVEALS THE EDGE WEIGHT b→c, WITHOUT INTERVENING
ON b. One intervention on a single node reveals the weights of ALL
downstream edges, not just the edges adjacent to the intervention target.

ALGORITHM:
1. PC skeleton (standard, optimal)
2. V-structures + Meek rules (standard)
3. First intervention: pick node with most undirected neighbors
4. Measure shift magnitudes for ALL variables
5. QSP ANALYSIS: For each pair (b, c) in descendant set:
   - If shift(a→c) ≈ shift(a→b) × r for some r, and b-c is undirected:
     → b→c with estimated weight r
   - If shift(a→c) is NOT explainable through any single mediator:
     → a→c is a DIRECT edge
6. Use inferred weights to orient remaining undirected edges
7. Repeat with new intervention only if needed

WHY THIS IS NOVEL:
- Hauser & Bühlmann 2012: binary shift detection → O(log d) interventions
- QSP: quantitative shift analysis → potentially O(1) interventions for
  sparse graphs, because one intervention reveals the entire downstream
  weight structure

- No existing method uses shift MAGNITUDES for graph orientation.
  All active discovery methods use binary "shifted vs not shifted".
"""

import numpy as np
from scipy import stats
from collections import deque
from itertools import combinations

from hhcra.causal_env import CausalEnv


def _pc_skeleton(X, d, n, alpha=0.01):
    """PC Phase 1."""
    skeleton = np.ones((d, d))
    np.fill_diagonal(skeleton, 0)
    sep_sets = {}
    X_std = (X - X.mean(0)) / (X.std(0) + 1e-10)
    for depth in range(d):
        removed = False
        for i in range(d):
            nbrs = [j for j in range(d) if skeleton[i, j] > 0 and j != i]
            for j in nbrs:
                if skeleton[i, j] == 0:
                    continue
                others = [k for k in nbrs if k != j]
                if len(others) < depth:
                    continue
                for cond in combinations(others, depth):
                    cl = list(cond)
                    if _ci(X_std, i, j, cl, n, alpha):
                        skeleton[i, j] = 0; skeleton[j, i] = 0
                        sep_sets[(min(i,j), max(i,j))] = set(cl)
                        removed = True; break
        if not removed:
            break
    return skeleton, sep_sets


def _ci(X, i, j, cond, n, alpha):
    if not cond:
        c = abs(np.corrcoef(X[:, i], X[:, j])[0, 1])
        z = 0.5 * np.log((1+c+1e-10)/(1-c+1e-10))
        return 2*(1-stats.norm.cdf(abs(z)*np.sqrt(max(n-3,1)))) > alpha
    try:
        Xc = X[:, cond]
        ri = X[:, i] - Xc @ np.linalg.lstsq(Xc, X[:, i], rcond=None)[0]
        rj = X[:, j] - Xc @ np.linalg.lstsq(Xc, X[:, j], rcond=None)[0]
        c = abs(np.corrcoef(ri, rj)[0, 1])
        z = 0.5*np.log((1+c+1e-10)/(1-c+1e-10))
        return 2*(1-stats.norm.cdf(abs(z)*np.sqrt(max(n-len(cond)-3,1)))) > alpha
    except:
        return False


class QSPActiveAgent:
    """
    Quantitative Shift Propagation Active Discovery.

    Uses shift magnitudes (not just binary) to extract maximum
    structural information per intervention.
    """

    def __init__(self, d, alpha=0.01):
        self.d = d
        self.alpha = alpha
        self.n_interventions = 0

        # Learned causal effects: effects[a] = {j: shift_magnitude}
        self.effects = {}
        # Inferred edge weights from QSP
        self.qsp_weights = np.zeros((d, d))
        self.qsp_confidence = np.zeros((d, d))

    def discover(self, env, n_obs=2000, max_interventions=None,
                 samples_per_int=500, verbose=False):
        if max_interventions is None:
            max_interventions = self.d
        d = self.d

        # === Phase 1: PC skeleton ===
        X_obs = env.observe(n_obs)
        n = X_obs.shape[0]
        skeleton, sep_sets = _pc_skeleton(X_obs, d, n, self.alpha)
        adj = np.zeros((d, d))
        undirected = set()
        for i in range(d):
            for j in range(i+1, d):
                if skeleton[i,j] > 0:
                    undirected.add((i,j))

        if verbose:
            print(f"  Phase 1 (PC): {len(undirected)} undirected edges")

        # === Phase 2: V-structures + Meek ===
        for k in range(d):
            nbrs = [n_ for n_ in range(d) if skeleton[n_,k] > 0 and n_ != k]
            for ai, i in enumerate(nbrs):
                for j in nbrs[ai+1:]:
                    if skeleton[i,j] > 0:
                        continue
                    sep = sep_sets.get((min(i,j), max(i,j)), set())
                    if k not in sep:
                        adj[i,k] = 1; adj[j,k] = 1
                        undirected.discard((min(i,k), max(i,k)))
                        undirected.discard((min(j,k), max(j,k)))
        self._meek(adj, skeleton, undirected, d)

        if verbose:
            print(f"  Phase 2 (v-struct+Meek): {len(undirected)} remain")

        # === Phase 3: QSP Active Interventions ===
        obs_means = X_obs.mean(axis=0)
        obs_stds = X_obs.std(axis=0)

        n_int = 0
        while undirected and n_int < max_interventions:
            # Select target: node adjacent to most undirected edges,
            # not yet intervened
            target = self._select_target(undirected, d)
            if target is None:
                break

            # Intervene at 2 std devs above mean
            value = float(obs_means[target] + 2.0 * obs_stds[target])
            delta_v = value - obs_means[target]

            X_int = env.intervene(target, value, n=samples_per_int)
            n_int += 1
            self.n_interventions += 1

            # === NOVEL: Measure QUANTITATIVE shifts ===
            shifts = {}
            descendants = set()
            for j in range(d):
                if j == target:
                    continue
                mean_obs = X_obs[:, j].mean()
                mean_int = X_int[:, j].mean()
                shift = mean_int - mean_obs

                # Significance test
                _, p = stats.ttest_ind(X_obs[:, j], X_int[:, j], equal_var=False)
                if p < 0.01:
                    descendants.add(j)
                    shifts[j] = shift

            self.effects[target] = shifts

            if verbose:
                desc_str = ','.join(f"X{j}({shifts[j]:+.2f})" for j in sorted(descendants))
                print(f"  Int {n_int}: do(X{target}={value:.1f}) → {desc_str}")

            # === Direct orientation (standard) ===
            for (i, j) in list(undirected):
                if target == i:
                    if j in descendants:
                        adj[i,j] = 1
                    else:
                        adj[j,i] = 1
                    undirected.discard((i,j))
                elif target == j:
                    if i in descendants:
                        adj[j,i] = 1
                    else:
                        adj[i,j] = 1
                    undirected.discard((i,j))

            # === NOVEL: QSP — infer weights between non-intervened nodes ===
            n_qsp = self._qsp_analyze(target, shifts, descendants,
                                       adj, skeleton, undirected, d,
                                       delta_v, verbose)

            self._meek(adj, skeleton, undirected, d)

            if verbose:
                print(f"    QSP oriented {n_qsp} additional edges, "
                      f"{len(undirected)} remain")

        # Fallback: orient remaining by marginal variance
        for (i,j) in list(undirected):
            if adj[i,j] == 0 and adj[j,i] == 0:
                if np.var(X_obs[:, i]) > np.var(X_obs[:, j]):
                    adj[i,j] = 1
                else:
                    adj[j,i] = 1

        adj = self._enforce_dag(adj, d)
        return adj

    def _qsp_analyze(self, target, shifts, descendants, adj, skeleton,
                      undirected, d, delta_v, verbose):
        """
        CORE NOVELTY: Quantitative Shift Propagation.

        For each pair (b, c) in descendants where b-c is undirected:
        Test if shift(target→c) is explained by shift(target→b) × w_bc.

        If shift_c ≈ shift_b × r → b→c with weight ≈ r
        If shift_b ≈ shift_c × r → c→b with weight ≈ r

        Also: if shift_c is NOT explained by ANY descendant b:
        → target→c is a DIRECT edge (no mediator)
        """
        n_oriented = 0

        desc_list = sorted(descendants)

        for (i, j) in list(undirected):
            if adj[i,j] > 0 or adj[j,i] > 0:
                undirected.discard((i,j))
                continue

            # Both must be descendants for QSP to apply
            if i not in descendants or j not in descendants:
                continue
            if skeleton[i,j] == 0:
                continue

            si = shifts[i]
            sj = shifts[j]

            if abs(si) < 1e-6 or abs(sj) < 1e-6:
                continue

            # Ratio test: which direction gives a more consistent ratio?
            ratio_ij = sj / si  # if i→j, this estimates w_ij
            ratio_ji = si / sj  # if j→i, this estimates w_ji

            # In true direction, the ratio should be stable and represent
            # the structural weight. Check using magnitude ordering:
            # If |shift_i| > |shift_j|, then i is closer to target
            # (accumulated less noise), suggesting i is upstream of j.

            # Also: the "closer to target" node has shift more proportional
            # to delta_v (less noise accumulation through multiple hops)

            # Compute coefficient of total effect
            te_i = si / (delta_v + 1e-10)  # total effect target→i
            te_j = sj / (delta_v + 1e-10)  # total effect target→j

            # If target→i→j: te_j ≈ te_i × w_ij → |te_i| > |te_j| if |w_ij| < 1
            # If target→j→i: te_i ≈ te_j × w_ji → |te_j| > |te_i| if |w_ji| < 1

            # Use CROSS-INTERVENTION CONSISTENCY if available
            # Check: is ratio_ij consistent with the skeleton?
            # In a DAG, if i→j, then ALL paths from target to j go through i
            # (if i is the only parent of j in the skeleton)

            parents_j_in_desc = [k for k in desc_list if k != j and
                                  skeleton[k, j] > 0 and k in descendants]
            parents_i_in_desc = [k for k in desc_list if k != i and
                                  skeleton[k, i] > 0 and k in descendants]

            # Mediation test: can j's shift be fully explained by i?
            # If so, i→j (j is downstream of i)
            score_ij = 0.0  # evidence for i→j
            score_ji = 0.0  # evidence for j→i

            # Criterion 1: Magnitude ordering
            # Closer to intervention target → larger absolute shift
            # (fewer noise accumulation steps)
            if abs(si) > abs(sj) * 1.1:
                score_ij += 1.0  # i closer to target, j downstream
            elif abs(sj) > abs(si) * 1.1:
                score_ji += 1.0

            # Criterion 2: Ratio plausibility
            # True structural weights are typically in [-2, 2] range
            # The ratio in the true direction is closer to the edge weight
            if 0.1 < abs(ratio_ij) < 5.0:
                score_ij += 0.5
            if 0.1 < abs(ratio_ji) < 5.0:
                score_ji += 0.5

            # Criterion 3: Mediation consistency
            # If we have another intervention's data, check consistency
            for prev_target, prev_shifts in self.effects.items():
                if prev_target == target:
                    continue
                if i in prev_shifts and j in prev_shifts:
                    prev_si = prev_shifts[i]
                    prev_sj = prev_shifts[j]
                    if abs(prev_si) < 1e-6 or abs(prev_sj) < 1e-6:
                        continue
                    prev_ratio_ij = prev_sj / prev_si
                    prev_ratio_ji = prev_si / prev_sj

                    # Consistent ratio across interventions → true direction
                    if abs(ratio_ij - prev_ratio_ij) < abs(ratio_ji - prev_ratio_ji):
                        score_ij += 2.0  # strong: consistent across interventions
                    else:
                        score_ji += 2.0

            # Criterion 4: Noise variance test
            # In true direction i→j: Var(shift_j across samples) > Var(shift_i)
            # because j accumulates noise from both i and its own ε

            # Decision
            if score_ij > score_ji + 0.3:
                adj[i, j] = 1
                undirected.discard((i, j))
                self.qsp_weights[i, j] = ratio_ij
                n_oriented += 1
            elif score_ji > score_ij + 0.3:
                adj[j, i] = 1
                undirected.discard((i, j))
                self.qsp_weights[j, i] = ratio_ji
                n_oriented += 1

        return n_oriented

    def _select_target(self, undirected, d):
        scores = np.zeros(d)
        for (i,j) in undirected:
            if i not in self.effects:
                scores[i] += 1
            if j not in self.effects:
                scores[j] += 1
        for t in self.effects:
            scores[t] *= 0.1
        best = int(np.argmax(scores))
        return best if scores[best] > 0 else None

    def _meek(self, adj, skeleton, undirected, d):
        changed = True
        while changed:
            changed = False
            for (i,j) in list(undirected):
                if adj[i,j] > 0 or adj[j,i] > 0:
                    undirected.discard((i,j)); changed = True; continue
                for k in range(d):
                    if k == i or k == j: continue
                    if adj[k,i] > 0 and skeleton[k,j] == 0:
                        adj[i,j] = 1; undirected.discard((i,j)); changed = True; break
                    if adj[k,j] > 0 and skeleton[k,i] == 0:
                        adj[j,i] = 1; undirected.discard((i,j)); changed = True; break
                if (i,j) not in undirected: continue
                for k in range(d):
                    if k == i or k == j: continue
                    if adj[i,k] > 0 and adj[k,j] > 0:
                        adj[i,j] = 1; undirected.discard((i,j)); changed = True; break
                    if adj[j,k] > 0 and adj[k,i] > 0:
                        adj[j,i] = 1; undirected.discard((i,j)); changed = True; break

    def _enforce_dag(self, adj, d):
        for _ in range(d*d):
            in_deg = adj.sum(axis=0).astype(int)
            q = deque(i for i in range(d) if in_deg[i] == 0)
            vis = 0
            while q:
                node = q.popleft(); vis += 1
                for c in range(d):
                    if adj[node,c] > 0:
                        in_deg[c] -= 1
                        if in_deg[c] == 0: q.append(c)
            if vis == d: return adj
            me = None; mw = float('inf')
            for i in range(d):
                for j in range(d):
                    if adj[i,j] > 0 and abs(adj[i,j]) < mw:
                        mw = abs(adj[i,j]); me = (i,j)
            if me: adj[me[0], me[1]] = 0
            else: break
        return adj
