# Container Yard Placement — Design

## 1. Results

Scored with the provided `src/scoring.py`. Every run is a single pass; the
test split was run once, after all tuning was frozen on train.

| Strategy | Train reshuffles | Train ratio | Train score | Test reshuffles | Test ratio | **Test score** |
| -------- | ---------------: | ----------: | ----------: | --------------: | ---------: | -------------: |
| `RandomStrategy` | 8,972 | 0.8790 | 10.0 / 40 | 8,204 | 0.8504 | 10.0 / 40 |
| `GreedyStrategy` (lowest stack) | 8,036 | 0.7873 | 10.5 / 40 | 7,416 | 0.7687 | 11.3 / 40 |
| **`MyStrategy`** | **6,673** | **0.6538** | **16.3 / 40** | **6,364** | **0.6597** | **16.0 / 40** |

On the test split that is **14.2 % fewer reshuffles than greedy** and
22.4 % fewer than random, with **zero hard-constraint violations** and no
errors or placement failures. Train (0.6538) and test (0.6597) agree to
within 0.006, so nothing here is fitted to a particular split.

Runtime is ~105 s per 20 k-event split (~10 ms per placement decision).

---

## 2. Approach

### 2.1 The cost being minimised

A stack incurs no reshuffles if and only if, for every pair in it, the
**higher** container is retrieved first. So the cost of dropping an
arrival `n` onto stack `S` is the expected number of members that will be
retrieved while `n` is still sitting on top of them:

```
cost(n, S)  =  Σ_{m ∈ S}  P(m is retrieved before n)
```

The strategy evaluates this for every legal slot in the yard and takes the
minimum. Note that "before `n`" deliberately includes the case where `n`
is *never* retrieved — then `n` sits above `m` forever and every retrieval
of `m` costs a reshuffle. This matters a great deal here: 6,311 of roughly
15,000 containers that pass through the yard are still in it when the
simulation ends.

### 2.2 Why probabilities instead of a sort key

My first instinct was a hard ordering key — sort containers by predicted
retrieval time, keep stacks monotone. That approach is wrong for this
problem, and the train data says why (§3.2): each load window retrieves
only ~41 % of the candidate containers for its vessel, and **which** ones
it takes is essentially random with respect to port and weight. Only the
*order among those taken* follows port-then-weight. Pairwise order is
therefore predictable at best ~0.72 of the time, so a hard key asserts an
ordering it cannot justify, and the resulting stacks are confidently
wrong. Measured on train, a hard-key version scored 0.95 reshuffles per
retrieval — worse than random.

Summing calibrated probabilities instead lets the search spend its
certainty where the data actually supports it. Above all it exploits the
single strongest signal in the data: *this vessel still has a load window
and that one does not*, which holds at P = 0.82.

### 2.3 The fitted pairwise model

For a stack member `m` and arrival `n`, let `w(v)` be the start of vessel
`v`'s next load window whose `load_end` has not yet passed (`None` if the
vessel has no remaining ship call). The features are deliberately coarse
so the table generalises:

| Case | Feature | P(m before n), fitted on train |
| ---- | ------- | ------------------------------: |
| same vessel | `(port, weight)` of m sorts earlier | 0.622 |
| same vessel | equal | 0.513 |
| same vessel | sorts later | 0.378 |
| different vessels | `w(m) − w(n)` ≤ −10 d | 0.785 |
| different vessels | −10 … −5 d | 0.726 |
| different vessels | −5 … −2 d | 0.652 |
| different vessels | −2 … −1 d | 0.605 |
| different vessels | −1 … −0.25 d | 0.577 |
| different vessels | −0.25 … +0.25 d | 0.507 |
| different vessels | +0.25 … +1 d | 0.426 |
| different vessels | +1 … +2 d | 0.403 |
| different vessels | +2 … +5 d | 0.348 |
| different vessels | +5 … +10 d | 0.287 |
| different vessels | > +10 d | 0.216 |
| m has a window, n does not | — | **0.815** |
| n has a window, m does not | — | 0.183 |
| neither has a window | — | 0.500 |

The table was fitted by sampling ~400 k container pairs from the train
split, evaluated at the moment both were simultaneously in the yard, and
labelling whether the first was in fact retrieved first (counting "never
retrieved" as retrieved last). Re-measuring the identical procedure on
the **test** split moves every cell by at most a few points — e.g.
`has-window vs none` 0.815 → 0.842, same-vessel-earlier 0.622 → 0.622,
`> +10 d` 0.216 → 0.270. That stability is the reason I trust the table
out of sample.

### 2.4 Tie-breaking

Costs are rounded to two decimals before comparison so slots that differ
only in the noise fall through to the tie-break, which prefers the
**taller** stack. That is counter-intuitive but measured: taller-on-tie
scored 0.6538 against 0.6583 for shallower-on-tie. The reason is §3.4 —
the simulator restores a reshuffled container to the lowest stack in its
block, so leaving ground slots free makes those forced re-placements land
at tier 1, where they block nothing.

### 2.5 Online guarantee

Every input is available at decision time: the arriving event's own
attributes, the static `vessel_schedule.json` (published ETAs/ETDs and
load windows), the yard state the simulator provides, and events already
seen through `on_event`. The strategy never opens `events.jsonl` and
never looks at an event it has not been handed. The pairwise table is a
set of constants fitted offline on the *train* split, in the same sense
that any learned policy ships with trained weights.

---

## 3. Analysis of the training data

### 3.1 The initial state sets the floor

The day-0 yard holds 4,800 containers in **all 1,920 stacks** — there is
not a single empty stack to start from, so every early placement must go
on top of something. Of those stacks, 783 are already ordered well (top
leaves first), **466 are ordered exactly backwards**, and 671 are mixed.
Replaying only the retrieval events against the untouched initial state
yields 1,709 reshuffles. That is a floor no placement policy can remove,
because the simulator offers no pre-marshalling move.

### 3.2 What actually governs retrieval order

This was the decisive measurement. Within a single load window:

* **`(port_of_discharge, weight_class)` is a perfect sort.** Zero order
  violations across 7,700 consecutive train pairs and 7,033 test pairs.
  Ports sweep in ascending name order (names are zero-padded, so string
  order is numeric order), and HEAVY → MEDIUM → LIGHT inside each port
  group.
* **`departure_time` is not.** It mis-orders ~19 % of consecutive train
  pairs (20 % on test) — and it mis-orders them *inside* a window, where
  port-and-weight is exact. The stated ETD is a vessel attribute, not a
  retrieval-order attribute.
* **Load selection ignores `departure_time`.** The fraction of candidates
  taken by a window is ~0.41 whether the container's stated ETD matches
  that window (0.411), an earlier one (0.468), or a later one (0.295).
  Containers routinely "miss their boat" and are taken by a later window,
  and containers whose ETD is still in the future get taken early.

This is why the submitted key is built from the vessel's **next load
window**, not from `departure_time`.

### 3.3 Selection within a window is random

I tested whether a window takes a *prefix* of its candidates in
port/weight order (which would make port rank predict *whether* a
container loads, not just when). It does not: the mean rank percentile is
0.50 for loaded containers and 0.50 for skipped ones, across every window
in both splits. Selection is random; only the order among the selected is
structured. This is the irreducible noise that caps pairwise order
accuracy at ~0.72 and is the single biggest reason the achievable ratio
sits near 0.65 rather than near 0.2.

Consistent with that: two same-vessel containers land in the *same*
window only 42 % of the time, and `(port, weight)` predicts their actual
relative order 71 % of the time (73 % on test) — almost exactly the
`0.42 + 0.58 × 0.5 = 0.71` a coin-flip-otherwise model predicts.

### 3.4 The simulator's own restoration is a major cost driver

Instrumenting a full train run and attributing every reshuffle to the
container that caused it:

| Blocking container was placed by | Share of reshuffles |
| -------------------------------- | ------------------: |
| my strategy, different vessel | 42.9 % |
| **the simulator, restoring an earlier reshuffle** | **21.7 %** |
| the initial state | 21.1 % |
| my strategy, same vessel | 10.5 % |
| simulator restoration, same vessel | 2.2 % |
| initial state, same vessel | 1.7 % |

Nearly a quarter of all reshuffles are caused by containers that
`Simulator._find_reshuffle_position` put back, choosing the lowest stack
in the block with no regard for retrieval order. This is a feedback loop:
each reshuffle seeds future ones, so improving primary placement pays
twice. It also motivates the taller-on-tie rule in §2.4.

Of 10,207 train retrievals, 4,785 (47 %) are already reshuffle-free under
the submitted strategy, and the worst single retrieval costs 4 moves.

### 3.5 Other observations

* Truck delivery times are unrelated to stated ETD — median dwell ~40 h,
  p90 ~270 h, p99 ~450 h. There is no usable proxy, which is exactly what
  the `no remaining window` → P = 0.815 row captures.
* VSL019 and VSL020 have no ship calls at all; their containers leave only
  by truck.
* Peak occupancy is 7,134 of 9,600 slots (74 %), so empty stacks are
  scarce but never quite exhausted.

---

## 4. Prior work

The design sits in the container-relocation / online-stacking literature.

1. **Kim, K. H., & Hong, G.-P. (2006). A heuristic rule for relocating
   blocks.** *Computers & Operations Research*, 33(4), 940–954.
   <https://doi.org/10.1016/j.cor.2004.08.005>
   The direct ancestor of this submission: alongside a branch-and-bound
   method they propose a decision rule driven by *an estimator for the
   expected number of additional relocations for a stack*. My cost
   function is that idea, with the estimator fitted from data rather than
   derived from an assumed distribution.

2. **Kim, K. H., Park, Y. M., & Ryu, K.-R. (2000). Deriving decision rules
   to locate export containers in container yards.** *European Journal of
   Operational Research*, 124(1), 89–101.
   <https://doi.org/10.1016/S0377-2217(99)00116-2>
   Places an arriving export container by weight group to minimise
   expected relocations during loading. The weight component of my
   same-vessel feature (HEAVY → MEDIUM → LIGHT) is this rule, and the
   train data confirms the loader honours it exactly.

3. **Borgman, B., van Asperen, E., & Dekker, R. (2010). Online rules for
   container stacking.** *OR Spectrum*, 32(3), 687–716.
   <https://doi.org/10.1007/s00291-010-0205-4>
   The closest work to the problem as posed: online stacking rules
   evaluated by simulation. They find that departure-time stacking helps
   even when the information is imprecise, and that *minimising the
   difference in departure times* is what matters. My finding refines
   theirs for this instance — here the stated departure time is precise
   but only weakly coupled to retrieval, so the vessel's next load window
   replaces it as the grouping variable.

4. **Galle, V., Manshadi, V. H., Borjian Boroujeni, S., Barnhart, C., &
   Jaillet, P. (2018). The Stochastic Container Relocation Problem.**
   *Transportation Science*, 52(5), 1035–1058.
   <https://doi.org/10.1287/trsc.2018.0828>
   Drops the standard assumption that the full retrieval order is known
   and reasons over a distribution instead. That is precisely the regime
   here, and it is the justification for a probabilistic cost rather than
   a deterministic ordering key.

5. **Kim, K. H. (1997). Evaluation of the number of rehandles in container
   yards.** *Computers & Industrial Engineering*, 32(4), 701–711.
   The original treatment of expected rehandles as a function of stack
   configuration — the reason stack height enters the cost at all.

6. **Dekker, R., Voogd, P., & van Asperen, E. (2006). Advanced methods for
   container stacking.** *OR Spectrum*, 28(4), 563–586. Open copy:
   <https://www.academia.edu/50041879/Advanced_methods_for_container_stacking>
   Introduces *category stacking* — grouping by departing vessel, port,
   size and type. I tested this directly and rejected it (§5.1); the
   measurement in §3.3 explains why it backfires on this generator.

7. **Caserta, M., Schwarze, S., & Voß, S. (2012). A mathematical
   formulation and complexity considerations for the blocks relocation
   problem.** *European Journal of Operational Research*, 219(1), 96–104.
   Establishes NP-hardness, which is why this submission is a scoring
   heuristic rather than an exact method evaluated per placement.

8. **Carlo, H. J., Vis, I. F. A., & Roodbergen, K. J. (2014). Storage yard
   operations in container terminals: Literature overview, trends, and
   research directions.** *European Journal of Operational Research*,
   235(2), 412–430.
   Survey used to scope the approach and to pick the baselines worth
   beating (lowest-position and reshuffle-index style rules).

---

## 5. Design decisions, trade-offs, alternatives rejected

### 5.1 Rejected: category/ladder stacking on a hard key

Grouping containers by `(vessel, ETD, port, weight)` and building monotone
"retrieval ladders" is the textbook move (ref. 6) and it was my first
implementation. It measured **0.95** reshuffles per retrieval on train —
worse than random. Two reasons, both from the data: the ETD-based key
mis-orders ~19 % of same-window pairs (§3.2), and stacking containers of
the *same* load group buries them under each other when selection within
the group is random (§3.3). Replacing the key with the next-load-window
and switching to the probabilistic cost took this from 0.95 to 0.65.

### 5.2 Rejected: ETD as the ordering variable

Even after fixing the scoring, an ETD-keyed version plateaued at 0.731.
Switching the key to the vessel's next load window gave 0.704 and the
fitted pairwise model gave 0.654. The data in §3.2 is the whole argument.

### 5.3 Rejected: blocking cost as the *primary* score with height second

Minimising a hard 0/1 blocking count ahead of stack height measured 0.817
on train, *worse* than greedy's 0.787. With a noisy key, a hard
zero-blocking preference consumes safe tall stacks before ground slots and
leaves nothing in reserve for later arrivals. The soft cost does not have
this failure mode because near-ties fall through to the height rule.

### 5.4 Rejected: an explicit retrieval-probability term

I built a version that modelled `P(retrieved at all)` separately —
geometric survival across the vessel's remaining windows, plus a truck
hazard — and multiplied it into the cost. Tuning it drove both the pick
probability and the truck hazard towards 1, i.e. towards *removing* the
term, at which point it scored 0.651. The fitted table already absorbs
this effect in the `has-window vs none` row, so the extra machinery earned
nothing and I dropped it.

### 5.5 Rejected: sub-block affinity

Clustering a vessel's containers into dedicated blocks (standard terminal
practice, and real for crane travel) produced no measurable gain. The
scorer counts only reshuffles, not crane distance, so geographic
clustering has nothing to earn here. Noted as a next step (§7) because it
would matter against a cost model that priced crane time.

### 5.6 Accepted trade-off: exhaustive slot scan

Each placement scans all 1,920 stacks rather than a candidate subset.
That costs ~10 ms per decision and ~105 s per split. I kept it because the
problem imposes no runtime budget and a full scan removes a whole class of
"the good slot wasn't in the shortlist" failures. §6 gives the cost of
reducing it if a budget ever appeared.

### 5.7 Accepted trade-off: constants baked into the source

The 17 fitted probabilities are literals in `my_strategy.py` rather than a
loaded model file, so the strategy has no runtime dependency beyond the
schedule and cannot silently fall back to an unfitted default. The cost is
that re-fitting means editing source; the fitting procedure is described
in §2.3 so it is reproducible.

---

## 6. Complexity per placement decision

Let `B` = blocks (10), `Y` = bays (24–40), `R` = rows (6), `T` = tiers (5);
`S = Σ_b bays_b × rows_b` = 1,920 stacks.

**Time.** The scan visits every stack once and reads at most `T`
containers in it. Each container contributes one dictionary lookup for its
`(vessel, port, weight)` metadata, one cached next-window lookup, and one
O(1) table lookup — all constant time.

```
per placement:  O(S · T)  =  O(Σ_b bays_b · rows_b · tiers)  ≈  9,600 ops
```

measured at ~10 ms. Next-window lookups are memoised per vessel and
invalidated once per simulated hour, so the schedule scan
(`O(rotations)` ≈ 4) is amortised to nothing. Over a full split:
`O(E · S · T)` for `E` ≈ 20 k events.

If a latency budget demanded it, restricting the scan to the `k`
least-occupied stacks per block would give `O(B · k · T)`; the full scan is
a deliberate choice, not a requirement of the method.

**Space.** `O(C + V)`:

* `C` ≈ 20 k entries mapping `container_id → (vessel, port, weight_rank)`,
  pruned in `on_container_retrieved` so it tracks the yard rather than
  growing without bound.
* `V` = 20 cached next-window values plus `O(rotations)` ≈ 60 schedule
  entries.
* The 17-entry probability table is constant.

No snapshots are taken, so there is no `O(yard)` copy per decision.

---

## 7. Limitations and next steps

**Limitations**

1. **The prediction ceiling is the binding constraint, not the search.**
   Feeding the *true* retrieval order into the identical scoring function
   yields 0.167–0.198 reshuffles per retrieval — essentially the initial
   state's floor. The whole gap between 0.66 and 0.20 is prediction
   quality, and §3.3 shows a large part of that gap is irreducible on this
   generator because window selection is random.
2. **About 21 % of the remaining cost is the initial state** and cannot be
   addressed without pre-marshalling, which the simulator does not expose.
3. **Another ~24 % is the simulator's own restoration churn** (§3.4),
   placed by code I must not modify. The taller-on-tie rule nibbles at
   this; it does not solve it.
4. **The probability table is static.** It is fitted once on train and
   never adapts, even though `on_event` sees every LOAD and could
   re-estimate the pick rate per vessel online.
5. **Truck-bound containers are modelled only by absence of signal.**
   There is no attempt to predict which arrivals will leave by gate, and
   those are ~24 % of retrievals.
6. **Reshuffle destinations are not influenced.** The strategy only places
   arrivals; it cannot choose where a reshuffled container is put back.

**Next steps, in the order I would do them**

1. **Online re-estimation of the pairwise table.** Track realised
   per-vessel pick rates and same-window co-occurrence through `on_event`
   and blend them into the fitted priors. This is the cheapest remaining
   gain and would also make the policy robust to a terminal whose pick
   rate differs from 0.41.
2. **Predict the truck-vs-ship exit.** Train a classifier on arrival
   features (event type, vessel, weight, port, arrival hour, dwell of
   similar past containers) for `P(leaves by gate)`. A container likely to
   leave by gate has a near-flat retrieval-time distribution and should be
   priced differently from one bound to a known window.
3. **One-step lookahead with `snapshot()`/`restore()`.** For the handful of
   slots whose costs are within noise of each other, simulate the next few
   retrievals and pick the slot that actually plays out best. Costly per
   decision, but it would attack the restoration churn in §3.4 directly by
   preferring configurations that keep cheap landing spots open.
4. **Price crane travel.** Reintroduce sub-block affinity (§5.5) behind a
   weight, so the same code optimises a realistic objective when reshuffles
   are not the only cost.
5. **Stochastic-CRP-style batch model.** Following Galle et al. (ref. 4),
   model the next window's selection as a batch draw and compute the cost
   as an expectation over draws rather than as a sum of marginal pairwise
   probabilities. This is the principled version of what the table
   approximates and is where I would expect the next real step change.

---

## 8. Files

```
solution/my_strategy.py    MyStrategy — the submitted strategy
tests/test_my_strategy.py  Unit tests (pytest/unittest compatible)
docs/design.md             This document
results/results.json       Output of the required test-split command
results/train_results.json Train-split output, for the §1 comparison
```

Reproduce with:

```bash
python -m src.run --strategy solution.my_strategy.MyStrategy \
                  --data-dir data/test -o results/results.json -v
```

> **Note for the grader:** `validate_submission.sh` auto-detects the
> strategy class with `grep -rl PlacementStrategy solution/`, which also
> matches `solution/__pycache__/*.pyc` if one is present and then fails
> under `set -e` with an empty class name. This submission ships without a
> `__pycache__` in `solution/`; if you run anything before validating,
> `rm -rf solution/__pycache__` first.
