# Cycle bound sweep status (campaign + journal_xcix)

Compiled from full sweeps of `puzzles/campaign` and `puzzles/journal_xcix` via
`bounds.cycles_lower_bound`, 90s per-puzzle timeout, 8 workers. `journal_cviii`
excluded — it's DLC content using mechanics not yet implemented (e.g. the
catalyst/proliferate interaction, see P282 below).

## MATCH — sound and tight (35 puzzles)

Campaign: P007–P021, P022, P024, P025–P030b, P031b, P032–P034, P036–P039 (33)
Journal_xcix: P054, P055, P056, P057, P062, P065, P085, P097, P242

## UNSOUND — c_lo exceeds record (6 puzzles, all journal_xcix)

| Puzzle | record | c_lo | overshoot |
|---|---|---|---|
| P095-Reactive-Gold | 55 | 60 | +5 |
| P240-Green-Vitriol | 40 | 52 | +12 |
| P241-Blue-Vitriol | 23 | 38 | +15 |
| P247-Luminous-Vapor | 26 | 29 | +3 |
| P250-Sophick-Mercury | 27 | 38 | +11 |
| P254-Reclaimed-Gold | 22 | 25 | +3 |

Real, confirmed bug (not noise) — a lower bound must never exceed an
achieved record. Not yet root-caused. P282-Balanced-Gold (journal_cviii,
excluded above) showed the same symptom, traced to `N_in`'s throughput
formula (`2*ceil(count/repeats)`) not accounting for catalytic reactions
(`proliferate_copper`, catalyst=copper) allowing more parallelism than the
formula assumes — worth checking first whether these 6 share that same root
cause.

## GAP — sound but loose (14 puzzles)

| Puzzle | record | c_lo | gap |
|---|---|---|---|
| P071-Synthesis-via-Alcohol | 21 | 19 | 2 |
| P074-Rat-Poison | 42 | 41 | 1 |
| P075-Fragrant-Powders | 116 | 39 | 77 |
| P076-Silver-Paint | 160 | 152 | 8 |
| P077-Aether-Detector | 134 | 13 | 121 |
| P078-Vapor-of-Levity | 54 | 40 | 14 |
| P079-Abrasive-Particles | 87 | 19 | 68 |
| P081-Eyedrops-of-Revelation | 100 | 54 | 46 |
| P082-Parade-Rocket-Fuel | 47 | 32 | 15 |
| P083-Special-Amaro | 113 | 13 | 100 |
| P084-Reconstructed-Solvent | 75 | 74 | 1 |
| P093-Conductive-Enamel | 134 | 63 | 71 |
| P105-Vanishing-Material | 119 | 78 | 41 |
| P106-Hyper-volatile-Gas | 22 | 21 | 1 |

P074–P084 (minus P080) are production (`_P`) category puzzles — known/expected
to be loose, the model minimizes g+c+i there instead of g+c+a. P071/P093/P105/
P106 are not production puzzles and haven't been investigated.

## ASSERT — no reachable state matches raw reagents (6 puzzles, journal_xcix)

P090-Lustre, P092-Lamplight-Gas, P094-Welding-Thermite, P101-Celestial-Thread,
P103-Electrum-Separation, P107-Quintessential-Medium

Not yet investigated — root cause unknown (may or may not be the same
divisibility/waste-atom class of bug found for P031b/P074/P082 earlier this
session).

## TIMEOUT — 90s cap hit (30 puzzles)

Campaign: P080-Viscous-Sludge
Journal_xcix: P058, P059, P060, P061, P063, P064, P066, P067, P068, P069,
P070, P072, P086, P087, P088, P089, P091b, P096, P098, P099, P100, P102,
P104, P108, P109, P243, P244, P245, P246, P248, P249, P251, P252, P253

journal_xcix's timeout rate (~50%) is far higher than campaign's — these
puzzles are structurally larger/harder for the current backward-search
approach. Not investigated per-puzzle; unclear whether any would resolve with
a longer timeout vs. genuinely needing an algorithmic improvement.

## SKIP — known combinatorial blowup, not rerun this sweep (5 puzzles, campaign)

P035, P042, P043 (ring puzzles), P040, P041 (large state-count puzzles) — see
memory `feedback_schematic_skip_ring_puzzles` / `project_bond_action_pooling_todo`.

## Next steps (unordered, not prioritized)

1. Root-cause the 6 UNSOUND cases — likely the highest-value fix, since an
   unsound bound is a correctness bug, not just a tightness gap.
2. Investigate the 6 ASSERT cases.
3. Investigate why journal_xcix times out so much more than campaign.
4. Investigate the 4 non-production GAP puzzles (P071, P093, P105, P106) —
   plausibly closeable given campaign's P031b-style investigation pattern.
