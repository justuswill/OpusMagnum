# PLAN.md — Toward (Provably) Optimal Solutions for Opus Magnum Puzzles

This document is a knowledge base + high-level plan for building a system that
searches for optimal (or record-level) solutions to Opus Magnum puzzles, using a
**choreography-first** approach (fix molecule trajectories, then synthesize the
machine), with `omsim`/`libverify` as the ground-truth simulator.

It is intended as a starting point for a coding agent to derive a concrete
implementation plan. Items marked **[VERIFY]** should be checked against the
actual source/docs before relying on them.

---

## 1. Goal and scope

- **Input:** an Opus Magnum puzzle (`.puzzle` file): reagent molecules, product
  molecules, allowed parts/glyphs/instructions.
- **Output:** a valid `.solution` file that the real game / omsim accepts, with
  the best achievable score in one or more metrics.
- **Metrics:** cost (g), cycles, area, instructions — plus combined categories
  (e.g. sum, products of metrics) used by the community leaderboards.
- **Two operating modes:**
  1. **Record hunting:** find the best solution we can (no proof needed).
  2. **Optimality proving:** exhaustively rule out anything below a bound
     (only feasible for very small puzzles / tight bounds; see §7.5).

Important reality check (from prior analysis): naive brute force over
instruction tapes is ~12^(arms × cycles) and is infeasible beyond toy sizes.
All value comes from (a) strong analytic lower bounds, (b) searching in the
choreography space with aggressive pruning, and (c) reducing machine synthesis
to constraint solving.

---

## 2. Required game-mechanics knowledge

The agent must build (or reuse) an exact model of:

- **Board:** infinite hex grid; axial/cube coordinates; 6 orientations.
- **Parts:**
  - Arms: normal arm (length 1–3), piston arm (extend/retract), track
    (straight/looping paths arms ride on), Van Berlo's wheel (demo puzzles).
    Multi-armed variants (2-, 3-, 6-armed) share one base.
  - Glyphs: bonding, multi-bonding, triplex bonding, unbonding, calcification,
    duplication, projection, purification, animismus, dispersion/unification
    (quintessence), disposal, equilibrium (decorative).
  - I/O: reagent inputs (spawn when clear), product outputs (accept matching
    molecule, need 6 completed products by default; infinite-product puzzles
    exist for custom puzzles).
- **Instructions (per arm, per cycle):** blank, grab, drop, rotate cw/ccw,
  pivot cw/ccw, extend, retract, track forward/back, plus tape control
  (repeat, reset, noop/period). Tape loops; overall cycle count depends on tape
  lengths and loop alignment.
- **Sub-cycle semantics:** order of operations within a cycle (grabs/drops,
  motion, glyph activation, input spawning, output consumption). Critical for
  correctness and for "overlap" tricks. Reference: Grimy's om_overlap README
  (https://github.com/Grimy/om_overlap).
- **Collision detection:** continuous rotational sweeps, hexagonal hitboxes,
  the exact rules the game uses. Reference: Luna's gist (linked from omsim
  README: https://gist.github.com/l-Luna/22a2595755f11d2acf0409f755cea9bb).
- **Metric definitions:**
  - Cost: sum of part costs (arm 20g, glyphs 10–40g, track 5g/hex, etc.).
  - Cycles: cycle on which the final required product is dropped/consumed.
  - Area: number of hexes ever swept/occupied by any atom/part during the run.
  - Instructions: total non-blank instructions across all tapes.
  - **[VERIFY]** exact edge cases (e.g. area contribution of arms vs atoms,
    track hexes) against omsim's implementation rather than folklore.

Do **not** re-derive these rules from the game by experiment; omsim encodes
them and is the ground truth for this project.

---

## 3. Tools and resources

### 3.1 omsim — the verifier (most important dependency)
- Repo: https://github.com/ianh/omsim (C).
- Simulates `.puzzle` + `.solution` pairs; reproduces game behavior including
  sub-cycle ordering and collision; used by the community leaderboards and
  tournaments as the source of truth.
- `make libverify.so` builds a shared library with an API **designed for bots**
  — see `verifier.h` in the repo for documentation. This is the intended
  integration point: link/FFI to libverify for fast programmatic
  verify-and-score of candidate solutions.
- Also computes steady-state throughput/"rate" metrics used in tournaments.
- Known gaps (per README, may be outdated): conduit "spooky action at a
  distance" not implemented; some track-reset edge cases differ. **[VERIFY]**
  current status.
- Repo also contains/points to `.puzzle` files for the built-in campaign
  puzzles (transcribed by community members) — use these as the test corpus.

### 3.2 File formats (shared with the game and OpusSolver)
- **`.puzzle`** and **`.solution`** are Zachtronics' binary formats, read and
  written by the game itself. omsim parses both (C parser in its source);
  OpusSolver (C#) also reads `.puzzle` and writes `.solution`.
- Community-written parsers/specs exist in several languages (omsim's source
  is the most battle-tested reference implementation). Recommended approach:
  either FFI into omsim's parser or port its parsing code, rather than
  reverse-engineering the format independently.
- Solutions written to the game's save directory
  (`Documents/My Games/Opus Magnum/<steamID>/`) load in-game; F10 reloads.
  Custom/workshop puzzle locations are documented in the OpusSolver README.
- **[VERIFY]** format version quirks (e.g. instruction encoding, part naming
  strings) by round-tripping: write a solution, have omsim simulate it, and
  load it in the game.

### 3.3 OpusSolver — existing constructive autosolver
- Repo: https://github.com/gtw123/OpusSolver (C#, command line;
  generates `.solution` from `.puzzle`). Uses lp_solve (ILP) and validates
  outputs via omsim/libverify.
- Useful to us for: `.puzzle`/`.solution` I/O code, the ILP "recipe" idea,
  the ArmController A* motion planner, and as a baseline solver to compare
  scores against.

### 3.4 Community knowledge & data
- Leaderboards / Pareto frontiers per puzzle and category:
  https://zlbb.faendir.com/ — gives target scores and (via linked records)
  example elite solutions. Best-known scores double as upper bounds and as
  regression targets.
- biggiemac42's blog (https://biggieblog.com/) — deep dives on cycle lower
  bounds (throughput + latency arguments), cost techniques (input suppression,
  partial glyph access), and the history of proving/achieving minimum cycles
  on all 36 campaign puzzles. These posts are effectively the spec for our
  analytic lower-bound module (§7.4).
- Grimy's om_overlap (sub-cycle ordering, overlap tricks), Luna's collision
  doc, jinyou's solution archive (large corpus of real solutions —
  http://jinyou.byethost5.com/ **[VERIFY]** still online; omsim README links it).
- The OM community Discord is where record verification bots run; not needed
  for implementation but useful context.

---

## 4. How OpusSolver works (high-level), for contrast

Constructive, template-based; **no global search**. Pipeline:

1. **Recipe (stoichiometry as ILP):** determine needed reactions from element
   mismatch between reagents and products; solve integer linear equations
   (one per element; variables = counts of reagents/reactions/products) with
   lp_solve; retry with scale factors 2–6; relax to inequalities (allowing
   waste atoms) if needed.
2. **Solution plan:** choose disassembler/assembler types per molecule based
   on shape/size; fix the order elements are produced/consumed.
3. **Element pipeline:** abstract generators (one per reagent / reaction /
   product) arranged in reaction order.
4. **Command dry-run:** output generator requests product atoms one at a time;
   requests propagate up the pipeline; produces an abstract command sequence
   (learn buffering needs, waste atoms, purification depth) with no geometry.
5. **Atom generators:** each abstract generator becomes concrete parts
   (arms/glyphs/track) + arm programs.
6. **Assemble program fragments; strip unused parts; write `.solution`.**

**Low-cost variant specifics:** a single main arm on a (preferably looping)
track at the center; generators are mostly single glyphs placed around it;
track = path through the union of all generators' access points (heuristics
favor loops and straight segments); product drop locations found by local
brute force over reachable, non-overlapping cells; arm motion planned by a
modified A* over (track cell, rotation, molecule pose, holding) states that
avoids collisions, unwanted bonds/reactions; buffer arms store out-of-order
atoms; waste handled via disposal or bonded waste chains. `--optimize` is a
**parameter sweep** (recipe, orderings, arm length, per-generator tweaks),
validating every combination with libverify and keeping the best.

Takeaway: OpusSolver = one rigid macro-architecture + ILP chemistry + local
A*. Fast and general, but structurally incapable of record scores (it always
fully disassembles, uses one arm, no overlap/pipelining tricks).

---

## 5. Our method: choreography-first search + machine synthesis

Core idea (from prior discussion): instead of enumerating instruction tapes
(12^(arms×cycles)), search over **molecule choreographies** and then
**synthesize** the machinery that realizes a choreography — the synthesis step
is nearly forced and reduces to constraint solving.

### 5.1 Representation
- A **choreography** = for every atom/molecule, its pose (position +
  orientation + bond structure) at every cycle, plus glyph events
  (bond formed at hexes (a,b) at cycle t, calcification at hex h at t, ...).
- Legal per-cycle molecule motions form a small closed set:
  - stationary (possibly waiting),
  - rotation by ±60° about some hex (an arm base),
  - pivot by ±60° about a grabbed atom,
  - straight-line translation by 1 hex (arm riding track; or piston
    extend/retract along the arm axis),
  - spawn (input) / despawn (output, disposal).
- This makes the choreography a path in a well-defined transition system.

### 5.2 Phase A — choreography search
- Search molecule trajectories from inputs to products under a score budget
  (cycles bound ⇒ horizon; area bound ⇒ bounding region; cost bound ⇒ limits
  on distinct glyph events / arm count implied later).
- Pruning:
  - **Throughput lower bounds** (inputs supply ≤ 1 atom / 2 cycles per input
    glyph; product needs N atoms ⇒ hard cycle floor) — kill timing branches.
  - **Symmetry canonicalization** (board rotations/reflections, molecule
    symmetries, time-shift of independent streams).
  - **Reachability/dominance:** discard partial choreographies dominated by
    another (same frontier state, ≤ cost commitments, ≤ time).
  - **Arm-realizability filter:** prune any motion event that no arm
    configuration could cause given already-committed geometry.
- Optionally seed with human/leaderboard blueprints ("paths given, timing
  free") — then Phase A reduces to **wait/delay insertion**, a small
  scheduling search (this is the user's original suggestion and the cheapest
  high-value mode).

### 5.3 Phase B — machine synthesis (CSP/SAT/ILP)
Given a choreography:
- **Glyphs are implied directly** by glyph events (a bond at (a,b) ⇒ bonder
  occupies exactly those hexes, oriented accordingly; similarly for all
  transmutation glyphs). Dedupe shared glyphs across events.
- **Arms:** each motion event has a small candidate set of explanations
  (arm base hex, arm length, grabbed atom, arm type, track segment). Solve an
  assignment problem: one consistent arm per event chain, no arm doing two
  things in one cycle, arm pose must evolve legally between its events
  (including un-shown repositioning moves in its idle cycles — these must
  also be collision-checked), grabs/drops inserted where hand-offs occur.
- Objective: minimize cost (arms + glyphs + track) or instructions, subject
  to realizing the choreography. Encode as SAT/ILP or specialized search.
- Emit instruction tapes from the assignment.

### 5.4 Phase C — verification & scoring
- Serialize to `.solution`; run through **libverify** for exact
  legality + metrics (never trust our own simulator for final answers; use an
  internal fast simulator only for pruning, and cross-check it against omsim
  on a corpus).
- Feed scores back: Pareto-archive of (cost, cycles, area, instructions);
  compare against zlbb best-knowns.

### 5.5 Optimality-proving mode (stretch goal; be honest about limits)
- Only attempt on tiny puzzles or tight sub-questions, e.g.:
  - "Given this exact layout, is there a tape ≤ c cycles?" (tape search with
    fixed geometry — feasible for small c),
  - "Is there any choreography meeting cycle bound B?" for very small atom
    counts/horizons, with exhaustive canonicalized Phase-A enumeration.
- Combine with analytic lower bounds (throughput, latency, part-count cost
  floors, geometric area floors). A metric is *proven optimal* when analytic
  lower bound == best found score, which is how the community proved all
  campaign cycle records; replicate those arguments programmatically as a
  lower-bound module.
- Do **not** attempt whole-puzzle exhaustive proofs at campaign scale; the
  choreography space is even larger than the tape space (~15^(molecules ×
  cycles)); it only wins when heavily pruned/seeded.

---

## 6. Suggested milestones for the implementation plan

1. **Infra:** build omsim + libverify; FFI bindings; parse/write `.puzzle`
   and `.solution` (reuse omsim/OpusSolver code); round-trip tests on the
   campaign corpus; score extraction.
2. **Internal fast simulator** (incremental, prunable, sub-cycle-accurate),
   differential-tested against omsim on thousands of solutions (jinyou
   archive / OpusSolver outputs / leaderboard solutions if obtainable).
3. **Lower-bound module:** throughput/latency cycle floors, cost floors
   (mandatory parts), naive area floors; report per-puzzle bounds vs zlbb
   best-knowns.
4. **Phase B synthesizer** first (it's the well-posed subproblem): input =
   hand-written choreography for a trivial puzzle, output = valid solution.
   Validate on 1–2 campaign puzzles by transcribing a known solution's
   choreography.
5. **Timing search** ("paths given, waits free") on top of Phase B.
6. **Phase A choreography search** with pruning; start with single-product,
   few-atom custom puzzles; scale ambitions by evidence.
7. **Pareto archive + benchmark harness** against leaderboard scores;
   (optional) exhaustive mode for micro-puzzles.

---

## 7. Risks / open questions

- Sub-cycle ordering and collision rules are the #1 correctness risk; treat
  Grimy's and Luna's docs + omsim source as normative and test heavily.
- Overlap/exotic tricks (allowed part overlaps in custom settings, input
  suppression, partial glyph access) expand the space; decide early whether
  v1 targets "vanilla" semantics only.
- Arm idle-time repositioning makes Phase B harder than pure assignment
  (arms must legally travel between their duty cycles) — likely needs its own
  path planner inside the CSP.
- Track, piston arms, and multi-arms enlarge the candidate-explanation sets in
  Phase B; consider staging support (fixed arms → +track → +pistons →
  +multiarms).
- Looping tapes: cycles metric depends on periodic tape structure; the
  choreography for "6 products" is usually periodic — exploit periodicity
  instead of unrolling the full horizon.
- Licensing/attribution: omsim, OpusSolver, om_overlap are community projects;
  check licenses before vendoring code.
