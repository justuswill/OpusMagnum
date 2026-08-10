"""
stoichiometry.py — recipe (stoichiometry) solver for Opus Magnum puzzles.

Step 1 of the OpusSolver-style pipeline (PLAN.md §4.1): before any
geometry/choreography is decided, figure out how many of each reagent, and
how many times each transformation glyph, are needed to supply every atom
the required outputs demand. Atom types are conserved quantities and
transformation glyphs are fixed-ratio reactions between them; the resulting
integer program is solved with an ILP solver (pulp/CBC), mirroring
OpusSolver's use of lp_solve for the same "recipe" step.

Reaction stoichiometry is read out of omsim/sim.c (the ground truth):
  - calcification:   elemental -> salt
  - duplication:     salt -> elemental, needs a never-consumed elemental catalyst
  - projection:      metal[N] + quicksilver -> metal[N+1]
  - purification:    2x metal[N] -> metal[N+1]
  - animismus:       2x salt -> vitae + mors
  - dispersion:      quintessence -> air+earth+fire+water
  - unification:     air+earth+fire+water -> quintessence
  - rejection:       metal[N] -> metal[N-1] + quicksilver
  - division:        metal[N] -> metal[ceil(N/2)] + metal[floor(N/2)]
  - proliferation:   metal[N] + quicksilver -> 2x metal[N], needs a never-consumed elemental catalyst

Van Berlo's/Ravari's wheels permanently hold one atom of each type in their
pool, frozen in place and never consumed; only for duplication/proliferation.
When present, the catalyst for those pool types is free/unlimited; otherwise
duplication/proliferation still work, but require >=1 atom of the catalyst
type sourced from a reagent (a single reserved atom, held as a permanent
template — it is never consumed, so it costs nothing beyond that one unit
regardless of how many times the reaction fires).

This step ignores geometry and cycle timing entirely (that's cycles_lower_bound
and later phases); it only asks "do the quantities balance."

ASSUMPTIONS (for now):
Optimal solutions
- achieve minimal sum over inputs used with a unique count allocated to each input (cant trade off input counts)
- don't use 100 more reactions than the minimum required
- can't save more than 1000 reactions with one additional input
"""

import warnings
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

import pulp

from puzzle_parser import (
    PuzzleFile,
    ATOM_NAMES, ATOM_SALT, ATOM_AIR, ATOM_EARTH, ATOM_FIRE, ATOM_WATER,
    ATOM_QUICKSILVER, ATOM_GOLD, ATOM_SILVER, ATOM_COPPER, ATOM_IRON,
    ATOM_TIN, ATOM_LEAD, ATOM_VITAE, ATOM_MORS, ATOM_QUINTESSENCE, ATOM_REPEAT,
    ELEMENTALS,
    PART_CALCIFICATION, PART_DUPLICATION, PART_PROJECTION, PART_PURIFICATION,
    PART_ANIMISMUS, PART_DISPERSION, PART_BARON, PART_RAVARI,
    PART_REJECTION, PART_DIVISION, PART_PROLIFERATION,
    BARON_WHEEL_ATOMS, RAVARI_WHEEL_ATOMS,
)

# Metal purity chain, least- to most-refined (matches the bit-shift direction
# used by omsim's projection/purification/rejection/division: lead is the
# base metal, gold the final one).
METAL_CHAIN = [ATOM_LEAD, ATOM_TIN, ATOM_IRON, ATOM_COPPER, ATOM_SILVER, ATOM_GOLD]
METAL_STEPS = list(zip(METAL_CHAIN, METAL_CHAIN[1:]))  # adjacent (lo, hi) pairs

# Purification (2:1) is the only amplifying reaction; climbing the full
# lead->gold chain needs up to 2**(len(chain)-1)x the top-tier demand. Bounds
# how large a catalytic reaction's firing count can get — sizes big-M below.
MAX_CHAIN_AMPLIFICATION = 2 ** (len(METAL_CHAIN) - 1)

# Fixed division outputs per input metal, derived from sim.c's bit-shift
# split (input metal -> two lower-tier metals). Lead excluded as input:
# there's nothing lower to split it into.
DIVISION_OUTPUTS = {
    ATOM_GOLD:   (ATOM_IRON, ATOM_IRON),
    ATOM_SILVER: (ATOM_TIN, ATOM_IRON),
    ATOM_COPPER: (ATOM_TIN, ATOM_TIN),
    ATOM_IRON:   (ATOM_LEAD, ATOM_TIN),
    ATOM_TIN:    (ATOM_LEAD, ATOM_LEAD),
}


@dataclass
class Reaction:
    name: str
    delta: Dict[int, int]            # atom_type -> net change per invocation
    catalyst: Optional[int] = None   # atom_type that must be present (unconsumed) but isn't free
    alt_reagent: Optional[List[int]] = None  # see solve_recipe_combined: a
    # combined "calcify_X_or_Y" synthetic reaction has no single fixed
    # reagent-side atom type — delta only carries its produced side
    # (+1 salt); this lists the alternative atom types schematic.py's
    # backward search may freely choose among when reversing one firing.


def build_reactions(pf: PuzzleFile, excluded_reactions: frozenset = frozenset()) -> List[Reaction]:
    """All transformation-glyph reactions usable in this puzzle.

    `excluded_reactions` drops specific named reactions even when their
    PART bit is set — needed for dispersion/unification, which share
    PART_DISPERSION in the game's own puzzle format (see that constant's
    "also unification" comment) despite being fully independent
    reactions; there's no other way to test "what if only unification
    were available" since the bit can't be split."""
    avail = pf.parts_available
    reactions: List[Reaction] = []

    if avail & PART_CALCIFICATION:
        for e in ELEMENTALS:
            reactions.append(Reaction(f"calcify_{ATOM_NAMES[e]}", {e: -1, ATOM_SALT: +1}))

    if avail & PART_DUPLICATION:
        for e in ELEMENTALS:
            free = bool(avail & PART_BARON) and e in BARON_WHEEL_ATOMS
            reactions.append(Reaction(
                f"duplicate_{ATOM_NAMES[e]}", {ATOM_SALT: -1, e: +1},
                catalyst=None if free else e,
            ))

    if avail & PART_PROJECTION:
        for lo, hi in METAL_STEPS:
            reactions.append(Reaction(
                f"project_{ATOM_NAMES[lo]}_to_{ATOM_NAMES[hi]}",
                {lo: -1, ATOM_QUICKSILVER: -1, hi: +1},
            ))

    if avail & PART_PURIFICATION:
        for lo, hi in METAL_STEPS:
            reactions.append(Reaction(
                f"purify_{ATOM_NAMES[lo]}_to_{ATOM_NAMES[hi]}",
                {lo: -2, hi: +1},
            ))

    if avail & PART_ANIMISMUS:
        reactions.append(Reaction(
            "animismus", {ATOM_SALT: -2, ATOM_VITAE: +1, ATOM_MORS: +1},
        ))

    if avail & PART_DISPERSION:
        reactions.append(Reaction(
            "dispersion",
            {ATOM_QUINTESSENCE: -1, ATOM_AIR: +1, ATOM_EARTH: +1, ATOM_FIRE: +1, ATOM_WATER: +1},
        ))
        reactions.append(Reaction(
            "unification",
            {ATOM_AIR: -1, ATOM_EARTH: -1, ATOM_FIRE: -1, ATOM_WATER: -1, ATOM_QUINTESSENCE: +1},
        ))

    if avail & PART_REJECTION:
        for lo, hi in METAL_STEPS:
            reactions.append(Reaction(
                f"reject_{ATOM_NAMES[hi]}_to_{ATOM_NAMES[lo]}",
                {hi: -1, lo: +1, ATOM_QUICKSILVER: +1},
            ))

    if avail & PART_PROLIFERATION:
        for m in METAL_CHAIN:
            free = bool(avail & PART_RAVARI) and m in RAVARI_WHEEL_ATOMS
            reactions.append(Reaction(
                f"proliferate_{ATOM_NAMES[m]}", {ATOM_QUICKSILVER: -1, m: +1},
                catalyst=None if free else m,
            ))

    if avail & PART_DIVISION:
        for m, (a, b) in DIVISION_OUTPUTS.items():
            delta = {m: -1}
            delta[a] = delta.get(a, 0) + 1
            delta[b] = delta.get(b, 0) + 1
            reactions.append(Reaction(f"divide_{ATOM_NAMES[m]}", delta))

    return [r for r in reactions if r.name not in excluded_reactions]


@dataclass
class RecipeResult:
    status: str
    reagent_counts: Dict[int, int] = field(default_factory=dict)   # input index -> count needed
    reagent_group_size: Dict[int, int] = field(default_factory=dict)  # input index -> # identical inputs it stands in for
    reaction_counts: Dict[str, int] = field(default_factory=dict)  # reaction name -> times invoked
    waste: Dict[int, int] = field(default_factory=dict)            # atom_type -> surplus produced
    extra_reactions: Dict[str, Reaction] = field(default_factory=dict)  # combined synthetic reactions (see solve_recipe_combined) that reaction_counts references but build_reactions doesn't define


def _build_recipe_lp(pf: PuzzleFile, prioritize_waste: bool = False, allow_extra_output_copies: bool = False,
                      excluded_reactions: frozenset = frozenset()):
    """Shared ILP construction for solve_recipe/solve_recipe_alternatives —
    see solve_recipe's docstring for the model. Returns
    (prob, x, y, waste, reagent_group_size, big_m), unsolved.

    `prioritize_waste` reorders the lexicographic objective to minimize
    waste first (see solve_recipe_min_waste) instead of last.

    `allow_extra_output_copies` adds one non-negative slack variable per
    *output molecule* (not per atom type) — how many extra whole copies of
    that molecule get delivered beyond the required demand, for free. A
    puzzle's output zones happily accept more than the minimum required
    deliveries, so an extra copy was never actually "waste" needing
    disposal. Crucially this is a whole-molecule slack: an extra copy of a
    bonded multi-atom output (e.g. one quicksilver bonded to one gold)
    pulls in an extra unit of *every* atom in it together, at that
    molecule's fixed ratio — it can't be used to launder one atom type's
    shortfall by overproducing only the *other* atom type in the same
    output molecule for free, the way a flat "any output atom type is free"
    rule could (see the P095 case in bounds.py's caller: quicksilver and
    gold are bonded 1:1 in the one output molecule, so an extra copy always
    costs an extra gold too — it can't hide arbitrary extra quicksilver on
    its own). Independent of `prioritize_waste` — only solve_recipe_min_waste
    turns it on, never solve_recipe/solve_recipe_combined's exact-demand
    accounting used elsewhere."""
    products_needed = pf.products_needed()

    demand: Dict[int, int] = {}
    for mol in pf.outputs:
        for atype, cnt in mol.atom_type_counts().items():
            # ATOM_REPEAT is a structural marker for repeating/polymer output
            # molecules (the length of the actual chain isn't fixed by the
            # puzzle file), not a real atom the puzzle grants a reagent for —
            # but it still occupies a real position/bonds in the output
            # molecule that a backward search has to reduce down to
            # something, so it's balanced like any other atom against the
            # single-atom repeat reagent parse_puzzle synthesizes for it.
            demand[atype] = demand.get(atype, 0) + cnt * products_needed

    # Reagents with identical atom composition are fungible — collapse each
    # group to a single variable keyed by its first index, so the LP doesn't
    # have to break an arbitrary tie between, say, two identical "water"
    # inputs (which would otherwise look like a meaningful choice).
    signature_of = {i: tuple(sorted(mol.atom_type_counts().items()))
                     for i, mol in enumerate(pf.inputs)}
    representative: Dict[tuple, int] = {}
    sig_counts: Dict[tuple, int] = {}
    for i, sig in signature_of.items():
        representative.setdefault(sig, i)
        sig_counts[sig] = sig_counts.get(sig, 0) + 1
    reagent_atoms = {i: pf.inputs[i].atom_type_counts() for i in set(representative.values())}
    reagent_group_size = {i: sig_counts[sig] for sig, i in representative.items()}
    reactions = build_reactions(pf, excluded_reactions)

    all_types = (set(demand)
                 | {t for atoms in reagent_atoms.values() for t in atoms}
                 | {t for r in reactions for t in r.delta})

    prob = pulp.LpProblem("recipe", pulp.LpMinimize)

    extra_copies = {}
    if allow_extra_output_copies:
        extra_copies = {j: pulp.LpVariable(f"extra_out_{j}", lowBound=0, cat="Integer")
                         for j in range(len(pf.outputs))}

    x = {i: pulp.LpVariable(f"reagent_{i}", lowBound=0, cat="Integer") for i in reagent_atoms}
    y = {r.name: pulp.LpVariable(f"reaction_{r.name}", lowBound=0, cat="Integer") for r in reactions}
    waste = {t: pulp.LpVariable(f"waste_{t}", lowBound=0, cat="Integer") for t in all_types}

    # A catalytic reaction (duplication/proliferation without a wheel) needs
    # one atom of its catalyst type reserved from raw reagent supply — that
    # atom is never consumed, so it costs nothing beyond being present once.
    catalyst_types = {r.catalyst for r in reactions if r.catalyst is not None}
    has_catalyst = {t: pulp.LpVariable(f"has_catalyst_{t}", cat="Binary") for t in catalyst_types}
    big_m = MAX_CHAIN_AMPLIFICATION * max(1, sum(demand.values())) + 100

    for t in all_types:
        supply = pulp.lpSum(x[i] * reagent_atoms[i].get(t, 0) for i in reagent_atoms)
        produced = pulp.lpSum(y[r.name] * r.delta.get(t, 0) for r in reactions)
        extras = pulp.lpSum(
            extra_copies[j] * pf.outputs[j].atom_type_counts().get(t, 0) for j in extra_copies
        )
        prob += supply + produced == demand.get(t, 0) + waste[t] + extras

    for r in reactions:
        if r.catalyst is not None:
            prob += y[r.name] <= big_m * has_catalyst[r.catalyst]
    for t, hc in has_catalyst.items():
        raw_supply = pulp.lpSum(x[i] * reagent_atoms[i].get(t, 0) for i in reagent_atoms)
        # A catalyst template doesn't have to come from a raw reagent — any
        # atom of that type the recipe produces along the way (e.g. a metal
        # reached via rejection) can just as well be the one held back and
        # never sent on. Added, never subtracted, so this only ever adds an
        # alternative way to satisfy hc — it can't make an already-feasible
        # raw-reagent-sourced catalyst infeasible, only free ones with no
        # raw supply of that type at all (duplication with no Baron wheel,
        # proliferation with no Ravari wheel — the only reactions with a
        # catalyst in the first place).
        produced = pulp.lpSum(y[r.name] * r.delta.get(t, 0) for r in reactions if r.delta.get(t, 0) > 0)
        prob += raw_supply + produced >= hc

    if prioritize_waste:
        # Least wasted atoms first, then fewest reactions, then fewest
        # reagents — used to test whether a *restricted* reaction set can
        # balance demand with zero waste at all, regardless of how many
        # reagents/reactions that costs.
        prob += (
            1_000_000 * pulp.lpSum(waste.values())
            + 1_000 * pulp.lpSum(y.values())
            + pulp.lpSum(x.values())
        )
    else:
        # Lexicographic-ish objective: fewest reagents spawned first, then
        # fewest reaction invocations, then least wasted atoms.
        prob += (
            1_000_000 * pulp.lpSum(x.values())
            + 1_000 * pulp.lpSum(y.values())
            + pulp.lpSum(waste.values())
        )

    return prob, x, y, waste, reagent_group_size, big_m


def _recipe_result(x, y, waste, reagent_group_size) -> RecipeResult:
    def val(v):
        return int(round(pulp.value(v) or 0))

    return RecipeResult(
        status="Optimal",
        reagent_counts={i: val(v) for i, v in x.items()},
        reagent_group_size=reagent_group_size,
        reaction_counts={name: val(v) for name, v in y.items() if val(v) > 0},
        waste={t: val(v) for t, v in waste.items() if val(v) > 0},
    )


def solve_recipe(pf: PuzzleFile) -> RecipeResult:
    """
    Solve for the cheapest (fewest reagents, then fewest reactions, then
    least waste) integer recipe that supplies every atom the required
    outputs demand, given the puzzle's available transformation glyphs.

    The objective only minimizes *totals* — it's indifferent to WHICH
    interchangeable reagent/reaction absorbs a given count (e.g. calcify_air
    vs calcify_fire when several elemental inputs are available), so CBC
    returns one arbitrary optimal recipe among possibly many. See
    solve_recipe_alternatives to enumerate the others.
    """
    prob, x, y, waste, reagent_group_size, _big_m = _build_recipe_lp(pf)
    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    status_str = pulp.LpStatus[status]
    if status_str != "Optimal":
        raise ValueError(f"no atom-balanced recipe found for {pf.name!r} (status={status_str})")

    return _recipe_result(x, y, waste, reagent_group_size)


def solve_recipe_min_waste(pf: PuzzleFile, excluded_reactions: frozenset = frozenset()) -> RecipeResult:
    """solve_recipe's model, but prioritizing least waste over fewest
    reagents/reactions — used by bounds.py to test whether a *restricted*
    set of reaction glyphs (typically just the ones already proven
    necessary) can balance demand with zero waste at all. A puzzle where
    every zero-waste recipe needs a reaction outside that restricted set
    still solves here (waste absorbs the imbalance instead), just with
    waste > 0 — the caller reads that as "restricted set is insufficient."

    Extra whole copies of an output molecule are free, never counted as
    waste (see _build_recipe_lp's allow_extra_output_copies) — only a
    genuine byproduct with no matching output, or a shortfall that can't be
    covered without breaking some other output molecule's fixed atom
    ratio, can register here.

    `excluded_reactions` (see build_reactions) lets a caller drop a
    specific named reaction even though its PART bit is set — the only
    way to test dispersion and unification independently, since they
    share PART_DISPERSION.
    """
    prob, x, y, waste, reagent_group_size, _big_m = _build_recipe_lp(
        pf, prioritize_waste=True, allow_extra_output_copies=True,
        excluded_reactions=excluded_reactions)
    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    status_str = pulp.LpStatus[status]
    if status_str != "Optimal":
        raise ValueError(f"no atom-balanced recipe found for {pf.name!r} (status={status_str})")

    return _recipe_result(x, y, waste, reagent_group_size)


def solve_recipe_alternatives(pf: PuzzleFile, limit: int = 20) -> List[RecipeResult]:
    """Every distinct optimal recipe (same total objective — same
    reagent/reaction/waste totals, just a different split across
    interchangeable choices) up to `limit`, not just the one arbitrary
    recipe solve_recipe returns — see its docstring for why more than one
    can exist. This matters beyond stoichiometry: schematic.py's backward
    search and bounds.py's L_spine computation are sensitive to WHICH
    specific reagents/reactions a recipe uses, so a different (equally
    cheap) recipe can yield a different — sometimes sounder — bound.

    Found by solving once to get the optimal objective value, pinning the
    objective to that value (so every subsequent solve is forced to stay
    at the same optimum rather than degrade to the next-best), then
    repeatedly re-solving with a "no-good" cut that forces at least one
    reagent/reaction variable to differ from every previously found
    solution — stops when no more exist (infeasible) or `limit` is hit."""
    prob, x, y, waste, reagent_group_size, big_m = _build_recipe_lp(pf)
    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[status] != "Optimal":
        raise ValueError(f"no atom-balanced recipe found for {pf.name!r} (status={pulp.LpStatus[status]})")

    def val(v):
        return int(round(pulp.value(v) or 0))

    best_objective = pulp.value(prob.objective)
    prob += prob.objective <= best_objective  # pin every future solve to this same optimum

    cut_vars = list(x.values()) + list(y.values())

    results = []
    for n in range(limit):
        status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
        if pulp.LpStatus[status] != "Optimal":
            break

        results.append(_recipe_result(x, y, waste, reagent_group_size))

        # No-good cut: force at least one reagent/reaction variable to
        # differ from this solution before the next solve. A single "diff"
        # binary per variable isn't enough — big-M only forces v==s when
        # diff==0, it doesn't force v!=s when diff==1 (the solver can just
        # leave v unchanged and still satisfy the cut, which is exactly why
        # the first version of this returned the same solution repeatedly).
        # Two binaries per variable, asserting v>s or v<s respectively when
        # set, correctly forces a genuine difference.
        diffs = []
        for idx, v in enumerate(cut_vars):
            s = val(v)
            up = pulp.LpVariable(f"_up_{n}_{idx}", cat="Binary")
            down = pulp.LpVariable(f"_down_{n}_{idx}", cat="Binary")
            prob += v >= s + 1 - big_m * (1 - up)
            prob += v <= s - 1 + big_m * (1 - down)
            prob += up + down <= 1
            diffs.append(up)
            diffs.append(down)
        prob += pulp.lpSum(diffs) >= 1

    return results


@dataclass
class RecipeFlexibility:
    essential: Dict[str, int]                    # reaction name -> fixed count every optimum needs
    flexible_groups: List[tuple]                  # (frozenset of interchangeable reaction names, combined total count)


def solve_recipe_flexibility(pf: PuzzleFile) -> RecipeFlexibility:
    """Characterizes the optimal recipe's degeneracy without enumerating
    every alternate optimum — solve_recipe_alternatives doesn't scale once
    the flexible count gets into the hundreds (e.g. P030b's 36-way split
    across calcify_air/earth/fire stalls CBC well before it's exhausted).

    Reactions that produce the exact same output atoms (their delta's
    positive side, e.g. every calcify_* reaction produces +1 salt
    regardless of which elemental it consumes) are candidate substitutes
    for each other. For each candidate group with 2+ members, every
    member's *max* achievable count is solved for, subject to the same
    balance constraints plus the objective pinned at its optimal value —
    this determines TRUE group membership (a candidate whose max is 0,
    e.g. calcify_air when no air reagent exists at all, isn't a real
    substitute and gets excluded) independent of whatever the arbitrary
    baseline solve happened to use, unlike testing only reactions the
    baseline already picked (which misses substitutes that started at 0,
    e.g. calcify_water when the baseline happened to use only
    calcify_fire). Groups with 2+ real members are FLEXIBLE — the
    individual split is meaningless, only the GROUP TOTAL (the baseline's
    combined count across the group) is invariant across every optimum.
    Everything else — lone reactions, or a "group" that collapses to one
    real member — is ESSENTIAL, required at exactly its baseline count."""
    prob, x, y, waste, reagent_group_size, _big_m = _build_recipe_lp(pf)
    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[status] != "Optimal":
        raise ValueError(f"no atom-balanced recipe found for {pf.name!r} (status={pulp.LpStatus[status]})")

    def val(v):
        return int(round(pulp.value(v) or 0))

    baseline = {name: val(v) for name, v in y.items()}
    best_objective = pulp.value(prob.objective)
    original_objective = prob.objective
    prob += original_objective <= best_objective  # pin every future solve to this same optimum

    reactions_by_name = {r.name: r for r in build_reactions(pf)}

    def produced_signature(name):
        return tuple(sorted((t, d) for t, d in reactions_by_name[name].delta.items() if d > 0))

    candidate_groups: Dict[tuple, list] = {}
    for name in y:
        candidate_groups.setdefault(produced_signature(name), []).append(name)

    essential = {}
    flexible_groups = []
    with warnings.catch_warnings():
        # Re-pointing prob.objective at a new expression each iteration is
        # deliberate here (see loop below), not a mistake pulp needs to
        # warn about every time.
        warnings.filterwarnings("ignore", message="Overwriting previously set objective")
        for names in candidate_groups.values():
            if len(names) == 1:
                if baseline[names[0]] > 0:
                    essential[names[0]] = baseline[names[0]]
                continue

            usable = []
            for name in names:
                prob += -y[name]  # replaces the objective: maximize this one variable
                st = prob.solve(pulp.PULP_CBC_CMD(msg=0))
                hi = val(y[name]) if pulp.LpStatus[st] == "Optimal" else baseline[name]
                if hi > 0:
                    usable.append(name)

            if len(usable) >= 2:
                flexible_groups.append((frozenset(usable), sum(baseline[n] for n in usable)))
            elif usable and baseline[usable[0]] > 0:
                essential[usable[0]] = baseline[usable[0]]
        prob += original_objective  # restore, tidy but not required after the loop

    return RecipeFlexibility(essential=essential, flexible_groups=flexible_groups)


def solve_recipe_combined(pf: PuzzleFile) -> RecipeResult:
    """solve_recipe's result, but with each flexible reaction group (see
    solve_recipe_flexibility) collapsed into ONE combined synthetic
    reaction (e.g. calcify_water_or_fire) carrying the group's whole
    combined total as its budget, instead of solve_recipe's one arbitrary
    per-member split — so schematic.py's backward search can freely choose
    which alternative atom type to produce at each firing, sharing one
    pool, rather than being stuck with only the specific reagent(s)
    solve_recipe happened to pick (this is what was making some bounds
    unsound — see P021/Courage Potion).

    Only calcify_* groups are supported — a flexible group containing
    anything else raises NotImplementedError, since combining any other
    reaction shape needs its own reversal logic in schematic.py that
    hasn't been written yet."""
    recipe = solve_recipe(pf)
    flex = solve_recipe_flexibility(pf)
    if not flex.flexible_groups:
        return recipe

    reactions_by_name = {r.name: r for r in build_reactions(pf)}
    reaction_counts = dict(recipe.reaction_counts)
    reagent_counts = dict(recipe.reagent_counts)
    extra_reactions = dict(recipe.extra_reactions)

    for names, total in flex.flexible_groups:
        if not all(n.startswith("calcify_") for n in names):
            raise NotImplementedError(
                f"flexible reaction group {sorted(names)} isn't all calcify_* — "
                "combining non-calcification reactions isn't implemented"
            )
        types = sorted(
            t for n in names for t, d in reactions_by_name[n].delta.items() if d < 0
        )
        combined_name = "calcify_" + "_or_".join(ATOM_NAMES[t] for t in types)
        for n in names:
            reaction_counts.pop(n, None)
        reaction_counts[combined_name] = total
        extra_reactions[combined_name] = Reaction(
            name=combined_name, delta={ATOM_SALT: 1}, alt_reagent=types,
        )

        # Rebalance the matching reagent counts proportionally to each
        # member's own duplicate-glyph count, instead of leaving whatever
        # arbitrary (often very lopsided) split solve_recipe's ILP happened
        # to settle on: schematic.py's L_spine computation still keys off
        # individual reagent counts per type even though the reaction
        # budget is now shared, so an unbalanced split needlessly inflates
        # latency for whichever type got overloaded.
        #
        # Only the *calcification-feedstock* portion of each type's count
        # is actually flexible — `total` (this group's reaction-firing
        # count) is exactly that portion, already excluding any atoms of
        # these types the output needs *directly* (bonded in, never
        # calcified). That direct-use amount is fixed per type (driven by
        # the output's own structure, from pf.outputs, not a free choice)
        # and must be added back unchanged, not redistributed — an earlier
        # version of this rebalanced the *whole* reagent total including
        # that fixed portion, which desynced reagent_counts from what any
        # real state in the graph could realize (P025 had 0% of its air
        # count and 100% of its water count as direct-use, so redistributing
        # both as if fully flexible produced an unreachable split).
        member_indices = [
            i for i, mol in enumerate(pf.inputs)
            if i in recipe.reagent_counts
            and len(mol.atom_type_counts()) == 1
            and next(iter(mol.atom_type_counts())) in types
        ]
        if member_indices:
            products_needed = pf.products_needed()
            demand = Counter()
            for mol in pf.outputs:
                for atype, cnt in mol.atom_type_counts().items():
                    demand[atype] += cnt * products_needed
            direct_use = {
                i: demand.get(next(iter(pf.inputs[i].atom_type_counts())), 0)
                for i in member_indices
            }

            repeats = {i: recipe.reagent_group_size.get(i, 1) for i in member_indices}
            total_repeats = sum(repeats.values())
            base = {i: (total * repeats[i]) // total_repeats for i in member_indices}
            remainder = total - sum(base.values())
            # Largest-remainder apportionment: hand the few leftover units
            # (from integer division) to whichever members' exact share
            # rounded down the most, so the total still adds up exactly.
            by_remainder = sorted(
                member_indices,
                key=lambda i: (total * repeats[i]) % total_repeats,
                reverse=True,
            )
            for i in by_remainder[:remainder]:
                base[i] += 1
            for i in member_indices:
                reagent_counts[i] = direct_use[i] + base[i]

    return replace(recipe, reagent_counts=reagent_counts, reaction_counts=reaction_counts,
                    extra_reactions=extra_reactions)


def solve_recipe_cheap(pf: PuzzleFile) -> RecipeResult:
    """solve_recipe's result, but with each flexible reaction group (see
    solve_recipe_flexibility) concentrated onto a single member instead of
    solve_recipe's one arbitrary split — the opposite of
    solve_recipe_combined's proportional-to-duplicate-count spread.

    That spread exists purely to maximize throughput for schematic.py's
    backward search / cycles_lower_bound (more physical reagent zones
    working in parallel means fewer cycles); cost_lower_bound/
    area_lower_bound's access-point count doesn't care about cycles at
    all, and a real solution can draw an arbitrarily large total from a
    single reagent zone over enough cycles, so minimizing the *number of
    distinct reagent zones actually used* is what actually minimizes cost
    — hence concentrating onto as few nodes as possible rather than
    spreading. Unlike solve_recipe_combined, the result needs no combined
    synthetic reaction (no ambiguity is left once concentrated), so this
    works for any flexible group, not just calcify_*."""
    recipe = solve_recipe(pf)
    flex = solve_recipe_flexibility(pf)
    if not flex.flexible_groups:
        return recipe

    reactions_by_name = {r.name: r for r in build_reactions(pf)}
    reaction_counts = dict(recipe.reaction_counts)
    reagent_counts = dict(recipe.reagent_counts)

    products_needed = pf.products_needed()
    demand = Counter()
    for mol in pf.outputs:
        for atype, cnt in mol.atom_type_counts().items():
            demand[atype] += cnt * products_needed

    for names, total in flex.flexible_groups:
        types = sorted(
            t for n in names for t, d in reactions_by_name[n].delta.items() if d < 0
        )
        member_indices = [
            i for i, mol in enumerate(pf.inputs)
            if i in recipe.reagent_counts
            and len(mol.atom_type_counts()) == 1
            and next(iter(mol.atom_type_counts())) in types
        ]
        if not member_indices:
            continue

        direct_use = {
            i: demand.get(next(iter(pf.inputs[i].atom_type_counts())), 0)
            for i in member_indices
        }
        # Prefer a member that already needs a reagent zone regardless
        # (direct_use > 0) — piling the whole flexible total onto it costs
        # no *extra* access point at all. Otherwise fall back to whichever
        # member solve_recipe's own arbitrary split already leaned on
        # most, disturbing the baseline as little as possible.
        chosen = max(member_indices, key=lambda i: (direct_use[i] > 0, recipe.reagent_counts.get(i, 0)))
        for i in member_indices:
            reagent_counts[i] = direct_use[i] + (total if i == chosen else 0)

        type_to_name = {
            next(t for t, d in reactions_by_name[n].delta.items() if d < 0): n
            for n in names
        }
        chosen_type = next(iter(pf.inputs[chosen].atom_type_counts()))
        for n in names:
            reaction_counts[n] = total if n == type_to_name[chosen_type] else 0

    return replace(recipe, reagent_counts=reagent_counts, reaction_counts=reaction_counts)


def necessary_inputs(pf: PuzzleFile, recipe: RecipeResult) -> set:
    """
    Representative input indices whose whole reagent shape is required for
    any atom-balanced recipe to exist at all — unlike recipe.reagent_counts,
    which only reflects one arbitrary optimal recipe solve_recipe happened to
    return.

    Reuses the already-solved `recipe` to skip most of the work instead of
    re-deriving everything from scratch:
      - non-representative duplicate indices (recipe.reagent_group_size has
        no entry for them) are never individually necessary by construction
        — a single reagent glyph is reusable without limit in the ILP, so a
        sibling copy of the same shape is always redundant — never tested.
      - a representative with recipe.reagent_counts[i] == 0 is already
        witnessed as unnecessary: `recipe` itself is a feasible solution that
        uses zero of it, so no new solve is needed to prove that.

    Only representatives recipe actually used (reagent_counts[i] > 0) are
    genuinely ambiguous — the solver could have picked one for cheapness
    rather than necessity — so only those get re-solved, with every copy of
    that shape (duplicates included) removed together: a single duplicate
    alone would never affect feasibility, only throughput, so the whole
    shape has to be dropped at once to test it honestly.

    Interchangeable groups: testing each ambiguous representative alone can
    make every member of a group look individually unnecessary — e.g. water,
    fire, and air are each individually removable given calcification + a
    wheel (any one still lets the others cover its atoms via salt), even
    though at least one of the three is mandatory. Rather than credit that
    group with 0 atoms (technically safe but needlessly loose — see BUG.md-
    style discussion), once a round finds nothing more individually
    necessary, this checks whether the whole remaining ambiguous group can be
    dropped *together*; if not, at least one member is required, so the
    cheapest (fewest atoms) is credited instead of 0, a warning is raised
    (this is exactly the kind of case worth eyeballing per-puzzle), and the
    remaining group is retested (fixing the cheapest one can resolve the rest
    without another joint-removal test).
    """
    def sibling_group(i: int) -> set:
        sig = tuple(sorted(pf.inputs[i].atom_type_counts().items()))
        return {j for j, mol in enumerate(pf.inputs)
                if tuple(sorted(mol.atom_type_counts().items())) == sig}

    def infeasible_without(remove: set) -> bool:
        trimmed = replace(pf, inputs=[m for j, m in enumerate(pf.inputs) if j not in remove])
        try:
            solve_recipe(trimmed)
            return False
        except ValueError:
            return True

    necessary = set()
    pending = [i for i, count in recipe.reagent_counts.items() if count > 0]

    while pending:
        still_ambiguous = []
        for i in pending:
            if infeasible_without(sibling_group(i)):
                necessary.add(i)
            else:
                still_ambiguous.append(i)

        if len(still_ambiguous) == len(pending):
            # No progress this round — everyone left looks individually
            # removable. Check if they're collectively necessary instead.
            if not still_ambiguous:
                break
            remove_all = set().union(*(sibling_group(i) for i in still_ambiguous))
            if not infeasible_without(remove_all):
                break  # truly none of the group is needed
            # technically a heuristic but bounds are still valid if this misses anything
            cheapest = min(still_ambiguous, key=lambda i: pf.inputs[i].atom_count())
            necessary.add(cheapest)
            still_ambiguous.remove(cheapest)

        pending = still_ambiguous

    return necessary


def describe_molecule(mol) -> str:
    counts = mol.atom_type_counts()
    return " + ".join(f"{n}x{ATOM_NAMES.get(t, t)}" for t, n in sorted(counts.items()))
