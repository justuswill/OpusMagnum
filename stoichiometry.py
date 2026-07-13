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


def build_reactions(pf: PuzzleFile) -> List[Reaction]:
    """All transformation-glyph reactions usable in this puzzle."""
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

    return reactions


@dataclass
class RecipeResult:
    status: str
    reagent_counts: Dict[int, int] = field(default_factory=dict)   # input index -> count needed
    reagent_group_size: Dict[int, int] = field(default_factory=dict)  # input index -> # identical inputs it stands in for
    reaction_counts: Dict[str, int] = field(default_factory=dict)  # reaction name -> times invoked
    waste: Dict[int, int] = field(default_factory=dict)            # atom_type -> surplus produced


def solve_recipe(pf: PuzzleFile) -> RecipeResult:
    """
    Solve for the cheapest (fewest reagents, then fewest reactions, then
    least waste) integer recipe that supplies every atom the required
    outputs demand, given the puzzle's available transformation glyphs.
    """
    products_needed = pf.products_needed()

    demand: Dict[int, int] = {}
    for mol in pf.outputs:
        for atype, cnt in mol.atom_type_counts().items():
            # ATOM_REPEAT is a structural marker for repeating/polymer output
            # molecules (the length of the actual chain isn't fixed by the
            # puzzle file), not a real atom — exclude it from the balance.
            if atype == ATOM_REPEAT:
                continue
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
    reactions = build_reactions(pf)

    all_types = (set(demand)
                 | {t for atoms in reagent_atoms.values() for t in atoms}
                 | {t for r in reactions for t in r.delta})

    prob = pulp.LpProblem("recipe", pulp.LpMinimize)

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
        prob += supply + produced == demand.get(t, 0) + waste[t]

    for r in reactions:
        if r.catalyst is not None:
            prob += y[r.name] <= big_m * has_catalyst[r.catalyst]
    for t, hc in has_catalyst.items():
        raw_supply = pulp.lpSum(x[i] * reagent_atoms[i].get(t, 0) for i in reagent_atoms)
        prob += raw_supply >= hc

    # Lexicographic-ish objective: fewest reagents spawned first, then
    # fewest reaction invocations, then least wasted atoms.
    prob += (
        1_000_000 * pulp.lpSum(x.values())
        + 1_000 * pulp.lpSum(y.values())
        + pulp.lpSum(waste.values())
    )

    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    status_str = pulp.LpStatus[status]
    if status_str != "Optimal":
        raise ValueError(f"no atom-balanced recipe found for {pf.name!r} (status={status_str})")

    def val(v):
        return int(round(pulp.value(v) or 0))

    return RecipeResult(
        status=status_str,
        reagent_counts={i: val(v) for i, v in x.items()},
        reagent_group_size=reagent_group_size,
        reaction_counts={name: val(v) for name, v in y.items() if val(v) > 0},
        waste={t: val(v) for t, v in waste.items() if val(v) > 0},
    )


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
