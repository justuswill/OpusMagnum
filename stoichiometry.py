"""
stoichiometry.py — recipe (stoichiometry) solver for Opus Magnum puzzles.

Figure out how many of each reagent, and how many times each transformation glyph,
are needed to supply every atom the required outputs demand. Atom types are conserved
quantities and transformation glyphs are fixed-ratio reactions between them; the resulting
integer program is solved with an ILP solver (pulp/CBC).

Reaction stoichiometry:
  - calcification:   elemental -> salt
  - duplication:     salt -> elemental, needs a never-consumed elemental catalyst
  - projection:      metal[N] + quicksilver -> metal[N+1]
  - purification:    2x metal[N] -> metal[N+1]
  - animismus:       2x salt -> vitae + mors
  - dispersion:      quintessence -> air+earth+fire+water
  - unification:     air+earth+fire+water -> quintessence
  - rejection:       metal[N] -> metal[N-1] + quicksilver
  - division:        metal[N] -> metal[ceil(N/2)] + metal[floor(N/2)]
  - proliferation:   quicksilver -> metal[N], needs a never-consumed metal[N] catalyst

Van Berlo's/Ravari's wheels permanently hold one atom of each type in their pool.
When present, the catalyst is free; otherwise duplication/proliferation requires
>=1 atom of the catalyst type to be produced.

ASSUMPTIONS:
- Optimal solutions don't contain redundant reactions, e.g. never both calcify_water and duplicate_water.
- Optimal solutions don't contain more net-positive reactions than needed, i.e. proliferate_N + reject_N pairs.

- heuristics big_m / pi_bound / scale are large enough.
"""

import math
from dataclasses import dataclass, field, replace
from os.path import commonprefix
from typing import Dict, List, Optional, Tuple, Union

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
    alternate_repeat_puzzle,
)


METAL_CHAIN = [ATOM_LEAD, ATOM_TIN, ATOM_IRON, ATOM_COPPER, ATOM_SILVER, ATOM_GOLD]
METAL_STEPS = list(zip(METAL_CHAIN, METAL_CHAIN[1:]))
MAX_CHAIN_AMPLIFICATION = 2 ** (len(METAL_CHAIN) - 1)
DIVISION_OUTPUTS = {
    ATOM_GOLD:   (ATOM_IRON, ATOM_IRON),
    ATOM_SILVER: (ATOM_TIN, ATOM_IRON),
    ATOM_COPPER: (ATOM_TIN, ATOM_TIN),
    ATOM_IRON:   (ATOM_LEAD, ATOM_TIN),
    ATOM_TIN:    (ATOM_LEAD, ATOM_LEAD),
}


@dataclass(eq=False)
class Reaction:
    name: str
    delta: Dict[int, int]            # atom_type -> net change per invocation
    catalyst: Optional[int] = None   # atom_type that must be present (unconsumed) but isn't free
    # (real_reaction_name, extra_delta) per alternative
    alternatives: Optional[List[Tuple[str, Dict[int, int]]]] = None


def build_reactions(pf: PuzzleFile, excluded_reactions: frozenset = frozenset()) -> List[Reaction]:
    """
    All transformation-glyph reactions usable in this puzzle, if they are not manually excluded.
    """
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
    # input index -> (min, max) count needed.
    reagent_counts: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    # input index -> # identical inputs it stands in for
    reagent_group_size: Dict[int, int] = field(default_factory=dict)
    # atom type -> (min, max) surplus produced.
    waste: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    # reaction -> (min, max) times invoked.
    reaction_counts: Dict[Reaction, Tuple[int, int]] = field(default_factory=dict)


def _build_recipe_lp(pf: PuzzleFile, objective: Union[None, str, tuple] = None,
                      pin_max: Optional[int] = None, allow_extra_output_copies: bool = False,
                      excluded_reactions: frozenset = frozenset(), constraints: bool = True):
    """
    Shared ILP construction for solve_recipe and friends — see
    solve_recipe's docstring for the model. Returns
    (prob, x, y, waste, reagent_group_size, pi), unsolved.

    `objective` selects what the LP optimizes (or, for the feasibility
    forms, just constrains — no objective is set at all, so callers that
    only check prob.solve()'s status get there without paying for an
    optimality proof):
      - None (default): most balanced reagent use first.
      - ("min"|"max"|"max>0", name_or_idx): minimize/maximize one reaction's (name), reagent's (int index),
                or waste type's (("waste", atype)) own count alone, "max>0" only checks feasibility.
      - ("waste", allowed_types, n): can non-allowed waste be bounded <= n?
      - ("min_zones",): minimize the number of reagents.
      - ("min_atoms",): minimize the total atom's of used reagents.
    `pin_max`, if given, caps every reagent's own count.
    `allow_extra_output_copies` allows extra output molecules not counting as waste.
    `constraints` forces no redundant-reaction/waste elimination.
    """
    products_needed = pf.products_needed()
    demand: Dict[int, int] = {}
    for mol in pf.outputs:
        for atype, cnt in mol.atom_type_counts().items():
            demand[atype] = demand.get(atype, 0) + cnt * products_needed

    # Detect duplicate inputs
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

    # Variables
    prob = pulp.LpProblem("recipe", pulp.LpMinimize)
    extra_copies = {}
    if allow_extra_output_copies:
        extra_copies = {j: pulp.LpVariable(f"extra_out_{j}", lowBound=0, cat="Integer")
                         for j in range(len(pf.outputs))}
    x = {i: pulp.LpVariable(f"reagent_{i}", lowBound=0, cat="Integer") for i in reagent_atoms}
    y = {r.name: pulp.LpVariable(f"reaction_{r.name}", lowBound=0, cat="Integer") for r in reactions}
    waste = {t: pulp.LpVariable(f"waste_{t}", lowBound=0, cat="Integer") for t in all_types - {ATOM_REPEAT}}
    catalyst_types = {r.catalyst for r in reactions if r.catalyst is not None}
    has_catalyst = {t: pulp.LpVariable(f"has_catalyst_{t}", cat="Binary") for t in catalyst_types}
    big_m = 10000

    # Game Constraints
    for t in all_types:
        supply = pulp.lpSum(x[i] * reagent_atoms[i].get(t, 0) for i in reagent_atoms)
        produced = pulp.lpSum(y[r.name] * r.delta.get(t, 0) for r in reactions)
        extras = pulp.lpSum(
            extra_copies[j] * pf.outputs[j].atom_type_counts().get(t, 0) for j in extra_copies
        )
        waste_term = waste[t] if t in waste else 0
        prob += supply + produced == demand.get(t, 0) + waste_term + extras
    for r in reactions:
        if r.catalyst is not None:
            prob += y[r.name] <= big_m * has_catalyst[r.catalyst]
    for t, hc in has_catalyst.items():
        raw_supply = pulp.lpSum(x[i] * reagent_atoms[i].get(t, 0) for i in reagent_atoms)
        produced = pulp.lpSum(
            y[r.name] * r.delta.get(t, 0)
            for r in reactions
            if r.delta.get(t, 0) > 0 and r.catalyst != t
        )
        prob += raw_supply + produced >= hc

    # Loop constraints, force the absence of redundant reactions:
    # add "chemical potential" per atom, and force all used reactions to decrease it,
    # possible via Gordan's theorem only if there is no net-zero loop.
    # Including waste as a pseudo-reaction also eliminates net-positive loops.
    # Including input as a pseudo-reaction eliminates redundant waste.
    pi_bound = 250
    pi = {t: pulp.LpVariable(f"pi_{t}", lowBound=-pi_bound, upBound=pi_bound) for t in waste}
    if constraints:
        min_grain: Dict[int, int] = {}
        for atoms in reagent_atoms.values():
            for t, c in atoms.items():
                if t in waste:
                    min_grain[t] = min(min_grain.get(t, c), c)
        loop_eps = {r.name: pulp.LpVariable(f"loop_eps_{r.name}", cat="Binary") for r in reactions}
        waste_eps = {t: pulp.LpVariable(f"waste_eps_{t}", cat="Binary") for t in waste}
        input_eps = {i: pulp.LpVariable(f"input_eps_{i}", cat="Binary") for i in x}
        for r in reactions:
            dmu = pulp.lpSum(coeff * pi[t] for t, coeff in r.delta.items() if t in pi)
            prob += y[r.name] <= big_m * loop_eps[r.name]
            prob += dmu <= -1 + big_m * (1 - loop_eps[r.name])
        for t in waste:
            thresh = min_grain.get(t, 1)
            prob += waste[t] <= (thresh - 1) + big_m * waste_eps[t]
            prob += -pi[t] <= -1 + big_m * (1 - waste_eps[t])
        for i, atoms in reagent_atoms.items():
            delta = {t: c for t, c in atoms.items() if t in waste}
            if not delta:
                continue
            dmu = pulp.lpSum(coeff * pi[t] for t, coeff in delta.items())
            prob += dmu <= -1 + big_m * (1 - input_eps[i])
            prob += x[i] <= big_m * input_eps[i]

    # Reagent pinning
    if pin_max is not None:
        for i in x:
            prob += x[i] <= pin_max

    # Objective
    reagent_max = pulp.LpVariable("reagent_max", lowBound=0, cat="Integer")
    for i in x:
        prob += x[i] <= reagent_max
    scale = 1000
    if objective is None:
        # Lexicographic-ish objective: most balanced reagent use first
        prob += (
            scale**3 * reagent_max
            + scale**2 * pulp.lpSum(x.values())
            + scale * pulp.lpSum(y.values())
            + pulp.lpSum(waste.values())
        )
    else:
        kind, *args = objective
        if kind == "waste":
            allowed_types, max_outside = args
            prob += pulp.lpSum(v for t, v in waste.items() if t not in allowed_types) <= max_outside
        elif kind in ("min_zones", "min_atoms"):
            reagents_used = {i: pulp.LpVariable(f"reagents_used_{i}", cat="Binary") for i in x}
            for i in x:
                prob += x[i] <= big_m * reagents_used[i]
            if kind == "min_zones":
                prob += pulp.lpSum(reagents_used.values())
            else:
                prob += pulp.lpSum(reagents_used[i] * sum(reagent_atoms[i].values()) for i in x)
        else:
            if isinstance(args[0], str):
                var = y[args[0]]
            elif isinstance(args[0], tuple) and args[0][0] == "waste":
                var = waste[args[0][1]]
            elif isinstance(args[0], int):
                var = x[args[0]]
            else:
                var = pulp.lpSum(y[n] for n in args[0])  # group of reaction names, joint sum
            if kind == "max>0":
                prob += var >= 1
            else:
                prob.sense = pulp.LpMaximize if kind == "max" else pulp.LpMinimize
                prob += var

    return prob, x, y, waste, reagent_group_size, pi


def _recipe_result(x, y, waste, reagent_group_size, reactions_by_name) -> RecipeResult:
    def val(v):
        return int(round(pulp.value(v) or 0))

    return RecipeResult(
        reagent_counts={i: (val(v), val(v)) for i, v in x.items()},
        reagent_group_size=reagent_group_size,
        reaction_counts={reactions_by_name[name]: (val(v), val(v))
                          for name, v in y.items() if val(v) > 0},
        waste={t: (val(v), val(v)) for t, v in waste.items() if val(v) > 0},
    )


def solve_recipe(pf: PuzzleFile) -> RecipeResult:
    """
    Return one representative inputs/reactions set that solves the puzzle, with minimal reagent sets.
    """
    prob, x, y, waste, reagent_group_size, _pi = _build_recipe_lp(pf, constraints=False)
    status = prob.solve(pulp.HiGHS(msg=0))
    status_str = pulp.LpStatus[status]
    if status_str != "Optimal":
        raise ValueError(f"no atom-balanced recipe found for {pf.name!r} (status={status_str})")
    reactions_by_name = {r.name: r for r in build_reactions(pf)}
    return _recipe_result(x, y, waste, reagent_group_size, reactions_by_name)


def recipe_waste_feasible(pf: PuzzleFile, *, waste_max: int = 0,
                           waste_types_allowed: frozenset = frozenset(),
                           excluded_reactions: frozenset = frozenset()) -> bool:
    """
    Check if a bounded waste solutions exist, given a set of reactions and allowed waste atoms.
    """
    prob, *_ = _build_recipe_lp(
        pf, objective=("waste", waste_types_allowed, waste_max), allow_extra_output_copies=True,
        excluded_reactions=excluded_reactions, constraints=False)
    status = prob.solve(pulp.HiGHS(msg=0))
    status_str = pulp.LpStatus[status]
    if status_str not in ("Optimal", "Infeasible"):
        raise RuntimeError
    return status_str == "Optimal"


def solve_recipe_flexibility(pf: PuzzleFile, slack: int = 0) -> RecipeResult:
    """
    Characterizes all inputs/reactions sets that solves the puzzle,
    with a minimal (+ slack) number of reagent sets.

    Reactions that produce at least one of the same output atoms,
    and could be used in a relevant solution, are pooled and report
    a group total instead of individual reaction counts.
    """
    def val(v):
        return int(round(pulp.value(v) or 0))

    prob_off, x_off, *_ = _build_recipe_lp(pf, constraints=False)
    status_off = prob_off.solve(pulp.HiGHS(msg=0))
    if pulp.LpStatus[status_off] != "Optimal":
        raise ValueError(f"no atom-balanced recipe found for {pf.name!r} (status={pulp.LpStatus[status_off]})")
    min_off = max((val(v) for v in x_off.values()), default=0)

    prob, x, y, waste, reagent_group_size, pi = _build_recipe_lp(pf)
    status = prob.solve(pulp.HiGHS(msg=0))
    if pulp.LpStatus[status] != "Optimal":
        raise ValueError(f"no atom-balanced recipe found for {pf.name!r} (status={pulp.LpStatus[status]})")
    min_on = max((val(v) for v in x.values()), default=0)
    if min_on != min_off:
        raise NotImplementedError(
            f"{pf.name!r}: Can't model puzzles that require redundant reactions to create catalysts.")

    allowed_reagent_max = max([val(v) for v in x.values()], default=0) + slack
    def _probe(objective):
        p, px, py, pw, _rgs, ppi = _build_recipe_lp(
            pf, objective=objective, pin_max=allowed_reagent_max)
        st = p.solve(pulp.HiGHS(msg=0))
        if pulp.LpStatus[st] not in ("Optimal", "Infeasible"):
            raise RuntimeError
        return px, py, pw, pulp.LpStatus[st]

    reagent_range = {}
    for i in x:
        px, _py, _pw, _status = _probe(("min", i))
        lo = val(px[i])
        px, _py, _pw, _status = _probe(("max", i))
        hi = val(px[i])
        reagent_range[i] = (lo, hi)

    waste_range = {}
    for t in waste:
        _px, _py, pw, _status = _probe(("min", ("waste", t)))
        lo = val(pw[t])
        _px, _py, pw, _status = _probe(("max", ("waste", t)))
        hi = val(pw[t])
        if hi > 0:
            waste_range[t] = (lo, hi)

    is_reaction_usable = {}
    reaction_range = {}
    for name in y:
        _px, _py, _pw, status = _probe(("max>0", name))
        is_reaction_usable[name] = status == "Optimal"
        if is_reaction_usable[name]:
            _px, py, _pw, _status = _probe(("min", name))
            lo = val(py[name])
            _px, py, _pw, _status = _probe(("max", name))
            hi = val(py[name])
            assert hi > 0
            reaction_range[name] = (lo, hi)

    # group reactions
    reactions_by_name = {r.name: r for r in build_reactions(pf)}
    def produced_types(name):
        return [t for t, d in reactions_by_name[name].delta.items() if d > 0]
    candidate_groups: Dict[int, list] = {}
    for name in y:
        for t in produced_types(name):
            group = candidate_groups.setdefault(t, [])
            if name not in group:
                group.append(name)
    for t, names in candidate_groups.items():
        usable = [n for n in names if is_reaction_usable[n] > 0]
        if len(usable) < 2:
            continue
        alternatives = []
        for n in usable:
            extra = {k: v for k, v in reactions_by_name[n].delta.items() if k != t}
            leftover = reactions_by_name[n].delta.get(t, 0) - 1
            if leftover:
                extra[t] = leftover
            alternatives.append((n, extra))
        usable_sorted = sorted(usable)
        prefix = commonprefix(usable_sorted)
        prefix = prefix[:prefix.rfind("_") + 1] if "_" in prefix else ""
        combined_name = prefix + "_or_".join(n[len(prefix):] for n in usable_sorted)
        _px, py, _pw, _status = _probe(("min", frozenset(usable)))
        lo_total = sum(val(py[n]) for n in usable)
        _px, py, _pw, _status = _probe(("max", frozenset(usable)))
        hi_total = sum(val(py[n]) for n in usable)
        base = sum(reaction_range[n][0] for n in usable)
        flex_lo, flex_hi = lo_total - base, hi_total - base
        if flex_hi:
            for n in usable:
                reaction_range[n] = (reaction_range[n][0], reaction_range[n][0])
                if reaction_range[n][0] == 0:
                    del reaction_range[n]
            reaction_range[Reaction(name=combined_name, delta={t: 1}, alternatives=alternatives)] = (flex_lo, flex_hi)
    for name in list(reaction_range):
        if isinstance(name, str):
            reaction_range[reactions_by_name[name]] = reaction_range.pop(name)

    return RecipeResult(
        reagent_counts=reagent_range,
        reagent_group_size=reagent_group_size,
        waste=waste_range,
        reaction_counts=reaction_range,
    )


def solve_recipe_fast(pf: PuzzleFile) -> List[Tuple[PuzzleFile, RecipeResult]]:
    """
    List of recipes with low number of input sets.
    Allow for several recipes with slower throughput if they could have lower latency.
    """
    def sweep(variant: PuzzleFile) -> List[RecipeResult]:
        results: List[RecipeResult] = []
        prev_key = None
        stable_run = 0
        slack = 0
        while True:
            combined = solve_recipe_flexibility(variant, slack=slack)
            reaction_key = tuple(sorted(r.name for r in combined.reaction_counts))
            reagent_key = tuple(sorted(
                (i, lo > 0)
                for i, (lo, hi) in combined.reagent_counts.items() if hi > 0
            ))
            key = (reaction_key, reagent_key)
            if key == prev_key:
                stable_run += 1
                if stable_run >= 3:
                    break
            else:
                stable_run = 0
                results.append(combined)
                prev_key = key
            slack += 1
        return results

    results = [(pf, r) for r in sweep(pf)]
    alt_pf = alternate_repeat_puzzle(pf)
    if alt_pf is not None:
        results.extend((alt_pf, r) for r in sweep(alt_pf))
    return results


def _solve_zone_objective(pf: PuzzleFile, kind: str) -> set:
    prob, x, *_ = _build_recipe_lp(pf, objective=(kind,), constraints=False)
    status = prob.solve(pulp.HiGHS(msg=0))
    status_str = pulp.LpStatus[status]
    if status_str != "Optimal":
        raise ValueError(f"no atom-balanced recipe found for {pf.name!r} (status={status_str})")
    return {i for i, v in x.items() if int(round(pulp.value(v) or 0)) > 0}


def necessary_inputs(pf: PuzzleFile) -> set:
    """
    Representative input indices used by a solution minimizing the number of reagents.
     """
    return _solve_zone_objective(pf, "min_zones")


def necessary_atoms(pf: PuzzleFile) -> set:
    """
    Representative input indices used by a solution minimizing total input atom footprint.
    """
    return _solve_zone_objective(pf, "min_atoms")


def describe_molecule(mol) -> str:
    counts = mol.atom_type_counts()
    return " + ".join(f"{n}x{ATOM_NAMES.get(t, t)}" for t, n in sorted(counts.items()))


def _count_label(rng: Tuple[int, int]) -> str:
    lo, hi = rng
    if hi > lo:
        return f"{lo}-{hi}x"
    return f"{hi}x"


def print_recipe(pf: PuzzleFile, result: RecipeResult) -> None:
    print("Recipe (stoichiometry):")
    for i, mol in enumerate(pf.inputs):
        lo, hi = result.reagent_counts.get(i, (0, 0))
        if hi:
            group_size = result.reagent_group_size.get(i, 1)
            suffix = f"[x{group_size}]" if group_size > 1 else ""
            print(f"  {_count_label((lo, hi)):>7}  input {i}{suffix}: {describe_molecule(mol)}")
    products_needed = pf.products_needed()
    for i, mol in enumerate(pf.outputs):
        print(f"  {products_needed:>4}x  output {i}: {describe_molecule(mol)}")
    if result.reaction_counts:
        print("  Transformations:")
        for r, (lo, hi) in sorted(result.reaction_counts.items(), key=lambda kv: kv[0].name):
            print(f"  {_count_label((lo, hi)):>7}  {r.name}")
    if result.waste:
        print("  Waste (surplus atoms produced beyond what's needed):")
        for atype, rng in sorted(result.waste.items()):
            print(f"  {_count_label(rng):>7}  {ATOM_NAMES.get(atype, atype)}")
