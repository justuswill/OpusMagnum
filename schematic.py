"""
schematic.py — backward state-space search from a puzzle's output molecule(s)
down toward its raw reagents.

Builds on the atom-count-level recipe (stoichiometry.solve_recipe) by adding
molecule *structure*: each state is a list of Molecule objects (same
hex-grid representation puzzle_parser uses) plus how many firings of each
transformation reaction remain. Starting from one unit copy of the output
molecule(s), two moves generate predecessor states:

  - un-bond: remove one existing bond. A bridge splits the molecule into
    two; a ring-closing edge just drops the bond and stays in one piece.
  - un-react: reverse one firing of an available reaction — remove the
    product-side atoms (matching types/counts) from wherever they sit,
    add fresh unbonded atoms for the reagent side, and spend one unit of
    that reaction's budget.

A BFS from the initial state enumerates every combination of unbond/un-react
moves reachable within the reaction budget — every structurally-distinct
way the recipe's atom bookkeeping could have been assembled, without
committing to geometry/cycle timing (out of scope here, as in
stoichiometry.py) — plus a third move, "wait" (_wait_neighbors), which ages
every free atom below _READY_AGE by one, modeling the idle cycles a
drop-and-create reaction's product atom needs before it can be picked up
again (_READY_AGE / Atom.age); nothing else about the state changes.

Termination: every unbond/un-react move strictly decreases (total bonds) +
(total reactions_left) — unbond removes bonds and adds none; un-react
spends budget and only adds unbonded atoms. "wait" leaves that sum
unchanged but strictly decreases a second measure, sum of
(_READY_AGE - age) over free atoms below _READY_AGE, and is never offered
once that's already zero (_wait_neighbors) — so the pair
(bonds+reactions_left, age deficit) strictly decreases every move, bounded
below by (0, 0): the graph is a finite DAG and the BFS always terminates.

Simplifications (deliberately out of scope):
  - An un-react move relabels its *first* consumed atom in place (position
    and bonds kept, matching how a glyph doesn't detach the atom) into the
    reaction's first produced type. Any additional needed atoms (e.g.
    purification's second lo atom, projection's quicksilver) spawn as a
    fresh single-atom molecule at (0, 0) — real multi-atom placement is a
    geometry/choreography question (PLAN.md Phase A/B), not modeled here.
  - Catalyst atoms (Reaction.catalyst) aren't modeled — solve_recipe
    already verified a catalyst-supplying reagent exists before handing us
    reaction_unit_counts, so every budgeted firing is assumed usable.
  - State dedup uses a color-refinement graph signature, not exact
    graph-isomorphism — collapses the overwhelming majority of symmetric
    duplicates without exact-canonicalization cost; pathologically
    symmetric molecules could rarely be treated as distinct when isomorphic.

    ASSUMPTIONS:
    reaction_counts // outputs_needed is sufficient to create optimal path
    the optimal path contains no more reactions than necessary
    - minimal number of unbonding reactions
"""

from collections import Counter, deque
from dataclasses import dataclass, replace
import math
from itertools import combinations, product as iproduct
from typing import Dict, Iterator, List, NamedTuple, Optional, Tuple, Union

from puzzle_parser import (
    PuzzleFile, Molecule, Atom, Bond, ATOM_NAMES,
    PART_BONDER, PART_BONDER_PRISMA, PART_BONDER_SPEED, PART_UNBONDER,
    BOND_NORMAL, BOND_TRIPLEX_R, BOND_TRIPLEX_K, BOND_TRIPLEX_Y,
)
from stoichiometry import RecipeResult, Reaction, build_reactions


@dataclass
class State:
    molecules: List[Molecule]
    reactions_left: Dict[str, int]  # reaction name -> firings still available

    def __repr__(self):
        """One line for the whole state: molecules (atom types/counts, bond
        count) separated by " | ", then remaining reaction budget."""
        mol_descs = []
        for mol in self.molecules:
            counts = mol.atom_type_counts()
            desc = ", ".join(f"{ATOM_NAMES.get(t, t)}x{c}" for t, c in sorted(counts.items()))
            if mol.bonds:
                desc += f" bonds={len(mol.bonds)}"
            mol_descs.append(desc)
        line = " | ".join(mol_descs)
        reactions_desc = ", ".join(f"{name}={count}"
                                    for name, count in sorted(self.reactions_left.items()) if count > 0)
        if reactions_desc:
            line += f"  [{reactions_desc}]"
        return line


class StateGraph:
    """Every state reachable from the initial state via unbond/un-react
    moves, plus which states are directly reachable from which. Nodes
    (states) are indexed by BFS discovery order; edges point from a state's
    index to each of its direct neighbors' indices, tagged with which move
    (bond/triple-bond/reaction name) produced them. When the same target is
    reachable via more than one kind of move, the tag with the lower
    _move_priority (plain bond preferred over triple-bond) wins."""

    def __init__(self):
        self.states: List[State] = []
        self.edges: Dict[int, List[int]] = {}
        self.edge_move: Dict[Tuple[int, int], str] = {}
        self._index: Dict[tuple, int] = {}
        self.input_state_idx: int = None
        # list, can have different atom ages
        self.input_state_indices: List[int] = []

    def add_state(self, state: State, key: tuple) -> int:
        idx = len(self.states)
        self.states.append(state)
        self.edges[idx] = []
        self._index[key] = idx
        return idx

    def add_edge(self, from_idx: int, to_idx: int, move: str):
        if to_idx not in self.edges[from_idx]:
            self.edges[from_idx].append(to_idx)
            self.edge_move[(from_idx, to_idx)] = move
        elif _move_priority(move) < _move_priority(self.edge_move[(from_idx, to_idx)]):
            self.edge_move[(from_idx, to_idx)] = move

    def __len__(self):
        return len(self.states)

    def __repr__(self):
        """One line per state: idx: state_str -> idx1[move1], idx2[move2],
        ... — every state matching the actual raw reagents (see
        input_state_indices) is marked idx* instead of idx."""
        lines = []
        for idx, state in enumerate(self.states):
            targets = ", ".join(f"{t}[{self.edge_move[(idx, t)]}]" for t in self.edges[idx])
            label = f"{idx}*" if idx in self.input_state_indices else f"{idx}"
            lines.append(f"{label}: {state!r} -> {targets}")
        return "\n".join(lines)


def _unit_scale(recipe: RecipeResult, products_needed: int) -> int:
    """Largest divisor of products_needed that also divides every nonzero
    reagent_counts/reaction_counts value (its max — the count any caller
    not showing the range uses) — the real "one unit" size when
    products_needed doesn't split those evenly (P031b: reagent_counts=
    {0: (6, 6)}, products_needed=18 -- 6/18 isn't an integer, but
    gcd(18, 6)=6 is, giving 3 output instances per unit instead of the
    naive 1)."""
    g = products_needed
    for _lo, hi in recipe.reagent_counts.values():
        if hi > 0:
            g = math.gcd(g, hi)
    for _lo, hi in recipe.reaction_counts.values():
        if hi > 0:
            g = math.gcd(g, hi)
    return g


def _reaction_unit_counts(recipe: RecipeResult, unit_scale: int) -> Dict[str, int]:
    """Per-unit reaction budget: the whole-run recipe.reaction_counts's max
    divided by unit_scale (_unit_scale). Shared by initial_state
    (output-side seed) and _seed_input_states (input-side seed) so both
    directions start from exactly the same budget — required for a
    forward and backward search to ever meet at a matching
    _state_signature."""
    return {r.name: hi // unit_scale for r, (_lo, hi) in recipe.reaction_counts.items() if hi > 0}


def initial_state(pf: PuzzleFile, recipe: RecipeResult) -> State:
    """products_needed // _unit_scale copies of each output molecule, with
    the per-unit reaction budget (_reaction_unit_counts), plus one free
    atom per unit for any recipe.waste type — needed to un-fire a reaction
    whose only tracked product is in the output, since its wasted
    co-product must also be present to reverse the firing (see P074/
    animismus/vitae)."""
    products_needed = pf.products_needed()
    g = _unit_scale(recipe, products_needed)
    batch = products_needed // g
    reaction_unit_counts = _reaction_unit_counts(recipe, g)
    molecules = [Molecule(atoms=list(mol.atoms), bonds=list(mol.bonds))
                 for mol in pf.outputs for _ in range(batch)]
    for atype, (_lo, hi) in recipe.waste.items():
        molecules.extend(Molecule(atoms=[Atom(type=atype, u=0, v=0)], bonds=[])
                          for _ in range(hi // g))
    return State(molecules, reaction_unit_counts)


_MAX_SEED_ALTERNATES = 200


def _seed_input_states(pf: PuzzleFile, recipe: RecipeResult, products_needed: int,
                        tracked_types: frozenset) -> Iterator[State]:
    """Every candidate raw-reagent starting state for the forward search —
    the mirror of initial_state, but built from pf.inputs/recipe.reagent_counts
    instead of pf.outputs/reaction_counts. Yields the baseline composition
    (matching recipe.reagent_counts exactly, one unit copy each) first, then
    — only for recipes with a flexible calcify_X_or_Y group
    (recipe.extra_reactions, see solve_recipe_fast) — every other valid
    way to split that group's shared total among its alternative types, up
    to _MAX_SEED_ALTERNATES: recipe.reagent_counts already encodes ONE such
    split (solve_recipe_fast's own proportional rebalance), which may
    not be the one any state in the backward graph actually realizes — same
    reason _matches_raw_reagents' exact=False fallback exists, just applied
    on the input side instead of the output side. Every yielded molecule of
    a tracked_types type is constructed already at _READY_AGE: a genuine
    raw reagent has no readiness requirement of its own (see
    _matches_raw_reagents), so it must present as already-matured or its
    _state_signature could never coincide with a backward node that
    (correctly) treats it as ready. Uses _unit_scale, not products_needed
    directly, as the divisor throughout — see initial_state's docstring
    for why (P031b: reagent_counts doesn't split evenly by products_needed
    alone)."""
    g = _unit_scale(recipe, products_needed)
    reaction_unit_counts = _reaction_unit_counts(recipe, g)
    # Every consumer here just wants "how many of this reagent" — the
    # min/max distinction reagent_counts carries is for range display and
    # <=-vs-== matching elsewhere (see reachable_states), not state
    # construction, so extract the max (the count a range's own ceiling
    # collapses to when there's no range) once up front.
    baseline_counts = {i: hi for i, (_lo, hi) in recipe.reagent_counts.items()}

    def make_state(counts: Dict[int, int]) -> State:
        molecules = []
        for i, count in counts.items():
            if count <= 0:
                continue
            src = pf.inputs[i]
            for _ in range(count // g):
                atoms = list(src.atoms)
                if len(atoms) == 1 and atoms[0].type in tracked_types:
                    atoms = [replace(atoms[0], age=_READY_AGE)]
                molecules.append(Molecule(atoms=atoms, bonds=list(src.bonds)))
        return State(molecules, dict(reaction_unit_counts))

    yield make_state(baseline_counts)

    yielded = 1
    for r in recipe.reaction_counts:
        if r.alternatives is None:
            continue
        types = _single_atom_alt_types(r)
        if yielded >= _MAX_SEED_ALTERNATES or not types:
            continue
        member_indices = [
            i for i, mol in enumerate(pf.inputs)
            if i in recipe.reagent_counts
            and len(mol.atom_type_counts()) == 1
            and next(iter(mol.atom_type_counts())) in types
        ]
        if len(member_indices) < 2:
            continue

        # The group's shared, genuinely-flexible total (per unit) — the
        # rest of each member's current reagent_counts is fixed direct-use
        # (bonded into the output as-is, never calcified) and must stay put.
        # direct_use[i] = recipe.reagent_counts[i] minus whatever share of
        # `total` solve_recipe_fast's own proportional split assigned it
        # — reconstructed from the same repeats-weighted apportionment it
        # uses, so this stays in sync without importing its internals.
        total = recipe.reaction_counts.get(r, (0, 0))[1] // g
        repeats = {i: recipe.reagent_group_size.get(i, 1) for i in member_indices}
        total_repeats = sum(repeats.values()) or 1
        base_guess = {i: (total * repeats[i]) // total_repeats for i in member_indices}
        remainder = total - sum(base_guess.values())
        by_remainder = sorted(member_indices, key=lambda i: (total * repeats[i]) % total_repeats, reverse=True)
        for i in by_remainder[:max(remainder, 0)]:
            base_guess[i] += 1
        direct_use = {i: baseline_counts[i] - base_guess[i] for i in member_indices}

        def compositions(total, n):
            if n == 1:
                yield (total,)
                return
            for first in range(total + 1):
                for rest in compositions(total - first, n - 1):
                    yield (first,) + rest

        for split in compositions(total, len(member_indices)):
            if yielded >= _MAX_SEED_ALTERNATES:
                break
            counts = dict(baseline_counts)
            for i, s in zip(member_indices, split):
                counts[i] = direct_use[i] + s
            if counts == baseline_counts:
                continue  # already yielded as the baseline
            yield make_state(counts)
            yielded += 1


# ── molecule graph helpers ───────────────────────────────────────────────

def _adjacency(atoms: List[Atom], bonds: List[Bond]) -> Dict[int, List[int]]:
    """atom index -> list of neighbor atom indices."""
    pos_to_index = {(a.u, a.v): i for i, a in enumerate(atoms)}
    adj: Dict[int, List[int]] = {i: [] for i in range(len(atoms))}
    for b in bonds:
        i = pos_to_index.get((b.from_u, b.from_v))
        j = pos_to_index.get((b.to_u, b.to_v))
        if i is None or j is None:
            continue
        adj[i].append(j)
        adj[j].append(i)
    return adj


def _split_molecule(atoms: List[Atom], bonds: List[Bond]) -> List[Molecule]:
    """Connected components of (atoms, bonds) as separate Molecule objects,
    each keeping its atoms' original positions and only the bonds fully
    inside it."""
    adj = _adjacency(atoms, bonds)
    seen = [False] * len(atoms)
    components: List[Molecule] = []
    for start in range(len(atoms)):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        members = {start}
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    members.add(v)
                    stack.append(v)
        comp_atoms = [atoms[i] for i in sorted(members)]
        comp_positions = {(atoms[i].u, atoms[i].v) for i in members}
        comp_bonds = [b for b in bonds
                      if (b.from_u, b.from_v) in comp_positions
                      and (b.to_u, b.to_v) in comp_positions]
        components.append(Molecule(atoms=comp_atoms, bonds=comp_bonds))
    return components


def _remove_atoms(atoms: List[Atom], bonds: List[Bond], remove_indices: set) -> List[Molecule]:
    """Drop the given atom indices (and any bond touching one of them), then
    split whatever's left into connected components."""
    remaining_atoms = [a for i, a in enumerate(atoms) if i not in remove_indices]
    kept_positions = {(a.u, a.v) for a in remaining_atoms}
    remaining_bonds = [b for b in bonds
                        if (b.from_u, b.from_v) in kept_positions
                        and (b.to_u, b.to_v) in kept_positions]
    return _split_molecule(remaining_atoms, remaining_bonds)


# ── canonical signature (BFS visited-set key) ────────────────────────────

# Axial hex directions, each one 60° from its neighbors in this list; index
# i and index (i+3)%6 are exact opposites — used to get the direction back
# from j to i given the direction from i to j.
_HEX_DIRS = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
_DIR_INDEX = {d: i for i, d in enumerate(_HEX_DIRS)}


def _rotate60(u: int, v: int, cu: int, cv: int, k: int) -> Tuple[int, int]:
    """(u, v) rotated k*60° counterclockwise about center (cu, cv), on this
    module's axial hex coordinates — derived from _HEX_DIRS being a
    60°-consecutive direction cycle: the linear map (u, v) -> (u+v, -u)
    sends each _HEX_DIRS entry to the next one, so it's exactly one 60°
    step; applying it k times is a k*60° rotation."""
    du, dv = u - cu, v - cv
    for _ in range(k % 6):
        du, dv = du + dv, -du
    return cu + du, cv + dv


def _molecule_rotational_group(mol: Molecule) -> Tuple[List[List[int]], List[Dict[int, int]]]:
    """(orbits, perms): orbits partitions mol's atom indices into
    equivalence classes under mol's own rotational symmetry (the largest
    found); perms is the list of nonidentity atom_idx -> atom_idx maps, one
    per nonzero rotation mapping mol onto itself exactly (same atom type at
    each position, same bonds). Rotations only, never reflections:
    molecule_signature is chirality-sensitive (mirror images get different
    signatures on purpose), so a mirror symmetry wouldn't make its two
    paired atoms interchangeable for pooling the way a rotation does
    (relabeling one vs. the other gives an actual mirror image, not an
    identical result).

    A nontrivial rotation has exactly one fixed point (its center), so
    candidate centers are restricted to mol's own atom positions — an
    off-atom center could only fix the molecule if no atom needs to map to
    itself, which doesn't arise for the puzzles this targets. Returns the
    identity-only (all-singleton orbits, no perms) result if no atom
    position yields a nontrivial symmetry.

    Not memoized on `mol`'s identity, for the same reason as
    molecule_signature: measured with an id-keyed cache on
    P025/P033/P034/P036 with no measurable improvement (within run-to-run
    noise, if anything slightly negative) — most calls are for distinct
    molecule content, not repeated hits on the same reference."""
    pos_to_idx = {(a.u, a.v): i for i, a in enumerate(mol.atoms)}
    bond_set = {(frozenset({(b.from_u, b.from_v), (b.to_u, b.to_v)}), b.type) for b in mol.bonds}

    best_perms: List[Dict[int, int]] = []
    for a0 in mol.atoms:
        cu, cv = a0.u, a0.v
        perms = []
        for k in range(1, 6):
            perm = {}
            ok = True
            for i, a in enumerate(mol.atoms):
                nu, nv = _rotate60(a.u, a.v, cu, cv, k)
                j = pos_to_idx.get((nu, nv))
                if j is None or mol.atoms[j].type != a.type:
                    ok = False
                    break
                perm[i] = j
            if not ok:
                continue
            new_bonds = {(frozenset({_rotate60(b.from_u, b.from_v, cu, cv, k),
                                      _rotate60(b.to_u, b.to_v, cu, cv, k)}), b.type)
                         for b in mol.bonds}
            if new_bonds == bond_set:
                perms.append(perm)
        if len(perms) > len(best_perms):
            best_perms = perms

    n = len(mol.atoms)
    if not best_perms:
        return [[i] for i in range(n)], []

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for perm in best_perms:
        for i, j in perm.items():
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values()), best_perms


def _directed_adjacency(atoms: List[Atom], bonds: List[Bond]) -> Dict[int, List[Tuple[int, int, int]]]:
    """atom index -> list of (hex direction index 0-5, bond.type bitmask,
    neighbor atom index) for each bond, direction taken from that atom's own
    point of view. Carrying the raw bond.type means a normal bond and any of
    the three triplex colors (BOND_NORMAL vs BOND_TRIPLEX_R/K/Y) are kept
    distinct, not just "bonded vs not"."""
    pos_to_index = {(a.u, a.v): i for i, a in enumerate(atoms)}
    adj: Dict[int, List[Tuple[int, int, int]]] = {i: [] for i in range(len(atoms))}
    for b in bonds:
        i = pos_to_index.get((b.from_u, b.from_v))
        j = pos_to_index.get((b.to_u, b.to_v))
        if i is None or j is None:
            continue
        d = _DIR_INDEX.get((atoms[j].u - atoms[i].u, atoms[j].v - atoms[i].v))
        if d is None:
            continue  # bond doesn't connect adjacent hexes — shouldn't happen, skip defensively
        adj[i].append((d, b.type, j))
        adj[j].append(((d + 3) % 6, b.type, i))
    return adj


def _cyclic_canonical(entries: List[Tuple[int, int, int]]) -> tuple:
    """Canonical encoding of one atom's (direction, bond kind, neighbor-color)
    entries that's invariant to which absolute direction the molecule happens
    to be rotated to, but keeps the *relative* angular gaps between bonds
    (60°/120°/180° — sharp/wide/straight are different), which specific bond
    kind sits at each gap (normal vs triplex-R/K/Y), and their cyclic order
    (so it stays chirality-sensitive: a neighbor arrangement and its mirror
    image get different canonical forms, since we only try rotations here,
    never a reversal). Direction alone is already a unique sort key — only
    one bond can occupy a given hex direction from an atom."""
    if not entries:
        return ()
    entries = sorted(entries)
    n = len(entries)
    dirs = [d for d, _, _ in entries]
    rest = [(kind, color) for _, kind, color in entries]
    best = None
    for start in range(n):
        rotated_dirs = tuple((dirs[(start + k) % n] - dirs[start]) % 6 for k in range(n))
        rotated_rest = tuple(rest[(start + k) % n] for k in range(n))
        candidate = (rotated_dirs, rotated_rest)
        if best is None or candidate < best:
            best = candidate
    return best


def molecule_signature(mol: Molecule, rounds: int = 4) -> tuple:
    """Rotation/translation/atom-order-invariant fingerprint via a few
    rounds of color refinement (Weisfeiler-Leman style) — see module
    docstring for why this is a heuristic, not an exact isomorphism test.
    Bond *directions and kinds* feed into each round via _cyclic_canonical,
    so e.g. an X-Y-X chain with its two X neighbors 60°/120°/180° apart at
    Y gets a different signature per angle, normal vs. triplex-color bonds
    are never conflated, and mirror-image arrangements stay distinct too.

    Deliberately *not* memoized on `mol`'s identity, despite being the
    hottest function in the whole search and despite _apply_mixed_actions
    reusing the same Molecule reference for untouched molecules — measured
    on P025/P033 with an id-keyed cache: consistently 6-11% *slower* end to
    end. Most calls are for genuinely distinct molecule content (freshly
    constructed, not shared references), so the hit rate is too low to pay
    for the lookup/store overhead. Left unmemoized rather than caching for
    cache's own sake."""
    n = len(mol.atoms)
    if n == 0:
        return ()
    adj = _directed_adjacency(mol.atoms, mol.bonds)
    colors = [a.type for a in mol.atoms]
    for _ in range(min(rounds, n)):
        colors = [
            hash((colors[i], _cyclic_canonical([(d, k, colors[j]) for d, k, j in adj[i]])))
            for i in range(n)
        ]
    return tuple(sorted(colors))


def _state_signature(state: State, tracked_types: frozenset) -> tuple:
    """Canonical BFS-dedup key for `state`. Distinct from molecule_signature
    itself (used everywhere else — _matches_raw_reagents, bounds.py's
    reagent matching): those callers need a free atom to match its puzzle
    input regardless of how many "wait" moves it's absorbed (a genuine raw
    reagent has no readiness requirement at all — see _READY_AGE), so age
    only feeds into *this* signature, which exists purely to tell the BFS
    apart two states that are one "wait" apart — since only one of them can
    reverse a drop-and-create reaction (_firing_atoms_ok), they really are
    distinct states with different onward moves, not duplicates to collapse.
    Restricted to `tracked_types` (the actual product types of this puzzle's
    drop-and-create reactions — see reachable_states) since age is never
    even looked at for any other type: including it in their signature too
    would only split otherwise-identical states apart for no reason,
    multiplying the graph without changing what's reachable from it."""
    mol_sigs = tuple(sorted(
        (molecule_signature(m), min(m.atoms[0].age, _READY_AGE))
        if len(m.atoms) == 1 and m.atoms[0].type in tracked_types
        else (molecule_signature(m), -1)
        for m in state.molecules
    ))
    react_sig = tuple(sorted((k, v) for k, v in state.reactions_left.items() if v > 0))
    return (mol_sigs, react_sig)


# ── neighbor generation ───────────────────────────────────────────────────

_MOVE_BOND = "bond"
_MOVE_BOND_R = "bond_r"
_MOVE_BOND_K = "bond_k"
_MOVE_BOND_Y = "bond_y"
_MOVE_TRIPLE_BOND = "triple-bond"
_TRIPLEX_COLOR_LABEL = {BOND_TRIPLEX_R: _MOVE_BOND_R, BOND_TRIPLEX_K: _MOVE_BOND_K, BOND_TRIPLEX_Y: _MOVE_BOND_Y}


class BondAction(NamedTuple):
    """One elementary bond-removal action (see _elementary_bond_actions):
    `touched` is the set of (mol_idx, atom_idx) pairs it grabs (used to
    check two actions combined into one move never share an atom);
    `remove_by_mol` maps molecule_idx -> set of (bond index into that
    molecule's bonds list, bit to clear from that bond's type) pairs —
    clearing a bit only drops the bond entry entirely once its type
    reaches 0 (see _apply_mixed_actions)."""
    label: str
    touched: frozenset
    remove_by_mol: Dict[int, set]


class ReactionAction(NamedTuple):
    """One elementary un-react firing (see _elementary_reaction_actions):
    `touched` == frozenset(firing). `firing` is the list of (mol_idx,
    atom_idx) atom instances this firing consumes — firing[0] is the atom
    relabeled in place (see _apply_mixed_actions); any further entries are
    extra co-reagents removed outright. `add_types` is the reaction's
    (already alternative-resolved, see _reverse_reaction_options) list of
    produced types — add_types[0] becomes firing[0]'s new type in place,
    add_types[1:] are spawned as fresh free atoms."""
    label: str
    touched: frozenset
    name: str
    firing: List[Tuple[int, int]]
    add_types: List[int]


Action = Union[BondAction, ReactionAction]


def _move_priority(move: str) -> int:
    if move in (_MOVE_BOND, _MOVE_BOND_R, _MOVE_BOND_K, _MOVE_BOND_Y):
        return 0
    if move == _MOVE_TRIPLE_BOND:
        return 2
    return 1  # a reaction name


def _bonds_by_direction(atoms: List[Atom], bonds: List[Bond]) -> Dict[int, Dict[int, int]]:
    """atom index -> {hex direction index 0-5: index into `bonds`}, direction
    taken from that atom's own point of view — like _directed_adjacency, but
    keeps the bond's index into `bonds` (not the neighbor atom) so a specific
    bond can be located and removed."""
    pos_to_index = {(a.u, a.v): i for i, a in enumerate(atoms)}
    result: Dict[int, Dict[int, int]] = {i: {} for i in range(len(atoms))}
    for bi, b in enumerate(bonds):
        i = pos_to_index.get((b.from_u, b.from_v))
        j = pos_to_index.get((b.to_u, b.to_v))
        if i is None or j is None:
            continue
        d = _DIR_INDEX.get((atoms[j].u - atoms[i].u, atoms[j].v - atoms[i].v))
        if d is None:
            continue
        result[i][d] = bi
        result[j][(d + 3) % 6] = bi
    return result


_TRIPLE_DIRECTION_SETS = [(0, 2, 4), (1, 3, 5)]


def _elementary_bond_actions(state: State, parts_available: int) -> List[BondAction]:
    """Every individual bonding action currently reversible in `state`: one
    single-bond removal (reverse of a Bonder — needs PART_BONDER, normal
    bonds only, label "bond"), or one single-*color* triplex removal
    (reverse of one Bonder-Prisma pass — needs PART_BONDER_PRISMA, label
    "bond_r"/"bond_k"/"bond_y"): a bond with 2-3 triplex colors set needs
    that many separate actions, one color at a time — you can't strip a
    full triplex bond in one move, only one color per pass. Or one
    triple-bond removal (reverse of a Multi-bonder, needs PART_BONDER_SPEED
    — 2 or 3 bonds around one atom, whichever of a qualifying triple's
    slots are occupied; only ever normal bonds). See BondAction for field
    meanings — touched is used to check that two actions combined into one
    move never share an atom (this is also exactly why two different
    colors of the *same* triplex bond can never combine into one move:
    they touch the same pair of atoms)."""
    actions = []
    for mi, mol in enumerate(state.molecules):
        pos_to_index = {(a.u, a.v): i for i, a in enumerate(mol.atoms)}
        for bi, b in enumerate(mol.bonds):
            ai = pos_to_index.get((b.from_u, b.from_v))
            aj = pos_to_index.get((b.to_u, b.to_v))
            if ai is None or aj is None:
                continue
            touched = frozenset({(mi, ai), (mi, aj)})
            if b.is_normal and (parts_available & PART_BONDER):
                actions.append(BondAction(_MOVE_BOND, touched, {mi: {(bi, BOND_NORMAL)}}))
            if parts_available & PART_BONDER_PRISMA:
                for color, label in _TRIPLEX_COLOR_LABEL.items():
                    if b.type & color:
                        actions.append(BondAction(label, touched, {mi: {(bi, color)}}))

        if parts_available & PART_BONDER_SPEED:
            by_dir = _bonds_by_direction(mol.atoms, mol.bonds)
            for ai in range(len(mol.atoms)):
                dirs = by_dir[ai]
                for triple in _TRIPLE_DIRECTION_SETS:
                    bond_indices = [dirs[d] for d in triple if d in dirs]
                    if len(bond_indices) < 2:
                        continue  # nothing to combine — single-bond action already covers this
                    if not all(mol.bonds[bi].is_normal for bi in bond_indices):
                        continue  # multi-bonder can't form triplex bonds
                    touched = {(mi, ai)}
                    for k in bond_indices:
                        b = mol.bonds[k]
                        neighbor_pos = ((b.to_u, b.to_v) if (b.from_u, b.from_v) == (mol.atoms[ai].u, mol.atoms[ai].v)
                                        else (b.from_u, b.from_v))
                        neighbor_ai = pos_to_index.get(neighbor_pos)
                        if neighbor_ai is not None:
                            touched.add((mi, neighbor_ai))
                    actions.append(BondAction(_MOVE_TRIPLE_BOND, frozenset(touched),
                                               {mi: {(k, BOND_NORMAL) for k in bond_indices}}))
    return actions


def _freshly_freed(components: List[Molecule]) -> List[Molecule]:
    """Force age=0 on every singleton among `components` — call this on the
    result of splitting a molecule that had bonds (so definitely had >1
    atom): any singleton coming out of that split just became free *this*
    move, however old the atom was before (see _READY_AGE / Atom.age)."""
    return [
        replace(m, atoms=[replace(m.atoms[0], age=0)]) if len(m.atoms) == 1 else m
        for m in components
    ]


_SINGLE_BOND_MOVES = {_MOVE_BOND, _MOVE_BOND_R, _MOVE_BOND_K, _MOVE_BOND_Y}


def _unbond_neighbors(state: State, parts_available: int) -> List[Tuple[State, str]]:
    """One single-bond removal (normal, or one triplex color) at a time —
    see _elementary_bond_actions."""
    return [(_apply_mixed_actions(state, [a]), a.label)
            for a in _elementary_bond_actions(state, parts_available)
            if a.label in _SINGLE_BOND_MOVES]


def _triple_unbond_neighbors(state: State, parts_available: int) -> List[Tuple[State, str]]:
    """One triple-bond (multi-bonder) removal at a time — see
    _elementary_bond_actions."""
    return [(_apply_mixed_actions(state, [a]), a.label)
            for a in _elementary_bond_actions(state, parts_available)
            if a.label == _MOVE_TRIPLE_BOND]


def _bond_orbit_bundles(local_actions: List[BondAction], group: List[int],
                         perms: List[Dict[int, int]]) -> List[List[BondAction]]:
    """Every atom-disjoint subset of one molecule's own bond-action rotation
    orbit `group` (indices into `local_actions` — see _bond_pool_catalogs),
    one representative per equivalence class under the orbit's own local
    rotation action, exactly the same canonicalize-by-min-over-perms
    approach _orbit_source_allocations uses for atom colorings. Removing
    bond A vs. its rotation-equivalent bond B (or the same choice of
    several) yields isomorphic results (same molecule_signature) since the
    molecule's own rotation maps one onto the other — so only one subset
    per class needs generating, not all of them. Orbit size is bounded by
    the rotation group's order (at most 6, orbit-stabilizer theorem), so
    brute-forcing all 2^d subsets stays cheap. The empty subset (bundle=[])
    is a real, needed yield — see _orbit_source_allocations for why: other
    independent catalogs need to be able to combine with "this orbit fires
    nothing"."""
    d = len(group)
    pos_of = {gi: p for p, gi in enumerate(group)}
    key_to_local = {(local_actions[gi].touched, local_actions[gi].label): gi for gi in group}

    local_perms = []
    for perm in perms:
        mapping = [None] * d
        ok = True
        for p, gi in enumerate(group):
            a = local_actions[gi]
            mapped = frozenset((m, perm[ai]) for m, ai in a.touched)
            partner = key_to_local.get((mapped, a.label))
            if partner is None:
                ok = False
                break
            mapping[p] = pos_of[partner]
        if ok:
            local_perms.append(tuple(mapping))

    bundles = []
    seen_canonical = set()
    for mask in range(1 << d):
        subset = tuple(p for p in range(d) if mask & (1 << p))
        touched_union = set()
        disjoint = True
        for p in subset:
            t = local_actions[group[p]].touched
            if t & touched_union:
                disjoint = False
                break
            touched_union |= t
        if not disjoint:
            continue
        canon = min([subset] + [tuple(sorted(perm[p] for p in subset)) for perm in local_perms])
        if canon in seen_canonical:
            continue
        seen_canonical.add(canon)
        bundles.append([local_actions[group[p]] for p in subset])
    return bundles


def _bond_pool_catalogs(state: State, parts_available: int, get_orbits=None,
                         exclude_molecules: frozenset = frozenset()
                         ) -> Tuple[List[BondAction], List[List[List[BondAction]]]]:
    """Splits state's elementary bond-removal actions (_elementary_bond_actions)
    into (remaining, catalogs): `remaining` has no rotational partner and
    goes through the caller's exact combinations() scan unchanged; `catalogs`
    has one joint entry per molecule with any poolable orbit — all of that
    molecule's size-2+ orbits combined and canonicalized together by
    _bond_orbit_bundles, then combined via iproduct the same way
    _combined_bond_reaction_neighbors already combines its reaction
    catalogs. Turns e.g. 15 simultaneously-removable, mostly-symmetric bonds
    from a 2^15 scan into a handful of small per-molecule products.

    Orbits must be joined per molecule, not canonicalized one at a time:
    if rotation r swaps both {a0,a1} and {b0,b1} at once, canonicalizing
    each independently only ever offers {a0,b0} (and its isomorphic image
    {a1,b1}) — never the genuinely distinct {a0,b1}, which silently
    vanishes. Joining into one group of 4 before canonicalizing recovers it
    (verified by a targeted brute-force case plus a full differential sweep
    against the pre-pooling baseline).

    `get_orbits`, if given, reuses the caller's own mi -> (orbits, perms)
    cache instead of recomputing _molecule_rotational_group.

    `exclude_molecules` sidesteps the same bug one level up: a molecule
    that also feeds a reaction's pool catalog has its reaction choices
    canonicalized independently of its bond choices under the same
    rotation group, so cross-joining the two drops the same mismatched-
    rotation combos. Falling back to exact bond enumeration for just those
    molecules avoids it — an exhaustive axis combined with a reduced one
    doesn't lose anything, only two independently-reduced axes do."""
    if get_orbits is None:
        _cache: Dict[int, tuple] = {}

        def get_orbits(mi):
            if mi not in _cache:
                _cache[mi] = _molecule_rotational_group(state.molecules[mi])
            return _cache[mi]

    actions = _elementary_bond_actions(state, parts_available)
    by_mol: Dict[int, List[int]] = {}
    for idx, a in enumerate(actions):
        mi = next(iter(a.touched))[0]
        by_mol.setdefault(mi, []).append(idx)

    remaining: List[BondAction] = []
    catalogs: List[List[List[BondAction]]] = []

    for mi, idxs in by_mol.items():
        if len(state.molecules[mi].atoms) < 2 or mi in exclude_molecules:
            remaining.extend(actions[i] for i in idxs)
            continue
        _atom_orbits, perms = get_orbits(mi)
        if not perms:
            remaining.extend(actions[i] for i in idxs)
            continue

        local_actions = [actions[i] for i in idxs]
        key_to_local = {(a.touched, a.label): pos for pos, a in enumerate(local_actions)}
        n = len(local_actions)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for pos, a in enumerate(local_actions):
            for perm in perms:
                mapped = frozenset((m, perm[ai]) for m, ai in a.touched)
                partner = key_to_local.get((mapped, a.label))
                if partner is not None:
                    rp, rq = find(pos), find(partner)
                    if rp != rq:
                        parent[rp] = rq

        groups: Dict[int, List[int]] = {}
        for pos in range(n):
            groups.setdefault(find(pos), []).append(pos)

        # All of this molecule's size-2+ orbits joined into ONE group before
        # canonicalizing (see docstring) — not one catalog per orbit.
        joint_group: List[int] = []
        for group in groups.values():
            if len(group) < 2:
                remaining.append(local_actions[group[0]])
            else:
                joint_group.extend(group)
        if joint_group:
            catalogs.append(_bond_orbit_bundles(local_actions, joint_group, perms))

    return remaining, catalogs


def _combined_bond_neighbors(state: State, parts_available: int) -> List[Tuple[State, str]]:
    """All ways to combine 2+ atom-disjoint elementary bonding actions
    (single-bond and/or triple-bond, possibly across different molecules)
    into one simultaneous move. This module doesn't track real glyph
    placement or cycle timing at all (out of scope, see module docstring),
    so nothing here distinguishes "several small glyphs firing at once,
    one per molecule/position" from "one action at a time" — the only real
    constraint is that a single atom can't be grabbed by two different
    bonding actions in the same move. That constraint is exactly why a
    straight x-y-z chain at 180° can't be undone in one combined move: a
    triple-bond centered on x never reaches z (it's not adjacent to x), and
    a plain bond removal on x-y already uses y, blocking a simultaneous y-z
    action in that same move — the two bonds have to be reversed one at a
    time instead.

    Rotationally-interchangeable actions (see _bond_pool_catalogs) are
    pooled rather than run through the exact combinations() scan below —
    only actions with no symmetry partner go through it directly, same
    split reaction-action pooling already uses."""
    actions, catalogs = _bond_pool_catalogs(state, parts_available)
    n = len(actions)
    individual_subsets = [[]]
    for size in range(1, n + 1):
        for combo in combinations(range(n), size):
            touched_union = set()
            disjoint = True
            for idx in combo:
                touched = actions[idx].touched
                if touched & touched_union:
                    disjoint = False
                    break
                touched_union |= touched
            if not disjoint:
                continue
            individual_subsets.append([actions[idx] for idx in combo])

    neighbors_out = []
    for pool_choice in iproduct(*catalogs) if catalogs else [()]:
        pool_bundle = [a for bundle in pool_choice for a in bundle]
        for ind_subset in individual_subsets:
            total = pool_bundle + ind_subset
            if len(total) < 2:
                continue  # size 0/1 already covered by _unbond_neighbors/_triple_unbond_neighbors
            touched_union = set()
            ok = True
            for a in total:
                if a.touched & touched_union:
                    ok = False
                    break
                touched_union |= a.touched
            if not ok:
                continue
            labels = [a.label for a in total]
            move = (f"{len(total)}x{labels[0]}" if all(l == labels[0] for l in labels)
                    else "+".join(labels))
            neighbors_out.append((_apply_mixed_actions(state, total), move))
    return neighbors_out


def _single_atom_alt_types(r: Reaction) -> Optional[List[int]]:
    """Alternative single reagent atom types, if every one of r's
    alternatives (see stoichiometry.Reaction.alternatives) is a pure
    single-atom substitution — consumes exactly one atom of its own type
    and nothing else, with r.delta itself carrying no consumed side of its
    own (only whatever's shared, e.g. calcify_water_or_fire's {SALT: +1}).
    None for a non-combined reaction, or a differently-shaped combo (e.g.
    a future purify_X_to_Y_or_reject_Z_to_Y, whose alternatives consume
    more than one atom and/or produce a bystander byproduct) — a
    structural check, not a name-based one, so it keeps recognizing
    calcify_X_or_Y correctly once other combo shapes exist alongside it.
    Callers that model flexibility as "which raw reagent molecule gets
    grabbed" (seeding alternate starting states, reagent-count
    bookkeeping) only make sense for this specific shape — a combo whose
    alternatives fire different real reactions using already-existing
    atoms, not raw reagents, needs a different mechanism entirely."""
    if not r.alternatives or any(d < 0 for d in r.delta.values()):
        return None
    types = []
    for _name, extra in r.alternatives:
        if len(extra) != 1:
            return None
        (t, c), = extra.items()
        if c != -1:
            return None
        types.append(t)
    return types


def _reverse_reaction_atoms(r: Reaction) -> Tuple[List[int], List[int]]:
    """(types to remove [product side], types to add [reagent side]) — one
    entry per atom instance, for reversing a single firing of r."""
    remove_types, add_types = [], []
    for atype, d in r.delta.items():
        if d > 0:
            remove_types.extend([atype] * d)
        elif d < 0:
            add_types.extend([atype] * (-d))
    return remove_types, add_types


def _reverse_reaction_options(r: Reaction) -> List[Tuple[str, List[int], List[int]]]:
    """(alt_name, remove_types, add_types) triples — like
    _reverse_reaction_atoms, but one option per alternative instead of one
    fixed (remove_types, add_types). Normally just one option (r's own name
    and fixed delta, unchanged from _reverse_reaction_atoms); a combined
    synthetic reaction (r.alternatives set — see stoichiometry.Reaction /
    solve_recipe_fast) has no single fixed result, so one option per
    alternative, all sharing the same combined reaction's budget — the
    caller tries each as a separate candidate move. Each alternative's full
    delta is r.delta merged with its own extra delta (Reaction.alternatives),
    so BOTH remove_types (product side) and add_types (reagent side) can
    vary per alternative, not just add_types — needed since a combo's
    members don't always even agree on what they produce (e.g. a
    purify_X_to_Y_or_reject_Z_to_Y combo: reject also makes a bystander
    quicksilver product purify doesn't). Each option keeps its own real
    member-reaction name (never the umbrella combined name) precisely so a
    caller can ask "is *this specific alternative* drop-and-create" rather
    than needing one answer for the whole group — calcify_X_or_Y's members
    always agree (none are), but project_X_to_Y_or_purify_X_to_Y's don't
    (only purification is)."""
    if not r.alternatives:
        remove_types, add_types = _reverse_reaction_atoms(r)
        return [(r.name, remove_types, add_types)]
    options = []
    for alt_name, extra in r.alternatives:
        full_delta = dict(r.delta)
        for atype, d in extra.items():
            full_delta[atype] = full_delta.get(atype, 0) + d
        remove_types, add_types = [], []
        for atype, d in full_delta.items():
            if d > 0:
                remove_types.extend([atype] * d)
            elif d < 0:
                add_types.extend([atype] * (-d))
        options.append((alt_name, remove_types, add_types))
    return options


def _elementary_actions_for_name(state: State, reactions: Dict[str, Reaction], name: str) -> List[ReactionAction]:
    """Every single k=1 un-react firing of `name` currently available and
    legal (see _firing_atoms_ok), one action per (alternative type, choice
    of atom instances) — one ReactionAction per literal instance-level
    combination, the same shape _elementary_bond_actions' BondAction uses
    so both can feed the same atom-disjoint combiner (see
    _combined_bond_reaction_neighbors). Used both for reactions that can
    never be pooled by that combiner's per-type catalogs (multi-atom
    firings, space-limited ones — see _elementary_reaction_actions) and as
    the exact fallback for a poolable-shaped name whose candidates this
    state conflict with one of those."""
    actions = []
    r = reactions[name]
    options = _reverse_reaction_options(r)
    for alt_name, remove_types, add_types in options:
        # A combined reaction's move label shows which concrete alternative
        # this action actually used, not the umbrella group name — and
        # _firing_atoms_ok is checked against that same alternative's real
        # name, not the umbrella one, since a combined group's members can
        # disagree on whether they're drop-and-create (project_X_to_Y isn't,
        # purify_X_to_Y is — see _reverse_reaction_options).
        for chosen in _choose_instances(state, remove_types):
            if not _firing_atoms_ok(state, alt_name, chosen):
                continue
            actions.append(ReactionAction(alt_name, frozenset(chosen), name, chosen, add_types))
    return actions


def _elementary_reaction_actions(state: State, reactions: Dict[str, Reaction]) -> List[ReactionAction]:
    """Every instance-level ReactionAction for a reaction that can never be
    pooled by _combined_bond_reaction_neighbors' per-type catalogs: a
    multi-atom firing (group_size > 1, several co-reagents need pairing up)
    or a space-limited one (projection's free-hex bookkeeping, see
    _is_space_limited) — both stay individually/exactly enumerated there.
    Single-atom, non-space-limited reactions are handled by that function's
    catalogs directly from `state` instead of being materialized here first
    (see its docstring) — this only covers what's left."""
    actions = []
    for name, budget in state.reactions_left.items():
        if budget <= 0:
            continue
        if _reaction_pool_key(state, reactions, name) is not None:
            continue  # handled by _combined_bond_reaction_neighbors' catalogs instead
        actions.extend(_elementary_actions_for_name(state, reactions, name))
    return actions


def _apply_mixed_actions(state: State, chosen: List[Action]) -> State:
    """Apply a combo of atom-disjoint elementary actions — some BondActions,
    some ReactionActions (add_types already resolved to whichever
    alternative that action picked — see _reverse_reaction_options) —
    simultaneously. Each firing relabels its own first consumed atom in
    place and removes/spawns the rest; bond removals just drop bonds. The
    sole place either kind of action is actually applied to a state — every
    neighbor-generating function builds action bundles and calls this."""
    bond_remove_by_mol: Dict[int, set] = {}
    relabel_by_mol: Dict[int, Dict[int, tuple]] = {}  # ai -> (new_type, is_drop_and_create)
    extra_remove_by_mol: Dict[int, set] = {}
    spawn_types = []
    reaction_uses: Counter = Counter()

    for action in chosen:
        if isinstance(action, ReactionAction):
            reaction_uses[action.name] += 1
            r_mi, r_ai = action.firing[0]
            relabel_by_mol.setdefault(r_mi, {})[r_ai] = (action.add_types[0], _is_drop_and_create(action.name))
            for mi, ai in action.firing[1:]:
                extra_remove_by_mol.setdefault(mi, set()).add(ai)
            spawn_types.extend(action.add_types[1:])
        else:
            for mi, bis in action.remove_by_mol.items():
                bond_remove_by_mol.setdefault(mi, set()).update(bis)

    new_molecules = []
    for mi, mol in enumerate(state.molecules):
        relabels = relabel_by_mol.get(mi, {})
        removed_bonds = bond_remove_by_mol.get(mi, set())
        removed_atoms = extra_remove_by_mol.get(mi, set())
        if not relabels and not removed_bonds and not removed_atoms:
            new_molecules.append(mol)
            continue

        new_atoms = list(mol.atoms)
        for ai, (new_type, is_dc) in relabels.items():
            old = new_atoms[ai]
            # A plain relabel carries the atom's age over (same atom,
            # continuous existence); a drop-and-create reversal starts both
            # sides fresh (genuine creation boundary).
            new_atoms[ai] = Atom(type=new_type, u=old.u, v=old.v, age=0 if is_dc else old.age)
        # removed_bonds is a set of (bond index, bit to clear) pairs (see
        # _elementary_bond_actions) — clearing a bit only drops a bond
        # entry once its type reaches 0, so one color of a multi-color
        # triplex bond can be removed while the others stay intact.
        clear_bits: Dict[int, int] = {}
        for bi, bit in removed_bonds:
            clear_bits[bi] = clear_bits.get(bi, 0) | bit
        new_bonds = []
        for bi, b in enumerate(mol.bonds):
            bits = clear_bits.get(bi, 0)
            if not bits:
                new_bonds.append(b)
                continue
            new_type = b.type & ~bits
            if new_type:
                new_bonds.append(replace(b, type=new_type))

        if removed_atoms:
            result = _remove_atoms(new_atoms, new_bonds, removed_atoms)
        else:
            result = _split_molecule(new_atoms, new_bonds)
        # A singleton here is only "freshly freed" (age reset) if it wasn't
        # already free-standing before this move.
        new_molecules.extend(_freshly_freed(result) if len(mol.atoms) > 1 else result)

    for atype in spawn_types:
        new_molecules.append(Molecule(atoms=[Atom(type=atype, u=0, v=0, age=0)], bonds=[]))

    new_reactions_left = dict(state.reactions_left)
    for name, uses in reaction_uses.items():
        new_reactions_left[name] -= uses

    return State(new_molecules, new_reactions_left)


def _reaction_pool_key(state: State, reactions: Dict[str, Reaction], name: str):
    """(atype, alt_options) if `name`'s single-atom reversal actions are
    poolable — safe to enumerate by *how many* (and *which kind of*) fire
    rather than materializing "which literal atom" per action first (see
    _atype_reaction_catalog) — else None. alt_options is a list of
    (alt_name, add_types) — every alternative's own real name (never a
    synthesized one), carried through so callers can use it directly as a
    fired action's display name instead of guessing it back from add_types
    (that guess only ever worked for calcify_X_or_Y, whose alternatives
    happen to be named f"calcify_{atom}" — it wouldn't generalize to
    other combo shapes).

    Requires group_size 1 (a lone atom in, no co-reagents to pair up), not
    space-limited (projection still needs its own free-hex bookkeeping per
    firing, so stays in the exact/individual path), every alternative
    sharing the exact same single-atom remove_types, AND every alternative
    agreeing on drop-and-create-ness — the pooling machinery below filters
    candidate atoms once per atype (see pool_names_by_type's is_dc), not
    per alternative, so a combined reaction whose own alternatives
    disagree internally (impossible for calcify_X_or_Y's members, which
    always agree; possible in principle for a future combo mixing e.g. a
    relabeling alternative with a drop-and-create one) can't be pooled
    uniformly and falls back to the exact/individual path instead — this
    is the single source of truth other poolability checks (e.g.
    _elementary_reaction_actions) defer to, rather than re-deriving it."""
    r = reactions[name]
    options = _reverse_reaction_options(r)
    remove_type_sets = {tuple(remove_types) for _alt_name, remove_types, _add_types in options}
    dc_flags = {_is_drop_and_create(alt_name) for alt_name, _remove_types, _add_types in options}
    if len(remove_type_sets) != 1 or len(dc_flags) != 1 or _is_space_limited(name):
        return None
    (remove_types,) = remove_type_sets
    if len(remove_types) != 1:
        return None
    return remove_types[0], [(alt_name, add_types) for alt_name, _remove_types, add_types in options]


def _free_source_allocations(candidates: List[Tuple[int, int]],
                              name_group: list, remaining_budget: Dict[str, int]):
    """Every joint way to split however many of `candidates` (0..len, all
    free-standing atoms of one shared type) go to each (name, alternative)
    in `name_group`, each name capped by remaining_budget[name] — yields
    (bundle, used), used mapping name -> how many candidates it claimed.

    Firing N of these free, same-type, single-atom, non-space-limited
    reversals together always yields the same resulting state no matter
    *which* N of the type's free instances are chosen — molecule_signature
    (state dedup's canonical key, see _state_signature) only sees each
    result's type, not its identity or board position (freshly spawned
    atoms are literally born at a dummy (0, 0) — see _apply_mixed_actions).
    So instead of the exponentially-many raw combinations of "which atom"
    per firing, this counts: how many go to each (name, alternative) — a
    composition problem, polynomial in len(candidates) rather than
    exponential in it."""
    M = len(candidates)

    def alt_splits(total, n_alts):
        if n_alts == 1:
            yield (total,)
            return
        for first in range(total + 1):
            for rest in alt_splits(total - first, n_alts - 1):
                yield (first,) + rest

    def rec(i, remaining):
        if i == len(name_group):
            yield [], {}
            return
        name, alt_options = name_group[i]
        cap = min(remaining_budget.get(name, 0), remaining)
        for t in range(cap + 1):
            for split in alt_splits(t, len(alt_options)):
                for rest_alloc, rest_used in rec(i + 1, remaining - t):
                    used = dict(rest_used)
                    if t:
                        used[name] = used.get(name, 0) + t
                    yield [(name, alt_options, split)] + rest_alloc, used

    for allocation, used in rec(0, M):
        offset = 0
        bundle = []
        for name, alt_options, split in allocation:
            for (alt_name, add_types), k in zip(alt_options, split):
                for _ in range(k):
                    mi, ai = candidates[offset]
                    offset += 1
                    firing = [(mi, ai)]
                    bundle.append(ReactionAction(alt_name, frozenset(firing), name, firing, add_types))
        yield bundle, used


def _orbit_source_allocations(mi: int, orbit: List[int],
                               perms: List[Dict[int, int]], name_group: list,
                               remaining_budget: Dict[str, int]):
    """Every canonical coloring of `orbit`'s atoms (all bonded inside
    molecule `mi`, all matching name_group's shared consumed type — orbits
    are always type-homogeneous, since a valid rotation requires matching
    types at every mapped position) among {unfired} + {(name, alternative)
    for every name in name_group and each alternative}, respecting each
    name's remaining_budget, one representative per equivalence class
    under the molecule's own rotation group `perms`. Yields (bundle, used)
    like _free_source_allocations.

    A rotation mapping the whole molecule onto itself also maps "coloring
    C" onto "coloring rotate(C)", giving both an identical (rotation-
    invariant) molecule_signature, so only one representative per class
    needs generating. Orbit size is bounded by the rotation group's order
    (at most 6, orbit-stabilizer theorem), so this stays cheap even
    brute-forced — a single-atom orbit is the degenerate d=1 case, where
    every coloring is already its own class, reducing to ordinary
    singleton-action enumeration."""
    d = len(orbit)
    orbit_index = {ai: pos for pos, ai in enumerate(orbit)}
    local_perms = []
    for perm in perms:
        local = [None] * d
        ok = True
        for pos, ai in enumerate(orbit):
            j = perm.get(ai)
            if j is None or j not in orbit_index:
                ok = False
                break
            local[pos] = orbit_index[j]
        if ok:
            local_perms.append(tuple(local))

    # symbol 0 = unfired; symbols 1.. = one per (name index, alternative index)
    alphabet: List[Tuple[Optional[int], Optional[int]]] = [(None, None)]
    for ni, (_name, alt_options) in enumerate(name_group):
        for alt_idx in range(len(alt_options)):
            alphabet.append((ni, alt_idx))
    n_symbols = len(alphabet)

    seen_canonical = set()
    for coloring in iproduct(range(n_symbols), repeat=d):
        counts: Dict[str, int] = {}
        over_budget = False
        for c in coloring:
            ni, _alt_idx = alphabet[c]
            if ni is None:
                continue
            name = name_group[ni][0]
            counts[name] = counts.get(name, 0) + 1
            if counts[name] > remaining_budget.get(name, 0):
                over_budget = True
                break
        if over_budget:
            continue
        canon = min([coloring] + [tuple(coloring[p] for p in perm) for perm in local_perms])
        if canon in seen_canonical:
            continue
        seen_canonical.add(canon)
        # The all-unfired coloring is a real, needed yield here (bundle=[],
        # used={}) — not filtered as a no-op the way a single source's
        # local dedup might suggest: _atype_reaction_catalog's joint
        # recursion over multiple sources needs every source, including
        # this one, to be able to explicitly contribute nothing so other
        # sources can independently contribute something (see
        # _free_source_allocations' t=0 case, which yields the same way).
        bundle = []
        used: Dict[str, int] = {}
        for pos, c in enumerate(canon):
            ni, alt_idx = alphabet[c]
            if ni is None:
                continue
            name, alt_options = name_group[ni]
            alt_name, add_types = alt_options[alt_idx]
            ai = orbit[pos]
            firing = [(mi, ai)]
            bundle.append(ReactionAction(alt_name, frozenset(firing), name, firing, add_types))
            used[name] = used.get(name, 0) + 1
        yield bundle, used


def _atype_reaction_catalog(state: State, name_group: list, sources: list):
    """Every joint allocation across ALL of a shared atom type's candidate
    `sources` (each ("free", candidates) or ("orbit", mi, orbit, perms) —
    see _free_source_allocations / _orbit_source_allocations) for every
    reaction name in `name_group` that consumes this type, as (bundle,
    has_reaction) pairs ready to feed _apply_mixed_actions alongside
    individually-enumerated actions. Each name's own reactions_left budget
    is shared across *every* source that offers it — sources are processed
    in sequence, each source's own usage reducing the remaining budget the
    next source sees, so e.g. a name with both free-standing and
    orbit-bonded candidates this state can draw from both without
    over-firing."""
    budgets = {name: state.reactions_left.get(name, 0) for name, _alt_options in name_group}

    def rec(idx, remaining_budget):
        if idx == len(sources):
            yield [], {}
            return
        source = sources[idx]
        if source[0] == "free":
            gen = _free_source_allocations(source[1], name_group, remaining_budget)
        else:
            _kind, mi, orbit, perms = source
            gen = _orbit_source_allocations(mi, orbit, perms, name_group, remaining_budget)
        for bundle, used in gen:
            next_remaining = dict(remaining_budget)
            for name, u in used.items():
                next_remaining[name] -= u
            for rest_bundle, rest_used in rec(idx + 1, next_remaining):
                combined_used = dict(used)
                for name, u in rest_used.items():
                    combined_used[name] = combined_used.get(name, 0) + u
                yield bundle + rest_bundle, combined_used

    for bundle, _used in rec(0, budgets):
        yield bundle, bool(bundle)


def _combined_bond_reaction_neighbors(state: State, parts_available: int,
                                       reactions: Dict[str, Reaction]) -> List[Tuple[State, str]]:
    """Every reaction-firing move: a lone firing, several simultaneous
    firings of one reaction, or an atom-disjoint combo of bonding actions
    with reaction firings / different reactions. The *only* generator of
    reaction-firing moves in this module — single and same-name-simultaneous
    firings are just the size-1/single-name cases of this combinatorics,
    not a separate path. Projection firings additionally need a free hex
    each (_is_space_limited) — several can combine, just only as many as
    there are distinct free hexes.

    Single-atom, non-space-limited reversals — the vast majority of
    available reaction actions in a typical state — are never materialized
    as raw per-instance actions: for each atom type such a reaction
    consumes, every current candidate (free-standing, or bonded inside a
    rotationally-symmetric molecule — _molecule_rotational_group) is
    grouped into one shared "how many, of which kind, fire" catalog
    (_atype_reaction_catalog) up front, class/orbit-native rather than
    enumerated instance-by-instance and pooled after the fact. 30
    simultaneously-reversible single-atom reactions would otherwise blow
    up the exact scan below to 2^30. Multi-atom firings, space-limited
    ones, and bonded targets with no orbit symmetry stay in that exact
    scan — as does any type whose catalog would need to share a reaction's
    budget with a conflicting non-poolable action of the same type (a
    multi-atom or space-limited reaction also touching a free atom of that
    type): simpler and safer to fall back to exact enumeration than risk a
    physical atom being claimed twice.

    Bond-removal actions get the same treatment via _bond_pool_catalogs,
    merged into the same `pool_catalogs` list: a symmetric molecule's
    interchangeable bonds are pooled by equivalence class instead of
    joining the exact scan, same rationale as the reaction case."""
    individual_reaction_actions = _elementary_reaction_actions(state, reactions)

    blocked_types = set()
    for action in individual_reaction_actions:
        for mi, ai in action.touched:
            if len(state.molecules[mi].atoms) == 1:
                blocked_types.add(state.molecules[mi].atoms[0].type)

    pool_names_by_type: Dict[int, list] = {}
    for name, budget in state.reactions_left.items():
        if budget <= 0:
            continue
        key = _reaction_pool_key(state, reactions, name)
        if key is None:
            continue
        atype, alt_options = key
        # _reaction_pool_key already required every alternative to agree on
        # drop-and-create-ness, so any one of them gives the right answer.
        is_dc = _is_drop_and_create(alt_options[0][0])
        pool_names_by_type.setdefault(atype, []).append((name, alt_options, is_dc))

    pool_catalogs = []
    molecule_orbits_cache: Dict[int, tuple] = {}
    reaction_orbit_molecules: set = set()

    def get_orbits(mi):
        if mi not in molecule_orbits_cache:
            molecule_orbits_cache[mi] = _molecule_rotational_group(state.molecules[mi])
        return molecule_orbits_cache[mi]

    for atype, name_group_raw in pool_names_by_type.items():
        dc_flags = {is_dc for _name, _add_types, is_dc in name_group_raw}
        if atype in blocked_types or len(dc_flags) > 1:
            for name, _add_types, _is_dc in name_group_raw:
                individual_reaction_actions.extend(_elementary_actions_for_name(state, reactions, name))
            continue

        is_dc = dc_flags.pop()
        name_group = [(name, alt_options) for name, alt_options, _is_dc in name_group_raw]

        free_candidates = [(mi, 0) for mi, mol in enumerate(state.molecules)
                            if len(mol.atoms) == 1 and mol.atoms[0].type == atype
                            and (not is_dc or mol.atoms[0].age >= _READY_AGE)]
        sources = [("free", free_candidates)] if free_candidates else []
        if not is_dc:
            for mi, mol in enumerate(state.molecules):
                if len(mol.atoms) < 2:
                    continue
                orbits, perms = get_orbits(mi)
                matching = [ai for orbit in orbits for ai in orbit if mol.atoms[ai].type == atype]
                if matching:
                    sources.append(("orbit", mi, matching, perms))
                    reaction_orbit_molecules.add(mi)

        if not sources:
            continue
        pool_catalogs.append(list(_atype_reaction_catalog(state, name_group, sources)))

    bond_actions, bond_catalogs = _bond_pool_catalogs(state, parts_available, get_orbits=get_orbits,
                                                        exclude_molecules=reaction_orbit_molecules)
    pool_catalogs.extend([[(bundle, False) for bundle in catalog] for catalog in bond_catalogs])

    individual_actions: List[Action] = bond_actions + individual_reaction_actions
    n = len(individual_actions)
    occupied = _occupied_positions(state)

    individual_subsets = [[]]
    for size in range(1, n + 1):
        for combo in combinations(range(n), size):
            touched_union = set()
            disjoint = True
            for idx in combo:
                touched = individual_actions[idx].touched
                if touched & touched_union:
                    disjoint = False
                    break
                touched_union |= touched
            if not disjoint:
                continue
            chosen = [individual_actions[idx] for idx in combo]
            name_counts = Counter(a.name for a in chosen if isinstance(a, ReactionAction))
            if any(count > state.reactions_left.get(name, 0) for name, count in name_counts.items()):
                continue
            individual_subsets.append(chosen)

    neighbors_out = []
    for pool_choice in iproduct(*pool_catalogs) if pool_catalogs else [()]:
        pool_bundle = [a for bundle, _has in pool_choice for a in bundle]
        pool_has_reaction = any(has for _bundle, has in pool_choice)
        for ind_subset in individual_subsets:
            total = pool_bundle + ind_subset
            has_reaction = pool_has_reaction or any(isinstance(a, ReactionAction) for a in ind_subset)
            if not has_reaction:
                continue

            touched_union = set()
            ok = True
            for a in total:
                if a.touched & touched_union:
                    ok = False
                    break
                touched_union |= a.touched
            if not ok:
                continue

            space_anchors = []
            for a in total:
                if isinstance(a, ReactionAction) and _is_space_limited(a.name):
                    mi, ai = a.firing[0]
                    space_anchors.append((state.molecules[mi].atoms[ai].u, state.molecules[mi].atoms[ai].v))
            if space_anchors and not _has_distinct_free_hexes(space_anchors, occupied):
                continue

            labels = [a.label for a in total]
            move = f"{len(total)}x{labels[0]}" if all(l == labels[0] for l in labels) else "+".join(labels)
            neighbors_out.append((_apply_mixed_actions(state, total), move))
    return neighbors_out


def _atom_instances_by_type(state: State) -> Dict[int, List[Tuple[int, int]]]:
    """atom type -> list of (molecule index, atom index) instances."""
    by_type: Dict[int, List[Tuple[int, int]]] = {}
    for mi, mol in enumerate(state.molecules):
        for ai, a in enumerate(mol.atoms):
            by_type.setdefault(a.type, []).append((mi, ai))
    return by_type


_DROP_AND_CREATE = {"animismus", "dispersion", "unification"}
_DROP_AND_CREATE_PREFIXES = ("purify_", "proliferate_", "divide_")


def _is_drop_and_create(name: str) -> bool:
    return name in _DROP_AND_CREATE or name.startswith(_DROP_AND_CREATE_PREFIXES)


# A drop-and-create reaction's product atom(s) can't be immediately re-grabbed
# for another drop-and-create firing the instant they appear — the glyph
# drops its inputs, fires, and only *then* is the result there to be picked
# back up, so a same-cycle re-use has nowhere to hide that pickup. If the
# atom instead sits free for a couple of "wait" moves first (see
# _wait_neighbors), that pickup overlaps with whatever else is happening
# during that idle time and costs nothing extra. READY_AGE=2 matches the
# rule: 0 the cycle it appears, 1 after one wait, 2 (ready) after two —
# capped there since a longer wait changes nothing further.
_READY_AGE = 2


def _firing_atoms_ok(state: State, name: str, firing: List[Tuple[int, int]]) -> bool:
    """True if `firing`'s atoms satisfy this reaction's unbonded
    requirements. Most reactions (calcification, projection, duplication,
    rejection, proliferation) relabel their first consumed atom in place,
    keeping its position *and bonds* — a transformation glyph applied to an
    atom already part of a molecule — so only the extras beyond index 0
    (separate co-reagents fed in from outside, e.g. projection's
    quicksilver or rejection's 2nd atom) need to be free-standing.
    animismus/dispersion/unification/purification are different: they drop
    every atom they consume and create fresh ones rather than transforming
    anything in place (there's no atom that "is" the output), so *all*
    their consumed atoms — not just the extras — must already be
    free-standing, *and* (see _READY_AGE) old enough to plausibly have sat
    around since a genuine firing produced them. An unbonded atom is exactly
    one whose molecule has just that 1 atom (state.molecules is always split
    into connected components — see _split_molecule)."""
    check_from = 0 if _is_drop_and_create(name) else 1
    if not all(len(state.molecules[mi].atoms) == 1 for mi, _ai in firing[check_from:]):
        return False
    if _is_drop_and_create(name):
        return all(state.molecules[mi].atoms[ai].age >= _READY_AGE for mi, ai in firing)
    return True


def _choose_instances(state: State, remove_types: List[int]):
    """Every way to pick distinct (molecule, atom) instances matching the
    required type multiset, one combination per yield."""
    needed = Counter(remove_types)
    by_type = _atom_instances_by_type(state)
    per_type_choices = []
    for atype, count in needed.items():
        candidates = by_type.get(atype, [])
        if len(candidates) < count:
            return  # this reaction's product side isn't present — can't reverse it
        per_type_choices.append(list(combinations(candidates, count)))
    for combo in iproduct(*per_type_choices):
        yield [inst for group in combo for inst in group]


def _is_space_limited(name: str) -> bool:
    """Rejection creates a brand-new quicksilver atom next to the metal atom
    it's relabeling — that hex can't be shared by two simultaneous firings,
    unlike e.g. calcification's plain in-place relabel. See
    _free_neighbor_positions / _has_distinct_free_hexes: the relabeled
    atom's own position is real, tracked board geometry (not an
    abstracted-away detail — see module docstring's Simplifications section
    on why only *its* position is real, not any second/third atom a
    multi-atom reaction also needs), so this can actually be checked rather
    than assumed unlimited."""
    return name.startswith("project_") or name.startswith("reject_")


def _occupied_positions(state: State) -> set:
    return {(a.u, a.v) for mol in state.molecules for a in mol.atoms}


def _free_neighbor_positions(pos: Tuple[int, int], occupied: set) -> List[Tuple[int, int]]:
    u, v = pos
    return [(u + du, v + dv) for du, dv in _HEX_DIRS if (u + du, v + dv) not in occupied]


def _has_distinct_free_hexes(anchors: List[Tuple[int, int]], occupied: set) -> bool:
    """True if every anchor position can be assigned its own distinct free
    (unoccupied) hex neighbor to place a simultaneous projection's result in
    — a small bipartite matching (Kuhn's algorithm; anchor counts here are
    always tiny) since two anchors can share a candidate free hex but not
    both actually use it."""
    candidates = [_free_neighbor_positions(a, occupied) for a in anchors]
    match_to_anchor: Dict[Tuple[int, int], int] = {}

    def try_assign(i, seen):
        for pos in candidates[i]:
            if pos in seen:
                continue
            seen.add(pos)
            if pos not in match_to_anchor or try_assign(match_to_anchor[pos], seen):
                match_to_anchor[pos] = i
                return True
        return False

    return all(try_assign(i, set()) for i in range(len(anchors)))


_MOVE_WAIT = "wait"


def _age_tracked_atoms(state: State, tracked_types: frozenset,
                        eligible_ids: Optional[frozenset] = None) -> Optional[State]:
    """Every free (single-atom-molecule) atom of a `tracked_types` type (the
    actual product types of this puzzle's drop-and-create reactions — see
    reachable_states) below _READY_AGE ages by 1 (capped there), nothing
    else about the state changes. None if nothing would change (every
    tracked free atom already mature). Split out from _wait_neighbors so
    _combine_with_wait can layer this same aging on top of an *already
    computed* bond/reaction move — this is the one piece of state-mutation
    logic both need, and the "is anything eligible" check has to be
    identical in both places or the two could disagree on when a "+wait"
    variant is legal.

    `eligible_ids`, if given, restricts aging to Molecule objects whose
    `id()` is in that set — see _combine_with_wait for why identity, not
    position: a move that just created or freed a tracked atom leaves it
    sitting at age 0 too, at the same dummy spawn coordinate _apply_mixed_actions
    always uses, so position alone can't tell it apart from a genuinely
    pre-existing age-0 atom that happens to already be there."""
    changed = False
    new_molecules = []
    for mol in state.molecules:
        a = mol.atoms[0] if len(mol.atoms) == 1 else None
        if (a is not None and a.type in tracked_types and a.age < _READY_AGE
                and (eligible_ids is None or id(mol) in eligible_ids)):
            changed = True
            new_molecules.append(replace(mol, atoms=[replace(a, age=a.age + 1)]))
        else:
            new_molecules.append(mol)
    if not changed:
        return None
    return State(new_molecules, dict(state.reactions_left))


def _wait_neighbors(state: State, tracked_types: frozenset) -> List[Tuple[State, str]]:
    """One "wait" move, alone: every eligible atom ages by 1 (_age_tracked_atoms)
    and nothing else about the state changes — modeling the real idle cycles
    a drop-and-create reaction's product atom (or any atom freshly freed by
    an unbond — see Atom.age) needs before it "matures" enough to be picked
    back up for another drop-and-create firing (_firing_atoms_ok). Omitted
    entirely once nothing would change: an edge to the identical state would
    be a self-loop, which would break the topological order bounds.py's path
    search relies on and contradicts the module docstring's termination
    argument (nothing it tracks would be decreasing)."""
    aged = _age_tracked_atoms(state, tracked_types)
    if aged is None:
        return []
    return [(aged, _MOVE_WAIT)]


def _combine_with_wait(state: State, moves: List[Tuple[State, str]],
                        tracked_types: frozenset) -> List[Tuple[State, str]]:
    """For every (predecessor state, move label) some *other* neighbor
    function already produced from `state`, also offer the variant where
    the wait-eligible atoms *already sitting in `state`* age by 1 in that
    same step — modeling that idle-cycle aging isn't a move of its own
    competing for the same cycle, it's real elapsed time that keeps passing
    no matter what else the puzzle's other glyphs are doing simultaneously.
    Without this, the search could only ever age tracked atoms in a move
    that does *nothing* else (_wait_neighbors alone), forcing every "do X,
    then wait for something unrelated to mature" sequence into two separate
    BFS steps even when X and the aging don't touch a single shared atom —
    inflating a path's length (and therefore bounds.py's L_spine latency
    figure) for no real reason.

    Restricted to *the same Molecule objects* already present in `state`
    *before* `move` ran — not just "whatever's eligible in move's own
    result" — because a move that itself creates or frees a tracked atom
    (a fresh drop-and-create product, or a singleton an unbond just split
    off — both start at age 0, see _apply_mixed_actions/_freshly_freed)
    would otherwise be indistinguishable from a genuinely pre-existing
    age-0 atom, letting it tick its first age-step in the very same move
    it was born — one real cycle's worth of aging manufactured for free.
    Identity (id()), not position, is what actually distinguishes them:
    every freshly spawned atom lands at the same dummy (0, 0)
    _apply_mixed_actions always uses, so position alone can collide with
    an unrelated, genuinely pre-existing atom that happens to already sit
    there too — id() can't, since _apply_mixed_actions reuses the exact
    same object reference for any molecule a move doesn't touch (see its
    `if not relabels and not removed_bonds and not removed_atoms:
    new_molecules.append(mol)` fast path) and always constructs a new one
    for anything it does. An atom `move` *consumes* is simply gone from
    its result and never considered either way."""
    eligible_ids = frozenset(
        id(mol) for mol in state.molecules
        if len(mol.atoms) == 1 and mol.atoms[0].type in tracked_types and mol.atoms[0].age < _READY_AGE
    )
    if not eligible_ids:
        return []
    out = []
    for nxt, move in moves:
        aged = _age_tracked_atoms(nxt, tracked_types, eligible_ids)
        if aged is not None:
            out.append((aged, f"{move}+wait"))
    return out


def neighbors(state: State, reactions: Dict[str, Reaction], parts_available: int,
              tracked_types: frozenset) -> List[Tuple[State, str]]:
    """Every (predecessor state, move label) pair reachable from `state` by
    one unbond, one triple-bond (multi-bonder) reversal, any atom-disjoint
    combination of 2+ of those, one or several simultaneous un-react moves
    (alone or atom-disjointly combined with bonding actions / other
    reactions — see _combined_bond_reaction_neighbors, the sole source of
    reaction-firing moves), one "wait" alone (_wait_neighbors), or any of
    the above combined with a simultaneous wait (_combine_with_wait) —
    unbond/triple-bond moves only apply if the puzzle actually grants that
    glyph. Move label is "bond", "triple-bond", "NxLABEL" for N combined
    same-kind moves, "label+label" for a mixed combination, "wait", the
    reaction name, or any of those with a trailing "+wait"."""
    other = (_unbond_neighbors(state, parts_available)
             + _triple_unbond_neighbors(state, parts_available)
             + _combined_bond_neighbors(state, parts_available)
             + _combined_bond_reaction_neighbors(state, parts_available, reactions))
    return (other
            + _wait_neighbors(state, tracked_types)
            + _combine_with_wait(state, other, tracked_types))


# ── forward search (meet-in-the-middle fallback) ─────────────────────────
#
# Everything below builds a small FORWARD (real-chronological) search from
# the puzzle's actual raw reagents, used only as a fallback when the
# backward search above finds no state matching them (see
# _forward_fallback and reachable_states). "react" is the mirror of
# un-react (_reverse_reaction_options, _choose_instances, _firing_atoms_ok,
# _is_space_limited/_has_distinct_free_hexes/_occupied_positions, and
# _apply_mixed_actions are all direction-agnostic and reused unchanged —
# none of them reference "product" vs. "reagent" by name, only "atoms this
# firing touches"). "debond" is a *new* action, not a reuse of
# _elementary_bond_actions: that backward machinery models undoing
# whichever *bonding* glyph (Bonder/Bonder-Prisma/Multi-bonder) built a
# structure, each at its own construction granularity (one triplex color
# per Bonder-Prisma pass, 2-3 bonds per Multi-bonder firing) — correct for
# that purpose, verified against omsim/sim.c, and left untouched. Forward
# debonding models the real Unbonder glyph (PART_UNBONDER) tearing apart
# the raw reagent's already-given bonds instead: one Unbonder action
# clears an entire bond — normal *and* every triplex color set on it —
# simultaneously (sim.c's UNBONDING case: `bond_direction(...)` spans
# NORMAL_BONDS and all three TRIPLEX_*_BONDS masks for one direction,
# cleared together via one `&= ~ab`), and there is no multi-junction
# equivalent for un-bonding at all (MULTI_BONDING is forward-only, bond
# creation, in sim.c).

def _elementary_forward_reaction_actions(state: State, reactions: Dict[str, Reaction],
                                          name: str) -> List[ReactionAction]:
    """Every single forward k=1 firing of `name` currently available and
    legal — the mirror of _elementary_actions_for_name: consumes the
    reagent side (`_reverse_reaction_options`'s add_types, tried as
    separate candidate alternatives for a combined reaction) and produces
    the product side (its remove_types, becoming the fired ReactionAction's
    own add_types — the same field _apply_mixed_actions already
    relabels/spawns from, regardless of which direction populated it)."""
    actions = []
    r = reactions[name]
    options = _reverse_reaction_options(r)
    for alt_name, produce_types, consume_types in options:
        for chosen in _choose_instances(state, consume_types):
            if not _firing_atoms_ok(state, alt_name, chosen):
                continue
            actions.append(ReactionAction(alt_name, frozenset(chosen), name, chosen, produce_types))
    return actions


_MOVE_DEBOND = "debond"


def _elementary_debond_actions(state: State, parts_available: int) -> List[BondAction]:
    """Every individual Unbonder firing currently available: one whole-bond
    removal — normal type and every triplex color currently set on it,
    cleared together (see this section's own docstring for why, verified
    against sim.c) — per bond. Gated solely on PART_UNBONDER, unlike the
    backward un-bond moves' PART_BONDER / PART_BONDER_PRISMA /
    PART_BONDER_SPEED gating (which key off which *bonding* glyph would
    have built the structure, not the Unbonder)."""
    if not (parts_available & PART_UNBONDER):
        return []
    actions = []
    for mi, mol in enumerate(state.molecules):
        pos_to_index = {(a.u, a.v): i for i, a in enumerate(mol.atoms)}
        for bi, b in enumerate(mol.bonds):
            ai = pos_to_index.get((b.from_u, b.from_v))
            aj = pos_to_index.get((b.to_u, b.to_v))
            if ai is None or aj is None:
                continue
            touched = frozenset({(mi, ai), (mi, aj)})
            actions.append(BondAction(_MOVE_DEBOND, touched, {mi: {(bi, b.type)}}))
    return actions


def _combined_forward_neighbors(state: State, reactions: Dict[str, Reaction],
                                 parts_available: int) -> List[Tuple[State, str]]:
    """Every atom-disjoint combo of debond actions and forward reaction
    firings — a single action is just the size-1 case of this, not a
    separate path (mirrors _combined_bond_reaction_neighbors' role on the
    backward side). Plain exact combinations() scan, no rotation-orbit
    pooling: forward's seed-derived states are small (no known puzzle
    hands the forward search a large symmetric raw-reagent molecule that
    would make 2^n prohibitive here), so the pooling optimization
    per-state combinatorics needs backward isn't needed here.

    Combining matters for more than just move count: a puzzle can need a
    debond and a reaction to fire *simultaneously* so that the atom the
    debond frees is immediately available (already aged/positioned) for
    the very next simultaneous debond+react pair, not one full extra step
    later. Confirmed on P024's true-optimal path — debond,
    debond+duplicate_air, debond+duplicate_earth, drop (L_spine 4) — which
    an uncombined one-action-at-a-time search can never produce; it always
    finds the longer debond, debond, debond, duplicate_air+duplicate_earth,
    drop (L_spine 5) instead, a real (if small) latency-bound overshoot."""
    debond_actions = _elementary_debond_actions(state, parts_available)
    react_actions: List[ReactionAction] = []
    for name, budget in state.reactions_left.items():
        if budget <= 0:
            continue
        react_actions.extend(_elementary_forward_reaction_actions(state, reactions, name))

    individual_actions: List[Action] = debond_actions + react_actions
    n = len(individual_actions)

    neighbors_out = []
    for size in range(1, n + 1):
        for combo in combinations(range(n), size):
            touched_union = set()
            disjoint = True
            for idx in combo:
                touched = individual_actions[idx].touched
                if touched & touched_union:
                    disjoint = False
                    break
                touched_union |= touched
            if not disjoint:
                continue
            chosen = [individual_actions[idx] for idx in combo]

            name_counts = Counter(a.name for a in chosen if isinstance(a, ReactionAction))
            if any(count > state.reactions_left.get(name, 0) for name, count in name_counts.items()):
                continue

            # Space-limited anchors are checked per molecule (own connected
            # component), never against the whole state — see
            # _elementary_forward_reaction_actions' caller docstring history
            # / reachable_states' P022 note — grouped by molecule since two
            # simultaneous anchors in the *same* molecule still need
            # distinct hexes (_has_distinct_free_hexes), but anchors in
            # different molecules never compete for each other's hexes.
            space_anchors_by_mol: Dict[int, List[Tuple[int, int]]] = {}
            for a in chosen:
                if isinstance(a, ReactionAction) and _is_space_limited(a.name):
                    mi, ai = a.firing[0]
                    pos = (state.molecules[mi].atoms[ai].u, state.molecules[mi].atoms[ai].v)
                    space_anchors_by_mol.setdefault(mi, []).append(pos)
            if space_anchors_by_mol:
                blocked = False
                for mi, anchors in space_anchors_by_mol.items():
                    occupied = {(atom.u, atom.v) for atom in state.molecules[mi].atoms}
                    if not _has_distinct_free_hexes(anchors, occupied):
                        blocked = True
                        break
                if blocked:
                    continue

            labels = [a.label for a in chosen]
            move = f"{len(chosen)}x{labels[0]}" if all(l == labels[0] for l in labels) else "+".join(labels)
            neighbors_out.append((_apply_mixed_actions(state, chosen), move))
    return neighbors_out


def _forward_neighbors(state: State, reactions: Dict[str, Reaction], parts_available: int,
                        tracked_types: frozenset) -> List[Tuple[State, str]]:
    """Every (successor state, move label) pair reachable from `state`
    (input-side, real-chronological) by any atom-disjoint combo of debond
    actions and forward reaction firings (_combined_forward_neighbors,
    including the single-action case), one "wait" alone, or any of those
    combined with a simultaneous wait (_combine_with_wait — same fix as
    the backward direction, applies equally forward)."""
    other = _combined_forward_neighbors(state, reactions, parts_available)
    return (other
            + _wait_neighbors(state, tracked_types)
            + _combine_with_wait(state, other, tracked_types))


_FORWARD_FALLBACK_STATE_CAP = 3000


def _forward_fallback(pf: PuzzleFile, recipe: RecipeResult, graph: StateGraph,
                       reactions: Dict[str, Reaction], parts_available: int,
                       tracked_types: frozenset, products_needed: int) -> bool:
    """Only called when the backward-only match search (reachable_states)
    found no state matching the raw reagents. Tries each _seed_input_states
    candidate in turn: a local forward BFS (react/debond/wait, via
    _forward_neighbors) from that seed, stopping expansion at any state
    whose *molecule content* (mol_sigs — the first half of
    _state_signature, ignoring reactions_left) already exists in `graph` (a
    "boundary" — the fuller backward neighbors() already computed that
    node's edges exhaustively and correctly, so it's never re-derived here)
    and capped at _FORWARD_FALLBACK_STATE_CAP locally-discovered states.

    Boundary matching deliberately ignores reactions_left: forward and
    backward each track it against their own full per-unit budget, in
    opposite consumption directions, so a real meeting point's two halves
    almost never agree on it even when the molecule content is identical
    (confirmed via a hand-traced P022 meeting point — post-debond,
    post-react forward state vs. the topologically identical backward
    state reached by pure un-bond: same atoms/bonds, reactions_left 0 vs.
    1). This is sound here specifically because any reaction a forward
    candidate already fired is one whose backward un-firing was, by
    construction, blocked at every node on this fallback's search path
    (that's the entire reason reachable_states' plain backward search
    failed and this fallback is running at all) — so the matched backward
    node's own nominally "unspent" budget for that reaction is dead weight
    it can never actually use going further backward, not a real
    double-spend risk. A candidate that never reaches a boundary fails
    outright — nothing from it is written into `graph`, only a confirmed
    connection is committed. Returns True (with
    graph.input_state_indices/input_state_idx set from every successful
    candidate's seed) if at least one candidate connects, False if every
    candidate fails."""
    boundary_index: Dict[tuple, List[int]] = {}
    for idx, s in enumerate(graph.states):
        boundary_index.setdefault(_state_signature(s, tracked_types)[0], []).append(idx)

    successful_seed_indices = []
    for seed in _seed_input_states(pf, recipe, products_needed, tracked_types):
        seed_key = _state_signature(seed, tracked_types)
        local_states: Dict[tuple, State] = {seed_key: seed}
        local_edges: List[Tuple[tuple, tuple, str]] = []
        boundary_edges: List[Tuple[tuple, int, str]] = []  # (from_key, backward graph idx, move)
        queue = deque([seed])
        while queue:
            if len(local_states) > _FORWARD_FALLBACK_STATE_CAP:
                break
            current = queue.popleft()
            current_key = _state_signature(current, tracked_types)
            for nxt, move in _forward_neighbors(current, reactions, parts_available, tracked_types):
                nxt_key = _state_signature(nxt, tracked_types)
                gidxs = boundary_index.get(nxt_key[0])
                if gidxs:
                    for gidx in gidxs:
                        boundary_edges.append((current_key, gidx, move))
                    continue  # nxt IS an existing backward node — don't re-add/re-expand it
                local_edges.append((current_key, nxt_key, move))
                if nxt_key not in local_states:
                    local_states[nxt_key] = nxt
                    queue.append(nxt)
        if not boundary_edges:
            continue  # this candidate never reached the backward graph — try the next one

        key_to_idx: Dict[tuple, int] = {}
        for key, state in local_states.items():
            key_to_idx[key] = graph._index[key] if key in graph._index else graph.add_state(state, key)
        for from_key, to_key, move in local_edges:
            # forward walk's edges point input-side -> output-side; the
            # graph's convention is output-side -> input-side (see
            # StateGraph docstring / reachable_states), so flip.
            graph.add_edge(key_to_idx[to_key], key_to_idx[from_key], move)
        for from_key, gidx, move in boundary_edges:
            graph.add_edge(gidx, key_to_idx[from_key], move)
        successful_seed_indices.append(key_to_idx[seed_key])

    if not successful_seed_indices:
        return False
    graph.input_state_indices = successful_seed_indices
    graph.input_state_idx = successful_seed_indices[0]
    return True


# ── BFS ────────────────────────────────────────────────────────────────────

def _matches_raw_reagents(state: State, pf: PuzzleFile, recipe: RecipeResult, products_needed: int,
                           exact: bool = False) -> bool:
    """True if `state` is exactly the puzzle's raw reagents, one unit copy
    each (per recipe.reagent_counts // _unit_scale — not products_needed
    directly, see initial_state's docstring for why). For a reagent
    covered by a combined reaction group (a recipe.reaction_counts entry
    with .alternatives set, see solve_recipe_fast), exact=False (the default) accepts ANY split of
    single-atom molecules among the group's alternative types as long as
    their combined count matches the group's total — some split has to be
    accepted, since solve_recipe's own one arbitrary split may not even be
    reachable — but bounds.py's L_spine computation still keys off
    recipe.reagent_counts per individual type, so reachable_states actually
    wants exact=True first (matching recipe.reagent_counts's specific
    per-type numbers precisely, when some reachable state realizes them)
    and only falls back to this looser exact=False match if no state does.
    A linear scan rather than a signature dict-lookup — reachable_states
    only needs this once, over however many states the BFS found.

    recipe.reagent_counts (see stoichiometry.RecipeResult.reagent_counts /
    solve_recipe_fast) is a (min, max) pair per index — min < max marks a
    reagent whose max is a ceiling, not an exact target, so a state using
    anywhere from 0 up to that many is accepted, not just an exact hit.
    This is a different kind of flexibility than the combined-group one
    above (which is exact on its *total*, just free about which specific
    type each unit is) — a project_X_to_Y_or_purify_X_to_Y combined
    reaction can genuinely need anywhere from the true minimum up to this
    ceiling worth of the varying resource, depending on which alternative
    a given firing chose, so pinning to an exact figure here would
    wrongly reject real, cheaper decompositions."""
    g = _unit_scale(recipe, products_needed)
    combined_types = {t for r in recipe.reaction_counts if r.alternatives is not None
                       for t in (_single_atom_alt_types(r) or [])}

    def is_combined_atom(mol):
        atoms = mol.atom_type_counts()
        return len(atoms) == 1 and not mol.bonds and next(iter(atoms)) in combined_types

    expected = Counter()
    upper_bound_sigs = set()
    for i, (lo, hi) in recipe.reagent_counts.items():
        mol = pf.inputs[i]
        if is_combined_atom(mol) and not exact:
            continue  # covered by combined_total instead of an exact per-reagent count
        sig = molecule_signature(mol)
        expected[sig] += hi // g
        if lo != hi:
            upper_bound_sigs.add(sig)

    combined_total = None
    if not exact:
        combined_total = sum(hi // g for i, (_lo, hi) in recipe.reagent_counts.items()
                              if is_combined_atom(pf.inputs[i]))

    remaining = Counter(expected)
    combined_found = 0
    for mol in state.molecules:
        if is_combined_atom(mol) and not exact:
            combined_found += 1
            continue
        sig = molecule_signature(mol)
        if remaining[sig] <= 0:
            return False
        remaining[sig] -= 1

    if exact:
        return all(v == 0 for v in remaining.values())
    # An upper-bound signature is satisfied by using anywhere from 0 up to
    # its expected ceiling (remaining[sig] > 0 there just means less than
    # the max was used, which is fine); every other signature still needs
    # an exact hit (remaining[sig] == 0).
    for sig, left in remaining.items():
        if left != 0 and sig not in upper_bound_sigs:
            return False
    return combined_found == combined_total


def reachable_states(pf: PuzzleFile, recipe: RecipeResult) -> StateGraph:
    """BFS over every state reachable from the output molecule(s) by
    unbond/un-react moves, respecting the per-unit reaction budget. Returns
    the graph of every distinct state found, and which states are directly
    reachable from which."""
    reactions = {r.name: r for r in build_reactions(pf)}
    reactions.update({r.name: r for r in recipe.reaction_counts if r.alternatives is not None})
    tracked_types = frozenset(
        atype
        for name, r in reactions.items()
        if _is_drop_and_create(name) and recipe.reaction_counts.get(r, (0, 0))[1] > 0
        for atype, d in r.delta.items()
        if d > 0
    )
    start = initial_state(pf, recipe)
    start_key = _state_signature(start, tracked_types)

    graph = StateGraph()
    start_idx = graph.add_state(start, start_key)
    queue = deque([(start, start_idx)])
    while queue:
        current, current_idx = queue.popleft()
        for nxt, move in neighbors(current, reactions, pf.parts_available, tracked_types):
            key = _state_signature(nxt, tracked_types)
            if key in graph._index:
                nxt_idx = graph._index[key]
            else:
                nxt_idx = graph.add_state(nxt, key)
                queue.append((nxt, nxt_idx))
            graph.add_edge(current_idx, nxt_idx, move)

    products_needed = pf.products_needed()
    match_indices = [i for i, s in enumerate(graph.states)
                      if _matches_raw_reagents(s, pf, recipe, products_needed, exact=True)]
    if not match_indices:
        match_indices = [i for i, s in enumerate(graph.states)
                          if _matches_raw_reagents(s, pf, recipe, products_needed)]
    if match_indices:
        graph.input_state_indices = match_indices
        graph.input_state_idx = match_indices[0]
    else:
        merged = _forward_fallback(pf, recipe, graph, reactions, pf.parts_available,
                                    tracked_types, products_needed)
        assert merged, (
            f"{pf.name!r}: no reachable state matches the actual raw reagents — "
            "the un-react/unbond move rules failed to find a valid full "
            "decomposition path that solve_recipe already proved exists"
        )
    return graph
