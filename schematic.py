"""
schematic.py — backward state-space search from a puzzle's output molecule(s)
down toward its raw reagents.

Builds on the atom-count-level recipe (stoichiometry.solve_recipe) by adding molecule
*structure*: each state is a list of Molecule objects (atoms on the hex grid,
same representation puzzle_parser uses everywhere else) plus how many
firings of each transformation reaction are still available. Starting from
one unit copy of the output molecule(s), two kinds of moves generate predecessor states:

  - un-bond: remove one existing bond. If it was a bridge, the molecule
    splits into two (the list grows by one); if it was a ring-closing edge,
    the molecule just loses that bond and stays in one piece.
  - un-react: reverse one firing of an available reaction — remove the
    atoms on its product side (matching types/counts) from wherever they
    sit in the current state, and add new, freshly-spawned, unbonded atoms
    for its reagent side. Spends one unit of that reaction's remaining
    budget.

A BFS from the initial state, following every neighbor, enumerates every
combination of unbond/un-react moves reachable within the reaction budget —
i.e. every structurally-distinct way the recipe's atom bookkeeping could
have been assembled, without yet committing to geometry/cycle timing (still
out of scope here, same as stoichiometry.py) — with one exception: a third
kind of move, "wait" (_wait_neighbors), ages every free atom below
_READY_AGE by one, modeling the real idle cycles a drop-and-create
reaction's product atom needs before it's old enough to be picked up again
(see _READY_AGE / Atom.age) — everything else about the state is untouched.

Termination: every unbond/un-react move strictly decreases (total bonds
across all molecules) + (total reactions_left) — unbond removes one bond
(triple-bond reversal removes three at once) and never adds any; un-react
spends one unit of budget and the atoms it adds are always unbonded. "wait"
is the one move that leaves that sum unchanged, but it strictly decreases a
second, lexicographically-following measure — sum of (_READY_AGE - age)
over every free atom below _READY_AGE — and is never offered at all once
that's already zero (see _wait_neighbors), so the lexicographic pair
(bonds+reactions_left, age deficit) still strictly decreases on every move,
bounded below by (0, 0): the state graph is still a finite DAG and the BFS
still always terminates.

Simplifications (deliberately out of scope for this module):
  - An un-react move relabels its *first* consumed atom in place (keeping
    its position and bonds, matching how a transformation glyph actually
    works — the atom doesn't detach from its molecule) and turns it into the
    reaction's first produced type. Any *additional* atoms the reaction
    needs beyond that one (e.g. purification's second lo atom, or
    projection's separately-needed quicksilver) have no real position yet,
    so they're spawned as a fresh single-atom molecule at local (0, 0).
    Where a transformation glyph would actually put multiple simultaneous
    inputs/outputs is a geometry/choreography question (PLAN.md Phase A/B),
    not a stoichiometric one — this is a simplification, not a claim that
    every atom in a multi-atom reaction ends up in the geometrically correct
    place relative to the others.
  - Catalyst atoms (stoichiometry.Reaction.catalyst) aren't modeled here —
    solve_recipe already verified a catalyst-supplying reagent exists before
    handing us reaction_unit_counts, so every budgeted firing is assumed
    usable.
  - State dedup uses a color-refinement graph signature (a few rounds of
    "recolor each atom by its own + its neighbors' colors"), not a true
    graph-isomorphism test — enough to collapse the overwhelming majority of
    symmetric duplicates (e.g. which of several identical salt atoms a move
    picked) without the cost of exact canonicalization; pathologically
    symmetric molecules could in rare cases be treated as distinct when
    they're actually isomorphic.

    ASSUMPTIONS:
    reaction_counts // ouputs_needed is sufficient to create optimal path
"""

from collections import Counter, deque
from dataclasses import dataclass, replace
from itertools import combinations, product as iproduct
from typing import Dict, List, Optional, Tuple

from puzzle_parser import (
    PuzzleFile, Molecule, Atom, Bond, ATOM_NAMES,
    PART_BONDER, PART_BONDER_PRISMA, PART_BONDER_SPEED,
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
        self.edge_move: Dict[Tuple[int, int], str] = {}  # (from_idx, to_idx) -> move label
        self._index: Dict[tuple, int] = {}  # canonical signature -> index
        self.input_state_idx: int = None  # set by reachable_states: input_state_indices[0], kept for display
        # Every state matching the actual raw reagents (see reachable_states)
        # — molecule_signature (unlike _state_signature) doesn't look at
        # Atom.age, so structurally-identical raw-reagent states can still
        # end up as distinct graph nodes differing only in how much a free
        # atom has "waited" (see _READY_AGE) along whichever path reached
        # them. A genuine raw-reagent grab has no readiness requirement of
        # its own, so bounds.py's path search must be free to start from
        # *any* of them and take whichever gives the best result, not just
        # the first one this BFS happened to discover.
        self.input_state_indices: List[int] = []
        # Memo for bounds._cadence_latency: (id(recipe), result), since a
        # graph is always built by reachable_states(pf, recipe) for one
        # specific recipe, and solve.py's SUM-tightening loop calls into it
        # repeatedly with that same (pf, recipe, graph) triple — recomputing
        # the same combinatorial path search on every iteration otherwise.
        self._cadence_latency_cache: Optional[Tuple[int, tuple]] = None

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


def initial_state(pf: PuzzleFile, recipe: RecipeResult) -> State:
    """One unit copy of each output molecule, with the per-unit reaction
    budget (the whole-run recipe.reaction_counts divided by products_needed,
    same floor-division scheme.build_unit_recipe uses)."""
    products_needed = pf.products_needed()
    reaction_unit_counts = {n: c // products_needed for n, c in recipe.reaction_counts.items() if c > 0}
    molecules = [Molecule(atoms=list(mol.atoms), bonds=list(mol.bonds)) for mol in pf.outputs]
    return State(molecules, reaction_unit_counts)


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
    Bond *directions and kinds* (not just which atoms are bonded) feed into
    each round via _cyclic_canonical, so e.g. an X-Y-X chain where the two X
    neighbors sit 60°/120°/180° apart at Y gets a different signature for
    each angle, a normal bond vs. any triplex color is never conflated with
    another, and mirror-image arrangements are kept distinct too."""
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

# Move-kind labels attached to each edge. Reaction moves use the reaction's
# own name as the label. Preference ranking (lower = preferred) used when
# the same target state is reachable via more than one kind of move — plain
# bond preferred over triple-bond, per instruction; reaction moves default
# to the middle since no preference was specified for them.
_MOVE_BOND = "bond"
_MOVE_BOND_R = "bond_r"
_MOVE_BOND_K = "bond_k"
_MOVE_BOND_Y = "bond_y"
_MOVE_TRIPLE_BOND = "triple-bond"
_TRIPLEX_COLOR_LABEL = {BOND_TRIPLEX_R: _MOVE_BOND_R, BOND_TRIPLEX_K: _MOVE_BOND_K, BOND_TRIPLEX_Y: _MOVE_BOND_Y}


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


# The two possible orientations of a triple-bond (multi-bonder) glyph
# centered on one atom: bonded to 3 neighbors simultaneously at alternating,
# 120°-apart directions — {0,2,4} or {1,3,5}. omsim/sim.c's MULTI_BONDING
# footprint ({1,0},{0,-1},{-1,1} relative to the center) is exactly the
# {0,2,4} case; {1,3,5} is the other rotation.
_TRIPLE_DIRECTION_SETS = [(0, 2, 4), (1, 3, 5)]


def _elementary_bond_actions(state: State, parts_available: int) -> List[Tuple[str, frozenset, Dict[int, set]]]:
    """Every individual bonding action currently reversible in `state`: one
    single-bond removal (reverse of a Bonder — needs PART_BONDER, normal
    bonds only, label "bond"), or one single-*color* triplex removal
    (reverse of one Bonder-Prisma pass — needs PART_BONDER_PRISMA, label
    "bond_r"/"bond_k"/"bond_y"): a bond with 2-3 triplex colors set needs
    that many separate actions, one color at a time — you can't strip a
    full triplex bond in one move, only one color per pass. Or one
    triple-bond removal (reverse of a Multi-bonder, needs PART_BONDER_SPEED
    — 2 or 3 bonds around one atom, whichever of a qualifying triple's
    slots are occupied; only ever normal bonds).

    Each action is (move_label, touched_atoms, remove_by_mol):
    touched_atoms is the set of (molecule_idx, atom_idx) pairs the action
    grabs — used to check that two actions combined into one move never
    share an atom (an atom can only be part of one bonding action at a
    time — this is also exactly why two different colors of the *same*
    triplex bond can never combine into one move: they touch the same pair
    of atoms). remove_by_mol maps molecule_idx -> set of (bond index into
    that molecule's bonds list, bit to clear from that bond's type) pairs
    the action removes — clearing a bit only drops the bond entry entirely
    once its type reaches 0 (see _apply_bond_removal)."""
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
                actions.append((_MOVE_BOND, touched, {mi: {(bi, BOND_NORMAL)}}))
            if parts_available & PART_BONDER_PRISMA:
                for color, label in _TRIPLEX_COLOR_LABEL.items():
                    if b.type & color:
                        actions.append((label, touched, {mi: {(bi, color)}}))

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
                    actions.append((_MOVE_TRIPLE_BOND, frozenset(touched),
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


def _apply_bond_removal(state: State, remove_by_mol: Dict[int, set]) -> List[Molecule]:
    """remove_by_mol maps molecule_idx -> set of (bond index, bit to clear)
    pairs (see _elementary_bond_actions) — clearing a bit only drops a bond
    entry entirely once its type reaches 0, so removing one color from a
    multi-color triplex bond leaves the other colors' connection intact."""
    new_molecules = []
    for mi, mol in enumerate(state.molecules):
        removals = remove_by_mol.get(mi)
        if not removals:
            new_molecules.append(mol)
            continue
        clear_bits: Dict[int, int] = {}
        for bi, bit in removals:
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
        new_molecules.extend(_freshly_freed(_split_molecule(mol.atoms, new_bonds)))
    return new_molecules


_SINGLE_BOND_MOVES = {_MOVE_BOND, _MOVE_BOND_R, _MOVE_BOND_K, _MOVE_BOND_Y}


def _unbond_neighbors(state: State, parts_available: int) -> List[Tuple[State, str]]:
    """One single-bond removal (normal, or one triplex color) at a time —
    see _elementary_bond_actions."""
    return [(State(_apply_bond_removal(state, remove_by_mol), dict(state.reactions_left)), label)
            for label, _touched, remove_by_mol in _elementary_bond_actions(state, parts_available)
            if label in _SINGLE_BOND_MOVES]


def _triple_unbond_neighbors(state: State, parts_available: int) -> List[Tuple[State, str]]:
    """One triple-bond (multi-bonder) removal at a time — see
    _elementary_bond_actions."""
    return [(State(_apply_bond_removal(state, remove_by_mol), dict(state.reactions_left)), label)
            for label, _touched, remove_by_mol in _elementary_bond_actions(state, parts_available)
            if label == _MOVE_TRIPLE_BOND]


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
    time instead."""
    actions = _elementary_bond_actions(state, parts_available)
    n = len(actions)
    neighbors_out = []
    for size in range(2, n + 1):
        for combo in combinations(range(n), size):
            touched_union = set()
            disjoint = True
            for idx in combo:
                touched = actions[idx][1]
                if touched & touched_union:
                    disjoint = False
                    break
                touched_union |= touched
            if not disjoint:
                continue
            remove_by_mol: Dict[int, set] = {}
            labels = []
            for idx in combo:
                label, _touched, per_mol = actions[idx]
                labels.append(label)
                for mi, bis in per_mol.items():
                    remove_by_mol.setdefault(mi, set()).update(bis)
            move = f"{size}x{labels[0]}" if all(l == labels[0] for l in labels) else "+".join(labels)
            new_molecules = _apply_bond_removal(state, remove_by_mol)
            neighbors_out.append((State(new_molecules, dict(state.reactions_left)), move))
    return neighbors_out


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


def _reverse_reaction_options(r: Reaction) -> Tuple[List[int], List[List[int]]]:
    """(remove_types, add_type_options) — like _reverse_reaction_atoms, but
    add_type_options is a list of alternatives for what the relabeled first
    atom can become. Normally just one option (the reaction's own fixed
    delta, unchanged from _reverse_reaction_atoms); a combined
    "calcify_X_or_Y" synthetic reaction (r.alt_reagent set — see
    stoichiometry.solve_recipe_combined) has no single fixed reagent-side
    atom, so one option per alternative type, all sharing the same
    reaction's budget — the caller tries each as a separate candidate
    move."""
    remove_types, add_types = _reverse_reaction_atoms(r)
    if not r.alt_reagent:
        return remove_types, [add_types]
    return remove_types, [[alt] for alt in r.alt_reagent]


def _elementary_reaction_actions(state: State, reactions: Dict[str, Reaction]) -> List[Tuple[str, frozenset, tuple]]:
    """Every single un-react firing (one reaction, one specific choice of
    atom instances, one choice of which alternative type to relabel to if
    the reaction is a combined "X_or_Y" one — see _reverse_reaction_options)
    currently available, in the same (label, touched_atoms, payload) shape
    _elementary_bond_actions uses so both can feed the same atom-disjoint
    combiner (see _combined_bond_reaction_neighbors) — payload is
    ("reaction", name, chosen, add_types) instead of a bond-removal dict.
    Only k=1 firings: batching several firings of the *same* reaction
    together is still _reaction_neighbors' own job (it composes with itself
    more directly); this exists so a firing can combine with bonds or with
    a *different* reaction, which neither existing path can produce."""
    actions = []
    for name, budget in state.reactions_left.items():
        if budget <= 0:
            continue
        r = reactions[name]
        remove_types, add_type_options = _reverse_reaction_options(r)
        for add_types in add_type_options:
            # See _reaction_neighbors: a combined "X_or_Y" reaction's move
            # label should show which concrete alternative this action
            # actually used, not the umbrella group name.
            display_name = f"calcify_{ATOM_NAMES[add_types[0]]}" if r.alt_reagent else name
            for chosen in _choose_instances(state, remove_types):
                if not _firing_atoms_ok(state, name, chosen):
                    continue
                actions.append((display_name, frozenset(chosen), ("reaction", name, chosen, add_types)))
    return actions


def _apply_mixed_actions(state: State, chosen: list) -> State:
    """Apply a combo of atom-disjoint elementary actions — some bonding
    actions (payload: {mol_idx: {(bond index, bit to clear), ...}}, from
    _elementary_bond_actions), some reaction firings (payload: ("reaction",
    name, firing, add_types), from _elementary_reaction_actions, add_types
    already resolved to whichever alternative that action picked — see
    _reverse_reaction_options) — simultaneously. Each firing relabels its
    own first consumed atom in place and removes/spawns the rest, exactly
    like a lone _reaction_neighbors firing; bond removals just drop bonds
    — see both for the individual rules this only combines."""
    bond_remove_by_mol: Dict[int, set] = {}
    relabel_by_mol: Dict[int, Dict[int, tuple]] = {}  # ai -> (new_type, is_drop_and_create)
    extra_remove_by_mol: Dict[int, set] = {}
    spawn_types = []
    reaction_uses: Counter = Counter()

    for _label, _touched, payload in chosen:
        if isinstance(payload, tuple) and payload[0] == "reaction":
            _, name, firing, add_types = payload
            reaction_uses[name] += 1
            r_mi, r_ai = firing[0]
            relabel_by_mol.setdefault(r_mi, {})[r_ai] = (add_types[0], _is_drop_and_create(name))
            for mi, ai in firing[1:]:
                extra_remove_by_mol.setdefault(mi, set()).add(ai)
            spawn_types.extend(add_types[1:])
        else:
            for mi, bis in payload.items():
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
            # See _reaction_neighbors: a plain relabel carries the atom's
            # age over (same atom, continuous existence); a drop-and-create
            # reversal starts both sides fresh (genuine creation boundary).
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
        # See _reaction_neighbors: a singleton here is only "freshly freed"
        # (age reset) if it wasn't already free-standing before this move.
        new_molecules.extend(_freshly_freed(result) if len(mol.atoms) > 1 else result)

    for atype in spawn_types:
        new_molecules.append(Molecule(atoms=[Atom(type=atype, u=0, v=0, age=0)], bonds=[]))

    new_reactions_left = dict(state.reactions_left)
    for name, uses in reaction_uses.items():
        new_reactions_left[name] -= uses

    return State(new_molecules, new_reactions_left)


def _combined_bond_reaction_neighbors(state: State, parts_available: int,
                                       reactions: Dict[str, Reaction]) -> List[Tuple[State, str]]:
    """Atom-disjoint combos that mix bonding actions with reaction firings
    (or different reactions with each other) into one simultaneous move —
    the gap _combined_bond_neighbors (bonds only) and _reaction_neighbors
    (same-reaction-name firings only) leave: an atom can be part of at most
    one bonding action or one reaction firing per move, but nothing stops
    unrelated glyphs elsewhere in the molecule from firing in the same
    cycle. Restricted to combos containing at least one reaction firing —
    all-bond combos are already _combined_bond_neighbors' job. Projection
    firings additionally need a free hex each (_is_space_limited) — several
    can still combine, just only as many as there are distinct free hexes
    to put results in."""
    bond_actions = _elementary_bond_actions(state, parts_available)
    reaction_actions = _elementary_reaction_actions(state, reactions)
    all_actions = bond_actions + reaction_actions
    n = len(all_actions)
    reaction_indices = set(range(len(bond_actions), n))
    occupied = _occupied_positions(state)

    neighbors_out = []
    for size in range(2, n + 1):
        for combo in combinations(range(n), size):
            if reaction_indices.isdisjoint(combo):
                continue  # pure-bond combo, already covered by _combined_bond_neighbors
            touched_union = set()
            disjoint = True
            for idx in combo:
                touched = all_actions[idx][1]
                if touched & touched_union:
                    disjoint = False
                    break
                touched_union |= touched
            if not disjoint:
                continue

            space_anchors = []
            for idx in combo:
                label, _touched, payload = all_actions[idx]
                if isinstance(payload, tuple) and payload[0] == "reaction" and _is_space_limited(payload[1]):
                    mi, ai = payload[2][0]
                    space_anchors.append((state.molecules[mi].atoms[ai].u, state.molecules[mi].atoms[ai].v))
            if space_anchors and not _has_distinct_free_hexes(space_anchors, occupied):
                continue

            chosen = [all_actions[idx] for idx in combo]
            labels = [all_actions[idx][0] for idx in combo]
            move = f"{size}x{labels[0]}" if all(l == labels[0] for l in labels) else "+".join(labels)
            neighbors_out.append((_apply_mixed_actions(state, chosen), move))
    return neighbors_out


def _atom_instances_by_type(state: State) -> Dict[int, List[Tuple[int, int]]]:
    """atom type -> list of (molecule index, atom index) instances."""
    by_type: Dict[int, List[Tuple[int, int]]] = {}
    for mi, mol in enumerate(state.molecules):
        for ai, a in enumerate(mol.atoms):
            by_type.setdefault(a.type, []).append((mi, ai))
    return by_type


_DROP_AND_CREATE = {"animismus", "dispersion", "unification"}
_DROP_AND_CREATE_PREFIXES = ("purify_",)


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
    """Projection needs its quicksilver atom *and* a free hex next to the
    atom it's projecting, to put the result in — that hex can't be shared
    by two simultaneous firings, unlike e.g. calcification's plain in-place
    relabel. See _free_neighbor_positions / _has_distinct_free_hexes: the
    relabeled atom's own position is real, tracked board geometry (not an
    abstracted-away detail — see module docstring's Simplifications section
    on why only *its* position is real, not any second/third atom a
    multi-atom reaction also needs), so this can actually be checked rather
    than assumed unlimited."""
    return name.startswith("project_")


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


def _reaction_neighbors(state: State, reactions: Dict[str, Reaction]) -> List[Tuple[State, str]]:
    """Reverses 1 up to `budget` firings of a reaction *simultaneously*, in
    one combined move. The relabeled atom's own position is real, tracked
    board geometry (see module docstring), so nothing stops several
    same-named firings at once *except* projection, which additionally
    needs its own free adjacent hex per firing (_is_space_limited) —
    everything else has no structural reason to force firings to happen one
    at a time. Each of the k firings gets its own independent application of
    the same relabel-in-place rule (its own first consumed atom keeps its
    own position/bonds and becomes the first produced type; only its own
    additional atoms, beyond the first, are removed/spawned fresh) — firings
    never share atoms with each other. Move label is "kxname" for k>1,
    otherwise plain "name" (unchanged from before)."""
    neighbors_out = []
    for name, budget in state.reactions_left.items():
        if budget <= 0:
            continue
        r = reactions[name]
        remove_types, add_type_options = _reverse_reaction_options(r)
        group_size = len(remove_types)
        is_dc = _is_drop_and_create(name)
        space_limited = _is_space_limited(name)
        occupied = _occupied_positions(state) if space_limited else None
        for add_types in add_type_options:
            # A combined "X_or_Y" reaction's move label should show which
            # concrete alternative this specific move actually used, not
            # the umbrella group name — e.g. "calcify_water", not
            # "calcify_fire_or_water" — since that's what a real solution
            # would actually be doing at this step.
            display_name = f"calcify_{ATOM_NAMES[add_types[0]]}" if r.alt_reagent else name
            for k in range(1, budget + 1):
                for chosen in _choose_instances(state, remove_types * k):
                    firings = [chosen[i * group_size:(i + 1) * group_size] for i in range(k)]

                    if not all(_firing_atoms_ok(state, name, firing) for firing in firings):
                        continue

                    if space_limited:
                        anchors = [(state.molecules[mi].atoms[ai].u, state.molecules[mi].atoms[ai].v)
                                   for mi, ai in (firing[0] for firing in firings)]
                        if not _has_distinct_free_hexes(anchors, occupied):
                            continue

                    relabel_by_mol: Dict[int, set] = {}
                    remove_by_mol: Dict[int, set] = {}
                    for firing in firings:
                        r_mi, r_ai = firing[0]
                        relabel_by_mol.setdefault(r_mi, set()).add(r_ai)
                        for mi, ai in firing[1:]:
                            remove_by_mol.setdefault(mi, set()).add(ai)

                    new_molecules = []
                    for mi, mol in enumerate(state.molecules):
                        relabel_ais = relabel_by_mol.get(mi, set())
                        extra_remove = remove_by_mol.get(mi, set())
                        if not relabel_ais and not extra_remove:
                            new_molecules.append(mol)
                            continue
                        new_atoms = list(mol.atoms)
                        for ai in relabel_ais:
                            old = new_atoms[ai]
                            # A plain relabel is the same atom continuing to
                            # exist (see _firing_atoms_ok) — its age carries
                            # over. A drop-and-create reversal crosses a
                            # genuine creation boundary (there's no atom that
                            # "is" the output), so both sides start fresh.
                            new_atoms[ai] = Atom(type=add_types[0], u=old.u, v=old.v,
                                                  age=0 if is_dc else old.age)
                        if extra_remove:
                            result = _remove_atoms(new_atoms, mol.bonds, extra_remove)
                        else:
                            result = [Molecule(atoms=new_atoms, bonds=list(mol.bonds))]
                        # A singleton coming out of this is only "freshly
                        # freed" (age reset, see _freshly_freed) if the atom
                        # wasn't already free-standing before this move —
                        # i.e. its original molecule had more than 1 atom.
                        # Otherwise it's the len==1 relabel-in-place case
                        # above, whose age decision already stands.
                        new_molecules.extend(_freshly_freed(result) if len(mol.atoms) > 1 else result)

                    for _ in range(k):
                        for atype in add_types[1:]:
                            new_molecules.append(Molecule(atoms=[Atom(type=atype, u=0, v=0, age=0)], bonds=[]))

                    new_reactions_left = dict(state.reactions_left)
                    new_reactions_left[name] -= k
                    label = f"{k}x{display_name}" if k > 1 else display_name
                    neighbors_out.append((State(new_molecules, new_reactions_left), label))
    return neighbors_out


_MOVE_WAIT = "wait"


def _wait_neighbors(state: State, tracked_types: frozenset) -> List[Tuple[State, str]]:
    """One "wait" move: every free (single-atom-molecule) atom of a
    `tracked_types` type (the actual product types of this puzzle's
    drop-and-create reactions — see reachable_states) below _READY_AGE ages
    by 1 (capped there), nothing else about the state changes. This is how
    a drop-and-create reaction's product atom (or any atom freshly freed by
    an unbond — see Atom.age) "matures" enough to be picked back up for
    another drop-and-create firing (_firing_atoms_ok) — real idle cycles the
    search has to actually spend, not a free pass. Restricted to
    tracked_types since age is never checked for any other type — offering
    a "wait" that only ages an irrelevant atom would just multiply the graph
    for no reason (see _state_signature). Omitted entirely once nothing
    would change (every tracked free atom already mature): an edge to the
    identical state would be a self-loop, which would break the topological
    order bounds.py's path search relies on and contradicts the module
    docstring's termination argument (nothing it tracks would be
    decreasing)."""
    changed = False
    new_molecules = []
    for mol in state.molecules:
        if len(mol.atoms) == 1 and mol.atoms[0].type in tracked_types and mol.atoms[0].age < _READY_AGE:
            changed = True
            new_molecules.append(replace(mol, atoms=[replace(mol.atoms[0], age=mol.atoms[0].age + 1)]))
        else:
            new_molecules.append(mol)
    if not changed:
        return []
    return [(State(new_molecules, dict(state.reactions_left)), _MOVE_WAIT)]


def neighbors(state: State, reactions: Dict[str, Reaction], parts_available: int,
              tracked_types: frozenset) -> List[Tuple[State, str]]:
    """Every (predecessor state, move label) pair reachable from `state` by
    one unbond, one triple-bond (multi-bonder) reversal, any atom-disjoint
    combination of 2+ of those, one (or several simultaneous, see
    _reaction_neighbors) un-react move, an atom-disjoint combination of
    bonding actions with reaction firings / different reactions with each
    other (see _combined_bond_reaction_neighbors), or one "wait" (see
    _wait_neighbors) — unbond/triple-bond moves only apply if the puzzle
    actually grants that glyph. Move label is "bond", "triple-bond",
    "NxLABEL" for N combined same-kind moves, "label+label" for a mixed
    combination, "wait", or the reaction name."""
    return (_unbond_neighbors(state, parts_available)
            + _triple_unbond_neighbors(state, parts_available)
            + _combined_bond_neighbors(state, parts_available)
            + _reaction_neighbors(state, reactions)
            + _combined_bond_reaction_neighbors(state, parts_available, reactions)
            + _wait_neighbors(state, tracked_types))


# ── BFS ────────────────────────────────────────────────────────────────────

def _matches_raw_reagents(state: State, pf: PuzzleFile, recipe: RecipeResult, products_needed: int,
                           exact: bool = False) -> bool:
    """True if `state` is exactly the puzzle's raw reagents, one unit copy
    each (per recipe.reagent_counts // products_needed). For a reagent
    covered by a combined reaction group (recipe.extra_reactions, see
    solve_recipe_combined), exact=False (the default) accepts ANY split of
    single-atom molecules among the group's alternative types as long as
    their combined count matches the group's total — some split has to be
    accepted, since solve_recipe's own one arbitrary split may not even be
    reachable — but bounds.py's L_spine computation still keys off
    recipe.reagent_counts per individual type, so reachable_states actually
    wants exact=True first (matching recipe.reagent_counts's specific
    per-type numbers precisely, when some reachable state realizes them)
    and only falls back to this looser exact=False match if no state does.
    A linear scan rather than a signature dict-lookup — reachable_states
    only needs this once, over however many states the BFS found."""
    combined_types = {t for r in recipe.extra_reactions.values() for t in (r.alt_reagent or [])}

    def is_combined_atom(mol):
        atoms = mol.atom_type_counts()
        return len(atoms) == 1 and not mol.bonds and next(iter(atoms)) in combined_types

    expected = Counter()
    for i, count in recipe.reagent_counts.items():
        mol = pf.inputs[i]
        if is_combined_atom(mol) and not exact:
            continue  # covered by combined_total instead of an exact per-reagent count
        expected[molecule_signature(mol)] += count // products_needed

    # A "raw, directly-needed-by-the-output" atom of a combined type and a
    # "calcification feedstock" atom of that same type are structurally
    # identical once fully decomposed — both end up as one standalone
    # unbonded atom. So the aggregate (exact=False) target isn't the
    # reaction's count (only the calcified portion), it's the *reagent*
    # total across the whole group (raw + feedstock together).
    combined_total = None
    if not exact:
        combined_total = sum(count // products_needed for i, count in recipe.reagent_counts.items()
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
    return combined_found == combined_total and all(v == 0 for v in remaining.values())


def reachable_states(pf: PuzzleFile, recipe: RecipeResult) -> StateGraph:
    """BFS over every state reachable from the output molecule(s) by
    unbond/un-react moves, respecting the per-unit reaction budget. Returns
    the graph of every distinct state found, and which states are directly
    reachable from which."""
    reactions = {r.name: r for r in build_reactions(pf)}
    reactions.update(recipe.extra_reactions)
    # Only these types ever have their age checked (_firing_atoms_ok, for
    # reversing a drop-and-create reaction) or matter for the BFS's own
    # dedup (_state_signature) — every other free atom's age tracking would
    # just be dead weight multiplying the graph for nothing (see
    # _wait_neighbors / _state_signature). Restricted to reactions this
    # puzzle's own recipe actually fires (reaction_counts > 0), not every
    # drop-and-create reaction build_reactions can construct from the parts
    # available — e.g. Purification generates one purify_X per consecutive
    # metal-chain pair regardless of whether this recipe's own chain needs
    # all of them, and a puzzle without the Glyph of Animismus granted at
    # all has no reachable way to ever fire (or reverse) it.
    tracked_types = frozenset(
        atype
        for name, r in reactions.items()
        if _is_drop_and_create(name) and recipe.reaction_counts.get(name, 0) > 0
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
    # Every state matching (exact=True preferred, see _matches_raw_reagents),
    # not just the first — molecule_signature ignores Atom.age, so the same
    # structural raw-reagent state can show up as several distinct graph
    # nodes differing only in how much "waiting" (_READY_AGE) a free atom
    # happened to accumulate along whichever path found it. A raw-reagent
    # grab itself has no readiness requirement, so bounds.py's path search
    # needs every one of them available to start from, and picks whichever
    # gives the best (smallest) result — not whichever this BFS discovered
    # first, which could be an unnecessarily-waited one.
    match_indices = [i for i, s in enumerate(graph.states)
                      if _matches_raw_reagents(s, pf, recipe, products_needed, exact=True)]
    if not match_indices:
        match_indices = [i for i, s in enumerate(graph.states)
                          if _matches_raw_reagents(s, pf, recipe, products_needed)]
    assert match_indices, (
        f"{pf.name!r}: no reachable state matches the actual raw reagents — "
        "the un-react/unbond move rules failed to find a valid full "
        "decomposition path that solve_recipe already proved exists"
    )
    graph.input_state_indices = match_indices
    graph.input_state_idx = match_indices[0]
    return graph
