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
out of scope here, same as stoichiometry.py).

Termination: every move strictly decreases (total bonds across all
molecules) + (total reactions_left) — unbond removes one bond (triple-bond
reversal removes three at once) and never adds any; un-react spends one unit
of budget and the atoms it adds are always unbonded. That sum is bounded
below by 0, so the state graph is a finite DAG and the BFS always terminates.

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
from dataclasses import dataclass
from itertools import combinations, product as iproduct
from typing import Dict, List, Tuple

from puzzle_parser import (
    PuzzleFile, Molecule, Atom, Bond, ATOM_NAMES,
    PART_BONDER, PART_BONDER_PRISMA, PART_BONDER_SPEED,
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
        self.input_state_idx: int = None  # set by reachable_states once the actual raw-reagent state is found

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
        ... — the state matching the actual raw reagents (see
        input_state_idx) is marked idx* instead of idx."""
        lines = []
        for idx, state in enumerate(self.states):
            targets = ", ".join(f"{t}[{self.edge_move[(idx, t)]}]" for t in self.edges[idx])
            label = f"{idx}*" if idx == self.input_state_idx else f"{idx}"
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


def _state_signature(state: State) -> tuple:
    mol_sigs = tuple(sorted(molecule_signature(m) for m in state.molecules))
    react_sig = tuple(sorted((k, v) for k, v in state.reactions_left.items() if v > 0))
    return (mol_sigs, react_sig)


# ── neighbor generation ───────────────────────────────────────────────────

# Move-kind labels attached to each edge. Reaction moves use the reaction's
# own name as the label. Preference ranking (lower = preferred) used when
# the same target state is reachable via more than one kind of move — plain
# bond preferred over triple-bond, per instruction; reaction moves default
# to the middle since no preference was specified for them.
_MOVE_BOND = "bond"
_MOVE_TRIPLE_BOND = "triple-bond"


def _move_priority(move: str) -> int:
    if move == _MOVE_BOND:
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
    single-bond removal (reverse of a Bonder/Bonder-Prisma — needs
    PART_BONDER for normal bonds, PART_BONDER_PRISMA for triplex, gated per
    bond kind), or one triple-bond removal (reverse of a Multi-bonder, needs
    PART_BONDER_SPEED — 2 or 3 bonds around one atom, whichever of a
    qualifying triple's slots are occupied; only ever normal bonds).

    Each action is (move_label, touched_atoms, remove_by_mol):
    touched_atoms is the set of (molecule_idx, atom_idx) pairs the action
    grabs — used to check that two actions combined into one move never
    share an atom (an atom can only be part of one bonding action at a
    time); remove_by_mol maps molecule_idx -> set of bond indices (into
    that molecule's bonds list) the action removes."""
    actions = []
    for mi, mol in enumerate(state.molecules):
        pos_to_index = {(a.u, a.v): i for i, a in enumerate(mol.atoms)}
        for bi, b in enumerate(mol.bonds):
            if b.is_normal and not (parts_available & PART_BONDER):
                continue
            if b.is_triplex and not (parts_available & PART_BONDER_PRISMA):
                continue
            ai = pos_to_index.get((b.from_u, b.from_v))
            aj = pos_to_index.get((b.to_u, b.to_v))
            if ai is None or aj is None:
                continue
            actions.append((_MOVE_BOND, frozenset({(mi, ai), (mi, aj)}), {mi: {bi}}))

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
                    actions.append((_MOVE_TRIPLE_BOND, frozenset(touched), {mi: set(bond_indices)}))
    return actions


def _apply_bond_removal(state: State, remove_by_mol: Dict[int, set]) -> List[Molecule]:
    new_molecules = []
    for mi, mol in enumerate(state.molecules):
        if mi in remove_by_mol:
            new_bonds = [b for k, b in enumerate(mol.bonds) if k not in remove_by_mol[mi]]
            new_molecules.extend(_split_molecule(mol.atoms, new_bonds))
        else:
            new_molecules.append(mol)
    return new_molecules


def _unbond_neighbors(state: State, parts_available: int) -> List[Tuple[State, str]]:
    """One single-bond removal at a time — see _elementary_bond_actions."""
    return [(State(_apply_bond_removal(state, remove_by_mol), dict(state.reactions_left)), label)
            for label, _touched, remove_by_mol in _elementary_bond_actions(state, parts_available)
            if label == _MOVE_BOND]


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


def _atom_instances_by_type(state: State) -> Dict[int, List[Tuple[int, int]]]:
    """atom type -> list of (molecule index, atom index) instances."""
    by_type: Dict[int, List[Tuple[int, int]]] = {}
    for mi, mol in enumerate(state.molecules):
        for ai, a in enumerate(mol.atoms):
            by_type.setdefault(a.type, []).append((mi, ai))
    return by_type


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


def _reaction_neighbors(state: State, reactions: Dict[str, Reaction]) -> List[Tuple[State, str]]:
    """Reverses 1 up to `budget` firings of a reaction *simultaneously*, in
    one combined move — this module doesn't track real glyph footprints or
    positions at all (out of scope, see module docstring), so nothing here
    distinguishes "one small glyph, applied many times" from "many firings
    at once"; there's no structural reason to force firings to happen one at
    a time. Each of the k firings gets its own independent application of
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
        remove_types, add_types = _reverse_reaction_atoms(r)
        group_size = len(remove_types)
        for k in range(1, budget + 1):
            for chosen in _choose_instances(state, remove_types * k):
                firings = [chosen[i * group_size:(i + 1) * group_size] for i in range(k)]

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
                        new_atoms[ai] = Atom(type=add_types[0], u=old.u, v=old.v)
                    if extra_remove:
                        new_molecules.extend(_remove_atoms(new_atoms, mol.bonds, extra_remove))
                    else:
                        new_molecules.append(Molecule(atoms=new_atoms, bonds=list(mol.bonds)))

                for _ in range(k):
                    for atype in add_types[1:]:
                        new_molecules.append(Molecule(atoms=[Atom(type=atype, u=0, v=0)], bonds=[]))

                new_reactions_left = dict(state.reactions_left)
                new_reactions_left[name] -= k
                label = f"{k}x{name}" if k > 1 else name
                neighbors_out.append((State(new_molecules, new_reactions_left), label))
    return neighbors_out


def neighbors(state: State, reactions: Dict[str, Reaction], parts_available: int) -> List[Tuple[State, str]]:
    """Every (predecessor state, move label) pair reachable from `state` by
    one unbond, one triple-bond (multi-bonder) reversal, any atom-disjoint
    combination of 2+ of those, or one (or several simultaneous, see
    _reaction_neighbors) un-react move — unbond/triple-bond moves only apply
    if the puzzle actually grants that glyph. Move label is "bond",
    "triple-bond", "NxLABEL" for N combined same-kind moves, "label+label"
    for a mixed combination, or the reaction name."""
    return (_unbond_neighbors(state, parts_available)
            + _triple_unbond_neighbors(state, parts_available)
            + _combined_bond_neighbors(state, parts_available)
            + _reaction_neighbors(state, reactions))


# ── BFS ────────────────────────────────────────────────────────────────────

def _inputs_state_key(pf: PuzzleFile, recipe: RecipeResult) -> tuple:
    """Canonical signature of the state matching the puzzle's actual raw
    reagent molecules (one unit copy each, per recipe.reagent_counts //
    products_needed, all reactions spent) — same signature function the BFS
    indexes by, so membership is an O(1) dict lookup."""
    products_needed = pf.products_needed()
    goal_molecules = []
    for i, count in recipe.reagent_counts.items():
        unit_count = count // products_needed
        mol = pf.inputs[i]
        for _ in range(unit_count):
            goal_molecules.append(Molecule(atoms=list(mol.atoms), bonds=list(mol.bonds)))
    return _state_signature(State(goal_molecules, {}))


def reachable_states(pf: PuzzleFile, recipe: RecipeResult) -> StateGraph:
    """BFS over every state reachable from the output molecule(s) by
    unbond/un-react moves, respecting the per-unit reaction budget. Returns
    the graph of every distinct state found, and which states are directly
    reachable from which."""
    reactions = {r.name: r for r in build_reactions(pf)}
    start = initial_state(pf, recipe)
    start_key = _state_signature(start)

    graph = StateGraph()
    start_idx = graph.add_state(start, start_key)
    queue = deque([(start, start_idx)])
    while queue:
        current, current_idx = queue.popleft()
        for nxt, move in neighbors(current, reactions, pf.parts_available):
            key = _state_signature(nxt)
            if key in graph._index:
                nxt_idx = graph._index[key]
            else:
                nxt_idx = graph.add_state(nxt, key)
                queue.append((nxt, nxt_idx))
            graph.add_edge(current_idx, nxt_idx, move)

    inputs_key = _inputs_state_key(pf, recipe)
    assert inputs_key in graph._index, (
        f"{pf.name!r}: no reachable state matches the actual raw reagents — "
        "the un-react/unbond move rules failed to find a valid full "
        "decomposition path that solve_recipe already proved exists"
    )
    graph.input_state_idx = graph._index[inputs_key]
    return graph
