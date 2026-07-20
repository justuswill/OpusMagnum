"""
Binary .puzzle file parser for Opus Magnum.
Format transcribed from omsim/parse.c and omsim/parse.h.
"""

import struct
from dataclasses import dataclass, field, replace
from typing import List, Optional


# Atom type byte values: the file stores the bit-position directly.
# decode_atom(byte) = 1 << byte; SALT = 1<<1, AIR = 1<<2, ... (from sim.h + decode.c)
ATOM_SALT        = 1
ATOM_AIR         = 2
ATOM_EARTH       = 3
ATOM_FIRE        = 4
ATOM_WATER       = 5
ATOM_QUICKSILVER = 6
ATOM_GOLD        = 7
ATOM_SILVER      = 8
ATOM_COPPER      = 9
ATOM_IRON        = 10
ATOM_TIN         = 11
ATOM_LEAD        = 12
ATOM_VITAE       = 13
ATOM_MORS        = 14
ATOM_REPEAT      = 15
ATOM_QUINTESSENCE = 16

ATOM_NAMES = {
    ATOM_SALT: "salt", ATOM_AIR: "air", ATOM_EARTH: "earth",
    ATOM_FIRE: "fire", ATOM_WATER: "water", ATOM_QUICKSILVER: "quicksilver",
    ATOM_GOLD: "gold", ATOM_SILVER: "silver", ATOM_COPPER: "copper",
    ATOM_IRON: "iron", ATOM_TIN: "tin", ATOM_LEAD: "lead",
    ATOM_VITAE: "vitae", ATOM_MORS: "mors",
    ATOM_REPEAT: "repeat", ATOM_QUINTESSENCE: "quintessence",
}

ELEMENTALS   = {ATOM_AIR, ATOM_EARTH, ATOM_FIRE, ATOM_WATER}
METALS       = {ATOM_GOLD, ATOM_SILVER, ATOM_COPPER, ATOM_IRON, ATOM_TIN, ATOM_LEAD}

# Bond type bits (from decode.c decode_bond_type)
BOND_NORMAL   = 0x1
BOND_TRIPLEX_R = 0x2
BOND_TRIPLEX_K = 0x4
BOND_TRIPLEX_Y = 0x8
BOND_ANY_TRIPLEX = BOND_TRIPLEX_R | BOND_TRIPLEX_K | BOND_TRIPLEX_Y

# Parts-available bitmask bits (from decode.c parts_available_bits_for_part_name)
PART_ARM1          = 1 << 0
PART_MULTI_ARM     = 1 << 1   # arm2/arm3/arm6
PART_PISTON        = 1 << 2
PART_TRACK         = 1 << 3
PART_BONDER        = 1 << 8
PART_UNBONDER      = 1 << 9
PART_BONDER_SPEED  = 1 << 10
PART_BONDER_PRISMA = 1 << 11
PART_CALCIFICATION = 1 << 12
PART_DUPLICATION   = 1 << 13
PART_PROJECTION    = 1 << 14
PART_PURIFICATION  = 1 << 15
PART_ANIMISMUS     = 1 << 16  # glyph-life-and-death
PART_DISPOSAL      = 1 << 17
PART_DISPERSION    = 1 << 18  # also unification
PART_REJECTION     = 1 << 19
PART_DIVISION      = 1 << 20
PART_PROLIFERATION = 1 << 21
PART_BARON         = 1 << 28  # van berlo's wheel
PART_RAVARI        = 1 << 29  # ravari's wheel

# Atom types permanently carried by each wheel.  All types in a pool are
# fungible with each other: the wheel can supply any one of them per rotation,
# so the throughput constraint applies to the pool total, not each type separately.
BARON_WHEEL_ATOMS  = frozenset({ATOM_SALT, ATOM_AIR, ATOM_EARTH, ATOM_FIRE, ATOM_WATER})
RAVARI_WHEEL_ATOMS = frozenset({ATOM_IRON, ATOM_COPPER, ATOM_SILVER, ATOM_GOLD, ATOM_LEAD, ATOM_TIN})


@dataclass
class Atom:
    type: int
    u: int
    v: int

    @property
    def name(self):
        return ATOM_NAMES.get(self.type, f"type{self.type}")


@dataclass
class Bond:
    type: int      # bit flags: BOND_NORMAL, BOND_TRIPLEX_*
    from_u: int
    from_v: int
    to_u: int
    to_v: int

    @property
    def is_normal(self):
        return bool(self.type & BOND_NORMAL)

    @property
    def is_triplex(self):
        return bool(self.type & BOND_ANY_TRIPLEX)


@dataclass
class Molecule:
    atoms: List[Atom] = field(default_factory=list)
    bonds: List[Bond] = field(default_factory=list)

    def atom_count(self):
        return len(self.atoms)

    def atom_type_counts(self):
        counts = {}
        for a in self.atoms:
            counts[a.type] = counts.get(a.type, 0) + 1
        return counts

    def has_normal_bonds(self):
        return any(b.is_normal for b in self.bonds)

    def has_triplex_bonds(self):
        return any(b.is_triplex for b in self.bonds)


@dataclass
class PuzzleFile:
    name: str
    parts_available: int
    inputs: List[Molecule] = field(default_factory=list)
    outputs: List[Molecule] = field(default_factory=list)
    output_scale: int = 1
    is_production: bool = False

    def products_needed(self):
        return 6 * self.output_scale


def mirror_repeat_molecule(mol: Molecule) -> Optional[Molecule]:
    """Alternate construction of a repeating output molecule: mirror its
    repeat marker(s) onto the opposite end instead of the one the puzzle
    file happens to depict. A real solution isn't required to build the
    polymer in that specific direction — trying both and keeping whichever
    gives the better bound covers the uncertainty (see
    puzzle_parser.alternate_repeat_puzzle / bounds.py's use of it).

    The atom every repeating molecule mirrors around is always the one
    sitting at local (0, 0) — Opus Magnum's own convention for where an
    output molecule's "seed" atom sits. For each bond the *original*
    repeat atom has (one per polymer "arm" — a repeat atom can have more
    than one, e.g. sitting on a ring, closing it), delete that repeat atom
    and its bonds, then add a fresh repeat atom next to (0, 0) at that same
    relative direction, with the same bond type. Returns None if the
    molecule has no repeat atom, no atom at (0, 0), a repeat atom with no
    bonds at all, or a mirrored position would land on an already-occupied
    hex — the mirror isn't geometrically valid for this molecule."""
    repeat_atoms = [a for a in mol.atoms if a.type == ATOM_REPEAT]
    if not repeat_atoms:
        return None
    if (0, 0) not in {(a.u, a.v) for a in mol.atoms}:
        return None

    occupied = {(a.u, a.v) for a in mol.atoms}
    remove_positions = set()
    new_repeat_atoms = []

    for r in repeat_atoms:
        r_pos = (r.u, r.v)
        touching = [b for b in mol.bonds
                    if (b.from_u, b.from_v) == r_pos or (b.to_u, b.to_v) == r_pos]
        if not touching:
            return None
        remove_positions.add(r_pos)
        for b in touching:
            other_pos = (b.to_u, b.to_v) if (b.from_u, b.from_v) == r_pos else (b.from_u, b.from_v)
            direction = (other_pos[0] - r_pos[0], other_pos[1] - r_pos[1])
            new_pos = direction  # anchor is (0, 0), so new position = (0,0) + direction
            if new_pos in occupied or any(p == new_pos for p, _ in new_repeat_atoms):
                return None
            new_repeat_atoms.append((new_pos, b.type))

    new_atoms = [a for a in mol.atoms if (a.u, a.v) not in remove_positions]
    new_bonds = [b for b in mol.bonds
                 if (b.from_u, b.from_v) not in remove_positions
                 and (b.to_u, b.to_v) not in remove_positions]
    for pos, btype in new_repeat_atoms:
        new_atoms.append(Atom(type=ATOM_REPEAT, u=pos[0], v=pos[1]))
        new_bonds.append(Bond(type=btype, from_u=0, from_v=0, to_u=pos[0], to_v=pos[1]))

    return Molecule(atoms=new_atoms, bonds=new_bonds)


def alternate_repeat_puzzle(pf: PuzzleFile) -> Optional[PuzzleFile]:
    """`pf` with every repeating output molecule's repeat marker(s)
    mirrored onto the opposite end instead (see mirror_repeat_molecule) —
    None if no output molecule has a repeat atom (nothing to try), or if
    mirroring isn't geometrically valid for one that does."""
    if not any(a.type == ATOM_REPEAT for mol in pf.outputs for a in mol.atoms):
        return None
    new_outputs = []
    for mol in pf.outputs:
        if not any(a.type == ATOM_REPEAT for a in mol.atoms):
            new_outputs.append(mol)
            continue
        mirrored = mirror_repeat_molecule(mol)
        if mirrored is None:
            return None
        new_outputs.append(mirrored)
    return replace(pf, outputs=new_outputs)


class _Parser:
    def __init__(self, data: bytes):
        self._d = data
        self._p = 0

    def u32(self):
        v, = struct.unpack_from('<I', self._d, self._p)
        self._p += 4
        return v

    def u64(self):
        v, = struct.unpack_from('<Q', self._d, self._p)
        self._p += 8
        return v

    def u8(self):
        v = self._d[self._p]
        self._p += 1
        return v

    def i8(self):
        v = struct.unpack_from('b', self._d, self._p)[0]
        self._p += 1
        return v

    def string(self):
        length = 0
        shift = 0
        while True:
            b = self.u8()
            length |= (b & 0x7F) << shift
            shift += 7
            if not (b & 0x80):
                break
        text = self._d[self._p:self._p + length].decode('utf-8', errors='replace')
        self._p += length
        return text

    def molecule(self):
        mol = Molecule()
        for _ in range(self.u32()):
            t = self.u8()
            u = self.i8()
            v = self.i8()
            mol.atoms.append(Atom(type=t, u=u, v=v))
        for _ in range(self.u32()):
            bt = self.u8()
            fu, fv = self.i8(), self.i8()
            tu, tv = self.i8(), self.i8()
            mol.bonds.append(Bond(type=bt, from_u=fu, from_v=fv, to_u=tu, to_v=tv))
        return mol

    def parse(self) -> PuzzleFile:
        if self.u32() != 3:
            raise ValueError("not a valid .puzzle file (wrong version)")
        name = self.string()
        _creator = self.u64()
        parts_available = self.u64()
        inputs  = [self.molecule() for _ in range(self.u32())]
        outputs = [self.molecule() for _ in range(self.u32())]
        output_scale = self.u32()
        is_production = bool(self.u8())
        if any(a.type == ATOM_REPEAT for mol in outputs for a in mol.atoms):
            # ATOM_REPEAT is a structural marker in the output, not a real
            # atom the puzzle file lists a reagent for — but schematic.py's
            # backward search still has to isolate and "grab" one from
            # somewhere when it fully un-bonds the output, so give it a
            # dedicated single-atom reagent to reduce down to.
            inputs.append(Molecule(atoms=[Atom(type=ATOM_REPEAT, u=0, v=0)], bonds=[]))
        return PuzzleFile(
            name=name,
            parts_available=parts_available,
            inputs=inputs,
            outputs=outputs,
            output_scale=output_scale,
            is_production=is_production,
        )


def parse_puzzle(path: str) -> PuzzleFile:
    with open(path, 'rb') as f:
        data = f.read()
    return _Parser(data).parse()
