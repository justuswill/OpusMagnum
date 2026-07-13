"""
Binary .puzzle file parser for Opus Magnum.
Format transcribed from omsim/parse.c and omsim/parse.h.
"""

import struct
from dataclasses import dataclass, field
from typing import List


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
