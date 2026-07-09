"""
Binary .puzzle file parser for Opus Magnum.
Format transcribed from omsim/parse.c and omsim/parse.h.
"""

import math
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


# ── Analytic lower bounds ────────────────────────────────────────────────────

def cycles_lower_bound(pf: PuzzleFile) -> tuple[int, str]:
    """
    Throughput (N) + Latency (L) + 1 lower bound on cycle count.
    Higher N may decrease L if using a different sequence of steps.
    Each input may have different N, L so one dominates.

    Throughput:
    ───────────
    Each input glyph spawns at most 1 molecule per 2 cycles.  Three bounds are
    computed and the tightest wins.

    1. Per-type bound (non-wheel types): for each atom type T in both inputs and
       outputs (excluding wheel types), c_lo >= ceil(needed_T * N / supply_T) * 2.

    1b. Per-wheel-pool bound: when a wheel is available, all atom types on that
       wheel are fungible — any one of them can be delivered per rotation.  The
       constraint applies to the pool total:
         c_lo >= ceil(pool_needed * N / pool_supply) * 2
       where pool_supply = total wheel-type input atoms, pool_needed = total
       wheel-type output atoms.  If pool_supply == 0 the wheel alone provides
       those atoms and this bound contributes 0.

    2. Total-atom bound: when neither duplication nor any wheel is available,
       every output atom must have come from an input pickup:
         c_lo >= ceil(total_output_atoms * N / total_input_atoms) * 2

    Latency:
    ────────


    refs
    https://biggieblog.com/battling-the-entire-world-in-opus-magnum/
    """
    products_needed = pf.products_needed()

    # Determine active wheel pools
    wheel_pools: list[tuple[frozenset, str]] = []
    if pf.parts_available & PART_BARON:
        wheel_pools.append((BARON_WHEEL_ATOMS, "baron-wheel"))
    if pf.parts_available & PART_RAVARI:
        wheel_pools.append((RAVARI_WHEEL_ATOMS, "ravari-wheel"))

    def pool_for(atype: int) -> frozenset | None:
        for pool, _ in wheel_pools:
            if atype in pool:
                return pool
        return None

    supply: dict[int, int] = {}
    for mol in pf.inputs:
        for atype, cnt in mol.atom_type_counts().items():
            supply[atype] = supply.get(atype, 0) + cnt

    c_lo = 0
    bottleneck_note = None

    # Bound 1: per-atom-type (skip wheel-type atoms; handled by pool bound)
    for out_mol in pf.outputs:
        for atype, needed_per_product in out_mol.atom_type_counts().items():
            if pool_for(atype) is not None:
                continue
            s = supply.get(atype, 0)
            if s <= 0:
                continue
            cycles = math.ceil(needed_per_product * products_needed / s) * 2
            if cycles > c_lo:
                c_lo = cycles
                bottleneck_note = f"throughput-limited by {ATOM_NAMES.get(atype, str(atype))}"

    # Bound 1b: per-wheel-pool (all wheel-type atoms in the pool are fungible)
    for pool, pool_name in wheel_pools:
        pool_out = sum(
            cnt
            for mol in pf.outputs
            for atype, cnt in mol.atom_type_counts().items()
            if atype in pool
        )
        pool_in = sum(
            cnt
            for mol in pf.inputs
            for atype, cnt in mol.atom_type_counts().items()
            if atype in pool
        )
        if pool_in <= 0 or pool_out <= 0:
            continue
        cycles = math.ceil(pool_out * products_needed / pool_in) * 2
        if cycles > c_lo:
            c_lo = cycles
            bottleneck_note = (f"throughput: {pool_out} {pool_name} atoms out, "
                               f"{pool_in} in per round")

    # Bound 2: total-atom (skip when duplication or any wheel is present)
    total_atom_note = None
    if not (pf.parts_available & PART_DUPLICATION) and not wheel_pools:
        total_out = sum(mol.atom_count() for mol in pf.outputs)
        total_in  = sum(mol.atom_count() for mol in pf.inputs)
        if total_in > 0:
            total_cycles = math.ceil(total_out * products_needed / total_in) * 2
            if total_cycles >= c_lo:
                total_atom_note = (f"throughput: {total_out} output atoms, "
                                   f"{total_in} input atoms/round")
            if total_cycles > c_lo:
                c_lo = total_cycles
                bottleneck_note = None

    note = bottleneck_note or total_atom_note or "unknown"
    return c_lo, note


def cost_lower_bound(pf: PuzzleFile) -> tuple[int, str]:
    """
    Minimum cost based on required parts.

    Every solution needs:
      - at least 1 arm (20g)
      - a bonder (10g) if any output has normal bonds not in any input
      - a triplex bonder (20g) if any output has triplex bonds not in any input
      - a calcification glyph (10g) if salt appears in any output but no salt (only elementals) in any input
      - an animismus glyph (20g) if vitae or mors appear in any output but not in any inputs
        - 4 additional tracks (20g) to access the glyph
      - a berlow wheel (30g) + duplication glyph (20g) = 50g total if any outputs contains (air/earth/fire/water) that is not in any input
      # ignored for now:
      - ravari wheel (30g) if
      - a projection if
      - an unbonder (10g) if ...
      - a purification
      - a dispersion
      - a unification
      - additional track if access points insufficient
        - full access + partial access (existing bonds or bonder) + no access (?) glyphs
        - access per arm + #track
        - channels

    refs:
    https://biggieblog.com/optimizing-cost-in-opus-magnum/
    # todo check track proofs: stamina potion, Very Dark Thread, Voltaic Coil

    These are hard minimums — a solution with any fewer parts would be invalid.
    """
    # Collect atom types and bond types across inputs and outputs
    input_atom_types: set[int] = set()
    for mol in pf.inputs:
        input_atom_types.update(a.type for a in mol.atoms)

    input_has_normal  = any(b.is_normal  for mol in pf.inputs for b in mol.bonds)
    input_has_triplex = any(b.is_triplex for mol in pf.inputs for b in mol.bonds)

    output_atom_types: set[int] = set()
    output_has_normal  = False
    output_has_triplex = False
    for mol in pf.outputs:
        output_atom_types.update(a.type for a in mol.atoms)
        if mol.has_normal_bonds():
            output_has_normal = True
        if mol.has_triplex_bonds():
            output_has_triplex = True

    # Arm: always need arm1 (20g) to pick up atoms from input glyphs.
    # Baron/ravari wheels can't grab or release (instructions 'f'/'r' are no-ops
    # for wheels) — they only ever hold their own fixed wheel atoms.
    g_lo = 20
    reasons = ["1×arm=20g"]

    # Wheel + duplication: required when baron-wheel atom types appear in output
    # but not in any input.  Baron (30g) + glyph-duplication (20g) = 50g extra.
    missing_baron = (
        bool(output_atom_types & BARON_WHEEL_ATOMS - input_atom_types)
        and bool(pf.parts_available & PART_BARON)
    )
    missing_ravari = (
        bool(output_atom_types & RAVARI_WHEEL_ATOMS - input_atom_types)
        and bool(pf.parts_available & PART_RAVARI)
    )

    if missing_baron:
        g_lo += 50  # baron=30g + glyph-duplication=20g
        reasons.append("1×baron=30g + 1×glyph-duplication=20g")
    if missing_ravari:
        g_lo += 50  # ravari=30g + glyph-duplication=20g
        reasons.append("1×ravari=30g + 1×glyph-duplication=20g")

    # Bonding glyphs
    if output_has_normal and not input_has_normal:
        g_lo += 10
        reasons.append("1×bonder=10g")

    if output_has_triplex and not input_has_triplex:
        g_lo += 20
        reasons.append("1×bonder-prisma=20g")

    # Animismus: vitae or mors in output but not in any input.
    # Animismus also requires 4 track hexes (20g) to give the arm access to the glyph.
    needs_vitae = ATOM_VITAE in output_atom_types and ATOM_VITAE not in input_atom_types
    needs_mors  = ATOM_MORS  in output_atom_types and ATOM_MORS  not in input_atom_types
    if needs_vitae or needs_mors:
        g_lo += 40  # animismus=20g + 4×track=20g
        reasons.append("1×animismus=20g + 4×track=20g")

    # Calcification: salt in output, only elemental-type sources available in input
    if (ATOM_SALT in output_atom_types and ATOM_SALT not in input_atom_types
            and bool(input_atom_types & ELEMENTALS)):
        g_lo += 10
        reasons.append("1×calcification=10g")

    return g_lo, " + ".join(reasons)


def area_lower_bound(pf: PuzzleFile) -> tuple[int, str]:
    """
    Molecule-size lower bound on area.

    The area metric counts every distinct hex ever occupied by any atom during
    the run.  When a molecule is fully assembled (dropped to output, or freshly
    spawned from input), its atoms occupy distinct hexes.  Therefore:

        a_lo >= max atom count over all molecules (inputs + outputs)

    This is a weak but provably correct lower bound.
    """
    all_mols = pf.inputs + pf.outputs
    if not all_mols:
        return 1, "no molecules"

    best_mol = max(all_mols, key=lambda m: m.atom_count())
    a_lo = best_mol.atom_count()
    kind = "output" if best_mol in pf.outputs else "input"
    return a_lo, f"largest {kind} molecule has {a_lo} atoms"

def instructions_lower_bound(pf: PuzzleFile) -> tuple[int, str]:
    """
    All non-trivial puzzles need grab, action, drop.
    4 is sufficient for all free space puzzles.

    refs:
    https://biggieblog.com/optimizing-instructions-in-opus-magnum/
    """
    return 3, ''
