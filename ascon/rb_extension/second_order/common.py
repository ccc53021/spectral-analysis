"""Shared Table 5 parameters; external masks are five literal uint64 words.

Column zero is the MSB of a word. A local S-box value orders x0,...,x4
from its MSB to LSB. Packed integers below concatenate literal words;
they deliberately do not use the legacy bit-reversed state representation.
"""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
SBOX = (4, 11, 31, 20, 26, 21, 9, 2, 27, 5, 8, 18, 29, 3, 6, 28,
        30, 19, 7, 14, 0, 13, 17, 24, 16, 12, 1, 25, 22, 10, 15, 23)
ROTATIONS = ((19, 28), (61, 39), (1, 6), (10, 17), (7, 41))
RC = (0xF0, 0xE1, 0xD2, 0xC3, 0xB4, 0xA5, 0x96, 0x87,
      0x78, 0x69, 0x5A, 0x4B)
MASK64 = (1 << 64) - 1
IV = 0x80400C0600000000
DELTA1_WORDS = (0, 0, 0, 0x8000000000000000, 0x8000000000000000)
DELTA2_WORDS = (0, 0, 0, 0x1000000000000000, 0x1000000000000000)
OUTPUT_WORDS = (0x0000000000002000, 0, 0, 0, 0)
ZERO = (0, 0, 0, 0, 0)


def default_config():
    return dict(target="table5_ascon128_5r_second_order", rounds=5,
                begin_round=0, iv=IV, delta1_words=DELTA1_WORDS,
                delta2_words=DELTA2_WORDS, output_words=OUTPUT_WORDS,
                equal_columns=(0, 3), mode="active_equal")


def target_config(target="5r"):
    """Select a confirmed Table 5 target."""
    cfg = default_config()
    if target in ("5r", "table5_ascon128_5r_second_order"):
        return cfg
    if target not in ("6r", "table5_ascon128_6r_second_order"):
        raise ValueError("target must be 5r or 6r")
    cfg.update(target="table5_ascon128_6r_second_order", rounds=6,
               delta2_words=(0, 0, 0, 0x0000000000040000, 0x0000000000040000),
               output_words=(0x0000000800000000, 0, 0, 0, 0),
               equal_columns=(0, 45))
    return cfg


def parse_words(values):
    words = tuple(int(x, 0) if isinstance(x, str) else int(x) for x in values)
    if len(words) != 5 or any(x < 0 or x > MASK64 for x in words):
        raise ValueError("expected five unsigned 64-bit words")
    return words


def words_hex(words):
    return [f"0x{x:016x}" for x in parse_words(words)]


def words_to_int(words):
    return sum(word << (64 * row) for row, word in enumerate(parse_words(words)))


def int_to_words(value):
    if int(value) < 0 or int(value) >> 320:
        raise ValueError("packed word mask must fit 320 bits")
    return tuple((int(value) >> (64 * row)) & MASK64 for row in range(5))


def column(words, col):
    if not 0 <= col < 64:
        raise ValueError("column must be in [0,64)")
    return sum(((word >> (63 - col)) & 1) << (4 - row)
               for row, word in enumerate(parse_words(words)))


def column_words(value, col):
    if not 0 <= value < 32 or not 0 <= col < 64:
        raise ValueError("invalid local mask or column")
    return tuple(((value >> (4 - row)) & 1) << (63 - col) for row in range(5))


def canonical_mask(words, config=None):
    """Return (free-input mask, constant sign) on the configured affine domain."""
    cfg = default_config() if config is None else config
    result = list(parse_words(words))
    phase = (result[0] & int(cfg["iv"])).bit_count() & 1
    result[0] = 0
    mode = cfg.get("mode", "active_equal")
    if mode not in ("active_equal", "unrestricted", "full_equal"):
        raise ValueError("unknown input domain")
    columns = range(64) if mode == "full_equal" else (
        cfg.get("equal_columns", (0, 3)) if mode == "active_equal" else ())
    eq = sum(1 << (63 - int(c)) for c in set(columns))
    result[3] ^= result[4] & eq
    result[4] &= MASK64 ^ eq
    return tuple(result), (-1 if phase else 1)


def validate_config(config):
    cfg = dict(config)
    cfg["rounds"] = int(cfg["rounds"])
    cfg["begin_round"] = int(cfg.get("begin_round", 0))
    cfg["iv"] = int(cfg["iv"], 0) if isinstance(cfg["iv"], str) else int(cfg["iv"])
    cfg["equal_columns"] = tuple(sorted(set(int(c) for c in cfg.get("equal_columns", (0, 3)))))
    if not 1 <= cfg["rounds"] <= 6 or cfg["begin_round"] != 0:
        raise ValueError("this specialized implementation supports rounds 1..6 from f0")
    if not 0 <= int(cfg["iv"]) <= MASK64:
        raise ValueError("IV must fit 64 bits")
    for key in ("delta1_words", "delta2_words", "output_words"):
        cfg[key] = parse_words(cfg[key])
    if cfg["delta1_words"][0] or cfg["delta2_words"][0]:
        raise ValueError("the differences must preserve the fixed IV")
    if any(not 0 <= int(c) < 64 for c in cfg.get("equal_columns", ())):
        raise ValueError("invalid equality column")
    canonical_mask(ZERO, cfg)
    for direction in (cfg["delta1_words"], cfg["delta2_words"]):
        mode = cfg.get("mode", "active_equal")
        cols = range(64) if mode == "full_equal" else (
            cfg.get("equal_columns", ()) if mode == "active_equal" else ())
        if any(((direction[3] ^ direction[4]) >> (63 - c)) & 1 for c in cols):
            raise ValueError("input direction must preserve equality constraints")
    return cfg


def serializable_config(config=None):
    cfg = validate_config(default_config() if config is None else config)
    return {**cfg, "iv": f"0x{cfg['iv']:016x}",
            **{k: words_hex(cfg[k]) for k in
               ("delta1_words", "delta2_words", "output_words")},
            "bit_order": "literal words; column 0 = MSB; S-box x0 = MSB",
            "round_constants": [f"0x{x:02x}" for x in RC[:cfg['rounds']]],
            "output_boundary": "after final S-box, before final linear layer"}
