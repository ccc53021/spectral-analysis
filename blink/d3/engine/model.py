"""Two distinct 4+6+4 events; evaluate actual pairs, never reset a difference.

All six linear connectors and their shared physical round keys are retained.
The restricted version checks every old two-layer endpoint. The endpoint-only
version checks only the three new block endpoints. Constants are not key vars.
"""
import numpy as np

from .conditioning import core, box64, connector64

PAIRS = [(core.U, core.T), (core.A, core.B), (core.C, core.D),
         (core.E, core.E), (core.D, core.C), (core.B, core.A), (core.T, core.U)]
RANGES = {'head': (0, 2), 'middle': (2, 5), 'tail': (5, 7), 'full': (0, 7)}
BLOCK_ENDS = (1, 4, 6)
CASES = {
    'old_maximum': {
        'X': 0x0000000010006000,
        'keys': [0x13198a2e03707344, 0x2254000043000000, 0,
                 0x0476540007440000, 0xd71577c1bd314b27]},
    'sat_witness': {
        'X': 0x7c0ac1a061aa71e8,
        'keys': [0x2874cc1e532a2747, 0x1062f52829c95f66,
                 0x18c078facb7bfa85, 0x73ae068f503c26b9,
                 0x90479091156bcc2d]},
}


def schedule(keys):
    if len(keys) != 5:
        raise ValueError('Exactly five physical 64-bit round-key words required.')
    k = [int(a) ^ b for a, b in zip(keys, core.RC)]
    kp = [int(a) ^ b for a, b in zip(keys, core.RCI)]
    boxes = [k[0], 0, k[3], 0, core.mix(kp[1]), 0, core.mix(kp[4])]
    joins = [(k[1], False), (k[2], False), (k[4], False),
             (kp[0], True), (kp[2], True), (kp[3], True)]
    return boxes, joins


def evaluate(x, keys, version='restricted', block='full', trace=False):
    if version not in ('restricted', 'endpoints'):
        raise ValueError(version)
    start, stop = RANGES[block]
    boxes, joins = schedule(keys)
    left = np.asarray(x, dtype=np.uint64)
    right = left ^ np.uint64(PAIRS[start][0])
    good = np.ones(left.shape, dtype=bool)
    states = []
    for j in range(start, stop):
        di = left ^ right
        left, right = box64(left, np.uint64(boxes[j])), box64(right, np.uint64(boxes[j]))
        do = left ^ right
        if version == 'restricted' or j in BLOCK_ENDS:
            good &= do == np.uint64(PAIRS[j][1])
        if trace:
            states.append((j, di.copy(), do.copy()))
        if j + 1 < stop:
            key, reverse = joins[j]
            left = connector64(left, np.uint64(key), reverse)
            right = connector64(right, np.uint64(key), reverse)
    return (good, left, right, states) if trace else good


def symbolic(circuit, x, keys, version='restricted', block='full'):
    """Return a logically equivalent pair circuit and required predicates.

    After a required fixed-difference equality, substitute that equality in
    the second value. This is sound only when ALL returned predicates are
    asserted; it avoids an unnecessarily expanded second 14-layer circuit.
    No omitted/unrequired internal difference is reset.
    """
    c = circuit
    start, stop = RANGES[block]
    boxes, joins = schedule(keys)
    left, right = x, x ^ c.c(PAIRS[start][0])
    conditions, old_conditions = [], []
    for j in range(start, stop):
        left, right = c.box(left, c.c(boxes[j])), c.box(right, c.c(boxes[j]))
        eq = left ^ right == c.c(PAIRS[j][1])
        old_conditions.append(eq)
        if version == 'restricted' or j in BLOCK_ENDS:
            conditions.append(eq)
            right = left ^ c.c(PAIRS[j][1])
        if j + 1 < stop:
            key, reverse = joins[j]
            left = c.connector(left, c.c(key), reverse)
            right = c.connector(right, c.c(key), reverse)
    return conditions, old_conditions, left, right


def verify():
    rng = np.random.default_rng(464)
    cases = 0
    for keys in [v['keys'] for v in CASES.values()]:
        x = rng.integers(0, 1 << 64, 128, dtype=np.uint64)
        x = np.append(x, np.uint64(CASES['sat_witness']['X']))
        p1, left, right, trace = evaluate(x, keys, trace=True)
        p2 = evaluate(x, keys, 'endpoints')
        assert np.all(~p1 | p2)
        master = sum(int(k) << (128 + 64 * i) for i, k in enumerate(keys))
        for i in range(len(x)):
            assert int(left[i]) == core.encrypt_zero_tweak(int(x[i]), master, grouped=False)
            assert int(right[i]) == core.encrypt_zero_tweak(int(x[i]) ^ core.U, master, grouped=False)
            cases += 2
    witness = CASES['sat_witness']
    assert evaluate(witness['X'], witness['keys'])
    assert evaluate(witness['X'], witness['keys'], 'endpoints')
    # Linear joins transport the fixed differential; no guessed differences.
    for j, (_, reverse) in enumerate(schedule([0] * 5)[1]):
        assert core.connector(PAIRS[j][1], 0, reverse) == PAIRS[j + 1][0]
    return {'passed': True, 'ungrouped_cipher_value_checks': cases,
            'full_route_known_success_in_both_models': True,
            'all_six_difference_connections_checked': True,
            'restricted_implies_endpoints_by_construction': True,
            'actual_connectors_retained': 6, 'zero_projected_connectors': 0}
