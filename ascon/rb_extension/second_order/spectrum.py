"""Finite-spectrum tools with explicit distinction between unknown and zero."""
from __future__ import annotations

import numpy as np

from common import ZERO, canonical_mask, int_to_words, parse_words, words_hex, words_to_int


def coefficient_map(result):
    if not result.get("complete", result.get("status") == "complete"):
        raise ValueError("an unfinished root is not a spectrum")
    coefficients = {}
    for item in result["terms"]:
        mask = parse_words(item["mask_words"])
        coefficients[mask] = coefficients.get(mask, 0.0) + float(item["coefficient"])
    return coefficients


def independent_basis(masks):
    rows, basis = {}, []
    for mask in masks:
        original = words_to_int(mask)
        reduced = original
        while reduced:
            pivot = reduced.bit_length() - 1
            if pivot not in rows:
                rows[pivot] = reduced
                basis.append(parse_words(mask))
                break
            reduced ^= rows[pivot]
    return basis


def span_masks(basis):
    if len(independent_basis(basis)) != len(basis):
        raise ValueError("basis must be linearly independent")
    values = [0]
    for mask in basis:
        packed = words_to_int(mask)
        values += [x ^ packed for x in values]
    return [int_to_words(x) for x in values]


def fwht(values):
    """Unnormalised FWHT: Fourier coefficients -> conditional expectations."""
    out = np.asarray(values, dtype=np.float64).copy()
    n = out.size
    if out.ndim != 1 or n == 0 or n & (n - 1):
        raise ValueError("FWHT requires a nonempty power-of-two vector")
    h = 1
    while h < n:
        blocks = out.reshape(-1, 2 * h)
        left, right = blocks[:, :h].copy(), blocks[:, h:].copy()
        blocks[:, :h], blocks[:, h:] = left + right, left - right
        h *= 2
    return out


def analyze_span(coefficients, basis, *, exact=False, max_classes=65536, config=None):
    """Do not publish a distribution when bounded-spectrum coordinates are absent.

    Presence of every requested coordinate is not a convergence proof either;
    a bounded model is always labelled as a diagnostic requiring experiments.
    max_classes is a resource guard, not a prescribed scientific dimension.
    """
    basis = [parse_words(mask) for mask in basis]
    if any(canonical_mask(mask, config) != (mask, 1) for mask in basis):
        raise ValueError("basis must use canonical free-input masks")
    if len(independent_basis(basis)) != len(basis):
        raise ValueError("dependent basis")
    count = 1 << len(basis)
    metadata = dict(dimension=len(basis), class_count=count,
                    basis=[words_hex(mask) for mask in basis],
                    exact_within_propagation_model=bool(exact))
    if count > max_classes:
        return dict(metadata, status="span_budget", distribution=None)
    targets = span_masks(basis)
    missing = [i for i, mask in enumerate(targets) if mask not in coefficients]
    if missing and not exact:
        return dict(metadata, status="unknown_coefficients", distribution=None,
                    missing_count=len(missing), missing_indices=missing)
    transformed = fwht([coefficients.get(mask, 0.0) for mask in targets])
    return dict(metadata, status="exact" if exact else "bounded_diagnostic",
                missing_count=len(missing), distribution=transformed.tolist(),
                minimum=float(transformed.min()), maximum=float(transformed.max()),
                minimum_index=int(transformed.argmin()), maximum_index=int(transformed.argmax()),
                zero_frequency=float(coefficients.get(ZERO, 0.0)),
                mean=float(transformed.mean()),
                within_correlation_bounds=bool(np.max(np.abs(transformed)) <= 1 + 1e-10))
