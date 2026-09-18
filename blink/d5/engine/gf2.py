"""Small GF(2) helpers used by the exact quotient computations."""


class RankAccumulator:
    """Incremental row space using highest-set-bit pivots."""

    def __init__(self, rows=()):
        self.pivots = {}
        for row in rows:
            self.add(row)

    @property
    def rank(self):
        return len(self.pivots)

    def reduce(self, value):
        value = int(value)
        while value:
            pivot = value.bit_length() - 1
            previous = self.pivots.get(pivot)
            if previous is None:
                break
            value ^= previous
        return value

    def add(self, value):
        value = self.reduce(value)
        if not value:
            return False
        self.pivots[value.bit_length() - 1] = value
        return True


class LinearCoordinates:
    """Coordinates in an explicitly ordered independent GF(2) basis."""

    def __init__(self, basis):
        self.basis = tuple(int(row) for row in basis)
        self.pivots = {}
        for index, row in enumerate(self.basis):
            value = row
            coordinate = 1 << index
            while value:
                pivot = value.bit_length() - 1
                previous = self.pivots.get(pivot)
                if previous is None:
                    self.pivots[pivot] = (value, coordinate)
                    break
                value ^= previous[0]
                coordinate ^= previous[1]
            if not value:
                raise ValueError("basis is linearly dependent")

    @property
    def dimension(self):
        return len(self.basis)

    def coordinate(self, value):
        value = int(value)
        coordinate = 0
        while value:
            pivot = value.bit_length() - 1
            previous = self.pivots.get(pivot)
            if previous is None:
                return None
            value ^= previous[0]
            coordinate ^= previous[1]
        return coordinate


def ordered_basis(weighted_rows):
    """Select a deterministic independent basis from ``(mask, weight)`` rows.

    Larger absolute Fourier coefficients are considered first.  This order is
    convenient for audit output only; the full quotient is independent of the
    selected basis.
    """
    span = RankAccumulator()
    selected = []
    for mask, coefficient in sorted(
        weighted_rows, key=lambda item: (-abs(int(item[1])), int(item[0]))
    ):
        if span.add(mask):
            selected.append(int(mask))
    return tuple(selected)

