import numpy as np


def get_kth_coordinates_of(a, k):
    """
    Say that we number the entries of the matrix from left to right then top to bottom.
    In what cell does the input integer k appear?

    >>> get_kth_coordinates_of(np.ones((3, 10)), 9)
    (0, 9)
    >>> get_kth_coordinates_of(np.ones((3, 10)), 10)
    (1, 0)
    """
    assert 0 <= k and k < a.size
    return (k // a.shape[1], k % a.shape[1])


def sample_coordinates_from(rng, a):
    """
    If we probability normalize all of the entries of a 2d array and sample from the
    corresponding categorical distribution, get the coordinates of the sampled entry.
    """
    m = np.arange(a.size).reshape(a.shape)
    return get_kth_coordinates_of(a, rng.choice(m.flat, p=a.flat / a.sum()))


class EventSampler:
    """
    Say we have a collection of entities, each of which can have a collection of events
    happen to them. These are rates of competing processes with
    exponentially-distributed time to events.

    In this implementation, rates are laid out in a matrix with events on the rows and
    things on the columns.
    """

    def __init__(self, rng):
        self.rng = rng
        self.rates = np.array([])

    def sample_time_to_next_event(self):
        """
        Sample the time to the next event.
        """
        return self.rng.exponential(1.0 / self.rates.sum())

    def sample_next_event(self):
        """
        Sample the next event.

        Returns a tuple (event_type, entity_idx)
        """
        return sample_coordinates_from(self.rng, self.rates)

    def drop(self, to_drop_idx: int):
        """
        Remove a given entity.
        """
        self.rates = np.delete(self.rates, to_drop_idx, axis=1)

    def append(self, rate_column: np.ndarray):
        """
        Append rates for a new entity.

        Return the index of the appended item.
        """
        to_add = rate_column.reshape(-1, 1)
        if self.rates.size == 0:
            self.rates = to_add
        else:
            self.rates = np.hstack((self.rates, to_add))
        return self.rates.shape[1]
