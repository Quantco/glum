"""Term-to-coefficient layout for fitted GLM models.

A :class:`TermLayout` is a lightweight, immutable view that records which
contiguous slice of the coefficient vector each model term occupies. It is
populated alongside ``term_names_`` after fitting and is intended to be the
single source of truth for code that needs to reason about *terms* rather than
individual coefficients (penalty assembly, term-level diagnostics, smooth
construction, etc.).

This module is intentionally small. It contains no formula logic and no
reference to ``formulaic`` or ``tabmat``; it builds the layout from the
per-coefficient term-name list that ``GeneralizedLinearRegressorBase`` already
constructs.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TermSlice:
    """One term and the coefficient indices it owns.

    ``start`` / ``stop`` are half-open and index the full coefficient vector,
    i.e. with the intercept at position 0 when ``fit_intercept`` is true.
    """

    name: str
    start: int
    stop: int
    kind: str

    @property
    def n_coefs(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True, slots=True)
class TermLayout:
    """Ordered view of terms and the coefficient slice each one owns.

    The layout is derived from the per-coefficient term-name list produced
    during fit and the intercept flag. Construction is O(n_features); after
    that all lookups are O(n_terms).
    """

    terms: tuple[TermSlice, ...]
    has_intercept: bool
    _by_name: dict[str, TermSlice]

    @classmethod
    def from_term_names(
        cls,
        term_names: Sequence[str],
        fit_intercept: bool,
    ) -> TermLayout:
        """Build a layout from a per-coefficient term-name list.

        ``term_names`` must have one entry per non-intercept coefficient, in
        coefficient order. Adjacent equal entries are collapsed into a single
        term slice. If ``fit_intercept`` is true, an ``"intercept"`` slice is
        prepended at index 0.
        """
        slices: list[TermSlice] = []
        offset = 1 if fit_intercept else 0
        if fit_intercept:
            slices.append(TermSlice("intercept", 0, 1, "intercept"))

        i = 0
        n = len(term_names)
        while i < n:
            j = i + 1
            current = term_names[i]
            while j < n and term_names[j] == current:
                j += 1
            slices.append(
                TermSlice(
                    name=str(current) if current is not None else f"_term_{i}",
                    start=offset + i,
                    stop=offset + j,
                    kind="term",
                )
            )
            i = j

        return cls(
            terms=tuple(slices),
            has_intercept=fit_intercept,
            _by_name={s.name: s for s in slices},
        )

    def __iter__(self) -> Iterator[TermSlice]:
        return iter(self.terms)

    def __len__(self) -> int:
        return len(self.terms)

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._by_name[key]
        return self.terms[key]

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return key in self._by_name

    @property
    def n_coefs(self) -> int:
        """Total number of coefficients spanned by the layout."""
        return self.terms[-1].stop if self.terms else 0

    @property
    def names(self) -> tuple[str, ...]:
        """Term names in coefficient order."""
        return tuple(term.name for term in self.terms)
