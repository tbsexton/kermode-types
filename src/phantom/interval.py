"""
Types for describing narrower sets of numbers than builtin numeric types like ``int``
and ``float``. Use the provided base classes to build custom intervals. For example, to
represent number in the closed range ``[0, 100]`` for a volume control you would define
a type like this:

.. code-block:: python

    class VolumeLevel(int, Inclusive, low=0, high=100):
        ...

There is also a set of concrete ready-to-use interval types provided, that use predicate
functions from :py:mod:`phantom.predicates.interval`.

.. code-block:: python

    def take_portion(portion: Portion, whole: Natural) -> float:
        return portion * whole

All interval types fully support pydantic and appropriately adds inclusive or exclusive
minimums and maximums to their schema representations.
"""

from __future__ import annotations

from typing import Any
from typing import TypeVar

from typing_extensions import Final
from typing_extensions import Protocol

from . import Phantom
from . import Predicate
from ._utils.misc import resolve_class_attr
from ._utils.types import Comparable
from ._utils.types import SupportsEq
from .predicates import interval
from .schema import Schema

N = TypeVar("N", bound=Comparable)
Derived = TypeVar("Derived", bound="Interval")


class IntervalCheck(Protocol):
    def __call__(self, a: N, b: N) -> Predicate[N]:
        ...


inf: Final = float("inf")
neg_inf: Final = float("-inf")


class Interval(Phantom[Comparable], bound=Comparable, abstract=True):
    """
    Base class for all interval types, providing the following class arguments:

    * ``check: IntervalCheck``
    * ``low: Comparable`` (defaults to negative infinity)
    * ``high: Comparable`` (defaults to positive infinity)

    Concrete subclasses must specify their runtime type bound as their first base.
    """

    __check__: IntervalCheck
    __low__: Comparable
    __high__: Comparable

    def __init_subclass__(
        cls,
        check: IntervalCheck | None = None,
        low: Comparable = neg_inf,
        high: Comparable = inf,
        **kwargs: Any,
    ) -> None:
        resolve_class_attr(cls, "__low__", low)
        resolve_class_attr(cls, "__high__", high)
        resolve_class_attr(cls, "__check__", check)
        if getattr(cls, "__check__", None) is None:
            raise TypeError(f"{cls.__qualname__} must define an interval check")
        super().__init_subclass__(
            predicate=cls.__check__(cls.__low__, cls.__high__),
            **kwargs,
        )

    @classmethod
    def parse(cls: type[Derived], instance: object) -> Derived:
        return super().parse(
            cls.__bound__(instance) if isinstance(instance, str) else instance
        )


def _format_limit(value: SupportsEq) -> str:
    if value == inf:
        return "∞"
    if value == neg_inf:
        return "-∞"
    return str(value)


class Exclusive(Interval, check=interval.exclusive, abstract=True):
    """Uses :py:func:`phantom.predicates.interval.exclusive` as ``check``."""

    @classmethod
    def __schema__(cls) -> Schema:
        return {
            **super().__schema__(),  # type: ignore[misc]
            "description": (
                f"A value in the exclusive range ({_format_limit(cls.__low__)}, "
                f"{_format_limit(cls.__high__)})."
            ),
            "exclusiveMinimum": cls.__low__ if cls.__low__ != neg_inf else None,
            "exclusiveMaximum": cls.__high__ if cls.__high__ != inf else None,
        }


class Inclusive(Interval, check=interval.inclusive, abstract=True):
    """Uses :py:func:`phantom.predicates.interval.inclusive` as ``check``."""

    @classmethod
    def __schema__(cls) -> Schema:
        return {
            **super().__schema__(),  # type: ignore[misc]
            "description": (
                f"A value in the inclusive range [{_format_limit(cls.__low__)}, "
                f"{_format_limit(cls.__high__)}]."
            ),
            "minimum": cls.__low__ if cls.__low__ != neg_inf else None,
            "maximum": cls.__high__ if cls.__high__ != inf else None,
        }


class ExclusiveInclusive(Interval, check=interval.exclusive_inclusive, abstract=True):
    """Uses :py:func:`phantom.predicates.interval.exclusive_inclusive` as ``check``."""

    @classmethod
    def __schema__(cls) -> Schema:
        return {
            **super().__schema__(),  # type: ignore[misc]
            "description": (
                f"A value in the half-open range ({_format_limit(cls.__low__)}, "
                f"{_format_limit(cls.__high__)}]."
            ),
            "exclusiveMinimum": cls.__low__ if cls.__low__ != neg_inf else None,
            "maximum": cls.__high__ if cls.__high__ != inf else None,
        }


class InclusiveExclusive(Interval, check=interval.inclusive_exclusive, abstract=True):
    """Uses :py:func:`phantom.predicates.interval.inclusive_exclusive` as ``check``."""

    @classmethod
    def __schema__(cls) -> Schema:
        return {
            **super().__schema__(),  # type: ignore[misc]
            "description": (
                f"A value in the half-open range [{_format_limit(cls.__low__)}, "
                f"{_format_limit(cls.__high__)})."
            ),
            "minimum": cls.__low__ if cls.__low__ != neg_inf else None,
            "exclusiveMaximum": cls.__high__ if cls.__high__ != inf else None,
        }


class Natural(int, InclusiveExclusive, low=0):
    """Represents integer values in the inclusive range ``[0, ∞)``."""

    @classmethod
    def __schema__(cls) -> Schema:
        return {
            **super().__schema__(),  # type: ignore[misc]
            "description": "An integer value in the inclusive range [0, ∞).",
        }


class NegativeInt(int, ExclusiveInclusive, high=0):
    """Represents integer values in the inclusive range ``(-∞, 0]``."""

    @classmethod
    def __schema__(cls) -> Schema:
        return {
            **super().__schema__(),  # type: ignore[misc]
            "description": "An integer value in the inclusive range (-∞, 0].",
        }


class Portion(float, Inclusive, low=0, high=1):
    """Represents float values in the inclusive range ``[0, 1]``."""

    @classmethod
    def __schema__(cls) -> Schema:
        return {
            **super().__schema__(),  # type: ignore[misc]
            "description": "A float value in the inclusive range [0, 1].",
        }
