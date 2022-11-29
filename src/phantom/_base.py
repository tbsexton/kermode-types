from __future__ import annotations

import abc
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import TypeVar
from typing import Annotated


from typing_extensions import Protocol
from typing_extensions import runtime_checkable

from beartype.door import die_if_unbearable, is_bearable, TypeHint
from beartype.vale._core._valecore import BeartypeValidator


@runtime_checkable
class InstanceCheckable(Protocol):
    def __instancecheck__(self, instance: object) -> bool:
        ...


class KermodeMeta(abc.ABCMeta):
    """
    Metaclass that defers __instancecheck__ to derived classes and prevents actual
    instance creation.
    """

    def __instancecheck__(self, instance: object) -> bool:
        if not issubclass(self, InstanceCheckable):
            return False
        return self.__instancecheck__(  # type: ignore[no-any-return,attr-defined]
            instance,
        )

    # With the current level of metaclass support in mypy it's unlikely that we'll be
    # able to make this context typed, hence the ignores.
    def __call__(cls, instance):  # type: ignore[no-untyped-def]
        return cls.parse(instance)  # type: ignore[attr-defined]


T = TypeVar("T", covariant=True)
Parser: TypeAlias = Callable[[object], T]
Derived = TypeVar("Derived", bound="KermodeBase")


class KermodeBase(metaclass=KermodeMeta):
    @classmethod
    def parse(cls: type[Derived], instance: object) -> Derived:
        """
        Parse an arbitrary value into a phantom type.
        :raises TypeError:
        """
        TypeHint(cls).die_if_unbearable(instance)

        return instance

    @classmethod
    @abc.abstractmethod
    def __instancecheck__(cls, instance: object) -> bool:
        ...


class UnresolvedClassAttribute(NotImplementedError):
    ...


def resolve_class_attr(
    cls: type,
    name: str,
    argument: object | None,
    required: bool = True,
) -> None:
    argument = getattr(cls, name, None) if argument is None else argument
    if argument is not None:
        setattr(cls, name, argument)
    elif required and not getattr(cls, "__abstract__", False):
        raise UnresolvedClassAttribute(
            f"Concrete phantom type {cls.__qualname__} must define class attribute "
            f"{name}."
        )


def get_beartype_parser(beartype: type[T] | Any) -> Parser[T]:
    def parser(instance: object) -> T:
        TypeHint(cls).die_if_unbearable(instance)
        return cast(T, instance)

    return parser


class Kermode(KermodeBase, Generic[T]):
    """
    Base class for predicate-based phantom types.
    **Class arguments**
    * ``validator: BearTypeValidator | None`` - Predicate function used for instance checks.
      Can be ``None`` if the type is abstract.
    * ``beartype: type[T] | None`` - Bound used to check values before passing them to the
      type's predicate function. This will often but not always be the same as the
      runtime type that values of the phantom type are represented as. If this is not
      provided as a class argument, it's attempted to be resolved in order from an
      implicit bound (any bases of the type that come before ``Phantom``), or inherited
      from super phantom types that provide a bound. Can be ``None`` if the type is
      abstract.
    """

    __validator__: BeartypeValidator
    # The bound of a phantom type is the type that its values will have at
    # runtime, so when checking if a value is an instance of a phantom type,
    # it's first checked to be within its bounds, so that the value can be
    # safely passed as argument to the predicate function.
    #
    # When subclassing, the bound of the new type must be a subtype of the bound
    # of the super class.
    __beartype__: TypeHint

    def __init_subclass__(
        cls,
        validator: BeartypeValidator | None = None,
        beartype: type[T] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init_subclass__(**kwargs)
        resolve_class_attr(cls, "__validator__", validator)

        resolve_class_attr(cls, "__beartype__", TypeHint(cls))
        cls._resolve_beartype(beartype)

    @classmethod
    def _interpret_implicit_bound(cls) -> BoundType:
        def discover_beartypes() -> Iterable[type]:
            for type_ in cls.__mro__:
                if type_ is cls:
                    continue
                if TypeHint(type_).issubclass(Kermode):
                    break
                yield type_
            else:  # pragma: no cover
                raise RuntimeError(f"{cls} is not a subclass of Phantom")

        types = tuple(discover_beartypes())
        if len(types) == 1:
            return types[0]
        return types

    @classmethod
    def _resolve_beartype(cls, class_arg: Any) -> None:
        inherited = getattr(cls, "__beartype__", None)
        implicit = cls._interpret_implicit_bound()
        validator = getattr(cls, "__validator__", name)
        if class_arg is not None:  # told explicitly in `__init_subclass__(bound=...)`
            bound = class_arg
        elif implicit:  # told implicitly in MyClass(implicit, Kermode)
            bound = implicit
        elif inherited is not None:  # inherited via MyClass(MyBase(Kermode))
            bound = inherited.hint
        else:
            return
        if validator is not None:
            bound = Annotated[bound, validator]
        if inherited is not None and not TypeHint(bound).is_subhint(inherited):
            raise BoundError(
                f"The bound of {cls.__qualname__} is not compatible with its "
                f"inherited bounds."
            )

        cls.__beartype__ = TypeHint(bound)

    @classmethod
    def __instancecheck__(cls, instance: object) -> bool:

        bound_parser: Parser[T] = get_beartype_parser(cls.__beartype__)
        try:
            instance = bound_parser(instance)
        except BoundError:
            return False
        return cls.__beartype__.is_bearable(instance)
