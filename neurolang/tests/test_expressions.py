from .. import expressions
from ..expressions import (
    Symbol, Constant,
    FunctionApplication
)

from ..expression_walker import ExpressionBasicEvaluator

import operator as op
import inspect
from typing import Set, Callable

C = Constant


def evaluate(expression, **kwargs):
    ebe = ExpressionBasicEvaluator()
    for k, v in kwargs.items():
        ebe.symbol_table[k] = v
    return ebe.walk(expression)


def test_symbol_application():
    a = Symbol('a')
    b = Symbol('b')

    oadd = C(op.add, type_=Callable[[int, int], int])
    osub = C(op.sub,  type_=Callable[[int, int], int])
    omul = C(op.mul,  type_=Callable[[int, int], int])
    c = oadd(C(2), C(3))
    assert c.functor == oadd
    assert all((e1.value == e2 for e1, e2 in zip(c.args, (2, 3))))
    assert evaluate(c) == 5

    fva = oadd(a, C(3))
    fvb = osub(fva, C(10))
    fvc = omul(fvb, b)
    fvd = FunctionApplication(a, None, kwargs={'c': b})
    fve = FunctionApplication(a, None, kwargs={'c': op.add(b, C(2))})

    assert a in fva._symbols and (len(fva._symbols) == 1)
    assert evaluate(fva, a=C(2)) == 5
    assert a in fvb._symbols and (len(fvb._symbols) == 1)
    assert evaluate(fvb, a=C(2)) == -5
    assert b in fvc._symbols and (len(fvc._symbols) == 2)
    assert isinstance(evaluate(fvc, b=C(2)), FunctionApplication)
    assert evaluate(
        evaluate(fvc, b=C(3)), a=C(2)
    ) == evaluate(fvc, a=C(2), b=C(3)) == -15
    return
    assert evaluate(
        evaluate(fvd, a=C(lambda *args, **kwargs: kwargs['c'])),
        b=C(2)
    ) == 2
    assert evaluate(
        evaluate(fvd, b=C(2)),
        a=lambda *args, **kwargs: kwargs['c']
    ) == 2
    assert evaluate(
        evaluate(fve, b=C(2)),
        a=lambda *args, **kwargs: kwargs['c']
    ) == 4


def test_symbol_method_and_operator():
    a = Symbol('a')
    fva = a.__len__()
    fvb = a - C(4, type_=int)
    fvc = C(4, type_=int) - a
    fve = a[C(2)]

    assert evaluate(a, a=C(1)) == 1
    assert evaluate(fva, a=C({1}, type_=Set[int])) == 1
    assert evaluate(fvb, a=C(1)) == -3
    assert evaluate(fvc, a=C(1)) == 3
    assert evaluate(fvc * C(2), a=C(1)) == 6
    assert evaluate(a | C(False), a=True)
    assert evaluate(
        fve,
        a=C([C(x) for x in (0, 1, 2)])
    ) == 2


def test_constant_method_and_operator():
    a = C(1, type_=int)
    fva = a + C(1)
    b = C({1}, type_=Set[int])
    fvb = b.__len__()
    fbc = b.union(C({C(1), C(2)}))

    assert a == 1
    assert evaluate(a) == 1
    assert evaluate(fva) == 2
    assert evaluate(fvb) == 1
    assert evaluate(fbc).value == {1, 2}


def test_symbol_wrapping():
    def f(a: int)->float:
        '''
        test help
        '''
        return 2. * int(a)

    fva = Constant(f)
    x = Symbol('x', type_=int)
    fvb = fva(x)
    assert fva.__annotations__ == f.__annotations__
    assert fva.__doc__ == f.__doc__
    assert fva.__name__ == f.__name__
    assert fva.__qualname__ == f.__qualname__
    assert fva.__module__ == f.__module__
    assert inspect.signature(fva) == inspect.signature(f)
    assert fvb.type == float
    assert x.type == int


def test_compatibility_for_pattern_matching():
    for symbol_name in dir(expressions):
        symbol = getattr(expressions, symbol_name)
        if not (
            type(symbol) == type and
            issubclass(symbol, expressions.Expression) and
            symbol != expressions.Expression
        ):
            continue
        signature = inspect.signature(symbol)
        argnames = [
            name for name, param in signature.parameters.items()
            if param.default == signature.empty
        ]
        args = (...,) * len(argnames)
        instance = symbol(*args)

        for argname in argnames:
            assert getattr(instance, argname) == ...