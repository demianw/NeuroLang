from ...expressions import Symbol
from .. import (Conjunction, Disjunction, ExistentialPredicate, Negation,
                UniversalPredicate, expression_processing)
from ...expression_walker import IdentityWalker


x = Symbol('x')
y = Symbol('y')
P = Symbol('P')
Q = Symbol('Q')


def test_flatten_conjunction():
    class FC(
        expression_processing.FlattenConjunctionDisjunctionWalker,
        IdentityWalker
    ):
        pass

    fc = FC()

    p1 = Conjunction((P(x), Q(y)))
    p2 = Disjunction((P(x), Q(x)))
    p3 = Negation(P(x))

    assert fc.walk(p1) == p1
    assert fc.walk(p2) == p2
    assert fc.walk(p3) == p3

    p4 = Conjunction((P(x), Conjunction((Q(y), P(y)))))
    assert set(fc.walk(p4).formulas) == set((P(x), Q(y), P(y)))

    p5 = Conjunction((Conjunction((P(x), Q(y))), P(y)))
    assert set(fc.walk(p5).formulas) == set((P(x), Q(y), P(y)))

    p6 = Conjunction((Conjunction((P(x), Conjunction((Q(y), Q(x))))), P(y)))
    assert set(fc.walk(p6).formulas) == set((P(x), Q(y), Q(x), P(y)))


def test_flatten_disjunction():
    class FC(
        expression_processing.FlattenConjunctionDisjunctionWalker,
        IdentityWalker
    ):
        pass

    fc = FC()

    p1 = Conjunction((P(x), Q(y)))
    p2 = Disjunction((P(x), Q(x)))
    p3 = Negation(P(x))

    assert fc.walk(p1) == p1
    assert fc.walk(p2) == p2
    assert fc.walk(p3) == p3

    p4 = Disjunction((P(x), Disjunction((Q(y), P(y)))))
    assert set(fc.walk(p4).formulas) == set((P(x), Q(y), P(y)))

    p5 = Disjunction((Disjunction((P(x), Q(y))), P(y)))
    assert set(fc.walk(p5).formulas) == set((P(x), Q(y), P(y)))

    p6 = Disjunction((Disjunction((P(x), Disjunction((Q(y), Q(x))))), P(y)))
    assert set(fc.walk(p6).formulas) == set((P(x), Q(y), Q(x), P(y)))
