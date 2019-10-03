from typing import AbstractSet, Callable, Tuple

import numpy as np
from pytest import raises

from ... import expression_walker as ew
from ... import expressions
from ...expressions import Constant, Expression, FunctionApplication, Symbol
from ...type_system import get_args, is_leq_informative
from ...utils import RelationalAlgebraFrozenSet
from ..basic_representation import DatalogProgram
from ..chase import (ChaseGeneral, ChaseNamedRelationalAlgebraMixin,
                     ChaseSemiNaive)
from ..expressions import Disjunction, Fact, Implication
from ..wrapped_collections import WrappedRelationalAlgebraSet

C_ = expressions.Constant
S_ = expressions.Symbol
Imp_ = Implication
F_ = Fact
Eb_ = expressions.ExpressionBlock
Disj_ = Disjunction


class Chase(ChaseNamedRelationalAlgebraMixin, ChaseSemiNaive, ChaseGeneral):
    pass


class ReplaceExpressionsByValues(ew.ReplaceExpressionsByValues):
    @ew.add_match(Constant[AbstractSet])
    def constant_abstract_set(self, constant_abstract_set):
        value = constant_abstract_set.value
        if isinstance(value, WrappedRelationalAlgebraSet):
            return value.unwrap()
        else:
            return super().constant_abstract_set(constant_abstract_set)


class Datalog(DatalogProgram, ew.ExpressionBasicEvaluator):
    @ew.add_match(
        FunctionApplication(Constant, ...),
        lambda exp: any(isinstance(a, Symbol) for a in exp.args)
    )
    def funct_over_symbols(self, expression):
        new_args = tuple()
        changed = False
        for arg in expression.args:
            if isinstance(arg, Symbol):
                new_arg = self.symbol_table.get(arg, arg)
                if new_arg is not arg:
                    changed |= True
                new_args += (new_arg,)

        if changed:
            return self.walk(FunctionApplication(expression.functor, new_args))
        else:
            return expression

    @ew.add_match(
        FunctionApplication(Constant[Callable], ...),
        lambda e:
        is_leq_informative(get_args(e.functor.type)[-1], AbstractSet[Tuple])
        and all(
            not isinstance(arg, Expression) or
            (
                isinstance(arg, Constant) and
                is_leq_informative(arg.type, AbstractSet[Tuple])
            )
            for arg in e.args
        )
    )
    def evaluate_function_vectorwise(self, function_application):
        functor = function_application.functor

        rebv = ReplaceExpressionsByValues(self.symbol_table)
        args = rebv.walk(function_application.args)
        kwargs = {
            k: rebv.walk(v)
            for k, v in function_application.kwargs.items()
        }

        result = functor.value(*args, **kwargs)
        result = WrappedRelationalAlgebraSet(result)
        result = Constant[get_args(functor.type)[-1]](result)
        return result

    @ew.add_match(
        FunctionApplication(Constant[Callable], ...),
        lambda e:
        is_leq_informative(get_args(e.functor.type)[-1], bool)
        and all(
            not isinstance(arg, Expression) or
            (
                isinstance(arg, Constant) and
                is_leq_informative(arg.type, AbstractSet[Tuple])
            )
            for arg in e.args
        )
    )
    def evaluate_function_elementwise(self, function_application):
        functor = function_application.functor

        rebv = ReplaceExpressionsByValues(self.symbol_table)
        args = rebv.walk(function_application.args)
        kwargs = {
            k: rebv.walk(v)
            for k, v in function_application.kwargs.items()
        }

        data = np.concatenate(
            np.broadcast_arrays(*(arg.asarray for arg in args)),
            axis=1
        )

        mask = np.apply_along_axis(
            lambda x: functor.value(*x, **kwargs),
            1, data
        )
        result = data[mask, :]
        result = WrappedRelationalAlgebraSet(result)
        result = Constant[
            AbstractSet[Tuple[get_args(functor.type)[:-1]]]
        ](result)
        return result

    def function_gt(
        self,
        x: AbstractSet[Tuple[int]], y: AbstractSet[Tuple[int]]
    ) -> AbstractSet[Tuple[int, int]]:
        data = np.concatenate(
            np.broadcast_arrays(x.asarray, y.asarray),
            axis=1
        )

        data = data[data[:, 0] > data[:, 1], ...]
        return RelationalAlgebraFrozenSet(data)

    def function_gt_elementwise(self, x: int, y: int) -> bool:
        return x > y


def test_function_right_type():
    dl = Datalog()

    gt = dl.symbol_table['gt']
    assert is_leq_informative(
        gt.type,
        Callable[
            [AbstractSet[Tuple], AbstractSet[Tuple]],
            AbstractSet[Tuple]
        ]
    )


def test_vectorised_works_on_RA_sets():
    s1 = RelationalAlgebraFrozenSet([0, 1, 2])
    s2 = RelationalAlgebraFrozenSet([0, -1, 1])
    s3 = RelationalAlgebraFrozenSet([0, 0, 1])
    s4 = RelationalAlgebraFrozenSet([-1])

    res = set(
        s1[i] + s2[i]
        for i in range(len(s1))
        if s1[i][0] > s2[i][0]
    )

    dl = Datalog()
    gtres = dl.function_gt(s1, s2)
    assert res == gtres

    with raises(ValueError):
        dl.function_gt(s1, s3)

    gtres = dl.function_gt(s1, s4)
    res = set(
        s1[i] + s4[0]
        for i in range(len(s1))
        if s1[i][0] > s4[0][0]
    )

    assert res == gtres


def test_vectorised_works_through_walking():
    dl = Datalog()
    s1 = S_('s1')
    s2 = S_('s2')
    dl.add_extensional_predicate_from_tuples(
        s1, ((i,) for i in [0, 1, 2])
    )
    dl.add_extensional_predicate_from_tuples(
        s2, ((i,) for i in [0, -1, 1])
    )

    s1_ = dl.symbol_table[s1].value
    s2_ = dl.symbol_table[s2].value

    res = C_[AbstractSet[Tuple[int, int]]](set(
        s1_[i] + s2_[i]
        for i in range(len(s1_))
        if s1_[i][0] > s2_[i][0]
    ))

    fun_app = S_('gt')(s1, s2)

    dl_res = dl.walk(fun_app)
    assert isinstance(dl_res.value, WrappedRelationalAlgebraSet)
    assert dl_res == res


def test_elementwise_works_on_sets_through_walking():
    dl = Datalog()
    s1 = S_('s1')
    s2 = S_('s2')
    dl.add_extensional_predicate_from_tuples(
        s1, ((i,) for i in [0, 1, 2])
    )
    dl.add_extensional_predicate_from_tuples(
        s2, ((i,) for i in [0, -1, 1])
    )

    s1_ = dl.symbol_table[s1].value
    s2_ = dl.symbol_table[s2].value

    res = C_[AbstractSet[Tuple[int, int]]](set(
        s1_[i] + s2_[i]
        for i in range(len(s1_))
        if s1_[i][0] > s2_[i][0]
    ))

    fun_app = S_('gt_elementwise')(s1, s2)

    dl_res = dl.walk(fun_app)
    assert isinstance(dl_res.value, WrappedRelationalAlgebraSet)
    assert dl_res == res
