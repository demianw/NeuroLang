from operator import and_, invert
from typing import AbstractSet, Callable, Tuple

from .datalog.expressions import Disjunction
from .expression_walker import add_match, expression_iterator
from .expressions import (Constant, FunctionApplication, NeuroLangException,
                          NonConstant, Symbol, is_leq_informative)
from .solver_datalog_naive import (DatalogBasic, Implication,
                                   extract_datalog_free_variables)
from .type_system import Unknown


class NegativeFact(Implication):
    '''This class defines negative facts. They are composed of an inverted
    antecedent and False in the consequent. It is not necessary that the
    initialization parameter is inverted.'''

    def __init__(self, antecedent):
        super().__init__(Constant(False), invert(antecedent))

    @property
    def fact(self):
        return self.antecedent

    def __repr__(self):
        return 'NegativeFact{{{} \u2190 {}}}'.format(
            repr(self.antecedent), True
        )


class DatalogBasicNegation(DatalogBasic):
    '''Datalog solver that implements negation. Adds the possibility of
    inverted terms when checking that expressions are in conjunctive
    normal form.'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.negated_symbols = {}

    @add_match(
        Implication(FunctionApplication[bool](Symbol, ...), NonConstant)
    )
    def statement_intensional(self, expression):
        consequent = expression.consequent
        antecedent = expression.antecedent

        self._check_implication(consequent, antecedent)

        consequent_symbols = consequent._symbols - consequent.functor._symbols

        if not consequent_symbols.issubset(antecedent._symbols):
            raise NeuroLangException(
                "All variables on the consequent need to be on the antecedent"
            )

        if consequent.functor.name in self.symbol_table:
            value = self.symbol_table[consequent.functor.name]
            self._is_previously_defined(value)
            disj = self.symbol_table[consequent.functor.name].literals
            self._is_in_idb(expression, disj)

        else:
            disj = tuple()

        if expression not in disj:
            disj += (expression, )

        self.symbol_table[consequent.functor.name] = Disjunction(disj)

        return expression

    def _check_implication(self, consequent, antecedent):
        if consequent.functor.name in self.protected_keywords:
            raise NeuroLangException(
                f'symbol {self.constant_set_name} is protected'
            )

        if not is_conjunctive_negation(antecedent):
            raise NeuroLangException(
                f'Expression {antecedent} is not conjunctive'
            )

    def _is_previously_defined(self, value):
        if (
            isinstance(value, Constant) and
            is_leq_informative(value.type, AbstractSet)
        ):
            raise NeuroLangException(
                'f{consequent.functor.name} has been previously '
                'defined as Fact or extensional database.'
            )

    def _is_in_idb(self, expression, eb):
        if (
            not isinstance(eb[0].consequent, FunctionApplication) or
            len(extract_datalog_free_variables(eb[0].consequent.args)
                ) != len(expression.consequent.args)
        ):
            raise NeuroLangException(
                f"{eb[0].consequent} is already in the IDB "
                f"with different signature."
            )

    @add_match(NegativeFact)
    def negative_fact(self, expression):
        fact = expression.fact.args[0]
        if fact.functor.name in self.protected_keywords:
            raise NeuroLangException(
                f'symbol {self.constant_set_name} is protected'
            )

        if any(not isinstance(a, Constant) for a in fact.args):
            raise NeuroLangException(
                'Facts can only have constants as arguments'
            )

        self._not_in_negated_symbol(fact)

        fact_set = self.negated_symbols[fact.functor.name]

        if isinstance(fact_set, Disjunction):
            raise NeuroLangException(
                f'{fact.functor.name} has been previously '
                'define as intensional predicate.'
            )

        fact_set.value.add(Constant(fact.args))

        return expression

    def _not_in_negated_symbol(self, fact):
        if fact.functor.name not in self.negated_symbols:
            if fact.functor.type is Unknown:
                c = Constant(fact.args)
                set_type = c.type
            elif isinstance(fact.functor.type, Callable):
                set_type = Tuple[fact.functor.type.__args__[:-1]]
            else:
                raise NeuroLangException('Fact functor type incorrect')

            self.negated_symbols[fact.functor.name] = \
                Constant[AbstractSet[set_type]](set())


def is_conjunctive_negation(expression):
    return all(
        not isinstance(exp, FunctionApplication) or (
            isinstance(exp, FunctionApplication) and
            ((
                isinstance(exp.functor, Constant) and
                (exp.functor.value is and_ or exp.functor.value is invert)
            ) or
             all(not isinstance(arg, FunctionApplication) for arg in exp.args))
        ) for _, exp in expression_iterator(expression)
    )
