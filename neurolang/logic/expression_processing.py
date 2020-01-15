from operator import and_, invert, or_
from typing import Any

from ..exceptions import NeuroLangException
from ..expression_walker import PatternWalker, ReplaceSymbolWalker, add_match
from ..expressions import Constant, FunctionApplication, Symbol
from ..utils import OrderedSet
from . import (FALSE, TRUE, Conjunction, Disjunction, ExistentialPredicate,
               Implication, LogicOperator, Negation, Quantifier, Union,
               UniversalPredicate)


class LogicSolver(PatternWalker):
    @add_match(Conjunction)
    def evaluate_conjunction(self, expression):
        unsolved_formulas = tuple()
        for formula in expression.formulas:
            solved_formula = self.walk(formula)
            if isinstance(solved_formula, Constant):
                if not bool(solved_formula.value):
                    return FALSE
            else:
                unsolved_formulas += (solved_formula,)

        if len(unsolved_formulas) == 0:
            return TRUE
        elif len(unsolved_formulas) == 1:
            return unsolved_formulas[0]
        else:
            return Conjunction(unsolved_formulas)

    @add_match(Disjunction)
    def evaluate_disjunction(self, expression):
        unsolved_formulas = tuple()
        for formula in expression.formulas:
            solved_formula = self.walk(formula)
            if isinstance(solved_formula, Constant):
                if bool(solved_formula.value):
                    return TRUE
            else:
                unsolved_formulas += (solved_formula,)

        if len(unsolved_formulas) == 0:
            return FALSE
        elif len(unsolved_formulas) == 1:
            return unsolved_formulas[0]
        else:
            return Disjunction(unsolved_formulas)

    @add_match(Negation)
    def evaluate_negation(self, expression):
        solved_formula = self.walk(expression.formula)
        if isinstance(solved_formula, Constant):
            return Constant[bool](not solved_formula.value)
        if isinstance(solved_formula, Negation):
            solved_formula = solved_formula.formula
        else:
            solved_formula = Negation(solved_formula)
        return solved_formula

    @add_match(Implication(..., TRUE))
    def evaluate_implication_true_antecedent(self, expression):
        return self.walk(expression.consequent)

    @add_match(Implication(..., FALSE))
    def evaluate_implication_false_antecedent(self, expression):
        return TRUE

    @add_match(Implication(TRUE, ...))
    def evaluate_implication_true_consequent(self, expression):
        return TRUE

    @add_match(Implication(FALSE, ...))
    def evaluate_implication_false_consequent(self, expression):
        return self.walk(Negation(expression.antecedent))

    @add_match(Implication)
    def evaluate_implication(self, expression):
        solved_antecedent = self.walk(expression.antecedent)
        if solved_antecedent is not expression.antecedent:
            return self.walk(
                Implication(expression.consequent, solved_antecedent)
            )

        solved_consequent = self.walk(expression.consequent)
        if (solved_consequent is not expression.consequent):
            return self.walk(Implication(solved_consequent, solved_antecedent))

        return expression


def is_logic_function_application(function_application):
    if not isinstance(function_application.functor, Constant):
        return False
    functor_value = function_application.functor.value
    return functor_value in (and_, or_, invert)


class TranslateToLogic(PatternWalker):
    @add_match(FunctionApplication(Constant[Any](and_), ...))
    def build_conjunction(self, conjunction):
        args = tuple()
        for arg in conjunction.args:
            new_arg = self.walk(arg)
            if isinstance(new_arg, Conjunction):
                args += new_arg.formulas
            else:
                args += (new_arg,)

        return self.walk(Conjunction(args))

    @add_match(FunctionApplication(Constant[Any](or_), ...))
    def build_disjunction(self, disjunction):
        args = tuple()
        for arg in disjunction.args:
            new_arg = self.walk(arg)
            if isinstance(new_arg, Disjunction):
                args += new_arg.formulas
            else:
                args += (new_arg,)

        return self.walk(Disjunction(args))

    @add_match(FunctionApplication(Constant[Any](invert), ...))
    def build_negation(self, inversion):
        arg = self.walk(inversion.args[0])
        return self.walk(Negation(arg))

    @add_match(
        LogicOperator,
        lambda expression: any(
            isinstance(arg, FunctionApplication) and
            is_logic_function_application(arg)
            for arg in expression.unapply()
        )
    )
    def translate_logic_operator(self, expression):
        args = expression.unapply()
        new_args = tuple()
        changed = False
        for arg in args:
            if (
                isinstance(arg, FunctionApplication) and
                is_logic_function_application(arg)
            ):
                new_arg = self.walk(arg)
            else:
                new_arg = arg

            if new_arg is not arg:
                changed = True

            new_args += (new_arg,)

        if changed:
            expression = self.walk(expression.apply(*new_args))
        return expression


class WalkLogicProgramAggregatingSets(PatternWalker):
    @add_match(Conjunction)
    def conjunction(self, expression):
        fvs = OrderedSet()
        for formula in expression.formulas:
            fvs |= self.walk(formula)
        return fvs

    @add_match(Union)
    def union(self, expression):
        return self.conjunction(expression)

    @add_match(Disjunction)
    def disjunction(self, expression):
        return self.conjunction(expression)

    @add_match(Negation)
    def negation(self, expression):
        return self.walk(expression.formula)


class ExtractFreeVariablesWalker(WalkLogicProgramAggregatingSets):
    @add_match(FunctionApplication)
    def extract_variables_fa(self, expression):
        args = expression.args

        variables = OrderedSet()
        for a in args:
            if isinstance(a, Symbol):
                variables.add(a)
            elif isinstance(a, FunctionApplication):
                variables |= self.walk(a)
            elif isinstance(a, Constant):
                pass
            else:
                raise NeuroLangException('Not a Datalog function application')
        return variables

    @add_match(Quantifier)
    def extract_variables_q(self, expression):
        return self.walk(expression.body) - expression.head._symbols

    @add_match(Implication)
    def extract_variables_s(self, expression):
        return (
            self.walk(expression.antecedent) -
            self.walk(expression.consequent)
        )

    @add_match(Symbol)
    def extract_variables_symbol(self, expression):
        return OrderedSet((expression, ))

    @add_match(Constant)
    def _(self, expression):
        return OrderedSet()


def extract_logic_free_variables(expression):
    """Extract variables from expression assuming it's in logic format.

    Parameters
    ----------
    expression : Expression


    Returns
    -------
        OrderedSet
            set of all free variables in the expression.
    """
    efvw = ExtractFreeVariablesWalker()
    return efvw.walk(expression)


class ExtractLogicPredicates(WalkLogicProgramAggregatingSets):
    @add_match(Symbol)
    def symbol(self, expression):
        return OrderedSet()

    @add_match(Constant)
    def constant(self, expression):
        return OrderedSet()

    @add_match(FunctionApplication)
    def extract_predicates_fa(self, expression):
        return OrderedSet([expression])

    @add_match(Negation)
    def negation(self, expression):
        return OrderedSet([expression])


def extract_logic_predicates(expression):
    """Extract predicates from expression
    knowing that it's in logic format

    Parameters
    ----------
    expression : Expression
        expression to extract predicates from


    Returns
    -------
    OrderedSet
        set of all predicates in the expression in lexicographical
        order.

    """
    edp = ExtractLogicPredicates()
    return edp.walk(expression)


class FlattenConjunctionDisjunctionWalker(PatternWalker):
    @add_match(
        Disjunction,
        lambda e: any(isinstance(f, Disjunction) for f in e.formulas)
    )
    def flatten_disjunction(self, expression):
        formulas = tuple()
        for f in expression.formulas:
            f = self.walk(f)
            if isinstance(f, Disjunction):
                formulas += f.formulas
            else:
                formulas += (f,)

        return self.walk(Disjunction(formulas))

    @add_match(
        Conjunction,
        lambda e: any(isinstance(f, Conjunction) for f in e.formulas)
    )
    def flatten_conjunction(self, expression):
        formulas = tuple()
        for f in expression.formulas:
            f = self.walk(f)
            if isinstance(f, Conjunction):
                formulas += f.formulas
            else:
                formulas += (f,)

        return self.walk(Conjunction(formulas))


class FreshQuantifiers(PatternWalker):
    @add_match(Quantifier)
    def quantifier(self, expression):
        fresh = expression.head.fresh()
        rsw = ReplaceSymbolWalker({expression.head: fresh})
        return expression.apply(
            fresh,
            self.walk(rsw(expression.body))
        )


class PushNegationWalker(PatternWalker):
    @add_match(Negation(Negation))
    def double_negation(self, expression):
        return self.walk(expression.formula.formula)

    @add_match(Negation(Conjunction))
    def negation_conjunction(self, expression):
        return self.walk(self.disjunction(
            tuple(Negation(f) for f in expression.formulas)
        ))

    @add_match(Negation(Disjunction))
    def negation_disjunction(self, expression):
        return self.walk(self.conjunction(
            tuple(Negation(f) for f in expression.formulas)
        ))


class PrenexNormalFormWalker(PatternWalker):
    @add_match(Quantifier)
    def quantifier(self, expression):
        body = expression.body
        walked_body = self.walk(body)
        if body is not walked_body:
            return self.walk(expression.apply(expression.head, walked_body))

    @add_match(
        Conjunction,
        lambda e: any(isinstance(f, Quantifier) for f in e.formulas)
    )
    def conjunction(self, expression):
        formulas, universals, existentials = \
            self.push_quantifiers_out(expression.formulas)

        expression = Conjunction(formulas)
        expression = self.assemble_quantifiers(
            universals, expression, existentials
        )

        return self.walk(expression)

    @add_match(
        Disjunction,
        lambda e: any(isinstance(f, Quantifier) for f in e.formulas)
    )
    def disjunction(self, expression):
        formulas, universals, existentials = \
            self.push_quantifiers_out(expression.formulas)

        expression = Disjunction(formulas)
        expression = self.assemble_quantifiers(
            universals, expression, existentials
        )

        return self.walk(expression)

    @add_match(
        Negation,
        lambda e: isinstance(e.formula, Quantifier)
    )
    def negation(self, expression):
        formulas, universals, existentials = \
            self.push_quantifiers_out([expression.formula])

        expression = Negation(formulas[0])
        expression = self.assemble_quantifiers(
            universals, expression, existentials
        )

        return self.walk(expression)

    def assemble_quantifiers(self, universals, expression, existentials):
        for quantified_variable in universals:
            expression = UniversalPredicate(quantified_variable, expression)
        for quantified_variable in existentials:
            expression = ExistentialPredicate(existentials)
        return expression

    def push_quantifiers_out(self, formulas_in):
        existentials = []
        universals = []
        formulas = tuple()
        for current_formula in formulas_in:
            f = current_formula
            while isinstance(f, Quantifier):
                quantified_variable = f.head
                body = f.body
                if not quantified_variable.is_fresh:
                    fresh = quantified_variable.fresh()
                    body = (
                        ReplaceSymbolWalker({quantified_variable: fresh})
                        .walk(body)
                    )
                    quantified_variable = fresh
                if isinstance(f, ExistentialPredicate):
                    existentials.append(quantified_variable)
                elif isinstance(f, UniversalPredicate):
                    universals.append(quantified_variable)
                f = body
            formulas += (f,)
        return formulas, universals, existentials
