from ... import expression_walker, expressions
from .. import Implication, Fact, DatalogProgram
from .. import magic_sets
from ..chase import Chase
from ..expressions import TranslateToLogic


C_ = expressions.Constant
S_ = expressions.Symbol
Imp_ = Implication
F_ = Fact
Eb_ = expressions.ExpressionBlock


class TranslateToDatalog(
    TranslateToLogic,
    expression_walker.IdentityWalker
):
    pass


class Datalog(
    DatalogProgram,
    expression_walker.ExpressionBasicEvaluator
):
    pass


def test_resolution_works():
    x = S_('X')
    y = S_('Y')
    z = S_('Z')
    anc = S_('anc')
    par = S_('par')
    q = S_('q')
    a = C_('a')
    b = C_('b')
    c = C_('c')
    d = C_('d')

    edb = Eb_([
        F_(par(a, b)),
        F_(par(b, c)),
        F_(par(c, d)),
    ])

    code = Eb_([
        Imp_(q(x), anc(a, x)),
        Imp_(anc(x, y), par(x, y)),
        Imp_(anc(x, y), anc(x, z) & par(z, y)),
    ])

    tr = TranslateToDatalog()
    dl = Datalog()
    dl.walk(tr.walk(code))
    dl.walk(tr.walk(edb))
    goal, mr = magic_sets.magic_rewrite(q(x), dl)

    dl = Datalog()
    dl.walk(tr.walk(mr))
    dl.walk(tr.walk(edb))

    solution = Chase(dl).build_chase_solution()
    assert solution[goal].value == {C_((e,)) for e in (b, c, d)}
