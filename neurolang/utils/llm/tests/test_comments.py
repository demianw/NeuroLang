"""The assistant writes Datalog annotated with comments."""

import pytest

from ..agent import extract_query
from ..prompts import system_prompt

PROGRAM = """\
% the studies that mention pain
pain(s) :- mentions(s, 'pain')
% the answer
ans(s) :- pain(s)"""


@pytest.mark.parametrize("strategy", ["icl", "react"])
def test_the_prompt_asks_for_comments_in_datalog(strategy):
    prompt = system_prompt("datalog", "study(study_id)", [], strategy)
    assert "% comment" in prompt
    assert "before each rule" in prompt


def test_the_prompt_does_not_ask_for_comments_in_squall():
    prompt = system_prompt("squall", "study(study_id)", [], "icl")
    assert "% comment" not in prompt


def test_a_commented_answer_keeps_its_comments():
    reply = f"```datalog\n{PROGRAM}\n```\nThe studies on pain."
    assert extract_query(reply, "datalog") == PROGRAM


def test_the_engine_runs_a_commented_program():
    from neurolang.frontend import NeurolangPDL

    from ..validation import check_query, run_preview

    nl = NeurolangPDL()
    nl.add_tuple_set([(1, "pain"), (2, "fear")], name="mentions")
    assert check_query(nl, PROGRAM, "datalog").ok
    assert run_preview(nl, PROGRAM, "datalog", 5).rows == 1
