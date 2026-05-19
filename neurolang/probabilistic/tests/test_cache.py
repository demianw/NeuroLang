import unittest
from typing import AbstractSet
from unittest.mock import patch

from ...expressions import Constant, Symbol
from ...logic import Conjunction
from ...utils.relational_algebra_set import NamedRelationalAlgebraFrozenSet
from .. import containment, dalvi_suciu_lift
from ..probabilistic_ra_utils import ProbabilisticFactSet


class TestProbabilisticCache(unittest.TestCase):
    """Test suite for probabilistic caching functionality."""

    def setUp(self):
        """Clear caches before each test."""
        containment.clear_cache()
        dalvi_suciu_lift.clear_cache()

    def tearDown(self):
        """Clear caches after each test."""
        containment.clear_cache()
        dalvi_suciu_lift.clear_cache()

    def test_containment_cache_is_populated(self):
        """Repeated identical containment checks should be cached."""
        R = Symbol("R")
        S = Symbol("S")
        x = Symbol("x")
        y = Symbol("y")
    
        q1 = Conjunction((R(x, y), S(y)))
        q2 = Conjunction((R(x, y), S(y)))
    
        # Verify cache is clear
        self.assertEqual(containment.is_contained.cache_info().hits, 0)
        self.assertEqual(containment.is_contained.cache_info().misses, 0)
    
        # first call populates cache
        containment.is_contained(q1, q2)
        self.assertEqual(containment.is_contained.cache_info().misses, 1)
        self.assertEqual(containment.is_contained.cache_info().hits, 0)
    
        # second identical call should hit
        containment.is_contained(q1, q2)
        self.assertEqual(containment.is_contained.cache_info().hits, 1)
    
    
    def test_dalvi_suciu_lift_cache_is_populated(self):
        """Repeated identical lifted-plan calls should be cached."""
        R = Symbol("R")
        x = Symbol("x")
        y = Symbol("y")
        query = Conjunction((R(x, y),))
    
        ra_set = Constant[AbstractSet](
            NamedRelationalAlgebraFrozenSet(
                iterable=[(1, 2), (3, 4)],
                columns=("a", "b"),
            )
        )
        fresh_R = Symbol("R_data")
        symbol_table = {
            R: ProbabilisticFactSet(fresh_R, Constant(0)),
            fresh_R: ra_set,
        }
    
        # Verify cache is clear
        self.assertEqual(len(dalvi_suciu_lift._dalvi_suciu_lift_cache), 0)
    
        plan1 = dalvi_suciu_lift.dalvi_suciu_lift(query, symbol_table)
        self.assertEqual(len(dalvi_suciu_lift._dalvi_suciu_lift_cache), 1)
    
        plan2 = dalvi_suciu_lift.dalvi_suciu_lift(query, symbol_table)
        self.assertEqual(len(dalvi_suciu_lift._dalvi_suciu_lift_cache), 1)
        # cached result should be the same object
        self.assertTrue(plan1 is plan2)
    
    
    def test_containment_cache_clear(self):
        """Cache clearing should reset the internal state."""
        R = Symbol("R")
        x = Symbol("x")
        q = Conjunction((R(x),))
    
        containment.is_contained(q, q)
        self.assertTrue(containment.is_contained.cache_info().misses >= 1)
    
        containment.clear_cache()
        self.assertEqual(containment.is_contained.cache_info().misses, 0)
        self.assertEqual(containment.is_contained.cache_info().hits, 0)