from collections import namedtuple
from collections.abc import Iterable

from neurolang.type_system import Unknown
from typing import Tuple

import logging
import numpy as np
import types
from neurolang.utils.relational_algebra_set.sql_helpers import (
    CreateTableAs,
    CreateView,
    map_dtype_to_sqla,
    SQLAEngineFactory,
)
import uuid

from . import abstract as abc
from ..various import log_performance
from .dask_helpers import DaskContextFactory
import pandas as pd
import dask.dataframe as dd

from sqlalchemy import (
    Table,
    column,
    table,
    MetaData,
    Index,
    func,
    and_,
    select,
    text,
    tuple_,
    literal_column,
    literal,
)
from sqlalchemy.sql import table, intersect, union, except_, exists
from sqlalchemy.dialects import postgresql

LOG = logging.getLogger(__name__)


class RelationalAlgebraFrozenSet(abc.RelationalAlgebraFrozenSet):

    _count = None
    _table_name = None
    _table = None

    def __init__(self, iterable=None):
        if isinstance(iterable, RelationalAlgebraFrozenSet):
            self._init_from(iterable)
        else:
            self._create_insert_table(iterable)

    @staticmethod
    def _new_name(prefix="table_"):
        return prefix + str(uuid.uuid4()).replace("-", "_")

    def _init_from(self, other):
        self._table_name = other._table_name
        self._table = other._table
        self._count = other._count

    def _create_insert_table(self, data):
        if data is not None:
            data = pd.DataFrame(data)
            data.columns = data.columns.astype(str)
            if len(data.columns) > 0:
                self._table_name = self._new_name()
                ddf = dd.from_pandas(data, npartitions=3)
                DaskContextFactory.get_context().create_table(
                    self._table_name, ddf
                )
                self._table = table(
                    self._table_name, *[column(c) for c in ddf.columns]
                )
            self._count = len(data)

    @classmethod
    def dee(cls):
        output = cls()
        output._count = 1
        return output

    @classmethod
    def dum(cls):
        return cls()

    @classmethod
    def create_view_from(cls, other):
        if not isinstance(other, cls):
            raise ValueError(
                "View can only be created from an object of the same class"
            )
        output = cls()
        output._init_from(other)
        return output

    def copy(self):
        if self.is_dee():
            return self.dee()
        elif self.is_dum():
            return self.dum()
        return type(self).create_view_from(self)

    def is_empty(self):
        if self._count is not None:
            return self._count == 0
        else:
            return self.fetch_one() is None

    def is_dum(self):
        return self.arity == 0 and self.is_empty()

    def is_dee(self):
        return self.arity == 0 and not self.is_empty()

    @property
    def arity(self):
        return len(self.columns)

    @property
    def columns(self):
        if self._table is None:
            return []
        return self._table.c.keys()

    @property
    def sql_columns(self):
        if self._table is None:
            return {}
        return self._table.c

    def __len__(self):
        if self._count is None:
            if self._table is not None:
                q = select([func.count()]).select_from(
                    select(self._table).distinct()
                )
                res = DaskContextFactory.sql(q).compute()
                self._count = 0 if len(res) == 0 else res.squeeze()
            else:
                self._count = 0
        return self._count

    def __iter__(self):
        values = self._fetchall(True).itertuples(name=None, index=False)
        for v in values:
            yield (tuple(v))

    def __contains__(self, element):
        if self.arity == 0:
            return False
        element = self._normalise_element(element)
        query = select(self._table)
        for c, v in element.items():
            query = query.where(self.sql_columns.get(c) == v)
        res = DaskContextFactory.sql(query).head(1)
        return len(res) > 0

    def _normalise_element(self, element):
        """
        Returns a dict representation of the element as col -> value.

        Parameters
        ----------
        element : Iterable[Any]
            the element to normalize

        Returns
        -------
        Dict[str, Any]
            the dict reprensentation of the element
        """
        if isinstance(element, dict):
            pass
        elif hasattr(element, "__iter__") and not isinstance(element, str):
            element = dict(zip(self.columns, element))
        else:
            element = dict(zip(self.columns, (element,)))
        return element

    def _create_view_from_query(self, query):
        output = type(self)()
        output._table = query.subquery()
        return output

    def selection(self, select_criteria):
        if self._table is None:
            return self.copy()

        query = select(self._table)
        if callable(select_criteria):
            lambda_name = self._new_name("lambda")
            SQLAEngineFactory.register_function(
                lambda_name,
                len(self.sql_columns),
                select_criteria,
                params=self.sql_columns.values(),
            )
            f_ = getattr(func, lambda_name)
            query = query.where(f_(*self.sql_columns))
        elif isinstance(
            select_criteria, abc.RelationalAlgebraStringExpression
        ):
            query = query.where(text(select_criteria))
        else:
            for k, v in select_criteria.items():
                if callable(v):
                    lambda_name = self._new_name("lambda")
                    c_ = self.sql_columns.get(str(k))
                    SQLAEngineFactory.register_function(
                        lambda_name, 1, v, [c_]
                    )
                    f_ = getattr(func, lambda_name)
                    query = query.where(f_(c_))
                elif isinstance(
                    select_criteria, abc.RelationalAlgebraStringExpression
                ):
                    query = query.where(text(v))
                else:
                    query = query.where(self.sql_columns.get(str(k)) == v)
        return self._create_view_from_query(query)

    def selection_columns(self, select_criteria):
        if self._table is None:
            return self.copy()
        query = select(*self.sql_columns).select_from(self._table)
        for k, v in select_criteria.items():
            query = query.where(
                self.sql_columns.get(str(k)) == self.sql_columns.get(str(v))
            )
        return self._create_view_from_query(query)

    def itervalues(self):
        raise NotImplementedError()

    def as_numpy_array(self):
        return self._fetchall(True).to_numpy()

    def as_pandas_dataframe(self):
        return self._fetchall(True)

    def _fetchall(self, drop_duplicates=False):
        if self.arity > 0 and self._table is not None:
            df = DaskContextFactory.sql(select(self._table)).compute()
            if drop_duplicates:
                df = df.drop_duplicates()
            return df
        elif self._count == 1:
            return pd.DataFrame([()])
        return pd.DataFrame([])

    def equijoin(self, other, join_indices=None):
        res = self._dee_dum_product(other)
        if res is not None:
            return res

        # Create an alias on the other table's name if we're joining on
        # the same table.
        ot = other._table
        if other._table_name == self._table_name:
            ot = ot.alias()

        join_cols = list(self.sql_columns) + [
            ot.c.get(str(i)).label(str(i + self.arity))
            for i in range(other.arity)
        ]
        query = select(*join_cols)

        if join_indices is not None and len(join_indices) > 0:
            on_clause = and_(
                *[
                    self.sql_columns.get(str(i)) == ot.c.get(str(j))
                    for i, j in join_indices
                ]
            )
            query = query.select_from(self._table.join(ot, on_clause))
        return self._create_view_from_query(query)

    def cross_product(self, other):
        return self.equijoin(other)

    def fetch_one(self):
        if self.arity > 0 and self._table is not None:
            # See https://dask-sql.readthedocs.io/en/latest/pages/sql.html?highlight=head#limitatons
            # for difference between limit in SQL and head in dask
            try:
                return next(
                    DaskContextFactory.sql(select(self._table))
                    .head(1)
                    .itertuples(name=None, index=False)
                )
            except StopIteration:
                return None
        elif self._count == 1:
            return tuple()
        return None

    def groupby(self, columns):
        raise NotImplementedError()

    def projection(self, *columns, reindex=True):
        if self.is_dum():
            return self.dum()
        elif self.is_dee() or len(columns) == 0:
            return self.dee()

        if reindex:
            proj_columns = [
                self.sql_columns.get(str(c)).label(str(i))
                for i, c in enumerate(columns)
            ]
        else:
            proj_columns = [self.sql_columns.get(str(c)) for c in columns]
        query = select(proj_columns).select_from(self._table)
        return self._create_view_from_query(query)

    def __repr__(self):
        t = self._table
        return "{}({})".format(type(self), t)

    def __eq__(self, other):
        if isinstance(other, RelationalAlgebraFrozenSet):
            if self.is_dee() or other.is_dee():
                res = self.is_dee() and other.is_dee()
            elif self.is_dum() or other.is_dum():
                res = self.is_dum() and other.is_dum()
            elif (
                self._table_name is not None
                and self._table_name == other._table_name
            ):
                res = True
            elif not self._equal_sets_structure(other):
                res = False
            else:
                select_left = select(self._table)
                select_right = select(
                    *[other.sql_columns.get(c) for c in self.columns]
                ).select_from(other._table)
                diff_left = select_left.except_(select_right)
                diff_right = select_right.except_(select_left)
                if len(DaskContextFactory.sql(diff_left).head(1)) > 0:
                    res = False
                elif len(DaskContextFactory.sql(diff_right).head(1)) > 0:
                    res = False
                else:
                    res = True
            return res
        else:
            return super().__eq__(other)

    def _equal_sets_structure(self, other):
        return set(self.columns) == set(other.columns)

    def _do_set_operation(self, other, sql_operator):
        if not self._equal_sets_structure(other):
            raise ValueError(
                "Relational algebra set operators can only be used on sets"
                " with same columns."
            )
        ot = other._table.alias()
        query = sql_operator(
            select(self._table),
            select([ot.c.get(c) for c in self.columns]).select_from(ot),
        )
        return self._create_view_from_query(query)

    def __and__(self, other):
        if not isinstance(other, RelationalAlgebraFrozenSet):
            return super().__and__(other)
        res = self._dee_dum_product(other)
        if res is not None:
            return res
        return self._do_set_operation(other, intersect)

    def __or__(self, other):
        if not isinstance(other, RelationalAlgebraFrozenSet):
            return super().__or__(other)
        res = self._dee_dum_sum(other)
        if res is not None:
            return res
        return self._do_set_operation(other, union)

    def __sub__(self, other):
        if not isinstance(other, RelationalAlgebraFrozenSet):
            return super().__sub__(other)
        if self.is_dee():
            if other.is_dee():
                return self.dum()
            return self.dee()
        if self._table is None or other._table is None:
            return self.copy()
        return self._do_set_operation(other, except_)

    def __hash__(self):
        if self._table is None:
            return hash((tuple(), None))
        return hash(tuple(self.columns, self.as_numpy_array().tobytes()))


class NamedRelationalAlgebraFrozenSet(
    RelationalAlgebraFrozenSet, abc.NamedRelationalAlgebraFrozenSet
):
    def __init__(self, columns=None, iterable=None):
        if isinstance(columns, RelationalAlgebraFrozenSet):
            iterable = columns
            columns = columns.columns
        self._count = None
        self._table = None
        self._init_columns = columns
        self._check_for_duplicated_columns(columns)
        if isinstance(iterable, RelationalAlgebraFrozenSet):
            if columns is None or columns == iterable.columns:
                self._init_from(iterable)
            else:
                self._init_from_and_rename(iterable, columns)
        elif columns is not None and len(columns) > 0:
            self._create_insert_table(iterable, columns)

    def _create_insert_table(self, data, columns=None):
        if data is None:
            data = []
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data, columns=columns)
        else:
            data.columns = list(columns)
        data.columns = data.columns.astype(str)
        data = data.drop_duplicates()
        self._table_name = self._new_name()
        ddf = dd.from_pandas(data, npartitions=3)
        DaskContextFactory.get_context().create_table(self._table_name, ddf)
        self._table = table(
            self._table_name, *[column(c) for c in ddf.columns]
        )
        self._count = len(data)

    @staticmethod
    def _check_for_duplicated_columns(columns):
        if columns is not None and len(set(columns)) != len(columns):
            columns = list(columns)
            dup_cols = set(c for c in columns if columns.count(c) > 1)
            raise ValueError(
                "Duplicated column names are not allowed. "
                f"Found the following duplicated columns: {dup_cols}"
            )

    def _init_from_and_rename(self, other, columns):
        if other._table is not None:
            query = select(
                [c.label(str(nc)) for c, nc in zip(other.sql_columns, columns)]
            ).select_from(other._table)
            self._table = query.subquery()
        self._count = other._count

    @classmethod
    def dee(cls):
        output = cls()
        output._count = 1
        return output

    @classmethod
    def dum(cls):
        return cls()

    @property
    def dummy_row_type(self):
        """
        Return dummy row_type of Tuple[Unknown * arity] to avoid calls
        to the DB.
        TODO: implement this for real.
        """
        if self.arity > 0:
            return Tuple[tuple(Unknown for _ in range(self.arity))]
        return Tuple

    @property
    def arity(self):
        return len(self.columns)

    @property
    def columns(self):
        if self._table is None:
            return (
                tuple()
                if self._init_columns is None
                else tuple(self._init_columns)
            )
        return tuple(self._table.c.keys())

    @property
    def sql_columns(self):
        if self._table is None:
            return {}
        return self._table.c

    def projection(self, *columns):
        return super().projection(*columns, reindex=False)

    def cross_product(self, other):
        res = self._dee_dum_product(other)
        if res is not None:
            return res
        if len(set(self.columns).intersection(set(other.columns))) > 0:
            raise ValueError(
                "Cross product with common columns " "is not valid"
            )

        query = select(self._table, other._table)
        return self._create_view_from_query(query)

    def naturaljoin(self, other):
        res = self._dee_dum_product(other)
        if res is not None:
            return res

        on = [c for c in self.columns if c in other.columns]
        if len(on) == 0:
            return self.cross_product(other)
        return self._do_join(other, on, isouter=False)

    def left_naturaljoin(self, other):
        """
        Same as naturaljoin with outher=True
        """
        on = [c for c in self.columns if c in other.columns]
        if len(on) == 0:
            return self
        return self._do_join(other, on, isouter=True)

    def _do_join(self, other, on, isouter=False):
        """
        Performs the join on the two sets.

        Parameters
        ----------
        other : NamedRelationalAlgebraFrozenSet
            The other set
        on : Iterable[sqlalchemy.Columns]
            The columns to join on
        isouter : bool, optional
            If True, performs a left outer join, by default False

        Returns
        -------
        NamedRelationalAlgebraFrozenSet
            The joined set
        """
        on_clause = and_(
            *[self._table.c.get(col) == other._table.c.get(col) for col in on]
        )
        select_cols = [self._table] + [
            other._table.c.get(col)
            for col in set(other.columns) - set(self.columns)
        ]
        # Create an alias on the other table's name if we're joining on the
        # same table.
        other_join_table = other._table
        if other._table_name == self._table_name:
            other_join_table = other_join_table.alias()
        query = select(*select_cols).select_from(
            self._table.join(other_join_table, on_clause, isouter=isouter)
        )
        return self._create_view_from_query(query)

    def __iter__(self):
        try:
            named_tuple_type = namedtuple("tuple", self.columns)
        except ValueError:
            # Invalid column names, just return a tuple
            return super().__iter__(self)
        values = self._fetchall(True).itertuples(name=None, index=False)
        for v in values:
            yield (named_tuple_type(**v))

    def equijoin(self, other, join_indices, return_mappings=False):
        raise NotImplementedError()

    def rename_column(self, src, dst):
        if (dst) in self.columns:
            raise ValueError(
                "Duplicated column names are not allowed. "
                f"{dst} is already a column name."
            )
        query = select(
            *[
                c.label(str(dst)) if c.name == src else c
                for c in self.sql_columns
            ]
        ).select_from(self._table)
        return self._create_view_from_query(query)

    def rename_columns(self, renames):
        # prevent duplicated destination columns
        self._check_for_duplicated_columns(renames.values())
        if not set(renames).issubset(self.columns):
            # get the missing source columns
            # for a more convenient error message
            not_found_cols = set(c for c in renames if c not in self.columns)
            raise ValueError(
                f"Cannot rename non-existing columns: {not_found_cols}"
            )
        query = select(
            *[
                c.label(str(renames.get(c.name))) if c.name in renames else c
                for c in self.sql_columns
            ]
        ).select_from(self._table)
        self._create_view_from_query(query)

    def aggregate(self, group_columns, aggregate_function):
        """
        Group by set values on group_columns, while applying aggregate
        functions.

        Parameters
        ----------
        group_columns : List[str, int]
            List of columns to group on
        aggregate_function : Union[Dict[str, Union[callable, str]],
                    List[tuple(str, str, Union[callable, str])]]
            dict of destination column name -> aggregate function

        Returns
        -------
        NamedRelationalAlgebraFrozenSet
            New set with aggregated values as columns

        Raises
        ------
        ValueError
            Raised on unsupported aggregate function
        """
        if isinstance(group_columns, str) or not isinstance(
            group_columns, Iterable
        ):
            group_columns = (group_columns,)
        if len(set(group_columns)) < len(group_columns):
            raise ValueError("Cannot group on repeated columns")

        distinct_sub_query = (
            select(self.sql_columns)
            .select_from(self._table)
            .distinct()
            .alias()
        )
        agg_cols = self._build_aggregate_functions(
            group_columns, aggregate_function, distinct_sub_query
        )
        groupby = [distinct_sub_query.c.get(str(c)) for c in group_columns]

        query = select(groupby + agg_cols).group_by(*groupby)
        return type(self).create_view_from_query(query, self._parent_tables)

    def _build_aggregate_functions(
        self, group_columns, aggregate_function, distinct_view
    ):
        """
        Create the list of aggregated destination columns.
        """
        if isinstance(aggregate_function, dict):
            agg_iter = ((k, k, v) for k, v in aggregate_function.items())
        elif isinstance(aggregate_function, (tuple, list)):
            agg_iter = aggregate_function
        else:
            raise ValueError(
                "Unsupported aggregate_function: {} of type {}".format(
                    aggregate_function, type(aggregate_function)
                )
            )
        un_grouped_cols = [
            c_ for c_ in distinct_view.c if c_.name not in group_columns
        ]
        agg_cols = []
        for dst, src, f in agg_iter:
            if src in distinct_view.c.keys():
                # call the aggregate function on only one column
                c_ = [distinct_view.c.get(src)]
            else:
                # call the aggregate function on all the non-grouped columns
                c_ = un_grouped_cols
            if isinstance(f, types.BuiltinFunctionType):
                f = f.__name__
            if callable(f):
                lambda_name = self._new_name("lambda")
                SQLAEngineFactory.register_aggregate(
                    lambda_name, len(c_), f, params=c_,
                )
                f_ = getattr(func, lambda_name)
            elif isinstance(f, str):
                f_ = getattr(func, f)
            else:
                raise ValueError(
                    f"Aggregate function for {src} needs "
                    "to be callable or a string"
                )
            agg_cols.append(f_(*c_).label(str(dst)))
        return agg_cols

    def extended_projection(self, eval_expressions):
        if self.is_dee():
            return self._extended_projection_on_dee(eval_expressions)
        elif self._table is None:
            return type(self)(
                columns=list(eval_expressions.keys()), iterable=[]
            )

        proj_columns = []
        for dst_column, operation in eval_expressions.items():
            if callable(operation):
                lambda_name = self._new_name("lambda")
                SQLAEngineFactory.register_function(
                    lambda_name,
                    len(self.sql_columns),
                    operation,
                    params=self.sql_columns.values(),
                )
                f_ = getattr(func, lambda_name)
                proj_columns.append(
                    f_(*self.sql_columns).label(str(dst_column))
                )
            elif isinstance(operation, abc.RelationalAlgebraStringExpression):
                if str(operation) != str(dst_column):
                    proj_columns.append(
                        literal_column(operation).label(str(dst_column))
                    )
                else:
                    proj_columns.append(self.sql_columns.get(str(operation)))
            elif isinstance(operation, abc.RelationalAlgebraColumn):
                proj_columns.append(
                    self.sql_columns.get(str(operation)).label(str(dst_column))
                )
            else:
                proj_columns.append(literal(operation).label(str(dst_column)))

        query = select(proj_columns).select_from(self._table)
        return type(self).create_view_from_query(query, self._parent_tables)

    def _extended_projection_on_dee(self, eval_expressions):
        """
        Extended projection when called on Dee to create set with
        constant values.
        """
        return type(self)(
            columns=eval_expressions.keys(),
            iterable=[eval_expressions.values()],
        )

    def fetch_one(self):
        if self.arity > 0 and self._table is not None:
            # See https://dask-sql.readthedocs.io/en/latest/pages/sql.html?highlight=head#limitatons
            # for difference between limit in SQL and head in dask
            try:
                return next(
                    DaskContextFactory.sql(select(self._table))
                    .head(1)
                    .itertuples(name="tuple", index=False)
                )
            except StopIteration:
                return None
        elif self._count == 1:
            return tuple()
        return None

    def to_unnamed(self):
        if self._table is not None:
            query = select(
                [c.label(str(i)) for i, c in enumerate(self.sql_columns)]
            ).select_from(self._table)
            return RelationalAlgebraFrozenSet.create_view_from_query(
                query, self._parent_tables, self._count
            )
        return RelationalAlgebraFrozenSet()

    def projection_to_unnamed(self, *columns):
        unnamed_self = self.to_unnamed()
        named_columns = self.columns
        columns = tuple(named_columns.index(c) for c in columns)
        return unnamed_self.projection(*columns)


class RelationalAlgebraSet(
    RelationalAlgebraFrozenSet, abc.RelationalAlgebraSet
):
    def add(self, value):
        raise NotImplementedError()

    def discard(self, value):
        raise NotImplementedError()

    def __ior__(self, other):
        raise NotImplementedError()

    def __isub__(self, other):
        raise NotImplementedError()