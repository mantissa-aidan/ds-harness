"""Compile a bounded relational plan into SELECT SQL. Raw SQL is never accepted."""

from __future__ import annotations

import re

from ds_harness.config import SelectQuery
from ds_harness.errors import PolicyError

IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_ ]{0,127}$")


def quote(identifier: str) -> str:
    if not IDENTIFIER.fullmatch(identifier):
        raise PolicyError("Unsupported SQL identifier; use a view with simple column names")
    return f"[{identifier}]"


def relation(name: str) -> str:
    parts = name.split(".")
    if not 1 <= len(parts) <= 2:
        raise PolicyError("Only table or schema.table names are allowed")
    return ".".join(quote(part) for part in parts)


def compile_select(query: SelectQuery, allowed_tables: list[str], limit: int,
                   dialect: str = "tsql") -> tuple[str, list]:
    if dialect not in {"tsql", "sqlite"} or limit < 1:
        raise PolicyError("Invalid SQL dialect or row limit")
    allowed = {name.casefold() for name in allowed_tables}

    def table(name):
        if name.casefold() not in allowed:
            raise PolicyError("Query references a table outside allowed_tables")
        return relation(name)

    aliases = {"t"}

    def column(name):
        parts = name.split(".")
        if len(parts) == 1:
            return f"[t].{quote(parts[0])}"
        if len(parts) == 2 and parts[0] in aliases:
            return f"{quote(parts[0])}.{quote(parts[1])}"
        raise PolicyError("Column must be unqualified or use a declared table alias")

    from_sql = f"{table(query.table)} AS [t]"
    for join in query.joins:
        if join.alias in aliases:
            raise PolicyError("Join aliases must be unique")
        left = column(join.left_column)
        aliases.add(join.alias)
        right = column(f"{join.alias}.{join.right_column}")
        from_sql += f" {join.kind.upper()} JOIN {table(join.table)} AS {quote(join.alias)}"
        from_sql += f" ON {left} = {right}"
    if query.measures or query.group_by:
        if query.columns and query.columns != query.group_by:
            raise PolicyError("Aggregate selection columns must equal group_by")
        selected = [column(c) for c in query.group_by]
    else:
        if query.joins and not query.columns:
            raise PolicyError("Joined queries require explicit columns with unique output names")
        selected = [column(c) for c in query.columns] if query.columns else ["[t].*"]
    for measure in query.measures:
        if measure.column == "*" and measure.aggregate != "count":
            raise PolicyError("Only COUNT can aggregate *")
        expr = "*" if measure.column == "*" else column(measure.column)
        selected.append(f"{measure.aggregate.upper()}({expr}) AS {quote(measure.alias)}")
    if not selected:
        raise PolicyError("Query has no selected columns")
    prefix = f"SELECT TOP ({int(limit)})" if dialect == "tsql" else "SELECT"
    sql = f"{prefix} {', '.join(selected)} FROM {from_sql}"
    predicates, parameters = [], []
    operators = {"eq": "=", "ne": "<>", "lt": "<", "lte": "<=", "gt": ">", "gte": ">="}
    for item in query.filters:
        col = column(item.column)
        if item.op in {"is_null", "not_null"}:
            predicates.append(f"{col} IS {'NOT ' if item.op == 'not_null' else ''}NULL")
        elif item.op == "in":
            if not isinstance(item.value, list) or not 1 <= len(item.value) <= 100:
                raise PolicyError("IN requires between 1 and 100 scalar values")
            predicates.append(f"{col} IN ({', '.join('?' for _ in item.value)})")
            parameters.extend(item.value)
        else:
            if item.value is None or isinstance(item.value, list):
                raise PolicyError("Comparison filters require a non-null scalar value")
            predicates.append(f"{col} {operators[item.op]} ?")
            parameters.append(item.value)
    if len(parameters) > 1000:
        raise PolicyError("Query exceeds parameter budget")
    if predicates:
        sql += " WHERE " + " AND ".join(predicates)
    if query.group_by:
        sql += " GROUP BY " + ", ".join(column(c) for c in query.group_by)
    if query.order_by:
        measure_names = {m.alias for m in query.measures}
        sql += " ORDER BY " + ", ".join(
            quote(c) if c in measure_names else column(c) for c in query.order_by
        )
    if dialect == "sqlite":
        sql += f" LIMIT {int(limit)}"
    return sql, parameters
