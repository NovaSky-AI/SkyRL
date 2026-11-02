"""Row-cap tests for the arctic_rl BIRD reward's ``_add_limit_to_query``.

The module under test is loaded by file path: it imports only the standard
library, while the ``integrations.arctic_rl`` package ``__init__`` pulls in
the trainer stack. This is the same loading the issue's repro uses.
"""

import importlib.util
import sqlite3
from pathlib import Path

import pytest

pytestmark = pytest.mark.integrations

_MODULE_PATH = Path(__file__).resolve().parents[3] / "integrations" / "arctic_rl" / "envs" / "bird_reward.py"
_spec = importlib.util.spec_from_file_location("bird_reward", _MODULE_PATH)
bird_reward = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bird_reward)

CAP = bird_reward.DEFAULT_LIMIT_NUMBER


def capped(query: str) -> str:
    return f"{query} LIMIT {CAP};"


def test_no_limit_gets_capped():
    assert bird_reward._add_limit_to_query("select a from t") == capped("select a from t")


def test_top_level_limit_is_unchanged():
    query = "select a from t limit 10"
    assert bird_reward._add_limit_to_query(query) == query


def test_top_level_limit_uppercase_and_offset_are_unchanged():
    for query in (
        "SELECT a FROM t LIMIT 10",
        "select a from t LIMIT 5, 10",
        "select a from t limit 10 offset 5",
    ):
        assert bird_reward._add_limit_to_query(query) == query


def test_limit_in_subquery_gets_capped():
    query = "select a from t where a = (select a from t order by a desc limit 1) or 1"
    assert bird_reward._add_limit_to_query(query) == capped(query)


def test_limit_in_line_comment_gets_capped():
    # The cap is inserted before the trailing comment so it is not itself
    # commented out.
    assert bird_reward._add_limit_to_query("select a from t -- LIMIT 1") == "select a from t LIMIT 5000; -- LIMIT 1"


def test_limit_in_block_comment_gets_capped():
    query = "select a from t /* LIMIT 1 */"
    assert bird_reward._add_limit_to_query(query) == capped(query)


def test_limit_in_string_literal_gets_capped():
    query = "select a from t where b = 'LIMIT 1'"
    assert bird_reward._add_limit_to_query(query) == capped(query)


def test_limit_in_string_literal_with_escaped_quote_gets_capped():
    query = "select a from t where b = 'it''s a LIMIT 1'"
    assert bird_reward._add_limit_to_query(query) == capped(query)


def test_limit_inside_identifier_gets_capped():
    query = "select credit_limit from t"
    assert bird_reward._add_limit_to_query(query) == capped(query)


def test_limit_in_quoted_identifier_gets_capped():
    query = 'select "limit" from t'
    assert bird_reward._add_limit_to_query(query) == capped(query)


def test_limit_in_cte_body_gets_capped_but_limit_after_cte_is_unchanged():
    query = "with x as (select a from t limit 5) select a from x"
    assert bird_reward._add_limit_to_query(query) == capped(query)
    outer = "with x as (select a from t) select a from x limit 5"
    assert bird_reward._add_limit_to_query(outer) == outer


def test_trailing_limit_of_union_is_unchanged():
    query = "select a from t union all select a from t limit 5"
    assert bird_reward._add_limit_to_query(query) == query


def test_execute_sql_row_count_is_bounded_for_subquery_limit(tmp_path):
    db_path = str(tmp_path / "t.db")
    conn = sqlite3.connect(db_path)
    conn.execute("create table t(a)")
    conn.executemany("insert into t values(?)", [(i,) for i in range(CAP + 1000)])
    conn.commit()
    conn.close()

    query = "select a from t where a = (select a from t order by a desc limit 1) or 1"
    result = bird_reward._execute_sql(db_path, bird_reward._add_limit_to_query(query))
    assert not isinstance(result, Exception)
    assert len(result) == CAP
