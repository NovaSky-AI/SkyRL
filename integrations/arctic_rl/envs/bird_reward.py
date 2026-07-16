# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
SQL reward function for BIRD RL training, adapted from SnowflakeDialectSQLRewardManagerV6b.

Uses SQLite execution instead of Snowflake. Compatible with verl's
custom_reward_function mechanism via compute_score().

Vendored from
``arctic_platform.rl.projects.txt2sql.bird_reward`` so this integration is
self-contained and doesn't require the Arctic-Platform private RL recipe
checkout. Bit-for-bit identical to the upstream reward function the verl
PR #6 benchmark uses; keep them in sync.

Reward scheme (matching V6b non-semantic-model behavior):
    1.0  - Predicted SQL produces the same result set as gold SQL
    0.1  - Predicted SQL executes successfully but produces wrong results,
           OR SQL was extracted successfully (format bonus)
    0.0  - No SQL extracted, SQL fails to execute, or timeout
"""

import json
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError

SQL_TIMEOUT = 30
DEFAULT_LIMIT_NUMBER = 5000
FORMAT_REWARD_BONUS = 0.1


# ---------------------------------------------------------------------------
# SQL extraction (mirrors V6b's extract_solution / _extract_sql_omnisql)
# ---------------------------------------------------------------------------


def _extract_sql_omnisql(message: str) -> str:
    """Extract SQL from ```sql ... ``` markdown blocks (last valid block)."""
    pattern = r"```sql\s*(.*?)\s*```"
    sql_blocks = re.findall(pattern, message, re.DOTALL)
    for block in reversed(sql_blocks):
        if len(block.strip()) > 6:
            return block.strip()
    return ""


def _extract_sql_generic_block(message: str) -> str:
    """Extract SQL from generic ``` ... ``` blocks containing SELECT."""
    blocks = re.findall(r"```\s*(.*?)\s*```", message, re.DOTALL)
    for block in reversed(blocks):
        if "SELECT" in block.upper() and len(block.strip()) > 6:
            return block.strip()
    return ""


def _extract_sql_analyst(message: str) -> str:
    """Extract SQL from ```json { "sql": "..." } ``` blocks."""
    block = re.search(r"```\s*json(.*?)```", message, re.DOTALL)
    if block is None:
        return ""
    json_str = block.group(1)
    idx_left = json_str.rfind("{")
    idx_right = json_str.find("}")
    if idx_left == -1 or idx_right == -1:
        return ""
    json_str = json_str[idx_left : idx_right + 1]
    try:
        return json.loads(json_str.replace("\\n", "\n").replace("\\'", "'"), strict=False).get("sql", "")
    except Exception:
        return ""


def _extract_sql_raw_select(message: str) -> str:
    """Fallback: extract a raw SELECT statement."""
    match = re.search(r"(SELECT\s+.+?)(?:\n\n|$)", message, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def extract_sql(response: str) -> str:
    """Extract SQL from model response, following V6b's extraction pipeline.

    1. Split on </think> to isolate the answer portion
    2. Try ```sql blocks
    3. Try generic ``` blocks with SELECT
    4. Try ```json blocks with {"sql": ...}
    5. Fallback to raw SELECT statement
    """
    if "</think>" in response:
        answer_part = response.split("</think>", 1)[1]
    else:
        answer_part = response

    sql = _extract_sql_omnisql(answer_part)
    if sql:
        return sql

    sql = _extract_sql_generic_block(answer_part)
    if sql:
        return sql

    sql = _extract_sql_analyst(answer_part)
    if sql:
        return sql

    return _extract_sql_raw_select(answer_part)


# ---------------------------------------------------------------------------
# Format validation (mirrors V6b's validate_response_structure)
# ---------------------------------------------------------------------------


def validate_response_format(response: str) -> bool:
    """Check that the response has exactly one <think>...</think> pair, properly nested."""
    start_positions = [m.start() for m in re.finditer(r"<think>", response)]
    end_positions = [m.start() for m in re.finditer(r"</think>", response)]

    if len(start_positions) != 1 or len(end_positions) != 1:
        return False
    return start_positions[0] < end_positions[0]


# ---------------------------------------------------------------------------
# LIMIT addition (mirrors V6b's _add_limit_to_query)
# ---------------------------------------------------------------------------


def _add_limit_to_query(query: str, limit: int = DEFAULT_LIMIT_NUMBER) -> str:
    """Add LIMIT clause if the query doesn't already have one."""
    if not query:
        return query
    upper = query.upper()
    if "LIMIT " in upper or "LIMIT\n" in upper or "LIMIT\t" in upper:
        return query
    return query.rstrip().rstrip(";").rstrip() + f" LIMIT {limit};"


# ---------------------------------------------------------------------------
# SQLite execution and comparison
# ---------------------------------------------------------------------------


def _execute_sql(db_path: str, sql: str, timeout: float = SQL_TIMEOUT) -> frozenset | Exception:
    """Execute SQL against a SQLite database and return result as frozenset of row tuples."""
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
        deadline = __import__("time").monotonic() + timeout

        def _check_cancel():
            if __import__("time").monotonic() > deadline:
                return 1
            return 0

        conn.set_progress_handler(_check_cancel, 1000)
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        conn.close()
        return frozenset(rows)
    except sqlite3.OperationalError as e:
        if "interrupt" in str(e).lower():
            return TimeoutError(f"SQL execution exceeded {timeout}s")
        return e
    except Exception as e:
        return e


def _execute_with_timeout(db_path: str, sql: str, timeout: float = SQL_TIMEOUT) -> frozenset | Exception:
    """Execute SQL with a timeout using a thread pool.

    Uses shutdown(wait=False) to avoid blocking if the SQLite thread is stuck.
    The SQLite progress handler provides cooperative cancellation.
    """
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_execute_sql, db_path, sql, timeout)
    try:
        return future.result(timeout=timeout + 2)
    except FuturesTimeoutError:
        return TimeoutError(f"SQL execution exceeded {timeout}s")
    except Exception as e:
        return e
    finally:
        executor.shutdown(wait=False)


def _compare_results(
    db_path: str,
    pred_sql: str,
    gold_sqls: list[str],
    timeout: float = SQL_TIMEOUT,
) -> tuple[float, bool]:
    """Execute and compare predicted SQL against all gold SQLs.

    Returns (reward, execution_success) matching V6b's non-semantic-model logic:
        - 1.0 if pred result == any gold result (frozenset match)
        - 0.1 if pred executes but doesn't match any gold
        - 0.0 if pred fails to execute

    Caches gold results within this call to avoid re-execution.
    """
    pred_sql_limited = _add_limit_to_query(pred_sql)
    pred_result = _execute_with_timeout(db_path, pred_sql_limited, timeout)

    if isinstance(pred_result, Exception):
        return 0.0, False

    gold_cache: dict[str, frozenset | Exception] = {}
    scores = []
    for gold_sql in gold_sqls:
        if gold_sql not in gold_cache:
            gold_sql_limited = _add_limit_to_query(gold_sql)
            gold_cache[gold_sql] = _execute_with_timeout(db_path, gold_sql_limited, timeout)
        gold_result = gold_cache[gold_sql]

        if isinstance(gold_result, Exception):
            scores.append(0.0)
            continue

        if pred_result == gold_result:
            scores.append(1.0)
        else:
            scores.append(0.1)

    return (max(scores) if scores else 0.0), True


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Compute reward score for SQL generation (verl custom_reward_function interface).

    Mirrors SnowflakeDialectSQLRewardManagerV6b logic with SQLite execution:
    1. Extract SQL from model response
    2. Validate response format (<think> tags)
    3. Execute predicted and gold SQL against SQLite
    4. Compare results (frozenset match)
    5. Apply format bonus

    Args:
        data_source: Dataset identifier (e.g. "bird")
        solution_str: Full model response text (decoded)
        ground_truth: Gold SQL query string
        extra_info: Dict with at minimum {"db_path": "/path/to/db.sqlite"}.
                    Optionally {"alternative_answers": [...]} for multiple gold SQLs.

    Returns:
        dict with "score" (float), "format_correct" (float), "execution_success" (float)
    """
    if extra_info is None:
        return {"score": 0.0, "format_correct": 0.0, "execution_success": 0.0}

    db_path = extra_info.get("db_path", "")
    if not db_path:
        return {"score": 0.0, "format_correct": 0.0, "execution_success": 0.0}

    pred_sql = extract_sql(solution_str)
    format_correct = float(bool(pred_sql) and validate_response_format(solution_str))

    if not pred_sql:
        return {"score": 0.0, "format_correct": 0.0, "execution_success": 0.0}

    alternative_answers = extra_info.get("alternative_answers")
    if alternative_answers and len(alternative_answers) > 0:
        gold_sqls = [str(s).strip() for s in alternative_answers if s and str(s).strip()]
    else:
        gold_sqls = [ground_truth] if ground_truth else []

    if not gold_sqls:
        return {"score": 0.0, "format_correct": format_correct, "execution_success": 0.0}

    reward, execution_success = _compare_results(db_path, pred_sql, gold_sqls)

    if format_correct:
        reward = max(reward, FORMAT_REWARD_BONUS)

    return {
        "score": reward,
        "format_correct": format_correct,
        "execution_success": float(execution_success),
    }
