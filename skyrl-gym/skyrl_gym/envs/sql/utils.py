"""
SynSQL reward calculation.

- Format reward: <think>...</think> <solution>...</solution>
- Outcome reward: check ground truth and predicted patch similarity
"""

import re
import sqlite3
import sys
import threading


THINK_START, THINK_END = "<think>", "</think>"
SQL_START, SQL_END = "<sql>", "</sql>"
SOLUTION_START, SOLUTION_END = "<solution>", "</solution>"
OBS_START, OBS_END = "<observation>", "</observation>"


# NOTE: bring back reward
def verify_format_and_extract(output: str):
    if output.count(SOLUTION_START) != 1:
        return False, None, None, None
    pre_solution, tail = output.split(SOLUTION_START, 1)

    if tail.count(SOLUTION_END) != 1:
        return False, None, None, None

    solution_text, _ = tail.split(SOLUTION_END, 1)

    if re.search(r"</?(think|sql|observation)\b", solution_text, re.I):
        return False, None, None, None

    thoughts = re.findall(r"<think>(.*?)</think>", output, re.S)
    if not thoughts:
        return False, None, None, None

    for m in re.finditer(r"</observation>", pre_solution, re.I):
        rest = pre_solution[m.end() :].lstrip()
        if not rest.lower().startswith(THINK_START):
            return False, None, None, None

    return True, thoughts, solution_text.strip(), None


def execute_sql_wrapper_single(db_file, sql, timeout, output_str):
    """
    Execute SQL with a timeout using a worker thread and conn.interrupt().
    Ensures long-running queries are interrupted and connections are closed.
    """
    res = (db_file, sql, None, 0)
    done = threading.Event()
    conn_holder = {"conn": None}

    def worker():
        nonlocal res
        conn = None
        try:
            conn = sqlite3.connect(db_file, check_same_thread=False)
            conn_holder["conn"] = conn
            cur = conn.cursor()
            conn.execute("BEGIN TRANSACTION;")
            cur.execute(sql)
            rows = frozenset(cur.fetchall())
            conn.rollback()
            res = (db_file, sql, rows, 1)
        except Exception:
            res = (db_file, sql, None, 0)
        finally:
            if conn is not None:
                try:
                    conn.rollback()
                except Exception:
                    pass
                try:
                    conn.close()
                except Exception:
                    pass
            done.set()

    threading.Thread(target=worker, daemon=True).start()

    try:
        if not done.wait(timeout):
            conn = conn_holder.get("conn")
            if conn is not None:
                try:
                    conn.interrupt()
                except Exception:
                    pass
            done.wait()
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception:
        res = (db_file, sql, None, 0)

    if isinstance(res, tuple):
        res = res + (output_str,)

    return res


def calculate_reward_single(completion, reference, db_file, timeout=30):
    reward = 0.0
    num_comparisons = 0

    is_valid, _, pred_sql, _ = verify_format_and_extract(completion)
    if not is_valid:
        reward = -1.0
        return reward
    else:
        num_comparisons += 1

    pred = execute_sql_wrapper_single(db_file, pred_sql, timeout, completion)
    ref = execute_sql_wrapper_single(db_file, reference, timeout, completion)

    _, _, pred_results, _, _ = pred
    _, _, gt_results, _, _ = ref

    if pred_results is not None and gt_results is not None and pred_results == gt_results:
        reward = 1.0
    else:
        reward = 0.0
    return reward


def compute_score_single(completion, reference, db_file):
    try:
        res = calculate_reward_single(completion, reference, db_file)
        return res
    except Exception as e:
        print(f"Unexpected error: {e}; Setting reward as 0")
        return 0
