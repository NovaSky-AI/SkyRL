import sqlite3
from func_timeout import func_timeout, FunctionTimedOut
import sys

def execute_sql(data_idx, db_file, sql):
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        conn.execute("BEGIN TRANSACTION;")
        cursor.execute(sql)
        execution_res = frozenset(cursor.fetchall())
        conn.rollback()
        conn.close()
        return data_idx, db_file, sql, execution_res, 1
    except Exception as e:
        print(f"Error executing SQL: {e}")
        conn.rollback()
        conn.close()
        return data_idx, db_file, sql, None, 0

def execute_sql_wrapper(data_idx, db_file, sql, timeout, output_str):
    try:
        res = func_timeout(timeout, execute_sql, args=(data_idx, db_file, sql))
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        print(f"Data index:{data_idx}\nSQL:\n{sql}\nTime Out!")
        print("-"*30)
        res = (data_idx, db_file, sql, None, 0)
    except Exception as e:
        print(f"Error executing SQL: {e}")
        res = (data_idx, db_file, sql, None, 0)

    # Append the output to the tuple
    if isinstance(res, tuple):
        res = res + (output_str,)
        
    return res