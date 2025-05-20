import sqlite3
import os
from func_timeout import func_timeout, FunctionTimedOut
import multiprocessing as mp
import sys
from time import perf_counter
import threading
import signal
    
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
        error_msg = f"{type(e).__name__}: {str(e)}"
        conn.rollback()
        conn.close()
        return data_idx, db_file, sql, error_msg, 0
        
def execute_sql_wrapper(data_idx, db_file, sql, timeout):
    try:
        res = func_timeout(timeout, execute_sql, args=(data_idx, db_file, sql))
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        print(f"Data index:{data_idx}\nSQL:\n{sql}\nTime Out!")
        print("-"*30)
        res = (data_idx, db_file, sql, "Function TimeOut", 0)
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        res = (data_idx, db_file, sql, error_msg, 0)
    return res
    
################################## Multi-Turn SQL Execution ##################################
def execute_sqls_parallel(db_files, pred_sqls, num_cpus=64, timeout=50):
    """
    Execute a list of SQL statements in parallel, returning their raw execution results.
    
    Args:
        db_files   (List[str]): paths to sqlite DB files
        pred_sqls  (List[str]): SQL strings to execute 
        num_cpus      (int): number of worker processes
        timeout       (int): per-query timeout in seconds
        
        Note: 1-1 correspondance of db_files to pred_sqls file 
    
    Returns:
        List[Tuple[int, str, str, Optional[frozenset], int]]:
            (data_idx, db_file, sql, execution_res, success_flag)
            
        # if success: return data_idx, db_file, sql, execution_res, 1
        # if failed: return data_idx, db_file, sql, "Error", 0
    """
    # prepare the argument tuples
    args = [
        (idx, db_file, sql, timeout)
        for idx, (db_file, sql) in enumerate(zip(db_files, pred_sqls))
    ]
    # spawn a pool and map
    print(f"[DEBUG - utils::execute_sqls_parallel]: executing {len(args)} SQL statements in parallel", flush=True)
    start = perf_counter()
    # NOTE (sumanthrh): SGLang has a custom sigchld handler, but it can produce confusing logs
    # during sql execution with regular pool termination. We temporarily disable the handler during sql execution. 
    original_sigchld_handler = signal.getsignal(signal.SIGCHLD)
    try:
        # Set to default handler to avoid spammy logs
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)
        # Use spawn method
        mp.set_start_method("spawn", force=True)
        with mp.Pool(processes=num_cpus) as pool:
            results = pool.starmap(execute_sql_wrapper, args)
    finally:
        # Restore the original handler
        signal.signal(signal.SIGCHLD, original_sigchld_handler)
    end = perf_counter()
    print(f"[DEBUG - utils::execute_sqls_parallel]: executed {len(args)} SQL statements in parallel in {end - start:.2f} seconds", flush=True)
    return results