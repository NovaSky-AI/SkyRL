from skyrl_gym.tools.core import tool, ToolGroup
import pandas as pd
import sqlite3
import sys
import os
import threading


class SQLCodeExecutorToolGroup(ToolGroup):
    def __init__(self, db_file_path: str):
        self.db_path = db_file_path
        super().__init__(name="SQLCodeExecutorToolGroup")

    @tool
    def sql(self, db_id, sql, turns_left, timeout=5) -> str:
        def _execute_sql_wrapper(db_file, sql, timeout=5) -> str:
            """
            Execute SQL with a timeout using a worker thread and conn.interrupt().
            Ensures long-running queries are interrupted and connections are closed.
            """
            res_holder = {"value": None}
            done = threading.Event()
            conn_holder = {"conn": None}

            def worker():
                conn = None
                result = None
                try:
                    conn = sqlite3.connect(db_file, check_same_thread=False)
                    conn_holder["conn"] = conn
                    cursor = conn.cursor()
                    conn.execute("BEGIN TRANSACTION;")
                    cursor.execute(sql)
                    execution_res = frozenset(cursor.fetchall())
                    conn.rollback()
                    result = execution_res
                except Exception as e:
                    result = f"Error executing SQL: {str(e)}, db file: {db_file}"
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
                    res_holder["value"] = result
                    done.set()

            threading.Thread(target=worker, daemon=True).start()

            timeout_occurred = False
            try:
                if not done.wait(timeout):
                    timeout_occurred = True
                    conn = conn_holder.get("conn")
                    if conn is not None:
                        try:
                            conn.interrupt()
                        except Exception:
                            pass
                    # Wait for worker to clean up its connection
                    done.wait()
            except KeyboardInterrupt:
                sys.exit(0)
            except Exception as e:
                res_holder["value"] = str(e)

            if timeout_occurred:
                return f"SQL Timeout:\n{sql}"

            res = res_holder["value"]
            if isinstance(res, frozenset):
                df = pd.DataFrame(res)
                res = df.to_string(index=False)
                # NOTE: observation too long, just truncate
                if len(res) > 9000:
                    truncated_df = df.head(50)
                    res = "Truncated to 50 lines since returned response too long: " + truncated_df.to_string(
                        index=False
                    )  # or index=True if you want row numbers
            else:
                res = str(res)

            return res

        # TODO (erictang000): move this logic up into the text2sql env, since this is more specific logic
        reminder_text = f"<reminder>You have {turns_left} turns left to complete the task.</reminder>"
        if sql is None:
            obs = "Your previous action is invalid. Follow the format of outputting thinking process and sql tool, and try again."
        else:
            db_file = os.path.join(self.db_path, db_id, db_id + ".sqlite")
            obs = _execute_sql_wrapper(db_file, sql, timeout)

        return f"\n\n<observation>{obs}\n{reminder_text}</observation>\n\n"
