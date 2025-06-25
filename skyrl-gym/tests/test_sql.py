"""
uv run --isolated --extra dev pytest tests/cpu/skyrl_gym/test_sql.py
"""

import skyrl_gym
import pytest
from unittest.mock import patch, MagicMock
from omegaconf import DictConfig

# Mock data for testing
MOCK_DB_RESULTS = {
    "SELECT name FROM employees WHERE id = 1;": [("John Doe",)],
    "SELECT name FROM employees WHERE id = 2;": [("Jane Smith",)],
    "GOLDEN_SQL;": [("John Doe",)],
}


@pytest.fixture
def mock_db_file():
    with patch("os.path.exists") as mock_exists:
        # Make the mock return True for any database file path
        mock_exists.return_value = True
        yield mock_exists


@pytest.fixture
def mock_sqlite_connection():
    with patch("sqlite3.connect") as mock_connect:
        # Create a mock connection
        mock_conn = MagicMock()
        mock_cursor = MagicMock()

        # Setup the mock cursor's execute and fetchall methods
        def mock_execute(sql, *args, **kwargs):
            if sql in MOCK_DB_RESULTS:
                mock_cursor.fetchall.return_value = MOCK_DB_RESULTS[sql]
            else:
                mock_cursor.fetchall.return_value = []

        mock_cursor.execute = mock_execute
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        yield mock_connect


@pytest.mark.parametrize(
    "step_1_output, step_2_output, ground_truth, expected",
    [
        # Test case 1: matching sql and solution
        (
            "<think>a</think>\n\n\
         <sql>SELECT name FROM employees WHERE id = 1;</sql>",
            "<think>b</think>\n\n\
         <solution>SELECT name FROM employees WHERE id = 1;</solution>",
            "SELECT name FROM employees WHERE id = 1;",
            1.0,
        ),
        # Test case 2: identical sql but different solution
        (
            "<think>a</think>\n\n\
         <sql>SELECT name FROM employees WHERE id = 1;</sql>",
            "<think>b</think>\n\n\
         <solution>SELECT name FROM employees WHERE id = 1;</solution>",
            "GOLDEN_SQL;",
            1.0,
        ),
        # Test case 3: solution is wrong
        (
            "<think>a</think>\n\n\
         <sql>SELECT name FROM employees WHERE id = 1;</sql>",
            "<think>b</think>\n\n\
         <solution>SELECT name FROM employees WHERE id = 2;</solution>",
            "SELECT name FROM employees WHERE id = 1;",
            0.0,
        ),
        # Test case 4: includes invalid sql msg
        (
            "<think>a</think>",  # no sql tag here
            "<think>b</think>\n\n\
         <solution>SELECT name FROM employees WHERE id = 1;</solution>",
            "SELECT name FROM employees WHERE id = 1;",
            1.0,
        ),
    ],
    ids=["correct_sql_and_solution", "golden_sql_same_result", "wrong_solution", "invalid_sql_msg"],
)
def test_compute_score(mock_db_file, mock_sqlite_connection, step_1_output, step_2_output, ground_truth, expected):
    extras = {
        "reward_spec": {"method": "rule", "ground_truth": "SELECT name FROM employees WHERE id = 1;"},
        "max_turns": 3,
        "db_id": "test_db",
        "data": "spider",
    }
    env = skyrl_gym.make("text2sql", env_config=DictConfig({"db_path": "/home/ray/default/sql_data"}), extras=extras)
    # Skip init() since it's not used in this test

    reminder_text = "<reminder>You have 2 turns left to complete the task.</reminder>"
    invalid_sql_message = (
        "Your previous action is invalid. Follow the format of outputting thinking process and sql tool, and try again."
    )

    output = env.step(step_1_output)
    obs1 = output["observations"]
    reward = output["reward"]

    # intermediate step reward is 0
    assert reward == 0.0
    # check reminder message
    assert reminder_text in obs1[0]["content"]
    if "<sql>" not in step_1_output:
        assert invalid_sql_message in obs1[0]["content"]
    output = env.step(step_2_output)
    reward = output["reward"]

    # Only assert reward when done is True
    if output["done"]:
        assert reward == expected
    else:
        assert reward == 0.0


@pytest.mark.parametrize(
    "output, parse_ground_truth",
    [
        # Test case 1: Valid SQL query
        ("<sql>SELECT name FROM employees WHERE id = 1;</sql>", "SELECT name FROM employees WHERE id = 1;"),
        # Test case 3: Invalid formatting
        ("<sql>SELECT name FROM employees", None),
    ],
)
def test_tool_parsing(mock_db_file, mock_sqlite_connection, output, parse_ground_truth):
    extras = {
        "reward_spec": {"method": "rule", "ground_truth": "SELECT name FROM employees WHERE id = 1;"},
        "max_turns": 2,
        "db_id": "test_db",
        "data": "spider",
    }
    env = skyrl_gym.make("text2sql", env_config=DictConfig({"db_path": "/home/ray/default/sql_data"}), extras=extras)
    # Skip init() since it's not used in this test

    # Step once and get the tool input in `info`
    output = env.step(output)
    info = output["metadata"]

    sql_query = info["tool_input"][1]  # SQL query is the second element in tool_input

    # assert it matches the parsed ground truth
    assert sql_query == parse_ground_truth


# Additional test for SQL execution
def test_sql_execution(mock_db_file, mock_sqlite_connection):
    extras = {
        "reward_spec": {"method": "rule", "ground_truth": "SELECT name FROM employees WHERE id = 1;"},
        "max_turns": 2,
        "db_id": "test_db",
        "data": "spider",
    }
    env = skyrl_gym.make("text2sql", env_config=DictConfig({"db_path": "/home/ray/default/sql_data"}), extras=extras)
    # Skip init() since it's not used in this test

    # Test a valid SQL query
    output = "<sql>SELECT name FROM employees WHERE id = 1;</sql>"
    output = env.step(output)
    observation = output["observations"]

    # Verify the SQL was executed
    mock_sqlite_connection.assert_called_once()

    # Verify the observation contains the expected result
    assert "John Doe" in observation[0]["content"]
