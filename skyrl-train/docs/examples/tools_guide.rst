Using Tools and ToolGroups in SkyRL-Gym
==========================

This guide shows how to use tools in SkyRL-Gym environments.

**What we're building:** An environment that uses tools to help agents perform tasks.

Core Concepts
-------------

- **Tool**: A single executable function
- **ToolGroup**: A collection of related tools that share the same context (states)
- **Environment Integration**: Tools are integrated into environments

Tool State Sharing and Modularity
---------------------------------

ToolGroups enable modular design with state sharing:

**State Sharing:**
- Tools in same ToolGroup share context (databases, caches, connections)
- Efficient resource reuse within groups
- Consistent state across tool executions

**Modular Benefits:**
- Use multiple ToolGroups for different LLM capabilities
- Independent development and testing
- Scale by adding/removing ToolGroups as needed

**Examples:**
- Database ToolGroup: Shared connection pool
- Search ToolGroup: Shared cache and index
- Code ToolGroup: Shared execution environment

Some tools need shared state, others work independently. This design enables scalable, modular environments.

The `@tool` Decorator
--------------------

The `@tool` decorator marks methods as executable tools:

.. code-block:: python

    from skyrl_gym.tools.core import tool

    class MyToolGroup(ToolGroup):
        def __init__(self):
            super().__init__(name="MyToolGroup")
        
        @tool
        def my_tool(self, input_param: str) -> str:
            """Execute a specific task."""
            return f"Processed: {input_param}"

The ToolGroup Base Class
-----------------------

`ToolGroup` provides tool management:

.. code-block:: python

    from skygym.tools.core import ToolGroup

    class MyToolGroup(ToolGroup):
        def __init__(self, name: str):
            super().__init__(name=name)

Core Methods:
- **`get_tool(name)`**: Get a tool by name
- **`execute_tool(name, *args, **kwargs)`**: Execute a tool

Built-in ToolGroups
------------------

SkyRL-Gym provides pre-built ToolGroups:

Python Code Execution
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from skygym.tools import PythonCodeExecutorToolGroup

    python_tools = PythonCodeExecutorToolGroup(timeout=15.0)
    result = python_tools.execute_tool("python", "print('Hello, World!')")

SQL Code Execution
~~~~~~~~~~~~~~~~~

.. code-block:: python

    from skygym.tools import SQLCodeExecutorToolGroup

    sql_tools = SQLCodeExecutorToolGroup(db_file_path="/path/to/databases")
    result = sql_tools.execute_tool("sql", "SELECT * FROM users")

Search ToolGroup
~~~~~~~~~~~~~~~

.. code-block:: python

    from skygym.tools import SearchToolGroup

    search_tools = SearchToolGroup(
        search_url="http://127.0.0.1:8000/retrieve"
    )
    result = search_tools.execute_tool("search", "Context to search")

Creating Custom ToolGroups
-------------------------

Basic Custom ToolGroup
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from skygym.tools.core import tool, ToolGroup

    class WeatherToolGroup(ToolGroup):
        def __init__(self, api_key: str):
            self.api_key = api_key
            super().__init__(name="WeatherToolGroup")
        
        @tool
        def get_weather(self, city: str) -> str:
            """Get current weather for a city."""
            # Implementation here
            return f"Weather in {city}: 20°C, sunny"

Environment Integration
----------------------

Tools are integrated into environments through `BaseTextEnv`:

Tool Initialization
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from skygym.envs.base_text_env import BaseTextEnv

    class MyEnvironment(BaseTextEnv):
        def __init__(self, env_config, extras):
            super().__init__()
            
            # Initialize tool groups
            python_tools = PythonCodeExecutorToolGroup(timeout=10.0)
            search_tools = SearchToolGroup()
            
            # Register tool groups
            self.init_tool_groups([python_tools, search_tools])

Tool Execution
~~~~~~~~~~~~~

Environments handle tool execution:

.. code-block:: python

    def step(self, action: str):
        # Parse action to extract tool call
        tool_group_name, tool_name, tool_input = self._parse_action(action)
        
        # Execute the tool
        observation = self._execute_tool(tool_group_name, tool_name, tool_input)
        
        return BaseTextEnvStepOutput(
            observations=[{"role": "user", "content": observation}],
            reward=reward,
            done=done,
            metadata=info
        )

Action Parsing
~~~~~~~~~~~~~

Parse agent actions to extract tool calls:

.. code-block:: python

    import re

    def _parse_action(self, action: str):
        # Parse tool blocks like <tool><tool_name>input</tool_name></tool>
        tool_block_match = re.search(r"<tool>(.*?)</tool>", action, re.DOTALL)
        if not tool_block_match:
            raise ValueError("No tool block found in action")
        
        tool_content = tool_block_match.group(1).strip()
        inner_tag_match = re.search(r"<(\w+)>(.*?)</\1>", tool_content, re.DOTALL)
        
        tool_name = inner_tag_match.group(1)
        tool_input = inner_tag_match.group(2).strip()
        
        tool_group_name = self.tool_to_toolgroup[tool_name]
        
        return tool_group_name, tool_name, [tool_input]

Using Multiple ToolGroups
-------------------------

Combine multiple ToolGroups for powerful environments:

.. code-block:: python

    class AdvancedEnvironment(BaseTextEnv):
        def __init__(self, env_config, extras):
            super().__init__()
            
            # Different ToolGroups with shared state
            self.db_tools = SQLCodeExecutorToolGroup(db_file_path="/path/to/databases")
            self.python_tools = PythonCodeExecutorToolGroup(timeout=10.0)
            self.search_tools = SearchToolGroup(search_url="http://127.0.0.1:8000/retrieve")
            self.custom_tools = MyCustomToolGroup(shared_config=extras.get("config"))
            
            # Register all tool groups
            self.init_tool_groups([self.db_tools, self.python_tools, self.search_tools, self.custom_tools])

**Benefits:**
- Comprehensive LLM capabilities (database, code, search, custom tools)
- Each ToolGroup manages its own resources and state
- Modular scaling - add/remove ToolGroups as needed
- Clean separation between different domains

Best Practices
--------------

Tool Design
~~~~~~~~~~

1. **Single Responsibility**: Each tool should have one purpose
2. **Error Handling**: Return meaningful error messages
3. **Timeout Protection**: Use timeouts to prevent hanging
4. **Input Validation**: Validate inputs before processing

ToolGroup Organization
~~~~~~~~~~~~~~~~~~~~~

1. **Logical Grouping**: Group tools that share similar state or context
2. **State Management**: Design shared state carefully - some tools need it, others don't
3. **Resource Efficiency**: Reuse connections and resources within ToolGroups
4. **Modular Design**: Keep ToolGroups independent and focused on specific domains

Environment Integration
~~~~~~~~~~~~~~~~~~~~~~

1. **Tool Registration**: Register tools during initialization
2. **Action Parsing**: Implement robust action parsing
3. **Error Recovery**: Provide graceful error recovery

Testing Tools
-------------

Test tools and environments:

.. code-block:: python

    import pytest
    from skygym.tools import PythonCodeExecutorToolGroup

    def test_python_tool_execution():
        """Test Python code execution tool."""
        tools = PythonCodeExecutorToolGroup(timeout=5.0)
        
        result = tools.execute_tool("python", "print('Hello, World!')")
        assert result == "Hello, World!"

API Reference
-------------

For detailed API documentation, see:

- :doc:`tools`: Core tool classes and methods
- :doc:`env`: Environment integration details

That's it! You've learned how to use tools in SkyRL-Gym environments. The same pattern works for any tool-based task you want to build. 