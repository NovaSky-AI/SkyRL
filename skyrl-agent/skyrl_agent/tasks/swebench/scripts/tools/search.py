#!/root/.venv/bin/python

"""
Description:
  Advanced code search tool that searches for code snippets, entities, and functions using graph-based indexing.
    •	Searches for code entities (classes, functions, files) by name or keywords using pre-built code indices.
    •	Can search for specific entities using qualified names (e.g., 'file.py:ClassName.method_name').
    •	Supports searching by line numbers within specific files to extract code context.
    •	Uses graph-based entity search combined with exact substring matching and BM25 semantic search.
    •	Filters results by file patterns or specific file paths.
    •	Returns structured results with code snippets, entity definitions, and relevant context.
  
  Search Strategy (in order):
    1. Entity search: Exact node ID or fuzzy entity name matching
       - Matches: 'file.py:ClassName.method' or just 'ClassName'
       - Only searches non-test files (from graph index)
    2. Exact substring match: ALWAYS performed (even if entity found)
       - Phase A: Searches in specified files (file_path_or_pattern)
       - Phase B: Searches all repo files if not found
    3. BM25 semantic search: Only if no entity AND no exact substring
       - Token-based semantic matching for related concepts
  
  Usage combinations:
    - search_terms only: searches across all Python files (or pattern-matched files)
    - search_terms + file_path_or_pattern: limits search to specific file(s) matching pattern

  Parameters:
    1.	search_terms (list of strings, optional)
  One or more search terms to find in the codebase. Can be entity names, keywords, qualified names like 'file.py:ClassName.method_name', or exact code substrings.
    2.	line_nums (list of integers, optional)
  Specific line numbers to extract context from when used with a specific file path.
    3.	file_path_or_pattern (string, optional)
  File pattern or specific file path to filter search scope. Defaults to '**/*.py'. If a specific file with line_nums, extracts context around those lines.
"""

import argparse
import collections
import fnmatch
import os
import pickle
import re
from collections import defaultdict
from typing import List, Optional
import ast

import networkx as nx

# Lightweight BM25 setup without NLTK to avoid tokenizer/punkt issues
try:
    from rank_bm25 import BM25Okapi

    CUSTOM_BM25_AVAILABLE = True
except Exception as e:
    CUSTOM_BM25_AVAILABLE = False
    print(f"Warning: BM25 backend not available: {e}")

# Simple, dependency-free tokenizer for code and text
_token_re = re.compile(r"[A-Za-z0-9_]+")


def _simple_tokenize(text: str):
    try:
        return _token_re.findall(text.lower())
    except Exception:
        return text.lower().split()


def is_skip_dir(dirname):
    """Check if directory should be skipped during graph building."""
    skip_dirs = [".github", ".git", "__pycache__", ".pytest_cache", "node_modules", ".venv", "tests"]
    for skip_dir in skip_dirs:
        if skip_dir in dirname:
            return True
    return False


class CodeAnalyzer(ast.NodeVisitor):
    def __init__(self, filename, source_code=None):
        self.filename = filename
        self.source_code = source_code  # Cache source code
        self.nodes = []
        self.node_name_stack = []
        self.node_type_stack = []

    def _get_end_lineno(self, node):
        """Get end line number with Python 3.7 fallback."""
        if hasattr(node, "end_lineno") and node.end_lineno is not None:
            return node.end_lineno

        # Fallback for Python 3.7: find the last line in the node's body
        # This is approximate but better than nothing
        try:
            last_lineno = node.lineno
            # Check if node has a body (classes and functions do)
            if hasattr(node, "body") and node.body:
                # Recursively find the last line number in the body
                for child in node.body:
                    if hasattr(child, "lineno"):
                        last_lineno = max(last_lineno, child.lineno)
                    if hasattr(child, "end_lineno"):
                        last_lineno = max(last_lineno, child.end_lineno)
            return last_lineno
        except:
            return node.lineno

    def visit_ClassDef(self, node):
        class_name = node.name
        full_class_name = ".".join(self.node_name_stack + [class_name])
        self.nodes.append(
            {
                "name": full_class_name,
                "type": NODE_TYPE_CLASS,
                "code": self._get_source_segment(node),
                "start_line": node.lineno,
                "end_line": self._get_end_lineno(node),
            }
        )

        self.node_name_stack.append(class_name)
        self.node_type_stack.append(NODE_TYPE_CLASS)
        self.generic_visit(node)
        self.node_name_stack.pop()
        self.node_type_stack.pop()

    def visit_FunctionDef(self, node):
        # if (
        #     self.node_type_stack
        #     and self.node_type_stack[-1] == NODE_TYPE_CLASS
        #     and node.name == '__init__'
        # ):
        #     return
        self._visit_func(node)

    def visit_AsyncFunctionDef(self, node):
        self._visit_func(node)

    def _visit_func(self, node):
        function_name = node.name
        full_function_name = ".".join(self.node_name_stack + [function_name])
        self.nodes.append(
            {
                "name": full_function_name,
                "parent_type": self.node_type_stack[-1] if self.node_type_stack else None,
                "type": NODE_TYPE_FUNCTION,
                "code": self._get_source_segment(node),
                "start_line": node.lineno,
                "end_line": self._get_end_lineno(node),
            }
        )

        self.node_name_stack.append(function_name)
        self.node_type_stack.append(NODE_TYPE_FUNCTION)
        self.generic_visit(node)
        self.node_name_stack.pop()
        self.node_type_stack.pop()

    def _get_source_segment(self, node):
        # Use cached source code instead of re-reading file
        if self.source_code is None:
            with open(self.filename, "r") as file:
                self.source_code = file.read()

        # ALWAYS extract lines directly from source to preserve exact formatting
        # This ensures exact match for later string-based file modifications
        lines = self.source_code.split("\n")
        if hasattr(node, "lineno"):
            start = node.lineno - 1  # Convert to 0-indexed
            end_lineno = self._get_end_lineno(node)
            end = end_lineno
            if start < len(lines) and end <= len(lines):
                return "\n".join(lines[start:end])
        return ""


def analyze_file(filepath):
    with open(filepath, "r") as file:
        code = file.read()
        # code = handle_edge_cases(code)
        try:
            tree = ast.parse(code, filename=filepath)
        except:
            raise SyntaxError
    analyzer = CodeAnalyzer(filepath, source_code=code)  # Pass source code to avoid re-reading
    try:
        analyzer.visit(tree)
    except RecursionError:
        pass
    return analyzer.nodes


def resolve_symlink(file_path):
    """Resolve the absolute path of a symbolic link."""
    if os.path.islink(file_path):
        relative_target = os.readlink(file_path)
        symlink_dir = os.path.dirname(os.path.dirname(file_path))
        absolute_target = os.path.abspath(os.path.join(symlink_dir, relative_target))
        if not os.path.exists(absolute_target):
            print(f"The target file does not exist: {absolute_target}")
            return None
        return absolute_target
    return None


def get_file_content(file_path):
    """Get file content with robust error handling and fallbacks."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        try:
            with open(file_path, "rb") as f:
                return f.read().decode("utf-8", errors="ignore")
        except Exception:
            return None


def build_graph(repo_path, fuzzy_search=True, global_import=False):
    import time

    start_time = time.time()
    print(f"Building graph for {repo_path}...")

    graph = nx.MultiDiGraph()
    file_nodes = {}
    file_count = 0

    ## add nodes
    graph.add_node("/", type=NODE_TYPE_DIRECTORY)
    dir_stack: List[str] = []
    dir_include_stack: List[bool] = []
    for root, _, files in os.walk(repo_path):
        # add directory nodes and edges
        dirname = os.path.relpath(root, repo_path)
        if dirname == ".":
            dirname = "/"
        elif is_skip_dir(dirname):
            continue
        else:
            graph.add_node(dirname, type=NODE_TYPE_DIRECTORY)
            parent_dirname = os.path.dirname(dirname)
            if parent_dirname == "":
                parent_dirname = "/"
            graph.add_edge(parent_dirname, dirname, type=EDGE_TYPE_CONTAINS)

        # in reverse step, remove directories that do not contain .py file
        while len(dir_stack) > 0 and not dirname.startswith(dir_stack[-1]):
            if not dir_include_stack[-1]:
                # print('remove', dir_stack[-1])
                graph.remove_node(dir_stack[-1])
            dir_stack.pop()
            dir_include_stack.pop()
        if dirname != "/":
            dir_stack.append(dirname)
            dir_include_stack.append(False)

        dir_has_py = False
        for file in files:
            if file.endswith(".py"):
                dir_has_py = True

                # add file nodes
                try:
                    file_path = os.path.join(root, file)
                    filename = os.path.relpath(file_path, repo_path)
                    if os.path.islink(file_path):
                        continue
                    else:
                        with open(file_path, "r") as f:
                            file_content = f.read()

                    graph.add_node(filename, type=NODE_TYPE_FILE, code=file_content)
                    file_nodes[filename] = file_path

                    nodes = analyze_file(file_path)
                    file_count += 1
                    if file_count % 10 == 0:
                        print(f"  Processed {file_count} files... (current: {filename})")
                        # break
                except (UnicodeDecodeError, SyntaxError):
                    # Skip the file that cannot decode or parse
                    continue

                # add function/class nodes
                for node in nodes:
                    full_name = f'{filename}:{node["name"]}'
                    graph.add_node(
                        full_name,
                        type=node["type"],
                        code=node["code"],
                        start_line=node["start_line"],
                        end_line=node["end_line"],
                    )

                # add edges with type=contains
                graph.add_edge(dirname, filename, type=EDGE_TYPE_CONTAINS)
                for node in nodes:
                    full_name = f'{filename}:{node["name"]}'
                    name_list = node["name"].split(".")
                    if len(name_list) == 1:
                        graph.add_edge(filename, full_name, type=EDGE_TYPE_CONTAINS)
                    else:
                        parent_name = ".".join(name_list[:-1])
                        full_parent_name = f"{filename}:{parent_name}"
                        graph.add_edge(full_parent_name, full_name, type=EDGE_TYPE_CONTAINS)

        # keep all parent directories
        if dir_has_py:
            for i in range(len(dir_include_stack)):
                dir_include_stack[i] = True

    # check last traversed directory
    while len(dir_stack) > 0:
        if not dir_include_stack[-1]:
            graph.remove_node(dir_stack[-1])
        dir_stack.pop()
        dir_include_stack.pop()

    global_name_dict = defaultdict(list)
    # if global_import:
    for node in graph.nodes():
        node_name = node.split(":")[-1].split(".")[-1]
        global_name_dict[node_name].append(node)
    graph.graph["global_name_dict"] = global_name_dict

    elapsed = time.time() - start_time
    print("Graph building complete!")
    print(f"  Files processed: {file_count}")
    print(f"  Total nodes: {graph.number_of_nodes()}")
    print(f"  Total edges: {graph.number_of_edges()}")
    print(f"  Time elapsed: {elapsed:.2f}s")
    return graph


def find_matching_files_from_list(file_list, file_pattern):
    """
    Find and return a list of file paths from the given list that match the given keyword or pattern.
    """
    # strip the double/single quotes if any
    if (file_pattern.startswith('"') and file_pattern.endswith('"')) or (
        file_pattern.startswith("'") and file_pattern.endswith("'")
    ):
        file_pattern = file_pattern[1:-1]

    repo_path = REPO_PATH if REPO_PATH else os.environ.get("REPO_PATH", "")
    if os.path.isabs(file_pattern):
        file_pattern = os.path.relpath(file_pattern, repo_path)
    else:
        file_pattern = os.path.normpath(file_pattern)
    if "*" in file_pattern or "?" in file_pattern or "[" in file_pattern:
        matching_files = fnmatch.filter(file_list, file_pattern)
    else:
        matching_files = [file for file in file_list if file_pattern in file]
    return matching_files


def merge_intervals(intervals):
    if not intervals:
        return []
    intervals.sort(key=lambda interval: interval[0])
    merged_intervals = [intervals[0]]
    for current in intervals[1:]:
        last = merged_intervals[-1]
        if current[0] <= last[1]:
            merged_intervals[-1] = (last[0], max(last[1], current[1]))
        else:
            merged_intervals.append(current)
    return merged_intervals


# Entity and dependency types used by searcher
NODE_TYPE_CLASS = "class"
NODE_TYPE_DIRECTORY = "directory"
NODE_TYPE_FILE = "file"
NODE_TYPE_FUNCTION = "function"

EDGE_TYPE_CONTAINS = "contains"

VALID_NODE_TYPES = [NODE_TYPE_FILE, NODE_TYPE_CLASS, NODE_TYPE_FUNCTION, NODE_TYPE_DIRECTORY]
VALID_EDGE_TYPES = [EDGE_TYPE_CONTAINS, "imports", "invokes", "inherits"]


class RepoEntitySearcher:
    """Retrieve Entity IDs and Code Snippets from Repository Graph"""

    def __init__(self, graph):
        self.G = graph
        self._global_name_dict = None
        self._global_name_dict_lowercase = None
        self._etypes_dict = {etype: i for i, etype in enumerate(VALID_EDGE_TYPES)}

    @property
    def global_name_dict(self):
        if self._global_name_dict is None:  # Compute only once
            _global_name_dict = defaultdict(list)
            for nid in self.G.nodes():
                if is_test_file(nid):
                    continue

                if nid.endswith(".py"):
                    fname = nid.split("/")[-1]
                    _global_name_dict[fname].append(nid)

                    name = nid[: -(len(".py"))].split("/")[-1]
                    _global_name_dict[name].append(nid)

                elif ":" in nid:
                    name = nid.split(":")[-1].split(".")[-1]
                    _global_name_dict[name].append(nid)

            self._global_name_dict = _global_name_dict
        return self._global_name_dict

    @property
    def global_name_dict_lowercase(self):
        if self._global_name_dict_lowercase is None:  # Compute only once
            _global_name_dict_lowercase = defaultdict(list)
            for nid in self.G.nodes():
                if is_test_file(nid):
                    continue

                if nid.endswith(".py"):
                    fname = nid.split("/")[-1].lower()
                    _global_name_dict_lowercase[fname].append(nid)

                    name = nid[: -(len(".py"))].split("/")[-1].lower()
                    _global_name_dict_lowercase[name].append(nid)

                elif ":" in nid:
                    name = nid.split(":")[-1].split(".")[-1].lower()
                    _global_name_dict_lowercase[name].append(nid)

            self._global_name_dict_lowercase = _global_name_dict_lowercase
        return self._global_name_dict_lowercase

    def has_node(self, nid, include_test=False):
        if not include_test and is_test_file(nid):
            return False
        return nid in self.G

    def get_node_data(self, nids, return_code_content=False, wrap_with_ln=True):
        rtn = []
        for nid in nids:
            node_data = self.G.nodes[nid]
            formatted_data = {
                "node_id": nid,
                "type": node_data["type"],
            }
            if node_data.get("code", ""):
                if "start_line" in node_data:
                    formatted_data["start_line"] = node_data["start_line"]
                    start_line = node_data["start_line"]
                elif formatted_data["type"] == NODE_TYPE_FILE:
                    start_line = 1
                    formatted_data["start_line"] = start_line
                else:
                    start_line = 1

                if "end_line" in node_data:
                    formatted_data["end_line"] = node_data["end_line"]
                    end_line = node_data["end_line"]
                elif formatted_data["type"] == NODE_TYPE_FILE:
                    end_line = len(node_data["code"].split("\n"))  # - 1
                    formatted_data["end_line"] = end_line
                else:
                    end_line = 1
                # load formatted code data
                if return_code_content and wrap_with_ln:
                    formatted_data["code_content"] = wrap_code_snippet(node_data["code"], start_line, end_line)
                elif return_code_content:
                    formatted_data["code_content"] = node_data["code"]
            rtn.append(formatted_data)
        return rtn

    def get_all_nodes_by_type(self, type):
        assert type in VALID_NODE_TYPES
        nodes = []
        for nid in self.G.nodes():
            if is_test_file(nid):
                continue
            if self.G.nodes[nid]["type"] == type:
                node_data = self.G.nodes[nid]
                if type == NODE_TYPE_FILE:
                    formatted_data = {
                        "name": nid,
                        "type": node_data["type"],
                        "content": node_data.get("code", "").split("\n"),
                    }
                elif type == NODE_TYPE_FUNCTION:
                    formatted_data = {
                        "name": nid.split(":")[-1],
                        "file": nid.split(":")[0],
                        "type": node_data["type"],
                        "content": node_data.get("code", "").split("\n"),
                        "start_line": node_data.get("start_line", 0),
                        "end_line": node_data.get("end_line", 0),
                    }
                elif type == NODE_TYPE_CLASS:
                    formatted_data = {
                        "name": nid.split(":")[-1],
                        "file": nid.split(":")[0],
                        "type": node_data["type"],
                        "content": node_data.get("code", "").split("\n"),
                        "start_line": node_data.get("start_line", 0),
                        "end_line": node_data.get("end_line", 0),
                        "methods": [],
                    }
                    dp_searcher = RepoDependencySearcher(self.G)
                    methods = dp_searcher.get_neighbors(
                        nid,
                        "forward",
                        ntype_filter=[NODE_TYPE_FUNCTION],
                        etype_filter=[EDGE_TYPE_CONTAINS],
                    )[0]
                    formatted_methods = []
                    for mid in methods:
                        mnode = self.G.nodes[mid]
                        formatted_methods.append(
                            {
                                "name": mid.split(".")[-1],
                                "start_line": mnode.get("start_line", 0),
                                "end_line": mnode.get("end_line", 0),
                            }
                        )
                    formatted_data["methods"] = formatted_methods
                nodes.append(formatted_data)
        return nodes


class RepoDependencySearcher:
    """Traverse Repository Graph"""

    def __init__(self, graph):
        self.G = graph
        self._etypes_dict = {etype: i for i, etype in enumerate(VALID_EDGE_TYPES)}

    def subgraph(self, nids):
        return self.G.subgraph(nids)

    def get_neighbors(
        self,
        nid,
        direction="forward",
        ntype_filter=None,
        etype_filter=None,
        ignore_test_file=True,
    ):
        nodes, edges = [], []
        if direction == "forward":
            for sn in self.G.successors(nid):
                if ntype_filter and self.G.nodes[sn]["type"] not in ntype_filter:
                    continue
                if ignore_test_file and is_test_file(sn):
                    continue
                for key, edge_data in self.G.get_edge_data(nid, sn).items():
                    etype = edge_data["type"]
                    if etype_filter and etype not in etype_filter:
                        continue
                    edges.append((nid, sn, self._etypes_dict[etype], {"type": etype}))
                    nodes.append(sn)

        elif direction == "backward":
            for pn in self.G.predecessors(nid):
                if ntype_filter and self.G.nodes[pn]["type"] not in ntype_filter:
                    continue
                if ignore_test_file and is_test_file(pn):
                    continue
                for key, edge_data in self.G.get_edge_data(pn, nid).items():
                    etype = edge_data["type"]
                    if etype_filter and etype not in etype_filter:
                        continue
                    edges.append((pn, nid, self._etypes_dict[etype], {"type": etype}))
                    nodes.append(pn)

        return nodes, edges


def traverse_tree_structure(
    G: nx.MultiDiGraph,
    start_node: str,
    direction: str,
    traversal_depth: int,
    entity_type_filter: Optional[List[str]],
    dependency_type_filter: Optional[List[str]],
):
    # Minimal placeholder for formatting outputs; not used by search_code_snippets
    return f"Traversal from {start_node} (depth={traversal_depth}, dir={direction})"


def is_test_file(path: str) -> bool:
    lowered = path.lower()
    return any(token in lowered for token in ["test/", "/tests/", "_test.py", "/test_"])


def wrap_code_snippet(code_content: str, start_line: int, end_line: int) -> str:
    """Wrap code snippet with line numbers."""
    lines = code_content.split("\n")
    wrapped_lines = []
    for i, line in enumerate(lines, start_line):
        wrapped_lines.append(f"{i:4d}|{line}")
    return "\n".join(wrapped_lines)


REPO_PATH: Optional[str] = None
GRAPH_INDEX_DIR: Optional[str] = None
ALL_FILE: Optional[list] = None
ALL_CLASS: Optional[list] = None
ALL_FUNC: Optional[list] = None

DP_GRAPH_ENTITY_SEARCHER: Optional[RepoEntitySearcher] = None
DP_GRAPH_DEPENDENCY_SEARCHER: Optional[RepoDependencySearcher] = None
DP_GRAPH: Optional[nx.MultiDiGraph] = None

# Lightweight BM25 cache (built lazily and reset when graph changes)
BM25_DOCS: Optional[List[List[str]]] = None
BM25_META: Optional[List[dict]] = None
BM25_MODEL: Optional["BM25Okapi"] = None


def parse_repo_index():
    global REPO_PATH, GRAPH_INDEX_DIR
    global DP_GRAPH_ENTITY_SEARCHER, DP_GRAPH_DEPENDENCY_SEARCHER, DP_GRAPH

    if not REPO_PATH:
        REPO_PATH = os.environ.get("REPO_PATH", "")
        # print('='*5, REPO_PATH, '='*5)

    if not GRAPH_INDEX_DIR:
        BASE_INDEX_DIR = os.path.join(REPO_PATH, "_index_data")
        GRAPH_INDEX_DIR = os.path.join(BASE_INDEX_DIR, "graph_index_v2.3")
        # print('='*5, BASE_INDEX_DIR, '='*5)

    if not DP_GRAPH:
        # setup graph traverser
        graph_index_file = os.path.join(GRAPH_INDEX_DIR, "code_graph.pkl")
        if not os.path.exists(graph_index_file):
            try:
                # Try to create the directory, but don't fail if we can't
                try:
                    os.makedirs(GRAPH_INDEX_DIR, exist_ok=True)
                except PermissionError:
                    print(f"Warning: Cannot create index directory {GRAPH_INDEX_DIR}, using in-memory graph only")

                G = build_graph(REPO_PATH, global_import=True)

                # Try to save graph
                try:
                    with open(graph_index_file, "wb") as f:
                        pickle.dump(G, f)
                except (PermissionError, OSError):
                    pass
            except Exception:
                import traceback

                traceback.print_exc()
                # Create an empty graph as fallback
                G = nx.MultiDiGraph()
        else:
            G = pickle.load(open(graph_index_file, "rb"))

        DP_GRAPH_ENTITY_SEARCHER = RepoEntitySearcher(G)
        DP_GRAPH_DEPENDENCY_SEARCHER = RepoDependencySearcher(G)
        DP_GRAPH = G

        # Reset BM25 cache whenever graph is (re)loaded
        global BM25_DOCS, BM25_META, BM25_MODEL
        BM25_DOCS = None
        BM25_META = None
        BM25_MODEL = None

        global ALL_FILE, ALL_CLASS, ALL_FUNC
        ALL_FILE = DP_GRAPH_ENTITY_SEARCHER.get_all_nodes_by_type(NODE_TYPE_FILE)
        ALL_CLASS = DP_GRAPH_ENTITY_SEARCHER.get_all_nodes_by_type(NODE_TYPE_CLASS)
        ALL_FUNC = DP_GRAPH_ENTITY_SEARCHER.get_all_nodes_by_type(NODE_TYPE_FUNCTION)

    # Custom BM25 implementation doesn't require pre-built indices
    # BM25Okapi builds index on-the-fly from documents


def get_current_repo_modules():
    return ALL_FILE, ALL_CLASS, ALL_FUNC


def get_graph_entity_searcher() -> RepoEntitySearcher:
    return DP_GRAPH_ENTITY_SEARCHER


def get_graph_dependency_searcher() -> RepoDependencySearcher:
    return DP_GRAPH_DEPENDENCY_SEARCHER


def get_graph():
    assert DP_GRAPH is not None
    return DP_GRAPH


def get_module_name_by_line_num(file_path: str, line_num: int):
    entity_searcher = get_graph_entity_searcher()
    dp_searcher = get_graph_dependency_searcher()

    cur_module = None
    if entity_searcher.has_node(file_path):
        module_nids, _ = dp_searcher.get_neighbors(file_path, etype_filter=[EDGE_TYPE_CONTAINS])
        module_ndatas = entity_searcher.get_node_data(module_nids)
        for module in module_ndatas:
            if module["start_line"] <= line_num <= module["end_line"]:
                cur_module = module
                break
        if cur_module and cur_module["type"] == NODE_TYPE_CLASS:
            func_nids, _ = dp_searcher.get_neighbors(cur_module["node_id"], etype_filter=[EDGE_TYPE_CONTAINS])
            func_ndatas = entity_searcher.get_node_data(func_nids, return_code_content=True)
            for func in func_ndatas:
                if func["start_line"] <= line_num <= func["end_line"]:
                    cur_module = func
                    break

    if cur_module:
        return cur_module
    return None


class QueryInfo:
    query_type: str = "keyword"
    term: Optional[str] = None
    line_nums: Optional[List] = None
    file_path_or_pattern: Optional[str] = None

    def __init__(
        self,
        query_type: str = "keyword",
        term: Optional[str] = None,
        line_nums: Optional[List] = None,
        file_path_or_pattern: Optional[str] = None,
    ):
        self.query_type = query_type
        if term is not None:
            self.term = term
        if line_nums is not None:
            self.line_nums = line_nums
        if file_path_or_pattern is not None:
            self.file_path_or_pattern = file_path_or_pattern

    def __str__(self):
        parts = []
        if self.term is not None:
            parts.append(f"term: {self.term}")
        if self.line_nums is not None:
            parts.append(f"line_nums: {self.line_nums}")
        if self.file_path_or_pattern is not None:
            parts.append(f"file_path_or_pattern: {self.file_path_or_pattern}")
        return ", ".join(parts)

    def __repr__(self):
        return self.__str__()


class QueryResult:
    file_path: Optional[str] = None
    format_mode: Optional[str] = "complete"
    nid: Optional[str] = None
    ntype: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    query_info_list: Optional[List[QueryInfo]] = None
    desc: Optional[str] = None
    message: Optional[str] = None
    warning: Optional[str] = None
    retrieve_src: Optional[str] = None

    def __init__(
        self,
        query_info: QueryInfo,
        format_mode: str,
        nid: Optional[str] = None,
        ntype: Optional[str] = None,
        file_path: Optional[str] = None,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        desc: Optional[str] = None,
        message: Optional[str] = None,
        warning: Optional[str] = None,
        retrieve_src: Optional[str] = None,
    ):
        self.format_mode = format_mode
        self.query_info_list = []
        self.insert_query_info(query_info)

        if nid is not None:
            self.nid = nid

        if ntype is not None:
            self.ntype = ntype
            if ntype in [NODE_TYPE_FILE, NODE_TYPE_CLASS, NODE_TYPE_FUNCTION]:
                self.file_path = nid.split(":")[0]

        if file_path is not None:
            self.file_path = file_path
        if start_line is not None and end_line is not None:
            self.start_line = start_line
            self.end_line = end_line

        if retrieve_src is not None:
            self.retrieve_src = retrieve_src

        if desc is not None:
            self.desc = desc
        if message is not None:
            self.message = message
        if warning is not None:
            self.warning = warning

    def insert_query_info(self, query_info: QueryInfo):
        self.query_info_list.append(query_info)

    def format_output(self, searcher):
        cur_result = ""
        if self.format_mode == "complete":
            node_data = searcher.get_node_data([self.nid], return_code_content=True)[0]
            ntype = node_data["type"]
            cur_result += f"Found {ntype} `{self.nid}`.\n"
            cur_result += "Source: " + self.retrieve_src + "\n"
            if "code_content" in node_data:
                if ntype in [NODE_TYPE_FUNCTION]:
                    cur_result += node_data["code_content"] + "\n"
                else:
                    content_size = node_data["end_line"] - node_data["start_line"]
                    if content_size <= 200:
                        cur_result += node_data["code_content"] + "\n"
                    else:
                        if ntype == NODE_TYPE_CLASS:
                            dp_searcher = RepoDependencySearcher(searcher.G)
                            method_nids, _ = dp_searcher.get_neighbors(
                                self.nid,
                                "forward",
                                ntype_filter=[NODE_TYPE_FUNCTION],
                                etype_filter=[EDGE_TYPE_CONTAINS],
                            )
                            if method_nids:
                                cur_result += "\nMethods:\n"
                                for mid in method_nids:
                                    method_name = mid.split(".")[-1]
                                    mnode = searcher.G.nodes[mid]
                                    cur_result += f'  - {method_name} (lines {mnode.get("start_line", "?")}-{mnode.get("end_line", "?")})\n'
                                cur_result += f"\nHint: Search for specific method like `{self.nid}.method_name` to see its implementation.\n"
                        elif ntype == NODE_TYPE_FILE:
                            # return class names
                            dp_searcher = RepoDependencySearcher(searcher.G)
                            class_nids, _ = dp_searcher.get_neighbors(
                                self.nid,
                                "forward",
                                ntype_filter=[NODE_TYPE_CLASS],
                                etype_filter=[EDGE_TYPE_CONTAINS],
                            )
                            if class_nids:
                                cur_result += "\nClasses:\n"
                                for cid in class_nids:
                                    class_name = cid.split(".")[-1]
                                    cnode = searcher.G.nodes[cid]
                                    cur_result += f'  - {class_name} (lines {cnode.get("start_line", "?")}-{cnode.get("end_line", "?")})\n'
                                cur_result += f"\nHint: Search for specific class like `{self.nid}.class_name` to see its implementation.\n"

        elif self.format_mode == "preview":
            node_data = searcher.get_node_data([self.nid], return_code_content=True)[0]
            ntype = node_data["type"]
            cur_result += f"Found {ntype} `{self.nid}`.\n"
            # show line range if available
            if "start_line" in node_data and "end_line" in node_data:
                cur_result += f'(lines {node_data["start_line"]}-{node_data["end_line"]})\n'
            cur_result += "Source: " + self.retrieve_src + "\n"
            if ntype == NODE_TYPE_FUNCTION:
                # hint
                cur_result += f"Hint: Search `{self.nid}` to see the full function code.\n"
            elif ntype in [NODE_TYPE_CLASS, NODE_TYPE_FILE]:
                content_size = node_data["end_line"] - node_data["start_line"]
                if content_size <= 50:
                    cur_result += node_data["code_content"] + "\n"
                else:
                    if ntype == NODE_TYPE_CLASS:
                        dp_searcher = RepoDependencySearcher(searcher.G)
                        method_nids, _ = dp_searcher.get_neighbors(
                            self.nid,
                            "forward",
                            ntype_filter=[NODE_TYPE_FUNCTION],
                            etype_filter=[EDGE_TYPE_CONTAINS],
                        )
                        if method_nids:
                            cur_result += "\nMethods:\n"
                            for mid in method_nids:
                                method_name = mid.split(".")[-1]
                                mnode = searcher.G.nodes[mid]
                                cur_result += f'  - {method_name} (lines {mnode.get("start_line", "?")}-{mnode.get("end_line", "?")})\n'
                            cur_result += f"\nHint: Search for specific method like `{self.nid}.method_name` to see its implementation.\n"
                    elif ntype == NODE_TYPE_FILE:
                        cur_result += f"Just show the structure of this {ntype} due to response length limitations:\n"
                        code_content = searcher.G.nodes[self.nid].get("code", "")
                        structure = get_skeleton(code_content)
                        cur_result += "```\n" + structure + "\n```\n"
                        cur_result += f"Hint: Search `{self.nid}.class_name` to get more information if needed.\n"
            elif ntype == NODE_TYPE_DIRECTORY:
                pass
        elif self.format_mode == "code_snippet":
            # If message exists, it's a summary (show message only, no code)
            if self.message and self.message.strip():
                cur_result += self.message + "\n"
                cur_result += "Source: " + self.retrieve_src + "\n"
            else:
                # Normal code snippet display
                if self.desc:
                    cur_result += self.desc + "\n"
                else:
                    cur_result += f"Found code snippet in file `{self.file_path}`.\n"
                cur_result += "Source: " + self.retrieve_src + "\n"
                node_data = searcher.get_node_data([self.file_path], return_code_content=True)[0]
                content = node_data["code_content"].split("\n")[1:-1]
                code_snippet = content[(self.start_line - 1) : self.end_line]
                code_snippet = "```\n" + "\n".join(code_snippet) + "\n```"
                cur_result += code_snippet + "\n"
        elif self.format_mode == "fold":
            node_data = searcher.get_node_data([self.nid], return_code_content=False)[0]
            self.ntype = node_data["type"]
            cur_result += f"Found {self.ntype} `{self.nid}`.\n"
        return cur_result


# get_skeleton is referenced by QueryResult.preview; provide a simple fallback
def get_skeleton(code_content: str) -> str:
    lines = code_content.splitlines()
    preview = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("class ") or stripped.startswith("def "):
            preview.append(stripped)
    return "\n".join(preview) if preview else "\n".join(lines[:50])


def get_code_block_by_line_nums(query_info, context_window=20):
    searcher = get_graph_entity_searcher()
    file_path = query_info.file_path_or_pattern
    line_nums = query_info.line_nums
    cur_query_results = []

    file_data = searcher.get_node_data([file_path], return_code_content=False)[0]
    line_intervals = []
    res_modules = []
    for line in line_nums:
        module_data = get_module_name_by_line_num(file_path, line)
        if not module_data:
            min_line_num = max(1, line - context_window)
            max_line_num = min(file_data["end_line"], line + context_window)
            line_intervals.append((min_line_num, max_line_num))
        elif module_data["node_id"] not in res_modules:
            query_result = QueryResult(
                query_info=query_info,
                format_mode="preview",
                nid=module_data["node_id"],
                ntype=module_data["type"],
                start_line=module_data["start_line"],
                end_line=module_data["end_line"],
                retrieve_src=f"Retrieved code context including {query_info.term}.",
            )
            cur_query_results.append(query_result)
            res_modules.append(module_data["node_id"])

    if line_intervals:
        line_intervals = merge_intervals(line_intervals)
        for interval in line_intervals:
            start_line, end_line = interval
            query_result = QueryResult(
                query_info=query_info,
                format_mode="code_snippet",
                nid=file_path,
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                retrieve_src=f"Retrieved code context including {query_info.term}.",
            )
            cur_query_results.append(query_result)
    return cur_query_results


def parse_node_id(nid: str):
    nfile = nid.split(":")[0]
    nname = nid.split(":")[-1]
    return nfile, nname


def search_entity_in_global_dict(term: str, include_files: Optional[List[str]] = None, prefix_term=None):
    searcher = get_graph_entity_searcher()
    if term.startswith(("class ", "Class")):
        term = term[len("class ") :].strip()
    elif term.startswith(("function ", "Function ")):
        term = term[len("function ") :].strip()
    elif term.startswith(("method ", "Method ")):
        term = term[len("method ") :].strip()
    elif term.startswith("def "):
        term = term[len("def ") :].strip()

    if term in searcher.global_name_dict:
        global_name_dict = searcher.global_name_dict
        nids = global_name_dict[term]
    elif term.lower() in searcher.global_name_dict_lowercase:
        term = term.lower()
        global_name_dict = searcher.global_name_dict_lowercase
        nids = global_name_dict[term]
    else:
        return None

    node_datas = searcher.get_node_data(nids, return_code_content=False)
    found_entities_filter_dict = collections.defaultdict(list)
    for ndata in node_datas:
        nfile, _ = parse_node_id(ndata["node_id"])
        if not include_files or nfile in include_files:
            prefix_terms = []
            candidite_prefixes = re.split(r"[./:]", ndata["node_id"].lower().replace(".py", ""))[:-1]
            if prefix_term:
                prefix_terms = prefix_term.lower().split(".")
            if not prefix_term or all([prefix in candidite_prefixes for prefix in prefix_terms]):
                found_entities_filter_dict[ndata["type"]].append(ndata["node_id"])
    return found_entities_filter_dict


def bm25_module_retrieve(
    query: str, include_files: Optional[List[str]] = None, search_scope: str = "all", similarity_top_k: int = 10
):
    # Simple entity-level retrieval using fuzzy matching on global names.
    searcher = get_graph_entity_searcher()
    candidates = []
    for nid in searcher.G.nodes:
        if query.lower() in nid.lower():
            candidates.append(nid)
    if include_files:
        candidates = [nid for nid in candidates if nid.split(":")[0] in include_files]
    return candidates[:similarity_top_k]


def search_entity(query_info, include_files: List[str] = None):
    term = query_info.term
    searcher = get_graph_entity_searcher()
    continue_search = True
    cur_query_results = []

    if searcher.has_node(term):
        continue_search = False
        query_result = QueryResult(
            query_info=query_info,
            format_mode="complete",
            nid=term,
            retrieve_src=f"Exact match found for entity name `{term}`.",
        )
        cur_query_results.append(query_result)

    if continue_search:
        found_entities_dict = search_entity_in_global_dict(term, include_files)
        if not found_entities_dict:
            found_entities_dict = search_entity_in_global_dict(term)

        use_sub_term = False
        used_term = term
        if not found_entities_dict and "." in term:
            try:
                prefix_term = ".".join(term.split(".")[:-1]).split()[-1]
            except IndexError:
                prefix_term = None
            split_term = term.split(".")[-1].strip()
            used_term = split_term
            found_entities_dict = search_entity_in_global_dict(split_term, include_files, prefix_term)
            if not found_entities_dict:
                found_entities_dict = search_entity_in_global_dict(split_term, prefix_term)
            if not found_entities_dict:
                use_sub_term = True
                found_entities_dict = search_entity_in_global_dict(split_term)

        if found_entities_dict:
            for ntype, nids in found_entities_dict.items():
                if not nids:
                    continue
                if ntype in [NODE_TYPE_FUNCTION, NODE_TYPE_CLASS, NODE_TYPE_FILE]:
                    if len(nids) <= 3:
                        node_datas = searcher.get_node_data(nids, return_code_content=True)
                        for ndata in node_datas:
                            query_result = QueryResult(
                                query_info=query_info,
                                format_mode="preview",
                                nid=ndata["node_id"],
                                ntype=ndata["type"],
                                start_line=ndata["start_line"],
                                end_line=ndata["end_line"],
                                retrieve_src=f"Match found for entity name `{used_term}`.",
                            )
                            cur_query_results.append(query_result)
                    else:
                        node_datas = searcher.get_node_data(nids, return_code_content=False)
                        for ndata in node_datas:
                            query_result = QueryResult(
                                query_info=query_info,
                                format_mode="fold",
                                nid=ndata["node_id"],
                                ntype=ndata["type"],
                                retrieve_src=f"Match found for entity name `{used_term}`.",
                            )
                            cur_query_results.append(query_result)
                        if not use_sub_term:
                            continue_search = False
                        else:
                            continue_search = True

    if continue_search:
        module_nids = bm25_module_retrieve(query=term, include_files=include_files)
        if not module_nids:
            module_nids = bm25_module_retrieve(query=term)
        module_datas = searcher.get_node_data(module_nids, return_code_content=True)
        showed_module_num = 0
        for module in module_datas[:5]:
            if module["type"] in [NODE_TYPE_FILE, NODE_TYPE_DIRECTORY]:
                query_result = QueryResult(
                    query_info=query_info,
                    format_mode="fold",
                    nid=module["node_id"],
                    ntype=module["type"],
                    retrieve_src="Retrieved entity using keyword search (approx).",
                )
                cur_query_results.append(query_result)
            elif showed_module_num < 3:
                showed_module_num += 1
                query_result = QueryResult(
                    query_info=query_info,
                    format_mode="preview",
                    nid=module["node_id"],
                    ntype=module["type"],
                    start_line=module["start_line"],
                    end_line=module["end_line"],
                    retrieve_src="Retrieved entity using keyword search (approx).",
                )
                cur_query_results.append(query_result)

    return (cur_query_results, continue_search)


def merge_query_results(query_results):
    priority = ["complete", "code_snippet", "preview", "fold"]
    merged_results = {}
    all_query_results: List[QueryResult] = []
    for qr in query_results:
        if qr.format_mode == "code_snippet":
            all_query_results.append(qr)
        elif qr.nid and qr.nid in merged_results:
            if qr.query_info_list[0] not in merged_results[qr.nid].query_info_list:
                merged_results[qr.nid].query_info_list.extend(qr.query_info_list)
            existing_format_mode = merged_results[qr.nid].format_mode
            if priority.index(qr.format_mode) < priority.index(existing_format_mode):
                merged_results[qr.nid].format_mode = qr.format_mode
                merged_results[qr.nid].start_line = qr.start_line
                merged_results[qr.nid].end_line = qr.end_line
                merged_results[qr.nid].retrieve_src = qr.retrieve_src
        elif qr.nid:
            merged_results[qr.nid] = qr
    all_query_results += list(merged_results.values())
    return all_query_results


def rank_and_aggr_query_results(query_results, fixed_query_info_list):
    query_info_list_dict = {}
    for qr in query_results:
        key = tuple(qr.query_info_list)
        if key in query_info_list_dict:
            query_info_list_dict[key].append(qr)
        else:
            query_info_list_dict[key] = [qr]

    def sorting_key(key):
        for i, fixed_query in enumerate(fixed_query_info_list):
            if fixed_query in key:
                return i
        return len(fixed_query_info_list)

    sorted_keys = sorted(query_info_list_dict.keys(), key=sorting_key)
    sorted_query_info_list_dict = {key: query_info_list_dict[key] for key in sorted_keys}

    priority = {"complete": 1, "preview": 2, "fold": 3, "code_snippet": 4}
    organized_dict = {}
    for key, values in sorted_query_info_list_dict.items():
        nested_dict = {priority_key: [] for priority_key in priority.keys()}
        for qr in values:
            if qr.format_mode in nested_dict:
                nested_dict[qr.format_mode].append(qr)
        organized_dict[key] = {k: v for k, v in nested_dict.items() if v}
    return organized_dict


def exact_substring_search(search_term: str, file_pattern: Optional[str] = None, max_files: int = 22):
    """
    Searches for exact substring match by scanning filesystem directly.
    Searches specified pattern first, then all .py files if not found.
    Includes test files.
    Returns dict: {file_path: [line_nums]}
    """
    repo_path = os.environ.get("REPO_PATH", "") or REPO_PATH
    if not repo_path or not os.path.exists(repo_path):
        return {}

    file_matches = {}  # {file_path: [line_numbers]}
    qlower = search_term.lower()

    # Normalize file_pattern (handle quotes, absolute paths, etc.)
    if file_pattern:
        # Strip quotes if any
        if (file_pattern.startswith('"') and file_pattern.endswith('"')) or (
            file_pattern.startswith("'") and file_pattern.endswith("'")
        ):
            file_pattern = file_pattern[1:-1]

        # Convert absolute path to relative
        if os.path.isabs(file_pattern):
            file_pattern = os.path.relpath(file_pattern, repo_path)
        else:
            file_pattern = os.path.normpath(file_pattern)

    # Build list of files to search
    target_files = []

    for root, dirs, files in os.walk(repo_path):
        # Skip hidden and irrelevant directories
        dirs[:] = [
            d
            for d in dirs
            if not d.startswith(".") and d not in [".git", "__pycache__", ".pytest_cache", "node_modules", ".venv"]
        ]

        for file in files:
            if not file.endswith(".py") or file.startswith("."):
                continue

            filepath = os.path.join(root, file)
            relative_path = os.path.relpath(filepath, repo_path)
            target_files.append((filepath, relative_path))

    # If file_pattern specified, filter and search those first
    if file_pattern and file_pattern != "**/*.py":
        pattern_files = []

        # Check if it's a glob pattern or substring match
        is_glob = "*" in file_pattern or "?" in file_pattern or "[" in file_pattern

        for filepath, relative_path in target_files:
            # Match against pattern
            if is_glob:
                # Use fnmatch for glob patterns
                if fnmatch.fnmatch(relative_path, file_pattern):
                    pattern_files.append((filepath, relative_path))
            else:
                # Use substring matching for non-glob patterns
                if file_pattern in relative_path:
                    pattern_files.append((filepath, relative_path))

        # Search pattern-matched files first
        for filepath, relative_path in pattern_files:
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    file_content = f.read()
                    lines = file_content.split("\n")

                    line_nums = []
                    for i, line in enumerate(lines, 1):
                        if qlower in line.lower():
                            line_nums.append(i)

                    if line_nums:
                        file_matches[relative_path] = line_nums

                        if len(file_matches) >= max_files:
                            return file_matches
            except (UnicodeDecodeError, PermissionError, OSError):
                continue

        # If found in pattern files, return immediately
        if file_matches:
            return file_matches

    # Search all files if not found or no pattern specified
    for filepath, relative_path in target_files:
        # Skip files already searched in pattern phase
        if relative_path in file_matches:
            continue

        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                file_content = f.read()
                lines = file_content.split("\n")

                line_nums = []
                for i, line in enumerate(lines, 1):
                    if qlower in line.lower():
                        line_nums.append(i)

                if line_nums:
                    file_matches[relative_path] = line_nums

                    if len(file_matches) >= max_files:
                        return file_matches
        except (UnicodeDecodeError, PermissionError, OSError):
            continue

    return file_matches


def bm25_semantic_only(query_info: QueryInfo, include_files: Optional[List[str]] = None, similarity_top_k: int = 10):
    """
    BM25 semantic search only (no exact substring matching).
    Uses token-based matching for semantic similarity.
    """
    if not CUSTOM_BM25_AVAILABLE:
        return []  # Fallback not available

    query = query_info.term or ""
    searcher = get_graph_entity_searcher()
    cur_query_results = []
    global BM25_DOCS, BM25_META, BM25_MODEL

    try:
        # Build BM25 index if needed
        if BM25_DOCS is None or BM25_META is None or BM25_MODEL is None:
            documents = []
            metadata = []
            for nid, data in searcher.G.nodes(data=True):
                if data.get("type") == "file" and "code" in data:
                    documents.append(_simple_tokenize(data["code"]))
                    metadata.append({"nid": nid, "file_path": nid, "type": "file"})
            if not documents:
                return []
            BM25_DOCS = documents
            BM25_META = metadata
            BM25_MODEL = BM25Okapi(BM25_DOCS)

        tokenized_query = _simple_tokenize(query)
        if not tokenized_query:
            return []

        # Get BM25 scores
        scores = BM25_MODEL.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: similarity_top_k * 2]

        # Process top-ranked files
        for idx in top_indices:
            if scores[idx] <= 0:
                continue

            meta = BM25_META[idx]
            file_path = meta["file_path"]
            if include_files and file_path not in include_files:
                continue

            node = searcher.G.nodes[file_path]
            code_content = node.get("code", "")
            if not code_content:
                continue

            lines = code_content.split("\n")
            q_tokens = [t for t in tokenized_query if len(t) >= 3]
            hit_line = None

            # Find line with most token matches
            for i, line in enumerate(lines, 1):
                low = line.lower()
                if any(t in low for t in q_tokens):
                    hit_line = i
                    break

            if hit_line is None:
                continue  # Skip files with no token matches

            start_line = max(1, hit_line - 2)
            end_line = min(len(lines), hit_line + 2)

            query_result = QueryResult(
                query_info=query_info,
                format_mode="code_snippet",
                nid=file_path,
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                retrieve_src="Retrieved using BM25 semantic search.",
            )
            cur_query_results.append(query_result)

        return cur_query_results[:3]

    except Exception as e:
        print(f"Error in BM25 semantic search: {e}")
        return []


def search_code_snippets(
    search_terms: Optional[List[str]] = None,
    line_nums: Optional[List] = None,
    file_path_or_pattern: Optional[str] = "**/*.py",
) -> str:
    try:
        parse_repo_index()
        files, _, _ = get_current_repo_modules()
        all_file_paths = [file["name"] for file in files]

        result = ""
        if file_path_or_pattern:
            include_files = find_matching_files_from_list(all_file_paths, file_path_or_pattern)
            if not include_files:
                include_files = all_file_paths
                result += f"No files found for file pattern '{file_path_or_pattern}'. Will search all files.\n...\n"
        else:
            include_files = all_file_paths
    except Exception as e:
        print(f"Error in search_code_snippets initialization: {e}")
        import traceback

        traceback.print_exc()
        return f"Error: Failed to initialize search. {str(e)}"

    try:
        query_info_list = []
        all_query_results = []

        if search_terms:
            filter_terms = []
            for term in search_terms:
                if is_test_file(term):
                    result += f"Do not support searching for test files: `{term}`.\n\n"
                else:
                    filter_terms.append(term)

            # Search each term independently
            for term in filter_terms:
                term = term.strip().strip(".")
                if not term:
                    continue

                query_info = QueryInfo(term=term)
                query_info_list.append(query_info)
                cur_query_results = []
                entity_found = False
                exact_substring_found = False

                # Step 1: Try entity search (exact node ID or fuzzy entity name)
                query_results, continue_search = search_entity(query_info=query_info, include_files=include_files)
                if query_results:
                    entity_found = True
                    cur_query_results.extend(query_results)

                # Step 2: ALWAYS try exact substring search (even if entity found)
                # User might want usages, not just definitions
                file_matches = exact_substring_search(term, file_pattern=file_path_or_pattern)
                if file_matches:
                    exact_substring_found = True
                    num_files = len(file_matches)

                    if num_files == 1:
                        # Single file: show line numbers and suggest str_replace_editor
                        file_path = list(file_matches.keys())[0]
                        line_nums = file_matches[file_path]

                        message = f"Found {len(line_nums)} match(es) in `{file_path}`:\n"
                        message += f'Lines: {", ".join(map(str, line_nums[:20]))}'
                        if len(line_nums) > 20:
                            message += f" ... and {len(line_nums) - 20} more."
                        message += "\n**Tip**: Use `str_replace_editor view` with view_range to see the code content."

                        query_result = QueryResult(
                            query_info=query_info,
                            format_mode="code_snippet",
                            nid=file_path,
                            file_path=file_path,
                            start_line=line_nums[0],
                            end_line=line_nums[0],
                            message=message,
                            retrieve_src="Exact substring match (single file).",
                        )
                        cur_query_results.append(query_result)

                    elif num_files <= 20:
                        # Multiple files (≤20): show file names and match counts
                        message = f"Found matches in {num_files} files:\n\n"
                        for file_path, line_nums in sorted(file_matches.items()):
                            message += f"- `{file_path}`: {len(line_nums)} match(es)"
                            message += "\n"
                        message += "\n**Tip**: Please narrow your search using `file_path_or_pattern` parameter."

                        # Create a single result summarizing all files
                        first_file = list(file_matches.keys())[0]
                        query_result = QueryResult(
                            query_info=query_info,
                            format_mode="code_snippet",
                            nid=first_file,
                            file_path=first_file,
                            start_line=1,
                            end_line=1,
                            message=message,
                            retrieve_src="Exact substring match (multiple files).",
                        )
                        cur_query_results.append(query_result)

                    else:
                        # Too many files (>20): suggest narrowing search
                        message = f"Found matches in {num_files} files (showing first 20):\n\n"
                        for file_path, line_nums in list(sorted(file_matches.items()))[:20]:
                            message += f"- `{file_path}`: {len(line_nums)} match(es)\n"
                        message += "\n**Warning**: More than 20 files matched. Please narrow your search using `file_path_or_pattern` parameter."

                        first_file = list(file_matches.keys())[0]
                        query_result = QueryResult(
                            query_info=query_info,
                            format_mode="code_snippet",
                            nid=first_file,
                            file_path=first_file,
                            start_line=1,
                            end_line=1,
                            message=message,
                            retrieve_src="Exact substring match (too many files).",
                        )
                        cur_query_results.append(query_result)

                # Step 3: BM25 semantic search ONLY if no entity AND no exact substring
                if not entity_found and not exact_substring_found:
                    query_results = bm25_semantic_only(query_info=query_info, include_files=include_files)
                    cur_query_results.extend(query_results)

                all_query_results.extend(cur_query_results)

        if file_path_or_pattern in all_file_paths and line_nums:
            if isinstance(line_nums, int):
                line_nums = [line_nums]
            file_path = file_path_or_pattern
            term = file_path + ":line " + ", ".join([str(line) for line in line_nums])
            query_info = QueryInfo(term=term, line_nums=line_nums, file_path_or_pattern=file_path)
            query_results = get_code_block_by_line_nums(query_info)
            all_query_results.extend(query_results)

        merged_results = merge_query_results(all_query_results)
        ranked_query_to_results = rank_and_aggr_query_results(merged_results, query_info_list)

        searcher = get_graph_entity_searcher()
        for query_infos, format_to_results in ranked_query_to_results.items():
            term_desc = ", ".join([f'"{query.term}"' for query in query_infos])
            result += f"##Searching for term {term_desc}...\n"
            result += "### Search Result:\n"
            cur_result = ""
            for format_mode, query_results in format_to_results.items():
                if format_mode == "fold":
                    cur_retrieve_src = ""
                    for qr in query_results:
                        if not cur_retrieve_src:
                            cur_retrieve_src = qr.retrieve_src
                        if cur_retrieve_src != qr.retrieve_src:
                            cur_result += "Source: " + cur_retrieve_src + "\n\n"
                            cur_retrieve_src = qr.retrieve_src
                        cur_result += qr.format_output(searcher)
                    cur_result += "Source: " + cur_retrieve_src + "\n"
                    if len(query_results) > 1:
                        cur_result += "Hint: Use more detailed query to get the full content of some if needed.\n"
                    else:
                        cur_result += f"Hint: Search `{query_results[0].nid}` for the full content if needed.\n"
                    cur_result += "\n"
                elif format_mode == "complete":
                    for qr in query_results:
                        cur_result += qr.format_output(searcher)
                        cur_result += "\n"
                elif format_mode == "preview":
                    filtered_results = []
                    grouped_by_file = defaultdict(list)
                    for qr in query_results:
                        if (qr.end_line - qr.start_line) < 100:
                            grouped_by_file[qr.file_path].append(qr)
                        else:
                            filtered_results.append(qr)
                    for file_path, results in grouped_by_file.items():
                        sorted_results = sorted(results, key=lambda qr: (qr.start_line, -qr.end_line))
                        max_end_line = -1
                        for qr in sorted_results:
                            if qr.end_line > max_end_line:
                                filtered_results.append(qr)
                                max_end_line = max(max_end_line, qr.end_line)
                    for qr in filtered_results:
                        cur_result += qr.format_output(searcher)
                        cur_result += "\n"
                elif format_mode == "code_snippet":
                    for qr in query_results:
                        cur_result += qr.format_output(searcher)
                        cur_result += "\n"
            cur_result += "\n\n"
            if cur_result.strip():
                result += cur_result
            else:
                result += "No locations found.\n\n"

        return result.strip()
    except Exception as e:
        print(f"Error in search_code_snippets execution: {e}")
        import traceback

        traceback.print_exc()
        return f"Error: Search failed. {str(e)}"


def main():
    parser = argparse.ArgumentParser(description="search_code_snippets tool")

    def parse_line_nums(line_str: str):
        # Remove brackets if present
        line_str = line_str.strip().strip("[]()")
        # Split on commas or whitespace
        parts = line_str.replace(",", " ").split()
        try:
            nums = [int(p) for p in parts]
        except ValueError:
            raise argparse.ArgumentTypeError(f"Could not convert {parts} to integers.")
        if not nums:
            raise argparse.ArgumentTypeError("At least one line number must be provided.")
        return nums

    parser.add_argument(
        "--file_path_or_pattern",
        type=str,
        default="**/*.py",
        help="Glob or file path to filter search scope. If specific file with --line_nums, extracts context around lines.",
    )
    parser.add_argument(
        "--search_terms",
        type=str,
        action="append",
        help="Repeatable search term(s). "
        "Example: --search_terms MyClass --search_terms file.py:Func "
        "or --search_terms '[MyClass, func_name]'",
    )

    parser.add_argument(
        "--line_nums",
        type=parse_line_nums,
        action="append",
        help="Repeatable line numbers when used with a specific file path. "
        "Example: --line_nums 42 --line_nums 56 "
        "or --line_nums '[42, 56, 78]'",
    )
    args = parser.parse_args()

    # Handle JSON-formatted search terms
    search_terms = args.search_terms
    if search_terms and len(search_terms) == 1:
        # Check if the single argument is a JSON array
        try:
            import json

            if search_terms[0].startswith("[") and search_terms[0].endswith("]"):
                search_terms = json.loads(search_terms[0])
        except (json.JSONDecodeError, ValueError):
            # Not JSON, use as-is
            pass

    try:
        output = search_code_snippets(
            search_terms=search_terms, line_nums=args.line_nums, file_path_or_pattern=args.file_path_or_pattern
        )
        print(output)
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
