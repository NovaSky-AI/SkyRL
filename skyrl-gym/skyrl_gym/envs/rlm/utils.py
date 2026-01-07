import textwrap


SAFE_BUILTINS = {
    # Core types and functions
    "print": print,
    "len": len,
    "str": str,
    "int": int,
    "float": float,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "bool": bool,
    "type": type,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "sorted": sorted,
    "reversed": reversed,
    "range": range,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "round": round,
    "any": any,
    "all": all,
    "pow": pow,
    "divmod": divmod,
    "chr": chr,
    "ord": ord,
    "hex": hex,
    "bin": bin,
    "oct": oct,
    "repr": repr,
    "ascii": ascii,
    "format": format,
    "hash": hash,
    "id": id,
    "iter": iter,
    "next": next,
    "slice": slice,
    "callable": callable,
    "hasattr": hasattr,
    "getattr": getattr,
    "setattr": setattr,
    "delattr": delattr,
    "dir": dir,
    "vars": vars,
    "bytes": bytes,
    "bytearray": bytearray,
    "memoryview": memoryview,
    "complex": complex,
    "object": object,
    "super": super,
    "property": property,
    "staticmethod": staticmethod,
    "classmethod": classmethod,
    "__import__": None,
    "open": None,
    # Exceptions
    "Exception": Exception,
    "BaseException": BaseException,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "AttributeError": AttributeError,
    "FileNotFoundError": FileNotFoundError,
    "OSError": OSError,
    "IOError": IOError,
    "RuntimeError": RuntimeError,
    "NameError": NameError,
    "ImportError": ImportError,
    "StopIteration": StopIteration,
    "AssertionError": AssertionError,
    "NotImplementedError": NotImplementedError,
    "ArithmeticError": ArithmeticError,
    "LookupError": LookupError,
    "Warning": Warning,
    # Blocked
    "input": None,
    "eval": None,
    "exec": None,
    "compile": None,
    "globals": None,
    "locals": None,
}


def truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0:
        return "", True
    if text is None:
        return "", False
    if len(text) <= max_chars:
        return text, False

    head = max(0, max_chars - 200)
    tail = min(200, max_chars // 5)
    if head + tail > max_chars:
        tail = max(0, max_chars - head)

    truncated = (
        text[:head] + f"\n... <truncated {len(text) - (head + tail)} chars> ...\n" + (text[-tail:] if tail else "")
    )
    return truncated, True


def print_box(title: str, content: str, *, color: str = "cyan", width: int = 96) -> None:
    """
    Print a simple colored box to the terminal (no extra deps).
    """
    if content is None:
        content = ""

    colors = {
        "reset": "\033[0m",
        "cyan": "\033[36m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "red": "\033[31m",
        "magenta": "\033[35m",
        "blue": "\033[34m",
        "gray": "\033[90m",
    }
    c = colors.get(color, colors["cyan"])
    reset = colors["reset"]

    # Clamp width so small terminals don't look terrible.
    width = max(60, min(int(width), 140))
    inner_w = width - 4

    def _wrap_lines(s: str) -> list[str]:
        if not s:
            return [""]
        out: list[str] = []
        for line in s.splitlines():
            if line.strip() == "":
                out.append("")
                continue
            out.extend(
                textwrap.wrap(
                    line,
                    width=inner_w,
                    replace_whitespace=False,
                    drop_whitespace=False,
                )
            )
        return out

    title_line = f" {title} "
    top = "┌" + "─" * (width - 2) + "┐"
    mid = "├" + "─" * (width - 2) + "┤"
    bot = "└" + "─" * (width - 2) + "┘"

    print(c + top + reset)
    print(c + "│" + reset + title_line[:inner_w].ljust(inner_w) + c + "│" + reset)
    print(c + mid + reset)
    for line in _wrap_lines(content):
        print(c + "│" + reset + line[:inner_w].ljust(inner_w) + c + "│" + reset)
    print(c + bot + reset)
