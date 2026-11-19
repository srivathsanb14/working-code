"""The candidate contract: what an LLM must write, and how it is checked and loaded.

A candidate is a short Python module:

    import numpy as np
    import dlib                                   # optional helper (LinearNetwork, pairs, softplus)

    PARAMS = {"name": (init, lo, hi), ...}        # global constants, shared by every building
    THETA  = {"name": (init, lo, hi), ...}        # per-scenario constants, re-fitted on each new building

    def simulate(geo, inputs, theta, p, T0):
        '''Return predicted room-air temperatures, shape (n, V), after each step.'''

Rules enforced here: imports limited to numpy, math and dlib; no file, process or
introspection builtins; finite bounds with lo < init < hi; at most MAX_PARAMS
constants in total.  This is a correctness filter, not a security boundary --
run untrusted candidates in a container.
"""
import ast
import math
import types

import numpy as np

from . import dlib

ALLOWED_IMPORTS = {"numpy", "math", "dlib"}
FORBIDDEN_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "input", "breakpoint",
                   "getattr", "setattr", "delattr", "os", "sys", "subprocess", "socket", "pathlib", "shutil", "importlib"}
MAX_PARAMS = 16
MAX_SOURCE_CHARS = 20000


class ContractError(Exception):
    pass


def check_source(src: str) -> ast.Module:
    if len(src) > MAX_SOURCE_CHARS:
        raise ContractError(f"source longer than {MAX_SOURCE_CHARS} characters")
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        raise ContractError(f"syntax error: {e}") from None
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    raise ContractError(f"import of '{a.name}' not allowed (allowed: numpy, math, dlib)")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                raise ContractError(f"import from '{node.module}' not allowed")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise ContractError(f"name '{node.id}' not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ContractError(f"dunder attribute '{node.attr}' not allowed")
    names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assigns = {t.id for n in tree.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
    for req in ("PARAMS", "THETA"):
        if req not in assigns:
            raise ContractError(f"missing top-level {req} dict")
    if "simulate" not in names:
        raise ContractError("missing function simulate(geo, inputs, theta, p, T0)")
    return tree


def complexity(tree: ast.Module) -> int:
    """AST node count, docstrings and the PARAMS/THETA tables excluded."""
    n = 0
    for top in tree.body:
        if isinstance(top, ast.Expr) and isinstance(getattr(top, "value", None), ast.Constant):
            continue
        if isinstance(top, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ("PARAMS", "THETA") for t in top.targets):
            continue
        if isinstance(top, (ast.Import, ast.ImportFrom)):
            continue
        for node in ast.walk(top):
            if isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant) and isinstance(node.value.value, str):
                continue
            n += 1
    return n


def _check_table(name, table):
    if not isinstance(table, dict):
        raise ContractError(f"{name} must be a dict")
    for k, v in table.items():
        if not (isinstance(v, (tuple, list)) and len(v) == 3):
            raise ContractError(f"{name}['{k}'] must be (init, lo, hi)")
        init, lo, hi = map(float, v)
        if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi and lo <= init <= hi):
            raise ContractError(f"{name}['{k}'] needs finite lo <= init <= hi, got {v}")


def load(src: str):
    """Validate and execute candidate source; return (module, complexity)."""
    tree = check_source(src)
    mod = types.ModuleType("candidate")
    safe_builtins = {k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k)
                     for k in ("abs", "min", "max", "range", "len", "sum", "float", "int", "dict", "list", "tuple", "zip",
                               "enumerate", "sorted", "set", "isinstance", "ValueError", "round", "any", "all", "map", "print",
                               "bool", "str", "Exception", "ZeroDivisionError")}

    def _imp(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "numpy":
            return np
        if name == "math":
            return math
        if name == "dlib":
            return dlib
        raise ImportError(name)
    safe_builtins["__import__"] = _imp
    mod.__dict__["__builtins__"] = safe_builtins
    exec(compile(tree, "<candidate>", "exec"), mod.__dict__)
    _check_table("PARAMS", mod.PARAMS)
    _check_table("THETA", mod.THETA)
    if len(mod.PARAMS) + len(mod.THETA) > MAX_PARAMS:
        raise ContractError(f"at most {MAX_PARAMS} constants in PARAMS + THETA")
    return mod, complexity(tree)
