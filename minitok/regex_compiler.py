"""Part A2: parser and Thompson construction for the project's regex syntax.

Notes on '*': on a single atom (a*, <digit>*) it means one-or-more;
on a parenthesized group ((a|b)*) it means zero-or-more.
"""
from __future__ import annotations

from dataclasses import dataclass
from .automata import EPS, NFA


class RegexSyntaxError(ValueError):
    """Raised for malformed regular expressions."""


@dataclass
class _Node:
    kind: str
    value: str = ""
    children: list["_Node"] | None = None


_LETTERS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
_DIGITS = set("0123456789")


class _Parser:
    def __init__(self, regex: str):
        self.regex = regex
        self.i = 0

    def parse(self) -> _Node:
        if not self.regex:
            raise RegexSyntaxError("empty regex")
        node = self.union()
        if self.i != len(self.regex):
            raise RegexSyntaxError(f"unexpected character {self.regex[self.i]!r} at position {self.i}")
        return node

    def union(self) -> _Node:
        parts = [self.concat()]
        while self.peek() == "|":
            self.i += 1
            if self.i >= len(self.regex) or self.peek() in "|)":
                raise RegexSyntaxError("union operator must have an expression on both sides")
            parts.append(self.concat())
        return parts[0] if len(parts) == 1 else _Node("union", children=parts)

    def concat(self) -> _Node:
        parts: list[_Node] = []
        while self.i < len(self.regex) and self.peek() not in "|)":
            parts.append(self.star())
        if not parts:
            raise RegexSyntaxError("missing expression")
        return parts[0] if len(parts) == 1 else _Node("concat", children=parts)

    def star(self) -> _Node:
        node = self.atom()
        while self.peek() == "*":
            self.i += 1
            if node.kind == "group":
                node = _Node("kleene", children=node.children)
            else:
                node = _Node("plus", children=[node])
        return node

    def atom(self) -> _Node:
        if self.i >= len(self.regex):
            raise RegexSyntaxError("missing expression")
        ch = self.peek()
        if ch == "(":
            self.i += 1
            node = self.union()
            if self.peek() != ")":
                raise RegexSyntaxError("unbalanced opening parenthesis")
            self.i += 1
            return _Node("group", children=[node])
        if ch == ")":
            raise RegexSyntaxError("unbalanced closing parenthesis")
        if ch == "<":
            # Only <letter> and <digit> are shorthand classes. A bare '<' is
            # a literal (e.g. OP: ...|<|>|<=|<>). Only a purely alphabetic
            # name like <word> is rejected as an unknown shorthand.
            for name in ("<letter>", "<digit>"):
                if self.regex.startswith(name, self.i):
                    self.i += len(name)
                    return _Node("class", name)
            end = self.regex.find(">", self.i + 1)
            inner = self.regex[self.i + 1:end] if end != -1 else ""
            if end != -1 and inner.isascii() and inner.isalpha():
                raise RegexSyntaxError(f"unknown shorthand class {self.regex[self.i:end + 1]}")
            self.i += 1
            return _Node("literal", "<")
        if ch == "\\":
            self.i += 1
            if self.i >= len(self.regex):
                raise RegexSyntaxError("trailing escape")
            literal = self.regex[self.i]
            self.i += 1
            literal = {"t": "\t", "n": "\n", "r": "\r"}.get(literal, literal)
            return _Node("literal", literal)
        if ch in "|*":
            raise RegexSyntaxError(f"unexpected operator {ch!r}")
        self.i += 1
        return _Node("literal", ch)

    def peek(self) -> str:
        return self.regex[self.i] if self.i < len(self.regex) else ""


def parse(regex: str) -> _Node:
    return _Parser(regex).parse()


class _Builder:
    def __init__(self):
        self.next_state = 0
        self.transitions: dict[int, dict[str, set[int]]] = {}
        self.alphabet: set[str] = set()

    def state(self) -> int:
        value = self.next_state
        self.next_state += 1
        return value

    def edge(self, source: int, symbol: str, target: int) -> None:
        self.transitions.setdefault(source, {}).setdefault(symbol, set()).add(target)
        if symbol != EPS:
            self.alphabet.add(symbol)

    def build(self, node: _Node) -> tuple[int, int]:
        if node.kind == "group":
            return self.build((node.children or [])[0])
        if node.kind == "literal":
            start, accept = self.state(), self.state()
            self.edge(start, node.value, accept)
            return start, accept
        if node.kind == "class":
            chars = _LETTERS if node.value == "<letter>" else _DIGITS
            start, accept = self.state(), self.state()
            for char in chars:
                self.edge(start, char, accept)
            return start, accept
        children = node.children or []
        if node.kind == "concat":
            fragments = [self.build(child) for child in children]
            for (_, left_accept), (right_start, _) in zip(fragments, fragments[1:]):
                self.edge(left_accept, EPS, right_start)
            return fragments[0][0], fragments[-1][1]
        if node.kind == "union":
            start, accept = self.state(), self.state()
            for child in children:
                child_start, child_accept = self.build(child)
                self.edge(start, EPS, child_start)
                self.edge(child_accept, EPS, accept)
            return start, accept
        if node.kind == "kleene":
            child_start, child_accept = self.build(children[0])
            start, accept = self.state(), self.state()
            self.edge(start, EPS, child_start)
            self.edge(start, EPS, accept)
            self.edge(child_accept, EPS, child_start)
            self.edge(child_accept, EPS, accept)
            return start, accept
        if node.kind == "plus":
            child_start, child_accept = self.build(children[0])
            start, accept = self.state(), self.state()
            self.edge(start, EPS, child_start)
            self.edge(child_accept, EPS, child_start)
            self.edge(child_accept, EPS, accept)
            return start, accept
        raise RegexSyntaxError(f"unsupported AST node: {node.kind}")


def compile_regex(regex: str) -> NFA:
    tree = parse(regex)
    builder = _Builder()
    start, accept = builder.build(tree)
    # Make sure every state (including the accept state, which has no
    # outgoing edges) has an entry, so automata.NFA never hits a KeyError.
    for s in range(builder.next_state):
        builder.transitions.setdefault(s, {})
    states = set(range(builder.next_state))
    return NFA(states, builder.alphabet, builder.transitions, start, {accept})

