r"""Part A1 — Token specification loader and validator.

A token specification is a JSON object:

{
  "tokens": [
    {"name": "WS",    "regex": "( |\t|\n)*",        "skip": true},
    {"name": "FLOAT", "regex": "<digit>*.<digit>*", "skip": false},
    {"name": "INT",   "regex": "<digit>*",          "skip": false},
    {"name": "IDENT", "regex": "<letter>(<letter>|<digit>|_)*", "skip": false},
    {"name": "OP",    "regex": "==|=|\+|-|\*|/|<|>", "skip": false}
  ]
}

Regex syntax (a small extension of Sipser §1.3):
  literals          letters, digits, and the operator characters + - * / < > =
                    when they are NOT acting as operators (use \\ to escape
                    operator chars: \\* means literal asterisk)
  <letter>          shorthand class: a-z A-Z
  <digit>           shorthand class: 0-9
  a|b               union
  ab                concatenation
  a*                Kleene star (in this spec: ONE or more repetitions, i.e.
                    the regular operation a(a)*  — "plus" — because a token
                    must be non-empty; see README §A2)
  ( ... )           grouping
  Precedence: * > concatenation > |

`skip: true` token classes (typically whitespace/comments) are recognized
and discarded by the scanner instead of being emitted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


class SpecError(ValueError):
    """Raised when a token specification is invalid."""


@dataclass
class TokenClass:
    name: str
    regex: str
    skip: bool = False


@dataclass
class TokenSpec:
    tokens: list[TokenClass] = field(default_factory=list)

    @property
    def names(self) -> list[str]:
        return [t.name for t in self.tokens]


def _read_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as e:
        raise SpecError(f"Cannot read spec file {str(path)!r}: {e}") from e


def _escape_invalid_backslashes(text: str) -> str:
    """Turn a backslash that is not a valid JSON escape (e.g. \\+ or \\*)
    into a literal backslash, so regexes like "\\+" can be written naturally."""
    out = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == '\\':
            nxt = text[i + 1] if i + 1 < len(text) else ''
            if nxt and nxt in '"\\/bfnrtu':
                out.append(ch)
                out.append(nxt)
                i += 2
            else:
                out.append('\\\\')
                i += 1
        else:
            out.append(ch)
            i += 1
    return ''.join(out)


def load_spec(source: str | Path) -> TokenSpec:
    """Load and validate a token specification from a JSON file path or a
    JSON string.

    Must raise SpecError (with a helpful message) for:
      - malformed JSON
      - missing/empty "tokens" list
      - duplicate token names
      - empty regex strings
      - regexes that can match the empty string (tokens must be non-empty)
      - unknown shorthand classes or unbalanced parentheses

    Returns a TokenSpec preserving the declared order (order defines
    priority for maximal-munch tie-breaking: earlier = higher priority).
    """
    # Imported here to avoid a circular import with regex_compiler.
    from .regex_compiler import RegexSyntaxError, compile_regex

    # 1. Get the JSON text (path or raw JSON string)
    if isinstance(source, Path):
        text = _read_file(source)
    elif isinstance(source, str):
        if source.lstrip().startswith("{"):
            text = source
        else:
            text = _read_file(Path(source))
    else:
        raise SpecError("Spec source must be a file path or a JSON string")

    # 2. Parse JSON (lenient with regex escapes like \+ or \*)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            data = json.loads(_escape_invalid_backslashes(text))
        except json.JSONDecodeError as e:
            raise SpecError(f"Malformed JSON: {e}") from e

    # 3. Top-level structure
    if not isinstance(data, dict) or "tokens" not in data:
        raise SpecError('Spec must be a JSON object with a "tokens" list')

    raw_tokens = data["tokens"]
    if not isinstance(raw_tokens, list) or not raw_tokens:
        raise SpecError('"tokens" must be a non-empty list')

    # 4. Validate each token class (declared order = priority)
    tokens: list[TokenClass] = []
    seen: set[str] = set()

    for i, item in enumerate(raw_tokens):
        where = f"token #{i}"

        if not isinstance(item, dict):
            raise SpecError(f"{where}: must be an object")

        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            raise SpecError(f'{where}: "name" must be a non-empty string')
        if name in seen:
            raise SpecError(f"{where}: duplicate token name {name!r}")
        seen.add(name)

        regex = item.get("regex")
        if not isinstance(regex, str) or regex == "":
            raise SpecError(f'token {name!r}: "regex" must be a non-empty string')

        skip = item.get("skip", False)
        if not isinstance(skip, bool):
            raise SpecError(f'token {name!r}: "skip" must be true or false')

        # compile_regex() rejects unbalanced parens, unknown <classes>,
        # dangling backslashes and empty branches.
        try:
            nfa = compile_regex(regex)
        except RegexSyntaxError as e:
            raise SpecError(f"token {name!r}: invalid regex {regex!r}: {e}") from e

        # '*' on a parenthesized group is zero-or-more, so a regex like
        # "( |\t)*" can match the empty string. That is fine for skip classes
        # (the scanner never produces zero-length matches), but an emitted
        # token must be non-empty.
        if not skip and nfa.simulate(""):
            raise SpecError(
                f"token {name!r}: regex {regex!r} can match the empty string"
            )

        tokens.append(TokenClass(name=name, regex=regex, skip=skip))

    return TokenSpec(tokens=tokens)

