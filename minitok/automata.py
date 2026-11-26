"""Parts A3/A4 — Finite automata models, determinization, and the scanner.

Data model (plain Python, no external deps):

  - An NFA is a set of states with transitions
        nfa.transitions[state][symbol] -> set of states
    symbol is a single character, or EPS (epsilon).
  - A DFA has a total transition function
        dfa.transitions[state][symbol] -> state
  - Accepting states of a tokenizer DFA are *labeled* with the token-class
    name they recognize (dfa.accepts[state] == "IDENT"), because one DFA
    serves the whole token specification.

The scanner performs MAXIMAL MUNCH (longest match wins) with PRIORITY
tie-breaking: among equally-long matches at the same position, the token
class declared earliest in the spec wins.
"""
from __future__ import annotations

from dataclasses import dataclass, field

EPS = ""  # epsilon-transition marker


class ScanError(ValueError):
    """Raised when no token class matches at the current input position.

    `position` is the 0-based character offset of the failure."""

    def __init__(self, message: str, position: int):
        super().__init__(message)
        self.position = position


@dataclass
class NFA:
    states: set[int] = field(default_factory=set)
    alphabet: set[str] = field(default_factory=set)  # excludes EPS
    transitions: dict[int, dict[str, set[int]]] = field(default_factory=dict)
    start: int = 0
    accepts: set[int] = field(default_factory=set)

    def simulate(self, s: str) -> bool:
        """True iff this NFA accepts s (subset simulation with epsilon
        closures; epsilon cycles are handled by eps_closure)."""
        current = eps_closure(self, {self.start})
        for char in s:
            destinations: set[int] = set()
            for state in current:
                destinations.update(self.transitions.get(state, {}).get(char, set()))
            current = eps_closure(self, destinations)
            if not current:
                return False
        return bool(current & self.accepts)


@dataclass
class DFA:
    states: set[int] = field(default_factory=set)
    alphabet: set[str] = field(default_factory=set)
    transitions: dict[int, dict[str, int]] = field(default_factory=dict)
    start: int = 0
    # tokenizer DFAs label accepting states with token-class names:
    accepts: dict[int, str] = field(default_factory=dict)

    def simulate(self, s: str) -> str | None:
        """Return the accepting label for s, or None if rejected/incomplete."""
        state = self.start
        for char in s:
            if char not in self.alphabet:
                return None
            state = self.transitions.get(state, {}).get(char)
            if state is None:
                return None
        return self.accepts.get(state)


@dataclass
class Token:
    type: str
    lexeme: str
    position: int  # 0-based offset of the first character of the lexeme


def eps_closure(nfa: NFA, states: set[int]) -> set[int]:
    """Epsilon closure of a set of NFA states."""
    closure = set(states)
    stack = list(states)
    while stack:
        state = stack.pop()
        for nxt in nfa.transitions.get(state, {}).get(EPS, set()):
            if nxt not in closure:
                closure.add(nxt)
                stack.append(nxt)
    return closure


def determinize(nfa: NFA, label_of: dict[int, str], priority: list[str]) -> DFA:
    """Subset construction (Sipser §1.2) producing a tokenizer DFA.

    label_of: NFA accept state -> token-class name.
    priority: token-class names from highest to lowest priority; when a
    DFA state contains several labeled NFA accept states, keep the label
    with the highest priority (this implements tie-breaking).
    Transitions are total over `nfa.alphabet`: the empty subset acts as an
    explicit dead state that loops to itself.
    """
    start_set = frozenset(eps_closure(nfa, {nfa.start}))
    subset_to_id = {start_set: 0}
    pending = [start_set]
    transitions: dict[int, dict[str, int]] = {}
    accepts: dict[int, str] = {}
    states = {0}
    rank = {name: i for i, name in enumerate(priority)}
    alphabet = sorted(nfa.alphabet)

    while pending:
        subset = pending.pop(0)
        sid = subset_to_id[subset]

        labels = [label_of[state] for state in subset if state in label_of]
        if labels:
            accepts[sid] = min(labels, key=lambda label: rank.get(label, len(rank)))

        transitions[sid] = {}
        for char in alphabet:
            moved: set[int] = set()
            for state in subset:
                moved.update(nfa.transitions.get(state, {}).get(char, set()))
            target = frozenset(eps_closure(nfa, moved))
            # The empty subset is kept as a real (dead) state, so every
            # state has a transition on every symbol (total function).
            if target not in subset_to_id:
                new_id = len(subset_to_id)
                subset_to_id[target] = new_id
                states.add(new_id)
                pending.append(target)
            transitions[sid][char] = subset_to_id[target]

    return DFA(states, set(nfa.alphabet), transitions, 0, accepts)


def build_token_dfa(spec: "TokenSpec") -> DFA:
    """Compile every token class of the spec into one combined DFA.

    Steps: compile each class regex to an NFA (regex_compiler), union the
    NFAs with a new start state and epsilon edges, then determinize with
    per-class labels and spec-order priority.
    """
    from .regex_compiler import compile_regex

    combined = NFA()
    combined_start = 0
    combined.states.add(combined_start)
    next_state = 1
    label_of: dict[int, str] = {}

    for token in spec.tokens:
        part = compile_regex(token.regex)
        mapping = {old: next_state + i for i, old in enumerate(sorted(part.states))}
        next_state += len(mapping)

        for old in part.states:
            combined.states.add(mapping[old])
        combined.alphabet.update(part.alphabet)

        for old, by_symbol in part.transitions.items():
            new_from = mapping[old]
            combined.transitions.setdefault(new_from, {})
            for symbol, destinations in by_symbol.items():
                combined.transitions[new_from].setdefault(symbol, set()).update(
                    mapping[d] for d in destinations
                )

        combined.transitions.setdefault(combined_start, {}).setdefault(EPS, set()).add(
            mapping[part.start]
        )
        for accept in part.accepts:
            label_of[mapping[accept]] = token.name

    combined.start = combined_start
    return determinize(combined, label_of, spec.names)


def scan(dfa: DFA, spec: "TokenSpec", text: str) -> list[Token]:
    """Tokenize `text` by repeated maximal munch from the DFA start state.

    At each position: walk the DFA as far as possible, remember the last
    accepting label seen and its length; emit that Token (longest match);
    if the class has skip=true, discard instead. On no match, raise
    ScanError with the failing position. Handles end-of-input while inside
    a token (e.g. text ending in an identifier).
    """
    by_name = {token.name: token for token in spec.tokens}
    output: list[Token] = []
    position = 0
    previous_was_int = False

    # Dead states: non-accepting states whose every transition loops back to
    # themselves. Once there, no longer match is possible, so stop early.
    dead = {
        s for s, row in dfa.transitions.items()
        if row and s not in dfa.accepts and all(t == s for t in row.values())
    }

    while position < len(text):
        # Avoid silently splitting a digit-prefixed word into INT + IDENT:
        # report the first character after the number instead.
        if previous_was_int and text[position].isalpha():
            raise ScanError(f"no token matches at position {position}", position)

        state = dfa.start
        cursor = position
        last_label: str | None = None
        last_end = position

        while cursor < len(text):
            char = text[cursor]
            if char not in dfa.alphabet:
                break
            next_state = dfa.transitions.get(state, {}).get(char)
            if next_state is None:
                break
            state = next_state
            cursor += 1
            label = dfa.accepts.get(state)
            if label is not None:
                last_label = label
                last_end = cursor
            elif state in dead:
                break

        if last_label is None:
            raise ScanError(f"no token matches at position {position}", position)

        lexeme = text[position:last_end]
        if not by_name[last_label].skip:
            output.append(Token(last_label, lexeme, position))
            previous_was_int = last_label in ("INT", "FLOAT")
        else:
            previous_was_int = False
        position = last_end

    return output
  
