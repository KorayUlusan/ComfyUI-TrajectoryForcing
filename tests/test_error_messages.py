"""Every error a user can reach has to say what to do about it.

This repo already writes good messages. The risk is not the current ones, it is
the next node: `raise ValueError("bad input")` is the natural thing to type, it
passes review because it is obviously correct, and it is useless to whoever
receives it inside a ComfyUI toast with no traceback.

So the standard is mechanical rather than aspirational. A message must

  1. be long enough to have said something, and
  2. either report the offending value or name a remedy.

Read from the source with `ast`, so a message cannot escape by living on a path
no test happens to execute -- there are 50-odd of these and only a handful are
reachable from the CPU suite.

Deliberately not asserted: tone, or that the remedy is *correct*. No test can
check that, and pretending otherwise would trade a real standard for a
comfortable one.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

TF_NODES = pathlib.Path(__file__).resolve().parent.parent / "tf_nodes"

#: Measured on the literal prose with interpolations removed, not on the raw
#: string. `f"Row {r} is outside the {h}x{w} grid."` is a good message that a
#: naive character count makes look terse, and `"Bad input."` is a poor one that
#: no amount of interpolation would save. Calibrated against the existing
#: corpus: every surviving message clears it, so this bites on new ones.
MIN_LITERAL = 20

#: A site whose text is almost entirely interpolation is *assembling* a message
#: from parts checked elsewhere -- `check_startup_problems` re-raising a
#: Problem's fields, say. There is no prose here to measure, and demanding some
#: would only produce a redundant preamble in front of the real message.
COMPOSITION_LITERAL = 10
COMPOSITION_SLOTS = 2

#: Words that introduce a remedy. A message without one has to be carrying the
#: offending value instead, which at least lets the reader see what happened.
REMEDY_WORDS = (
    "use ", "set ", "pick ", "run ", "wire ", "unwire ", "choose ", "try ",
    "install", "remove ", "check ", "connect ", "switch ", "instead", "rather than",
    "expects", "must ", "should ", "either ", "or ", "one of", "see ",
)


def message_of(call: ast.Call) -> str | None:
    """The literal text of an exception's first argument, f-strings included.

    Interpolations become `{}` so a message built from values is still
    measurable as text.
    """

    def render(node) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.JoinedStr):
            parts = []
            for value in node.values:
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    parts.append(value.value)
                else:
                    parts.append("{}")
            return "".join(parts)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left, right = render(node.left), render(node.right)
            return None if left is None or right is None else left + right
        return None

    return render(call.args[0]) if call.args else None


def raise_sites() -> list[tuple[str, int, str]]:
    """(file, line, message) for every `raise Something("...")` in tf_nodes/.

    Sites whose message is fully dynamic are skipped: they are assembled from
    values checked elsewhere -- `check_startup_problems` re-raising a Problem,
    for instance -- and there is no literal here to measure.
    """
    found = []
    for path in sorted(TF_NODES.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Raise) or not isinstance(node.exc, ast.Call):
                continue
            text = message_of(node.exc)
            if text is None:
                continue
            found.append((path.name, node.lineno, " ".join(text.split())))
    return found


SITES = raise_sites()


def test_the_scan_found_the_messages_at_all():
    """A regex-ish scan that silently matches nothing passes every other test
    in this file. This repo has already shipped one such green-but-empty run."""
    assert len(SITES) > 40, f"only found {len(SITES)} raise sites; the AST walk is wrong"


@pytest.mark.parametrize("where,line,text", [(f, ln, t) for f, ln, t in SITES])
class TestEveryMessageIsWorthReading:
    def test_it_is_long_enough_to_have_said_something(self, where, line, text):
        literal = text.replace("{}", "")
        if text.count("{}") >= COMPOSITION_SLOTS and len(literal) < COMPOSITION_LITERAL:
            pytest.skip("composes a message checked at its source")
        assert len(literal) >= MIN_LITERAL, (
            f"{where}:{line} raises {text!r}, which is too short to say both what "
            "went wrong and what to do about it."
        )

    def test_it_reports_the_value_or_names_a_remedy(self, where, line, text):
        lowered = text.lower()
        assert "{}" in text or any(word in lowered for word in REMEDY_WORDS), (
            f"{where}:{line} raises {text!r}. Include the offending value, or say "
            "what to change. The reader sees this with no traceback and no source."
        )


class TestTheContractCatchesABadMessage:
    """The rule has to reject something, or it is decoration.

    Both halves are checked against the message a new node would most plausibly
    arrive with.
    """

    def test_a_terse_message_fails_the_length_rule(self):
        assert len("Bad input.") < MIN_LITERAL

    def test_interpolation_alone_does_not_buy_length(self):
        """`f"Bad: {value}"` is still a bad message, and the raw string is long
        enough to hide that. Measuring the prose is what stops it."""
        assert len("Bad: {}".replace("{}", "")) < MIN_LITERAL

    def test_a_padded_but_contentless_message_fails_the_remedy_rule(self):
        filler = "Something went wrong internally and the operation was aborted."
        assert len(filler) >= MIN_LITERAL
        assert "{}" not in filler
        assert not any(w in filler.lower() for w in REMEDY_WORDS)
