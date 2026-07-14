"""Unit tests for the band-math saved-expression persister (issue #625).

The expressions are a plain list of strings backed by ApplicationState, so the
persister round-trips them directly; an absent section leaves the store
untouched, and non-string junk is dropped and reported.
"""

import json

import tests.context  # noqa: F401

from wiser.project.persisters.bandmath import load_bandmath, save_bandmath


class _FakeAppState:
    def __init__(self, expressions=None):
        self._expressions = list(expressions or [])

    def get_bandmath_expressions(self):
        return list(self._expressions)

    def set_bandmath_expressions(self, expressions):
        self._expressions = list(expressions)


def _round_trip(manifest):
    return json.loads(json.dumps(manifest))


def test_expressions_round_trip():
    src = _FakeAppState(["b1 + b2", "b3 / b4"])
    manifest = {}
    save_bandmath(src, manifest)
    assert manifest["bandmath_expressions"] == ["b1 + b2", "b3 / b4"]

    dst = _FakeAppState()
    assert load_bandmath(_round_trip(manifest), dst) == []
    assert dst.get_bandmath_expressions() == ["b1 + b2", "b3 / b4"]


def test_absent_section_leaves_store_untouched():
    dst = _FakeAppState(["keep"])
    assert load_bandmath({}, dst) == []
    assert dst.get_bandmath_expressions() == ["keep"]


def test_non_string_entries_dropped():
    manifest = {"bandmath_expressions": ["ok", 5, None, "b1 + b2"]}
    dst = _FakeAppState()
    dropped = load_bandmath(manifest, dst)
    assert dst.get_bandmath_expressions() == ["ok", "b1 + b2"]
    assert dropped == [5, None]


def test_non_list_section_reported():
    dst = _FakeAppState()
    dropped = load_bandmath({"bandmath_expressions": "not-a-list"}, dst)
    assert dropped == ["not-a-list"]
    assert dst.get_bandmath_expressions() == []
