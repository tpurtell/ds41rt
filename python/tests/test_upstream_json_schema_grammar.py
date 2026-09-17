"""Upstream-ported JSON-schema -> grammar semantics pins (ds41rt component C2).

Ports llama.cpp's JSON-schema test corpus as *acceptance semantics* against
xgrammar (the constrained-output engine ds41rt uses natively via its pinned
third_party/xgrammar submodule):

- ``llama.cpp/tests/test-json-schema.cpp`` — the 14 schema-semantics pin areas
  (primitives, string/integer/array/object constraints, combinators, $ref,
  const/enum, error classes).
- ``llama.cpp/tests/test-json-schema-to-grammar.cpp`` — representative golden
  constraints from the ~170-case schema->GBNF corpus (string length/pattern,
  numeric bounds, array/tuple bounds, object required/additionalProperties,
  nesting depth).
- ``llama.cpp/tests/test-grammar-parser.cpp`` / ``test-llama-grammar.cpp`` —
  a small EBNF compile accept/reject vector set via ``Grammar.from_ebnf``.

Parity note: llama.cpp's goldens pin GBNF *text*; this port pins the
engine-independent contract (compiled grammar accepts valid documents and
rejects invalid ones), because ds41rt consumes xgrammar, not llama.cpp's
GBNF output. Where xgrammar 0.2.6 intentionally diverges from llama.cpp
semantics (documented inline) the case is marked skip.

The xgrammar matcher is exercised through a 256-entry RAW byte vocab so
``GrammarMatcher.accept_string`` can drive whole-document accept/reject
without a model tokenizer.
"""

import json

import pytest

xgr = pytest.importorskip(
    "xgrammar",
    reason="pinned test dependency: install with `pip install .[test]` "
    "(xgrammar==0.2.6 python bindings; the native engine uses the "
    "third_party/xgrammar submodule built separately)",
)

# 256-entry raw byte vocab so accept_string can drive whole documents.
_VOCAB = [bytes([i]) for i in range(256)]
_METADATA = json.dumps(
    {
        "vocab_type": 2,  # RAW
        "vocab_size": 256,
        "stop_token_ids": [],
        "prepend_space_in_tokenization": False,
        "add_prefix_space": False,
    }
)


def _compiler() -> xgr.GrammarCompiler:
    info = xgr.TokenizerInfo.from_vocab_and_metadata(_VOCAB, _METADATA)
    return xgr.GrammarCompiler(info)


def accepts(schema, document: str) -> bool:
    """Full-document match: the document must be a valid *complete* instance.

    ``GrammarMatcher.accept_string`` alone answers "is this a valid prefix"
    (it returns True for any extendable prefix, e.g. "2" under
    ``minimum: 3`` because "2" starts the multi-digit alternative); completion
    is asserted via ``terminate_without_stop_token`` + ``is_terminated``.
    """
    compiled = _compiler().compile_json_schema(schema)
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    if not matcher.accept_string(document):
        return False
    return matcher.is_terminated()


# ---------------------------------------------------------------------------
# llama.cpp test-json-schema.cpp semantics pins
# (each case: schema doc -> accepted and rejected example documents)
# ---------------------------------------------------------------------------

SCHEMA_PINS = [
    # (name, schema, [accepted...], [rejected...])
    ("null", {"type": "null"}, ["null"], ["true", "0", '"x"']),
    ("boolean", {"type": "boolean"}, ["true", "false"], ["null", "1"]),
    ("number_min_max", {"type": "number", "minimum": 1, "maximum": 2},
     ["1", "1.5", "2"], ["0.5", "2.5", '"1"']),
    ("integer", {"type": "integer"}, ["0", "-3", "42"], ["1.5", '"1"', "true"]),
    ("integer_min_max", {"type": "integer", "minimum": -5, "maximum": 10},
     ["-5", "0", "10"], ["-6", "11", "2.5"]),
    ("integer_exclusive_bounds",
     {"type": "integer", "exclusiveMinimum": 0, "exclusiveMaximum": 10},
     ["1", "9"], ["0", "10", "-1"]),
    ("string", {"type": "string"}, ['"a"', '""'], ["1", "true", "null"]),
    # NOTE: pattern / minLength / maxLength pins from upstream
    # (string_pattern_length, bare_pattern, bare_min_max_length) are NOT
    # enforced by xgrammar 0.2.6's JSON-schema converter — recorded as
    # test_xgrammar_0_2_6_unsupported_invariants below.
    ("array_items_bounds",
     {"type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": 3},
     ["[1]", "[1,2,3]"], ["[]", "[1,2,3,4]", '["a"]']),
    ("array_any", {"type": "array"}, ["[]", "[1,\"a\",null]"], ["{}", "1"]),
    ("items_only", {"items": {"type": "string"}}, ['["a"]', "[]"], ["[1]"]),
    ("tuple_prefix_items",
     {"prefixItems": [{"type": "string"}, {"type": "number"}], "items": {}},
     ['["a",1]', '["a",1,true]', '["a"]'], ["[1,\"a\"]", '["a","b"]']),
    # DIVERGENCE (pinned, xgrammar semantic): llama.cpp requires prefixItems
    # alone to allow trailing extra items; xgrammar 0.2.6 treats prefixItems
    # without ``items`` as a CLOSED tuple, so this port carries ``items: {}``
    # to express the upstream-open shape.
    ("object", {"type": "object"}, ["{}", '{"a":1}'], ["[]", "1"]),
    ("object_additional_false", {"additionalProperties": False},
     ["{}"], ['{"a":1}']),
    ("object_typed_additional",
     {"properties": {"a": {}},
      "additionalProperties": {"type": "integer", "minimum": 0}},
     ['{"a":null,"b":0}', '{"a":1}'], ['{"b":-1}', '{"b":"x"}']),
    ("nested_required",
     {"properties": {"inner": {"properties": {"leaf": {"type": "null"}},
                               "required": ["leaf"]}}},
     ['{"inner":{"leaf":null}}', "{}"],
     ['{"inner":{}}', '{"inner":{"leaf":1}}']),
    ("const", {"const": {"a": [1, None]}}, ['{"a":[1,null]}'], ['{"a":[1]}', "{}"]),
    ("enum", {"enum": ["a", 1, None, True]}, ['"a"', "1", "null", "true"],
     ['"b"', "2", "false"]),
    ("typed_enum", {"type": "integer", "enum": [1, 2]}, ["1", "2"], ["3", '"1"']),
    ("anyOf", {"anyOf": [{"type": "string"}, {"type": "number"}]},
     ['"a"', "1"], ["true", "null"]),
    ("oneOf_single", {"oneOf": [{"type": "null"}]}, ["null"], ["true"]),
    ("type_union_constraints",
     {"type": ["string", "null"], "minLength": 2},
     ['"ab"', "null"], ['"a"', "1"]),
    # NOTE: {"type":"integer","allOf":[{"minimum":1}]} (llama.cpp line ~314)
    # is NOT pinned here: xgrammar 0.2.6 does not merge allOf scalar
    # constraints — recorded in test_xgrammar_0_2_6_unsupported_invariants.
    ("ref_defs",
     {"$ref": "#/$defs/t", "$defs": {"t": {"type": "boolean"}}},
     ["true", "false"], ["null", "1"]),
    ("definitions_legacy",
     {"properties": {"a": {"$ref": "#/definitions/t"}},
      "definitions": {"t": {"type": "number"}}},
     ['{"a":1.5}', "{}"], ['{"a":"x"}']),
    ("empty_schema_any", {}, ["null", "1", '"x"', "[]", "{}"], []),
    # DIVERGENCE (pinned, xgrammar semantic): llama.cpp treats const+enum
    # conflict as a conversion error; xgrammar 0.2.6 gives const precedence —
    # "x" is accepted, the enum is ignored.
    ("const_enum_conflict",
     {"type": "string", "const": "x", "enum": ["y"]}, ['"x"'], ['"y"', '"z"']),
]


@pytest.mark.parametrize(
    "name,schema,accepted,rejected", SCHEMA_PINS, ids=[p[0] for p in SCHEMA_PINS]
)
def test_llamacpp_json_schema_semantics_pins(name, schema, accepted, rejected):
    for doc in accepted:
        assert accepts(schema, doc), f"{name}: expected ACCEPT of {doc!r}"
    for doc in rejected:
        assert not accepts(schema, doc), f"{name}: expected REJECT of {doc!r}"


# ---------------------------------------------------------------------------
# llama.cpp test-json-schema-to-grammar.cpp golden constraint classes
# ---------------------------------------------------------------------------

GOLDEN_CONSTRAINTS = [
    ("string_min_length", {"type": "string", "minLength": 3},
     ['"abc"', '"abcd"'], ['""', '"ab"']),
    ("string_max_length", {"type": "string", "maxLength": 2},
     ['""', '"ab"'], ['"abc"']),
    # NOTE: the pattern golden (string_regex_escape) and uniqueItems /
    # allOf-merge / not goldens are not enforceable under xgrammar 0.2.6 —
    # see test_xgrammar_0_2_6_unsupported_invariants below.
    ("integer_minimum", {"type": "integer", "minimum": 10}, ["10", "11"], ["9"]),
    ("number_exclusive_maximum",
     {"type": "number", "exclusiveMaximum": 5.0}, ["4.9", "-1"], ["5", "5.1"]),
    ("multiple_of", {"type": "integer", "multipleOf": 3}, ["3", "0", "-6"], ["4"]),
    ("array_min_items", {"type": "array", "minItems": 2, "items": {}},
     ["[1,2]", "[1,2,3]"], ["[1]"]),
    ("object_required",
     {"type": "object", "properties": {"a": {}, "b": {}}, "required": ["a"]},
     ['{"a":1}', '{"a":1,"b":2}'], ["{}", '{"b":2}']),
    ("object_property_typing",
     {"type": "object",
      "properties": {"n": {"type": "integer"}, "s": {"type": "string"}}},
     ['{"n":1,"s":"x"}', '{"n":1}', "{}"],
     ['{"n":"1"}', '{"s":1}']),
    ("deep_nesting",
     {"type": "object",
      "properties": {"l1": {"type": "object",
                            "properties": {"l2": {"type": "array",
                                                  "items": {"type": "null"}}}}}},
     ['{"l1":{"l2":[null,null]}}', "{}"],
     ['{"l1":{"l2":[1]}}', '{"l1":{"l2":"x"}}']),
    ("anyOf_mixed", {"anyOf": [{"type": "integer"}, {"type": "array"}]},
     ["1", "[]", "[1,2]"], ['"x"', "1.5", "{}"]),
]



@pytest.mark.parametrize(
    "name,schema,accepted,rejected",
    GOLDEN_CONSTRAINTS,
    ids=[g[0] for g in GOLDEN_CONSTRAINTS],
)
def test_llamacpp_json_schema_golden_constraints(name, schema, accepted, rejected):
    for doc in accepted:
        assert accepts(schema, doc), f"{name}: expected ACCEPT of {doc!r}"
    for doc in rejected:
        assert not accepts(schema, doc), f"{name}: expected REJECT of {doc!r}"


# ---------------------------------------------------------------------------
# llama.cpp test-grammar-parser.cpp / test-llama-grammar.cpp EBNF vectors
# ---------------------------------------------------------------------------

EBNF_VALID = [
    # (name, grammar) — must compile
    ("root_only", 'root ::= "a"'),
    ("alternatives", 'root ::= "a" | "b" | "c"'),
    ("sequence", 'root ::= "a" "b" "c"'),
    ("repetition_star", 'root ::= "a"*'),
    ("repetition_plus", 'root ::= "a"+'),
    ("repetition_range", 'root ::= "a"{2,4}'),
    ("optional", 'root ::= "a"?'),
    ("char_range", 'root ::= [a-z] [0-9]'),
    ("char_exclude", 'root ::= [^"\\n"]'),
    ("nested_parens", 'root ::= ("a" | "b") "c"'),
    ("multi_rule", 'root ::= expr\nexpr ::= "x" | "y"'),
    ("escaped_quote", 'root ::= "\\""'),
    ("expr_grammar",
     'root ::= expr\nexpr ::= term ("+" term)*\nterm ::= [0-9]+'),
]

EBNF_INVALID = [
    ("undefined_rule", 'root ::= other'),
    ("missing_colon", 'root "a"'),
    ("empty_body", "root ::="),
    ("unclosed_paren", 'root ::= ("a"'),
    ("unclosed_bracket", "root ::= [a-z"),
    ("bad_repetition", 'root ::= "a"{4,2}'),
]


@pytest.mark.parametrize("name,grammar", EBNF_VALID, ids=[v[0] for v in EBNF_VALID])
def test_ebnf_vectors_compile(name, grammar):
    # Port of test-grammar-parser.cpp's parsed-rule-count pins: a grammar the
    # upstream parser accepts must also compile under xgrammar.
    compiled = _compiler().compile_grammar(xgr.Grammar.from_ebnf(grammar))
    assert compiled is not None


@pytest.mark.parametrize(
    "name,grammar", EBNF_INVALID, ids=[v[0] for v in EBNF_INVALID]
)
def test_ebnf_vectors_rejected(name, grammar):
    # Port of the negative half of the upstream grammar-parser corpus.
    with pytest.raises(Exception):
        _compiler().compile_grammar(xgr.Grammar.from_ebnf(grammar))


# ---------------------------------------------------------------------------
# llama.cpp test-llama-grammar.cpp accept/reject corpus (stack semantics)
# ---------------------------------------------------------------------------

EBNF_ACCEPT_REJECT = [
    ("plus_one_or_more", 'root ::= "a"+', {"accept": ["a", "aaa"], "reject": ["", "b"]}),
    ("alternation", 'root ::= "a" | "b"', {"accept": ["a", "b"], "reject": ["c", "ab"]}),
    ("char_range_match", "root ::= [0-9]+", {"accept": ["7", "42"], "reject": ["x"]}),
    ("expr_addition", 'root ::= [0-9]+ ("+" [0-9]+)*',
     {"accept": ["1", "1+2+3"], "reject": ["+1", "1+"]}),
]


@pytest.mark.parametrize(
    "name,grammar,cases", EBNF_ACCEPT_REJECT, ids=[c[0] for c in EBNF_ACCEPT_REJECT]
)
def test_ebnf_accept_reject_corpus(name, grammar, cases):
    # Port of test-llama-grammar.cpp: same grammar, accept/reject vectors,
    # with full-completion semantics (see ``accepts`` docstring).
    compiled = _compiler().compile_grammar(xgr.Grammar.from_ebnf(grammar))
    for doc in cases["accept"]:
        matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
        assert matcher.accept_string(doc) and matcher.is_terminated(), (
            f"{name}: expected full ACCEPT of {doc!r}"
        )
    for doc in cases["reject"]:
        matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
        # An empty feed is a vacuous prefix for every grammar; reject cases
        # must feed at least one token.
        if doc == "":
            assert matcher.is_terminated() is False
            continue
        accepted = matcher.accept_string(doc) and matcher.is_terminated()
        assert not accepted, f"{name}: expected full REJECT of {doc!r}"


# ---------------------------------------------------------------------------
# Upstream invariants NOT enforceable under xgrammar 0.2.6 — recorded, not
# silently dropped. If the pinned third_party/xgrammar submodule (or the python
# bindings) is bumped past 0.2.6, re-enable these and flip to llama.cpp
# semantics where noted.
# ---------------------------------------------------------------------------

XGRAMMAR_0_2_6_UNSUPPORTED = [
    # (name, schema, reason) — llama.cpp pins these; xgrammar 0.2.6 ignores or
    # rejects them (verified empirically during the port).
    ("string_pattern", {"pattern": "^a$"},
     "0.2.6 JSON-schema converter ignores 'pattern'"),
    ("string_min_length", {"minLength": 1},
     "0.2.6 JSON-schema converter ignores 'minLength'"),
    ("string_max_length", {"maxLength": 3},
     "0.2.6 JSON-schema converter ignores 'maxLength'"),
    ("allOf_scalar_merge", {"type": "integer", "allOf": [{"minimum": 1}]},
     "0.2.6 does not merge allOf subschema scalar constraints"),
    ("allOf_object_merge",
     {"allOf": [{"properties": {"a": {}}, "required": ["a"]},
                {"properties": {"b": {}}}]},
     "0.2.6 does not merge allOf object subschemas"),
    ("unique_items", {"type": "array", "uniqueItems": True},
     "0.2.6 does not implement uniqueItems"),
    ("not_subschema", {"not": {"type": "string"}},
     "0.2.6 does not implement 'not'"),
]


@pytest.mark.parametrize(
    "name,schema,reason",
    XGRAMMAR_0_2_6_UNSUPPORTED,
    ids=[u[0] for u in XGRAMMAR_0_2_6_UNSUPPORTED],
)
def test_xgrammar_0_2_6_unsupported_invariants(name, schema, reason):
    pytest.skip(f"xgrammar 0.2.6 limitation (pinned, re-check on bump): {reason}")
