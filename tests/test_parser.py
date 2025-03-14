import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import main as loglang


def parse_and_serialize(text):
    tokens = loglang.Tokenizer(text).tokenize_all()
    parser = loglang.Parser(tokens)
    ast = parser.parse()
    assert ast is not None
    assert len(ast.expressions) == 1
    formal = loglang.node_to_formal_string(ast.expressions[0], loglang.syntax_ascii)
    return formal


def test_parse_atoms():
    assert parse_and_serialize("a") == "a"
    assert parse_and_serialize("dad(a)") == "dad(a)"


def test_parse_unary():
    assert parse_and_serialize("¬ a") == "!a"
    assert parse_and_serialize("¬a") == "!a"


def test_parse_binary():
    assert parse_and_serialize("a ∧ b") == "(a && b)"
    assert parse_and_serialize("a | b") == "(a || b)"
    assert parse_and_serialize("a → b") == "(a -> b)"
    assert parse_and_serialize("a ↔ b") == "(a <-> b)"
    assert parse_and_serialize("a <- b") == "(a <- b)"


def test_parse_nested_and():
    assert parse_and_serialize("a ∧ b ∧ c") == "(a && b && c)"
    assert parse_and_serialize("(a ∧ b) ∧ c") == "(a && b && c)"
    assert parse_and_serialize("(a ∧ b) ∧ (c ∧ d)") == "(a && b && c && d)"
    assert parse_and_serialize("a ∧ (b ∧ c)") == "(a && b && c)"
    assert parse_and_serialize("a ∧ b ∧ c ∧ b ∧ c") == "(a && b && c && b && c)"


def test_parse_nested_or():
    assert parse_and_serialize("a ∨ b ∨ c") == "(a || b || c)"
    assert parse_and_serialize("a ∨ b ∨ c ∨ b ∨ c") == "(a || b || c || b || c)"
    assert parse_and_serialize("(a ∨ b) ∨ c") == "(a || b || c)"
    assert parse_and_serialize("(a -> b) ∨ c") == "((a -> b) || c)"
    assert parse_and_serialize("a ∨ (b ∨ c)") == "(a || b || c)"


def test_parse_quantifiers():
    assert (
        parse_and_serialize("∀x∀y(larger(x, y)) ∧ ∀x∀y(larger(y, z))")
        == "(@forall x (@forall y (larger(x, y))) && @forall x (@forall y (larger(y, z))))"
    )


def test_complex_expressions():
    assert (
        parse_and_serialize(
            "∀x∀y(larger(x,y) → ¬larger(y,x)) ∧ ∀x∀y∀z(larger(x,y) ∧ larger(y,z)→ larger(x,z))"
        )
        == "(@forall x (@forall y (larger(x, y) -> !larger(y, x))) && @forall x (@forall y (@forall z ((larger(x, y) && larger(y, z)) -> larger(x, z)))))"
    )
