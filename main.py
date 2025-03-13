from __future__ import annotations

import copy
import os
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Callable

ANSII_FG_GREEN = "\033[92m"
ANSII_FG_RED = "\033[91m"
ANSII_RESET = "\033[0m"


class TokenKind(Enum):
    INVALID = 0
    IDENTIFIER = 1
    TRUE = 2
    FALSE = 3
    LEFT_PAREN = 4
    RIGHT_PAREN = 5
    COMMA = 6
    COLON = 7
    COMMENT = 8
    EQUALS = 9
    NOT_EQUALS = 10
    GREATER = 11
    GREATER_OR_EQUAL = 12
    LESS = 13
    LESS_OR_EQUAL = 14
    PLUS = 15
    MINUS = 16
    MULTIPLY = 17
    DIVIDE = 18
    MODULO = 19
    EXPONENT = 20

    NOT = 100
    AND = 101
    OR = 102
    LEFT_IMPLIES = 103
    RIGHT_IMPLIES = 104
    EQUIV = 105

    FOR_ALL = 200
    EXISTS = 201
    WHERE = 202
    IN = 203

    DIRECTIVE = 300

    EOF = 999

    def __repr__(self) -> str:
        return super().__repr__().split(":")[0][1:]


@dataclass
class Token:
    kind: TokenKind
    source: str

    def is_quantifier(self) -> bool:
        return self.kind in [TokenKind.FOR_ALL, TokenKind.EXISTS]

    def is_logical_binary_op(self) -> bool:
        return self.kind in [
            TokenKind.AND,
            TokenKind.OR,
            TokenKind.LEFT_IMPLIES,
            TokenKind.RIGHT_IMPLIES,
            TokenKind.EQUIV,
        ]

    def is_binary_op(self) -> bool:
        return self.kind in [
            TokenKind.EQUALS,
            TokenKind.NOT_EQUALS,
            TokenKind.GREATER,
            TokenKind.GREATER_OR_EQUAL,
            TokenKind.LESS,
            TokenKind.LESS_OR_EQUAL,
            TokenKind.AND,
            TokenKind.OR,
            TokenKind.PLUS,
            TokenKind.MINUS,
            TokenKind.MULTIPLY,
            TokenKind.DIVIDE,
            TokenKind.MODULO,
            TokenKind.EXPONENT,
        ]

    def is_logical_unary_op(self) -> bool:
        return self.kind == TokenKind.NOT or self.kind == TokenKind.MINUS

    def is_unary_op(self) -> bool:
        return self.kind == TokenKind.NOT or self.kind == TokenKind.MINUS

    def priority_logical(self) -> int:
        match self.kind:
            case TokenKind.NOT | TokenKind.MINUS:
                return 3
            case TokenKind.FOR_ALL:
                return 3
            case TokenKind.EXISTS:
                return 3
            case TokenKind.AND:
                return 2
            case TokenKind.OR:
                return 1
            case TokenKind.LEFT_IMPLIES:
                return 0
            case TokenKind.RIGHT_IMPLIES:
                return 0
            case TokenKind.EQUIV:
                return 0
            case _:
                return -1

    def priority_regular(self) -> int:
        match self.kind:
            case TokenKind.NOT:
                return 3
            case TokenKind.EXPONENT:
                return 3
            case TokenKind.EQUALS:
                return 2
            case TokenKind.NOT_EQUALS:
                return 2
            case TokenKind.GREATER:
                return 2
            case TokenKind.GREATER_OR_EQUAL:
                return 2
            case TokenKind.LESS:
                return 2
            case TokenKind.LESS_OR_EQUAL:
                return 2
            case TokenKind.MULTIPLY:
                return 2
            case TokenKind.DIVIDE:
                return 2
            case TokenKind.MODULO:
                return 2
            case TokenKind.AND:
                return 1
            case TokenKind.PLUS:
                return 1
            case TokenKind.MINUS:
                return 1
            case TokenKind.OR:
                return 0
            case _:
                return -1


def match_start(source: str, patterns: list[str]) -> str | None:
    for pattern in patterns:
        if source.startswith(pattern):
            return pattern

    return None


class Tokenizer:
    source: str
    position: int

    def __init__(self, source: str):
        self.source = source
        self.position = 0
        self.read_position = 0

    def __peek_char(self) -> str | None:
        if self.position >= len(self.source):
            return None
        return self.source[self.position]

    def __skip_whitespace(self):
        while self.position < len(self.source) and self.source[self.position].isspace():
            self.position += 1

    def __read_identifier(self) -> str:
        start = self.position
        while self.position < len(self.source):
            c = self.source[self.position]
            if c.isalpha() or c.isdigit() or (c in ["_"]):
                self.position += 1
            else:
                break

        return self.source[start : self.position]

    def next_token(self) -> Token:
        self.__skip_whitespace()
        if self.position >= len(self.source):
            return Token(TokenKind.EOF, "")

        c = self.source[self.position]
        match c:
            case x if x.isalpha() or x.isdigit() or x in ["_"]:
                ident = self.__read_identifier()
                if ident == "true":
                    return Token(TokenKind.TRUE, ident)
                elif ident == "false":
                    return Token(TokenKind.FALSE, ident)
                elif ident == "in":
                    return Token(TokenKind.IN, ident)
                else:
                    return Token(TokenKind.IDENTIFIER, ident)
            case "0":
                self.position += 1
                return Token(TokenKind.FALSE, "0")
            case "1":
                self.position += 1
                return Token(TokenKind.TRUE, "1")
            case "(":
                self.position += 1
                return Token(TokenKind.LEFT_PAREN, c)
            case ")":
                self.position += 1
                return Token(TokenKind.RIGHT_PAREN, c)
            case ",":
                self.position += 1
                return Token(TokenKind.COMMA, c)
            case ":":
                self.position += 1
                return Token(TokenKind.COLON, c)
            case "+":
                self.position += 1
                return Token(TokenKind.PLUS, c)
            case "*":
                self.position += 1
                return Token(TokenKind.MULTIPLY, c)
            case "%":
                self.position += 1
                return Token(TokenKind.MODULO, c)
            case "^":
                self.position += 1
                return Token(TokenKind.EXPONENT, c)
            case "-":
                self.position += 1
                if (
                    self.position + 1 < len(self.source)
                    and self.source[self.position] == ">"
                ):
                    self.position += 1
                    return Token(TokenKind.RIGHT_IMPLIES, c + ">")

                return Token(TokenKind.MINUS, c)
            case "=":
                self.position += 1
                if self.position < len(self.source):
                    next_char = self.source[self.position]
                else:
                    next_char = None

                if next_char == "=":
                    self.position += 1
                    return Token(TokenKind.EQUALS, "==")
                elif next_char == ">":
                    self.position += 1
                    return Token(TokenKind.RIGHT_IMPLIES, "=>")

                return Token(TokenKind.INVALID, c)
            case "!":
                self.position += 1
                if self.position < len(self.source):
                    next_char = self.source[self.position]
                else:
                    next_char = None

                if next_char == "=":
                    self.position += 1
                    return Token(TokenKind.NOT_EQUALS, "!=")

                return Token(TokenKind.NOT, c)
            case "<":
                self.position += 1

                next_char = self.__peek_char()
                if next_char == "=":
                    self.position += 1

                    if self.__peek_char() == ">":
                        self.position += 1
                        return Token(TokenKind.EQUIV, "<=>")

                    return Token(TokenKind.LESS_OR_EQUAL, "<=")
                elif next_char == "-":
                    self.position += 1

                    if self.__peek_char() == ">":
                        self.position += 1
                        return Token(TokenKind.EQUIV, "<->")

                    return Token(TokenKind.LEFT_IMPLIES, "<-")

                return Token(TokenKind.LESS, "<")
            case ">":
                self.position += 1
                if self.__peek_char() == "=":
                    self.position += 1
                    return Token(TokenKind.GREATER_OR_EQUAL, ">=")

                return Token(TokenKind.GREATER, ">")
            case "¬" | "~":
                self.position += 1
                return Token(TokenKind.NOT, c)
            case "&" | "∧":
                if (
                    self.position + 1 < len(self.source)
                    and self.source[self.position + 1] == "&"
                ):
                    self.position += 2
                    return Token(TokenKind.AND, "&&")

                self.position += 1
                return Token(TokenKind.AND, c)
            case "|" | "∨":
                if (
                    self.position + 1 < len(self.source)
                    and self.source[self.position + 1] == "|"
                ):
                    self.position += 2
                    return Token(TokenKind.OR, "||")

                self.position += 1
                return Token(TokenKind.OR, c)
            case "∃":
                self.position += 1
                return Token(TokenKind.EXISTS, c)
            case "∀":
                self.position += 1
                return Token(TokenKind.FOR_ALL, c)
            case "@":
                self.position += 1
                ident = self.__read_identifier()
                if ident in ["forall"]:
                    return Token(TokenKind.FOR_ALL, "@" + ident)
                elif ident in ["exists"]:
                    return Token(TokenKind.EXISTS, "@" + ident)
                elif ident in ["where"]:
                    return Token(TokenKind.WHERE, "@" + ident)
                else:
                    return Token(TokenKind.INVALID, "@" + ident)
            case "#":
                self.position += 1
                ident = self.__read_identifier()
                return Token(TokenKind.DIRECTIVE, "#" + ident)
            case "↔" | "⇔":
                self.position += 1
                return Token(TokenKind.EQUIV, c)
            case "→" | "⇒":
                self.position += 1
                return Token(TokenKind.RIGHT_IMPLIES, c)
            case "←" | "⇐":
                self.position += 1
                return Token(TokenKind.LEFT_IMPLIES, c)
            case "/":
                start = self.position
                self.position += 1
                if self.__peek_char() == "/":
                    self.position += 1
                    while (
                        self.position < len(self.source)
                        and self.source[self.position] != "\n"
                    ):
                        self.position += 1
                    return Token(TokenKind.COMMENT, self.source[start : self.position])
                return Token(TokenKind.DIVIDE, "/")
            case other:
                self.position += 1
                return Token(TokenKind.INVALID, other)

    def tokenize_all(self) -> list[Token]:
        tokens = []
        while (token := self.next_token()).kind != TokenKind.EOF:
            tokens.append(token)

        return tokens


@dataclass
class NodeBase:
    token: Token


@dataclass
class IdentifierNode(NodeBase):
    dynamic_param: bool = False


@dataclass
class BinOpNode(NodeBase):
    left: Node
    right: Node


@dataclass
class UnaryOpNode(NodeBase):
    child: Node


@dataclass
class BoolNode(NodeBase):
    value: bool


@dataclass
class DirectiveNode(NodeBase):
    params: list[Node]


@dataclass
class WhereClauseNode(NodeBase):
    condition: Node
    child: Node


class QuantifierKind(Enum):
    FOR_ALL = 0
    EXISTS = 1

    def __repr__(self) -> str:
        return super().__repr__().split(":")[0][1:]


@dataclass
class QuantifierNode(NodeBase):
    kind: QuantifierKind
    variables: list[IdentifierNode]
    directive: Node | None
    child: Node


@dataclass
class PredicateNode(NodeBase):
    params: list[IdentifierNode]


@dataclass
class AtomicFormula(NodeBase):
    pass


Node = (
    IdentifierNode
    | BinOpNode
    | UnaryOpNode
    | BoolNode
    | DirectiveNode
    | WhereClauseNode
    | QuantifierNode
    | PredicateNode
    | AtomicFormula
)


@dataclass
class Ast:
    atoms: set[tuple[str, int]]
    symbols: set[str]
    expressions: list[Node]


class Parser:
    tokens: list[Token]
    position: int

    _atoms: set[tuple[str, int]]
    _symbols: set[str]
    _dynamic_params: list[str]

    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.position = 0
        self._atoms = set()
        self._symbols = set()
        self._dynamic_params = []

    def __peek(self, distance=0) -> Token:
        if self.position + distance >= len(self.tokens):
            return Token(TokenKind.EOF, "")
        return self.tokens[self.position + distance]

    def __advance(self):
        self.position += 1

    def __parse_quantifier(self) -> Node:
        token = self.__peek()
        if token.kind == TokenKind.FOR_ALL:
            kind = QuantifierKind.FOR_ALL
        elif token.kind == TokenKind.EXISTS:
            kind = QuantifierKind.EXISTS
        else:
            assert False, "Invalid quantifier"

        self.__advance()

        variables: list[IdentifierNode] = []
        while self.__peek().kind == TokenKind.IDENTIFIER:
            variables.append(IdentifierNode(self.__peek()))
            self.__advance()

            self._dynamic_params.append(variables[-1].token.source)

            if self.__peek().kind == TokenKind.COMMA:
                self.__advance()
                continue
            else:
                break

        if self.__peek().kind == TokenKind.IN:
            self.__advance()
            directive = self.__parse_directive()
        else:
            directive = None

        if self.__peek().kind == TokenKind.COLON:
            self.__advance()

        child = self.__parse_logical_expression(token.priority_logical())

        for _ in variables:
            self._dynamic_params.pop()

        return QuantifierNode(token, kind, variables, directive, child)

    def __parse_atom(self) -> Node:
        name = self.__peek()
        assert name.kind == TokenKind.IDENTIFIER, "Expected identifier got " + str(name)
        self.__advance()

        if self.__peek().kind != TokenKind.LEFT_PAREN:
            self._atoms.add((name.source, 0))
            return AtomicFormula(name)
        self.__advance()

        params = []
        while self.__peek().kind != TokenKind.RIGHT_PAREN:
            param = self.__parse_expression()
            params.append(param)

            if self.__peek().kind == TokenKind.COMMA:
                self.__advance()
        self.__advance()

        self._atoms.add((name.source, len(params)))
        return PredicateNode(name, params)

    def __parse_directive(self) -> Node:
        token = self.__peek()
        assert token.kind == TokenKind.DIRECTIVE, "Expected directive"
        self.__advance()

        if self.__peek().kind == TokenKind.LEFT_PAREN:
            self.__advance()
            params = []
            while self.__peek().kind != TokenKind.RIGHT_PAREN:
                params.append(self.__parse_expression())
                next_token = self.__peek()
                if next_token.kind == TokenKind.COMMA:
                    self.__advance()
                    continue
                elif next_token.kind == TokenKind.RIGHT_PAREN:
                    break
                else:
                    raise Exception(f"Invalid directive parameters: {next_token}")
            self.__advance()
        else:
            params = []

        # TODO: Refactor this
        match token.source:
            case "#symbols":
                for param in params:
                    match param:
                        case IdentifierNode(token, _):
                            self._symbols.add(param.token.source)
                        case DirectiveNode(token, directive_params):
                            match token.source:
                                case "#range":
                                    values = [
                                        int(p.token.source) for p in directive_params
                                    ]
                                    for value in range(*values):
                                        self._symbols.add(str(value))
            case _:
                pass

        return DirectiveNode(token, params)

    def __parse_where_clause(self) -> Node:
        token = self.__peek()
        assert token.kind == TokenKind.WHERE, "Expected where"
        self.__advance()

        condition = self.__parse_expression()

        if self.__peek().kind == TokenKind.COLON:
            self.__advance()

        child = self.__parse_logical_expression()
        return WhereClauseNode(token, condition, child)

    def __parse_logical_expression(self, current_priority=0) -> Node:
        token = self.__peek()
        if token.is_quantifier():
            child = self.__parse_quantifier()
        elif token.kind == TokenKind.WHERE:
            child = self.__parse_where_clause()
        elif token.kind == TokenKind.IDENTIFIER:
            child = self.__parse_atom()
        elif token.is_logical_unary_op():
            self.__advance()
            child = UnaryOpNode(
                token, self.__parse_logical_expression(token.priority_logical())
            )
        elif token.kind == TokenKind.LEFT_PAREN:
            self.__advance()
            child = self.__parse_logical_expression(0)
            if self.__peek().kind != TokenKind.RIGHT_PAREN:
                raise Exception("Expected right paren")
            self.__advance()
        elif token.kind == TokenKind.DIRECTIVE:
            return self.__parse_directive()
        else:
            raise Exception(f"Invalid expression got: {token}")

        op = self.__peek()
        while op.is_logical_binary_op() and op.priority_logical() >= current_priority:
            self.__advance()
            second_child = self.__parse_logical_expression(op.priority_logical())
            child = BinOpNode(op, child, second_child)
            op = self.__peek()

        assert child is not None

        return child

    def __parse_expression(self, current_priority=0) -> Node:
        token = self.__peek()
        if token.kind == TokenKind.IDENTIFIER:
            self.__advance()
            child = IdentifierNode(token)
            if token.source in self._dynamic_params:
                child.dynamic_param = True
            else:
                self._symbols.add(token.source)
        elif token.kind == TokenKind.LEFT_PAREN:
            self.__advance()
            child = self.__parse_expression(0)
            if self.__peek().kind != TokenKind.RIGHT_PAREN:
                raise Exception("Expected right paren")
            self.__advance()
        elif token.kind == TokenKind.TRUE:
            self.__advance()
            child = BoolNode(token, True)
        elif token.kind == TokenKind.FALSE:
            self.__advance()
            child = BoolNode(token, False)
        elif token.kind == TokenKind.DIRECTIVE:
            return self.__parse_directive()
        elif token.is_unary_op():
            self.__advance()
            unary_child = self.__parse_expression(token.priority_regular())
            child = UnaryOpNode(token, unary_child)
        else:
            raise Exception("Invalid expression")

        op = self.__peek()
        while op.is_binary_op() and op.priority_regular() >= current_priority:
            self.__advance()
            second_child = self.__parse_expression(op.priority_regular())
            child = BinOpNode(op, child, second_child)
            op = self.__peek()

        return child

    def parse(self) -> Ast:
        nodes = []
        while self.__peek().kind != TokenKind.EOF:
            nodes.append(self.__parse_logical_expression())
        return Ast(self._atoms, self._symbols, nodes)


Syntax = Callable[[Token], str]


def syntax_ascii(token: Token) -> str:
    match token.kind:
        case TokenKind.INVALID:
            return "??"
        case TokenKind.IDENTIFIER:
            return token.source
        case TokenKind.TRUE:
            return "True"
        case TokenKind.FALSE:
            return "False"
        case TokenKind.LEFT_PAREN:
            return "("
        case TokenKind.RIGHT_PAREN:
            return ")"
        case TokenKind.COMMA:
            return ","
        case TokenKind.COLON:
            return ":"
        case TokenKind.COMMENT:
            return "//"
        case TokenKind.EQUALS:
            return "=="
        case TokenKind.NOT_EQUALS:
            return "!="
        case TokenKind.GREATER:
            return ">"
        case TokenKind.GREATER_OR_EQUAL:
            return ">="
        case TokenKind.LESS:
            return "<"
        case TokenKind.LESS_OR_EQUAL:
            return "<="
        case TokenKind.PLUS:
            return "+"
        case TokenKind.MINUS:
            return "-"
        case TokenKind.MULTIPLY:
            return "*"
        case TokenKind.DIVIDE:
            return "/"
        case TokenKind.MODULO:
            return "%"
        case TokenKind.EXPONENT:
            return "**"
        case TokenKind.NOT:
            return "!"
        case TokenKind.AND:
            return "&&"
        case TokenKind.OR:
            return "||"
        case TokenKind.LEFT_IMPLIES:
            return "<-"
        case TokenKind.RIGHT_IMPLIES:
            return "->"
        case TokenKind.EQUIV:
            return "<->"
        case TokenKind.FOR_ALL:
            return "@forall"
        case TokenKind.EXISTS:
            return "@exists"
        case TokenKind.WHERE:
            return "@where"
        case TokenKind.IN:
            return "in"
        case TokenKind.DIRECTIVE:
            return token.source
        case TokenKind.EOF:
            return ""


def node_to_formal_string(node: Node, syntax: Syntax) -> str:
    match node:
        case IdentifierNode(token, _):
            return syntax(token)
        case BoolNode(token, _):
            return syntax(token)
        case UnaryOpNode(token, child):
            return syntax(token) + node_to_formal_string(child, syntax)
        case BinOpNode(token, left, right):
            return (
                "("
                + node_to_formal_string(left, syntax)
                + " "
                + syntax(token)
                + " "
                + node_to_formal_string(right, syntax)
                + ")"
            )
        case DirectiveNode(token, params):
            if len(params) == 0:
                return syntax(token)

            return (
                syntax(token)
                + "("
                + ", ".join([node_to_formal_string(param, syntax) for param in params])
                + ")"
            )

        case QuantifierNode(token, _, variables, directive, child):
            child_str = node_to_formal_string(child, syntax)
            if child_str[0] != "(":
                child_str = f"({child_str})"

            if directive is not None:
                return f"{syntax(token)} {', '.join([node_to_formal_string(var, syntax) for var in variables])} in {node_to_formal_string(directive, syntax)}{child_str}"
            return f"{syntax(token)} {', '.join([node_to_formal_string(var, syntax) for var in variables])} {child_str}"
        case WhereClauseNode(token, condition, child):
            return f"{syntax(token)}({node_to_formal_string(condition, syntax)}) ({node_to_formal_string(child, syntax)})"
        case PredicateNode(name, params):
            return f"{syntax(name)}({', '.join([node_to_formal_string(param, syntax) for param in params])})"
        case AtomicFormula(token):
            return syntax(token)


AstTransformer = Callable[[Node], Node]


def node_transform(node: Node, transformers: list[AstTransformer]) -> Node:
    for transformer in transformers:
        node = transformer(node)

    match node:
        case IdentifierNode(_, _):
            return node
        case BinOpNode(_, left, right) as bin_op:
            bin_op.left = node_transform(left, transformers)
            bin_op.right = node_transform(right, transformers)
            return bin_op
        case UnaryOpNode(_, child) as unary_op:
            unary_op.child = node_transform(child, transformers)
            return unary_op
        case BoolNode(_, _):
            return node
        case DirectiveNode(_, _):
            return node
        case WhereClauseNode(_, condition, child) as where:
            where.condition = node_transform(condition, transformers)
            where.child = node_transform(child, transformers)
            return where
        case QuantifierNode(_, _, _, _, child) as quantifier:
            quantifier.child = node_transform(child, transformers)
            return quantifier
        case PredicateNode(_, _):
            return node
        case AtomicFormula(_):
            return node


@dataclass
class Context:
    ast: Ast
    dynamic_values: list[tuple[str, str]]


def python_expr_regular(ctx: Context, expression: Node) -> str:
    match expression:
        case IdentifierNode(token):
            return token.source
        case BoolNode(token, value):
            return str(value)
        case UnaryOpNode(token, child):
            match token.kind:
                case TokenKind.NOT:
                    return f"not ({python_expr_regular(ctx, child)})"
                case _:
                    raise Exception("Invalid unary operator")
        case BinOpNode(token, left, right):
            match token.kind:
                case TokenKind.EQUALS:
                    op = "=="
                case TokenKind.NOT_EQUALS:
                    op = "!="
                case TokenKind.GREATER:
                    op = ">"
                case TokenKind.GREATER_OR_EQUAL:
                    op = ">="
                case TokenKind.LESS:
                    op = "<"
                case TokenKind.LESS_OR_EQUAL:
                    op = "<="
                case TokenKind.AND:
                    op = "and"
                case TokenKind.OR:
                    op = "or"
                case TokenKind.PLUS:
                    op = "+"
                case TokenKind.MINUS:
                    op = "-"
                case TokenKind.MULTIPLY:
                    op = "*"
                case TokenKind.DIVIDE:
                    op = "/"
                case TokenKind.MODULO:
                    op = "%"
                case TokenKind.EXPONENT:
                    op = "**"
                case _:
                    raise Exception("Invalid binary operator")

            left = python_expr_regular(ctx, left)
            right = python_expr_regular(ctx, right)
            return f"({left} {op} {right})"
        case DirectiveNode(token, child):
            match token.source:
                case "#range":
                    params = [python_expr_regular(ctx, param) for param in child]
                    return f"range({', '.join(params)})"
                case _:
                    raise Exception("Invalid directive")
        case _:
            raise Exception("Invalid node")


def z3_expr_logical(ctx: Context, expression: Node) -> str:
    match expression:
        case IdentifierNode(token, dynamic_param):
            if dynamic_param:
                for name, value in ctx.dynamic_values[::-1]:
                    if token.source == name:
                        return value
            return token.source
        case DirectiveNode(token, params):
            return ""
        case PredicateNode(name, params):
            return f"{name.source}({', '.join([python_expr_regular(ctx, param) for param in params])})"
        case AtomicFormula(token):
            return token.source
        case UnaryOpNode(token, child):
            match token.kind:
                case TokenKind.NOT | TokenKind.MINUS:
                    return f"Not({z3_expr_logical(ctx, child)})"
                case _:
                    raise Exception("Invalid unary operator")
        case BinOpNode(token, left, right):
            match token.kind:
                case TokenKind.AND:
                    return f"And({z3_expr_logical(ctx, left)}, {z3_expr_logical(ctx, right)})"
                case TokenKind.OR:
                    return f"Or({z3_expr_logical(ctx, left)}, {z3_expr_logical(ctx, right)})"
                case TokenKind.LEFT_IMPLIES:
                    return f"Implies({z3_expr_logical(ctx, right)}, {z3_expr_logical(ctx, left)})"
                case TokenKind.RIGHT_IMPLIES:
                    return f"Implies({z3_expr_logical(ctx, left)}, {z3_expr_logical(ctx, right)})"
                case TokenKind.EQUIV:
                    return (
                        f"{z3_expr_logical(ctx, left)} == {z3_expr_logical(ctx, right)}"
                    )
                case _:
                    raise Exception(f"Invalid binary operator: {token}")
        case QuantifierNode(token, kind, variables, directive, child):
            variable_names = [variable.token.source for variable in variables]
            ctx.dynamic_values.extend(zip(variable_names, variable_names))
            child = z3_expr_logical(ctx, child)

            if directive is not None:
                iterable = python_expr_regular(ctx, directive)
            else:
                iterable = "all_symbols"

            match kind:
                case QuantifierKind.FOR_ALL:
                    result = f"forall({iterable}, {len(variables)}, lambda {', '.join(variable_names)}: {child})"
                case QuantifierKind.EXISTS:
                    result = f"exists({iterable}, {len(variables)}, lambda {', '.join(variable_names)}: {child})"
            ctx.dynamic_values = ctx.dynamic_values[: -len(variables)]
            return result
        case WhereClauseNode(token, condition, child):
            condition = python_expr_regular(ctx, condition)
            child = z3_expr_logical(ctx, child)
            return f"where({condition}, lambda: {child})"
        case _:
            raise Exception("Invalid node")


z3_template = r"""
import itertools
import sys
from z3 import And, Not, Bool, Implies, Or, Solver

%%%

# -------------------
# Helper function
# -------------------

def Equiv(a, b):
    return And(Implies(a, b), Implies(b, a))


def forall(all_symbols, repeat, fn):
    values = []
    for sym in itertools.product(all_symbols, repeat=repeat):
        sub = fn(*sym)
        if sub is not None:
            values.append(sub)
    return And(values)


def exists(all_symbols, repeat, fn):
    values = []
    for sym in itertools.product(all_symbols, repeat=repeat):
        sub = fn(*sym)
        if sub is not None:
            values.append(sub)
    return Or(values)


def where(condition, fn):
    if condition:
        return fn()

# ANSII colors
LIGHT_GREEN = "\033[92m"
LIGHT_RED = "\033[91m"
CYAN = "\033[96m"
RESET = "\033[0m"

def get_name_for_symbol_value(value):
    for sname in symbol_names:
        search = eval(sname)
        if search == value:
            return sname
    return "?"

def parse_model_key(key):
    key = str(key)
    parts = key.split("(")
    if len(parts) == 1:
        return key, None
    name, args = parts
    args = args[:-1].split(", ")

    return name, args

def format_model_key(key):
    name, args = parse_model_key(key)
    if args is None:
        return name
    return f"{name}({', '.join(args)})"

def getchar():
    import sys, tty, termios
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def solve(solver):
    solutions = 0
    while True:
        result = solver.check()

        if result.r == 1:
            print(LIGHT_GREEN + "Satisfiable" + RESET)
            solutions += 1

            inverse_to_add = []
            model = solver.model()
            keys = sorted([k for k in model], key=str)
            for key in keys:
                if model[key]:
                    inverse_to_add.append(Not(Bool(str(key))))
                else:
                    inverse_to_add.append(Bool(str(key)))
                
                formatted_key = format_model_key(key)
                print(f"{formatted_key} = {model[key]}")

            solver.add(Or(inverse_to_add))

            print(CYAN + "Press ; for next result..." + RESET, end="")
            sys.stdout.flush()

            ch = getchar()
            print("\r" + " " * 30, end="")
            print()
            if ch != ";":
                break


        else:
            print(LIGHT_RED + "Unsatisfiable" + RESET)
            print(f"Total solutions: {solutions}")
            break

if __name__ == "__main__":
    main()
"""


def python_excape_name(name: str) -> str:
    if name[0].isdigit():
        return f"_{name}"

    if name in ["and", "or", "not", "def", "return", "lambda"]:
        return f"_{name}"

    return name


def z3_generate(ctx: Context) -> str:
    ident = " " * 4
    lines = []

    # generate predicates
    letters = "abcdefghijklmnopqrstuvwxyz"
    for name, arity in sorted(ctx.ast.atoms):
        if arity == 0:
            lines.append(f"{python_excape_name(name)} = Bool('{name}')")
            continue
        str_params = ", ".join([f"str({i})" for i in letters[:arity]])
        params = ", ".join(letters[:arity])
        signature = f"def {name}({params}):"
        body = ident + f"return Bool('{name}(' + ', '.join([{str_params}]) + ')')"
        lines.extend([signature, body, ""])

    # generate symbols
    for symbol in sorted(ctx.ast.symbols):
        lines.append(f"{python_excape_name(symbol)} = '{symbol}'")

    all_symbols = (
        "set([" + ", ".join([python_excape_name(x) for x in ctx.ast.symbols]) + "])"
    )
    lines.append(f"all_symbols = {all_symbols}")
    symbol_names = ", ".join([f"'{python_excape_name(s)}'" for s in ctx.ast.symbols])
    lines.append(f"symbol_names = [{symbol_names}]")

    lines.append("")

    # main function
    lines.append("def main():")
    fn_body = [
        "solver = Solver()",
        "",
    ]
    for node in ctx.ast.expressions:
        fn_body.append(f"solver.add({z3_expr_logical(ctx, node)})")
    fn_body.append("")
    fn_body.append("solve(solver)")
    lines.extend([ident + line for line in fn_body])

    content = "\n".join(lines)
    result = z3_template.replace("%%%", content)
    return result.strip() + "\n"


class SharedCounter:
    value: int

    def __init__(self):
        self.value = 0

    def next(self) -> int:
        self.value += 1
        return self.value


@dataclass
class TableauNode:
    value: bool
    node: Node
    parent: TableauNode | None = None
    id: int = -1
    # The nodes that are present in beta rules,
    # are duplicated in alpha rules to make my life easier. So we don't print them twice.
    hide_as_alpha: bool = False

    def parent_id(self) -> int:
        if self.parent is not None:
            return self.parent.id
        return -1

    def with_id(self, id: int | SharedCounter) -> TableauNode:
        if isinstance(id, SharedCounter):
            id = id.next()
        self.id = id
        return self

    def with_new_id(self, id: int | SharedCounter) -> TableauNode:
        if self.id != -1:
            return self

        if isinstance(id, SharedCounter):
            id = id.next()
        self.id = id
        return self


def tableau_node_to_formal_string(node: TableauNode, syntax: Syntax) -> str:
    if node.value:
        prefix = "T"
    else:
        prefix = "F"
    return f"{prefix} {node_to_formal_string(node.node, syntax)}"


def tableau_rule_to_formal_string(rule: TableauRule, syntax: Syntax) -> str:
    match rule:
        case TableauAlphaRule(node) as rule:
            return f"({node.id}) Alpha {tableau_node_to_formal_string(node, syntax)}  (from: {node.parent_id()})"
        case TableauBetaRule(left, right):
            return f"({left.id}) Beta {tableau_node_to_formal_string(left, syntax)} {tableau_node_to_formal_string(right, syntax)}"


@dataclass
class TableauAlphaRule:
    node: TableauNode

    def parent_id(self) -> int:
        return self.node.parent_id()


@dataclass
class TableauBetaRule:
    left: TableauNode
    right: TableauNode

    def parent_id(self) -> int:
        assert self.left.parent_id() == self.right.parent_id()
        return self.left.parent_id()


TableauRule = TableauAlphaRule | TableauBetaRule


def tableau_rule_get_nodes(rule: TableauRule) -> list[TableauNode]:
    match rule:
        case TableauAlphaRule(node):
            return [node]
        case TableauBetaRule(left, right):
            return [left, right]


def tableau_preprocess(node: Node) -> Node:
    match node:
        case BinOpNode(Token(TokenKind.LEFT_IMPLIES, _), left, right):
            left = tableau_preprocess(left)
            right = tableau_preprocess(right)
            return BinOpNode(Token(TokenKind.RIGHT_IMPLIES, "->"), right, left)
        case BinOpNode(Token(TokenKind.EQUIV, _), left, right):
            left = tableau_preprocess(left)
            right = tableau_preprocess(right)
            return BinOpNode(
                Token(TokenKind.AND, "&&"),
                BinOpNode(Token(TokenKind.RIGHT_IMPLIES, "->"), left, right),
                BinOpNode(Token(TokenKind.RIGHT_IMPLIES, "->"), right, left),
            )
        case BinOpNode(token, left, right):
            return BinOpNode(token, tableau_preprocess(left), tableau_preprocess(right))
        case UnaryOpNode(token, child):
            return UnaryOpNode(token, tableau_preprocess(child))
        case _:
            return node


def tableau_node_expand(node: TableauNode) -> list[TableauRule]:
    match node.value, node.node:
        case (True, BinOpNode(Token(TokenKind.AND, _), left, right)):
            return [
                TableauAlphaRule(TableauNode(True, left, node)),
                TableauAlphaRule(TableauNode(True, right, node)),
            ]
        case (False, BinOpNode(Token(TokenKind.OR, _), left, right)):
            return [
                TableauAlphaRule(TableauNode(False, left, node)),
                TableauAlphaRule(TableauNode(False, right, node)),
            ]
        case (False, BinOpNode(Token(TokenKind.RIGHT_IMPLIES, _), left, right)):
            return [
                TableauAlphaRule(TableauNode(True, left, node)),
                TableauAlphaRule(TableauNode(False, right, node)),
            ]
        case (True, UnaryOpNode(Token(TokenKind.NOT, _), child)):
            return [TableauAlphaRule(TableauNode(False, child, node))]
        case (False, UnaryOpNode(Token(TokenKind.NOT, _), child)):
            return [TableauAlphaRule(TableauNode(True, child, node))]
        case (False, BinOpNode(Token(TokenKind.AND, _), left, right)):
            return [
                TableauBetaRule(
                    TableauNode(False, left, node), TableauNode(False, right, node)
                )
            ]
        case (True, BinOpNode(Token(TokenKind.OR, _), left, right)):
            return [
                TableauBetaRule(
                    TableauNode(True, left, node), TableauNode(True, right, node)
                )
            ]
        case (True, BinOpNode(Token(TokenKind.RIGHT_IMPLIES, _), left, right)):
            return [
                TableauBetaRule(
                    TableauNode(False, left, node), TableauNode(True, right, node)
                )
            ]
        case (_, AtomicFormula(_)):
            return []
        case (_, PredicateNode(_, _)):
            return []
        case other:
            raise Exception(f"Unsupported node for tableau expansion: {other}")


@dataclass
class Tableau:
    rules: list[TableauAlphaRule]
    closed: tuple[TableauNode, TableauNode] | None
    beta_rule: TableauBetaRule | None
    beta_left: Tableau | None
    beta_right: Tableau | None


def tableau_print(tableau: Tableau, syntax: Syntax, indentation=0):
    indent = " " * indentation * 4
    for rule in tableau.rules:
        if rule.node.hide_as_alpha:
            continue
        print(indent + tableau_rule_to_formal_string(rule, syntax))

    if tableau.closed is not None:
        print(
            indent + ANSII_FG_GREEN + "Closed",
            # tableau_node_to_formal_string(tableau.closed[0], syntax),
            f"({tableau.closed[0].id})",
            # tableau_node_to_formal_string(tableau.closed[1], syntax),
            f"({tableau.closed[1].id})",
            ANSII_RESET,
        )
        return

    if (
        tableau.beta_rule is not None
        and tableau.beta_left is not None
        and tableau.beta_right is not None
    ):
        print(
            indent
            + f"({tableau.beta_rule.left.id}) Beta "
            + tableau_node_to_formal_string(tableau.beta_rule.left, syntax_ascii)
            + f"  (from: {tableau.beta_rule.left.parent_id()})"
        )
        tableau_print(tableau.beta_left, syntax, indentation + 1)
        print(
            indent
            + f"({tableau.beta_rule.right.id}) Beta "
            + tableau_node_to_formal_string(tableau.beta_rule.right, syntax_ascii)
            + f"  (from: {tableau.beta_rule.right.parent_id()})"
        )
        tableau_print(tableau.beta_right, syntax, indentation + 1)
        return

    # The tableau was left open
    print(indent + ANSII_FG_RED + "Open" + ANSII_RESET)


def find_conflicting_node(
    node: TableauNode, seen_formulas: dict[str, TableauNode]
) -> TableauNode | None:
    inverse = copy.copy(node)
    inverse.value = not inverse.value
    inverse_str = tableau_node_to_formal_string(inverse, syntax_ascii)

    if inverse_str in seen_formulas:
        return seen_formulas[inverse_str]

    return None


def tableau_generate(
    nodes: list[TableauNode],
    pending_beta_rules: list[TableauBetaRule],
    seen_formulas: dict[str, TableauNode],
    id_source: SharedCounter,
) -> Tableau:
    # This has to be a shallow copy
    nodes = [copy.copy(node) for node in nodes]

    alpha_rules: list[TableauAlphaRule] = []

    for node in nodes:
        node = node.with_new_id(id_source)
        alpha_rules.append(TableauAlphaRule(node))

        if conflicting_node := find_conflicting_node(node, seen_formulas):
            return Tableau(alpha_rules, (node, conflicting_node), None, None, None)

        seen_formulas[tableau_node_to_formal_string(node, syntax_ascii)] = node

    beta_rules: list[TableauBetaRule] = [copy.copy(rule) for rule in pending_beta_rules]

    # Expand all alpha rules
    while nodes:
        node = nodes.pop(0)
        rules = tableau_node_expand(node)
        for rule in rules:
            match rule:
                case TableauAlphaRule(node):
                    node = node.with_id(id_source)
                    nodes.append(node)
                    alpha_rules.append(TableauAlphaRule(node))

                    if conflicting_node := find_conflicting_node(node, seen_formulas):
                        return Tableau(
                            alpha_rules, (node, conflicting_node), None, None, None
                        )

                    seen_formulas[tableau_node_to_formal_string(node, syntax_ascii)] = (
                        node
                    )
                case TableauBetaRule(_, _) as beta_rule:
                    beta_rules.append(beta_rule)

    # pick the firs beta rule
    if len(beta_rules) == 0:
        return Tableau(alpha_rules, None, None, None, None)

    beta_rule = beta_rules.pop(0)
    beta_rule.left = beta_rule.left.with_id(id_source)
    beta_rule.left.hide_as_alpha = True
    beta_rule.right = beta_rule.right.with_id(id_source)
    beta_rule.right.hide_as_alpha = True

    left_seen = copy.copy(seen_formulas)
    left_tableau = tableau_generate([beta_rule.left], beta_rules, left_seen, id_source)

    right_seen = copy.copy(seen_formulas)
    right_tableau = tableau_generate(
        [beta_rule.right], beta_rules, right_seen, id_source
    )

    return Tableau(alpha_rules, None, beta_rule, left_tableau, right_tableau)


def tableau_prune_rec(tableau: Tableau, required_ids: set[int]):
    if tableau.closed is not None:
        required_ids.add(tableau.closed[0].id)
        required_ids.add(tableau.closed[1].id)

    if (
        tableau.beta_rule is not None
        and tableau.beta_left is not None
        and tableau.beta_right is not None
    ):
        tableau_prune_rec(tableau.beta_left, required_ids)
        tableau_prune_rec(tableau.beta_right, required_ids)

    i = len(tableau.rules) - 1
    while i >= 0:
        rule = tableau.rules[i]

        if rule.node.id not in required_ids:
            tableau.rules.pop(i)
        else:
            required_ids.add(rule.parent_id())

        i -= 1


def tableau_run(expressions: list[Node]):
    nodes = []
    for expr in expressions:
        expr = tableau_preprocess(expr)
        print(node_to_formal_string(expr, syntax_ascii))
        nodes.append(TableauNode(True, expr))
    print("----------------")
    tableau = tableau_generate(nodes, [], dict(), SharedCounter())
    tableau_prune_rec(tableau, set())
    tableau_print(tableau, syntax_ascii)


def main():
    if len(sys.argv) != 3:
        print("Usage: python main.py <subcommand> <options>", file=sys.stderr)
        print("python3 main.py transpile <source_file>", file=sys.stderr)
        print("python3 main.py solve <source_file>", file=sys.stderr)
        sys.exit(1)

    subcommand = sys.argv[1]
    source_filename = sys.argv[2]
    with open(source_filename, "r") as f:
        source = f.read()

    tokenizer = Tokenizer(source)

    tokens = []
    while (token := tokenizer.next_token()).kind != TokenKind.EOF:
        if token.kind == TokenKind.COMMENT:
            continue
        tokens.append(token)

    parser = Parser(tokens)
    ast = parser.parse()

    ctx = Context(ast, [])

    if subcommand == "transpile":
        z3_code = z3_generate(ctx)
        print(z3_code)
    elif subcommand == "solve":
        z3_code = z3_generate(ctx)
        with open("loglang_out.py", "w") as f:
            f.write(z3_code)
        os.execvp("python3", ["python3", "loglang_out.py"])
    elif subcommand == "tableau":
        tableau_run(ast.expressions)


if __name__ == "__main__":
    main()
