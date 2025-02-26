import os
import sys
from dataclasses import dataclass
from enum import Enum


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

    NOT = 100
    AND = 101
    OR = 102
    LEFT_IMPLIES = 102
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
        ]

    def is_logical_unary_op(self) -> bool:
        return self.kind == TokenKind.NOT

    def is_unary_op(self) -> bool:
        return self.kind == TokenKind.NOT

    def priority_logical(self) -> int:
        match self.kind:
            case TokenKind.NOT:
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
            case TokenKind.AND:
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
            case "-":
                self.position += 1
                if (
                    self.position + 1 < len(self.source)
                    and self.source[self.position] == ">"
                ):
                    self.position += 1
                    return Token(TokenKind.RIGHT_IMPLIES, c + ">")

                return Token(TokenKind.NOT, c)
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
                        return Token(TokenKind.RIGHT_IMPLIES, "->")

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
                return Token(TokenKind.INVALID, "/")
            case other:
                self.position += 1
                return Token(TokenKind.INVALID, other)

    def tokenize_all(self) -> list[Token]:
        tokens = []
        while (token := self.next_token()).kind != TokenKind.EOF:
            tokens.append(token)

        return tokens


@dataclass
class Node:
    token: Token


@dataclass
class IdentifierNode(Node):
    dynamic_param: bool = False
    pass


@dataclass
class BinOpNode(Node):
    left: Node
    right: Node


@dataclass
class UnaryOpNode(Node):
    child: Node


@dataclass
class BoolNode(Node):
    value: bool


@dataclass
class DirectiveNode(Node):
    params: list[Node]
    pass


@dataclass
class WhereClauseNode(Node):
    condition: Node
    child: Node


class QuantifierKind(Enum):
    FOR_ALL = 0
    EXISTS = 1

    def __repr__(self) -> str:
        return super().__repr__().split(":")[0][1:]


@dataclass
class QuantifierNode(Node):
    kind: QuantifierKind
    variables: list[IdentifierNode]
    directive: Node | None
    child: Node


@dataclass
class PredicateNode(Node):
    params: list[IdentifierNode]


@dataclass
class Ast:
    predicates: set[tuple[str, int]]
    symbols: set[str]
    expressions: list[Node]


class Parser:
    tokens: list[Token]
    position: int

    _predicates: set[tuple[str, int]]
    _symbols: set[str]
    _dynamic_params: list[str]

    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.position = 0
        self._predicates = set()
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

    def __parse_predicate(self) -> Node:
        name = self.__peek()
        assert name.kind == TokenKind.IDENTIFIER, "Expected identifier got " + str(name)
        self.__advance()

        if self.__peek().kind != TokenKind.LEFT_PAREN:
            raise Exception("Expected left paren")
        self.__advance()

        params = []
        while self.__peek().kind == TokenKind.IDENTIFIER:
            param = self.__peek()
            params.append(IdentifierNode(param))
            self.__advance()

            if param.source in self._dynamic_params:
                params[-1].dynamic_param = True
            else:
                self._symbols.add(params[-1].token.source)

            if self.__peek().kind == TokenKind.COMMA:
                self.__advance()
                continue
            elif self.__peek().kind == TokenKind.RIGHT_PAREN:
                self.__advance()
                break
            else:
                raise Exception("Invalid predicate parameter list")

        self._predicates.add((name.source, len(params)))
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
                if self.__peek().kind == TokenKind.COMMA:
                    self.__advance()
                    continue
                elif self.__peek().kind == TokenKind.RIGHT_PAREN:
                    break
                else:
                    raise Exception("Invalid directive parameters")
            self.__advance()
        else:
            params = []

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
            child = self.__parse_predicate()
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
        return Ast(self._predicates, self._symbols, nodes)


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
                case _:
                    raise Exception("Invalid binary operator")

            left = python_expr_regular(ctx, left)
            right = python_expr_regular(ctx, right)
            return f"({left} {op} {right})"
        case DirectiveNode(token, child):
            match token.source:
                case "#range":
                    params = [python_expr_regular(ctx, param) for param in child]
                    return f"map(str, range({', '.join(params)}))"
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
        case PredicateNode(name, params):
            return f"{name.source}({', '.join([z3_expr_logical(ctx, param) for param in params])})"
        case UnaryOpNode(token, child):
            match token.kind:
                case TokenKind.NOT:
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
    name, args = key.split("(")
    args = args[:-1].split(", ")

    return name, args

def format_model_key(key):
    name, args = parse_model_key(key)
    args = [get_name_for_symbol_value(arg) for arg in args]
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


def z3_generate(ctx: Context) -> str:
    ident = " " * 4
    lines = []

    # generate predicates
    letters = "abcdefghijklmnopqrstuvwxyz"
    for name, arity in sorted(ctx.ast.predicates):
        params = ", ".join(list(letters[:arity]))
        signature = f"def {name}({params}):"
        body = ident + f"return Bool('{name}(' + ', '.join([{params}]) + ')')"
        lines.extend([signature, body, ""])

    # generate symbols
    for symbol in sorted(ctx.ast.symbols):
        lines.append(f"{symbol} = '{symbol}'")

    all_symbols = "set([" + ", ".join(ctx.ast.symbols) + "])"
    lines.append(f"all_symbols = {all_symbols}")
    symbol_names = ", ".join([f"'{s}'" for s in ctx.ast.symbols])
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

    z3_code = z3_generate(ctx)

    if subcommand == "transpile":
        print(z3_code)
    elif subcommand == "solve":
        with open("loglang_out.py", "w") as f:
            f.write(z3_code)
        os.execvp("python3", ["python3", "loglang_out.py"])


if __name__ == "__main__":
    main()
