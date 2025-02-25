import itertools
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

    NOT = 10
    AND = 11
    OR = 12
    LEFT_IMPLIES = 13
    RIGHT_IMPLIES = 14
    EQUIV = 15

    FOR_ALL = 20
    EXISTS = 21

    FLAG = 30

    EOF = 99

    def __repr__(self) -> str:
        return super().__repr__().split(":")[0][1:]


@dataclass
class Token:
    kind: TokenKind
    source: str

    def is_quantifier(self) -> bool:
        return self.kind in [TokenKind.FOR_ALL, TokenKind.EXISTS]

    def is_binary_op(self) -> bool:
        return self.kind in [
            TokenKind.AND,
            TokenKind.OR,
            TokenKind.LEFT_IMPLIES,
            TokenKind.RIGHT_IMPLIES,
            TokenKind.EQUIV,
        ]

    def is_unary_op(self) -> bool:
        return self.kind == TokenKind.NOT

    def priority(self) -> int:
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
            case x if x.isalpha():
                ident = self.__read_identifier()
                if ident == "true":
                    return Token(TokenKind.TRUE, ident)
                elif ident == "false":
                    return Token(TokenKind.FALSE, ident)
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
            case "!" | "¬" | "~" | "-":
                self.position += 1

                if (
                    self.position + 1 < len(self.source)
                    and self.source[self.position] == ">"
                ):
                    self.position += 1
                    return Token(TokenKind.LEFT_IMPLIES, c + ">")

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
                else:
                    return Token(TokenKind.INVALID, "@" + ident)
            case "#":
                self.position += 1
                ident = self.__read_identifier()
                match ident:
                    case "expand":
                        return Token(TokenKind.FLAG, "#" + ident)
                    case _:
                        raise Exception("Unsupported flag")
            case other:
                rest = self.source[self.position :]
                if m := match_start(rest, ["<->", "↔", "<=>", "⇔"]):
                    self.position += len(m)
                    return Token(TokenKind.EQUIV, m)
                elif m := match_start(rest, ["→", "=>", "⇒"]):
                    self.position += len(m)
                    return Token(TokenKind.RIGHT_IMPLIES, m)
                elif m := match_start(rest, ["<-", "←", "<=", "⇐"]):
                    self.position += len(m)
                    return Token(TokenKind.LEFT_IMPLIES, m)
                elif m := match_start(rest, ["//"]):
                    start = self.position
                    self.position += len(m)
                    while (
                        self.position < len(self.source)
                        and self.source[self.position] != "\n"
                    ):
                        self.position += 1
                    return Token(TokenKind.COMMENT, self.source[start : self.position])

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
class FlagNode(Node):
    child: Node
    pass


class QuantifierKind(Enum):
    FOR_ALL = 0
    EXISTS = 1

    def __repr__(self) -> str:
        return super().__repr__().split(":")[0][1:]


@dataclass
class QuantifierNode(Node):
    kind: QuantifierKind
    variables: list[IdentifierNode]
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
            elif self.__peek().kind == TokenKind.COLON:
                self.__advance()
                break
            else:
                break

        child = self.__parse_expression(token.priority())

        for _ in variables:
            self._dynamic_params.pop()

        return QuantifierNode(token, kind, variables, child)

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

    def __parse_expression(self, current_priority=0) -> Node:
        token = self.__peek()
        if token.is_quantifier():
            child = self.__parse_quantifier()
        elif token.kind == TokenKind.IDENTIFIER:
            child = self.__parse_predicate()
        elif token.is_unary_op():
            self.__advance()
            child = UnaryOpNode(token, self.__parse_expression(token.priority()))
        elif token.kind == TokenKind.LEFT_PAREN:
            self.__advance()
            child = self.__parse_expression(0)
            if self.__peek().kind != TokenKind.RIGHT_PAREN:
                raise Exception("Expected right paren")
            self.__advance()
        elif token.kind == TokenKind.FLAG:
            self.__advance()
            child = self.__parse_expression(current_priority)
            return FlagNode(token, child)
        else:
            raise Exception(f"Invalid expression got: {token}")

        op = self.__peek()
        while op.is_binary_op() and op.priority() >= current_priority:
            self.__advance()
            child = BinOpNode(op, child, self.__parse_expression(op.priority()))
            op = self.__peek()

        assert child is not None

        return child

    def parse(self) -> Ast:
        nodes = []
        while self.__peek().kind != TokenKind.EOF:
            nodes.append(self.__parse_expression())
        return Ast(self._predicates, self._symbols, nodes)


class Flags:
    expand: bool = False


@dataclass
class Context:
    ast: Ast
    dynamic_values: list[tuple[str, str]]
    flags: Flags


def z3_expr(ctx: Context, expression: Node) -> str:
    match expression:
        case FlagNode(token, child):
            match token.source:
                case "#expand":
                    ctx.flags.expand = True
                    return z3_expr(ctx, child)
                case _:
                    raise Exception("Invalid flag")
        case IdentifierNode(token, dynamic_param):
            if dynamic_param:
                for name, value in ctx.dynamic_values[::-1]:
                    if token.source == name:
                        return value
            return token.source
        case PredicateNode(name, params):
            return (
                f"{name.source}({', '.join([z3_expr(ctx, param) for param in params])})"
            )
        case UnaryOpNode(token, child):
            match token.kind:
                case TokenKind.NOT:
                    return f"Not({z3_expr(ctx, child)})"
                case _:
                    raise Exception("Invalid unary operator")
        case BinOpNode(token, left, right):
            match token.kind:
                case TokenKind.AND:
                    return f"And({z3_expr(ctx, left)}, {z3_expr(ctx, right)})"
                case TokenKind.OR:
                    return f"Or({z3_expr(ctx, left)}, {z3_expr(ctx, right)})"
                case TokenKind.LEFT_IMPLIES:
                    return f"Implies({z3_expr(ctx, right)}, {z3_expr(ctx, left)})"
                case TokenKind.RIGHT_IMPLIES:
                    return f"Implies({z3_expr(ctx, left)}, {z3_expr(ctx, right)})"
                case TokenKind.EQUIV:
                    return f"{z3_expr(ctx, left)} == {z3_expr(ctx, right)}"
                case _:
                    raise Exception("Invalid binary operator")
        case QuantifierNode(token, kind, variables, child):
            if ctx.flags.expand:
                children_enumerations = []
                variable_names = [variable.token.source for variable in variables]
                for values in itertools.product(ctx.ast.symbols, repeat=len(variables)):
                    ctx.dynamic_values.extend(zip(variable_names, values))
                    children_enumerations.append(z3_expr(ctx, child))
                    ctx.dynamic_values = ctx.dynamic_values[: -len(variables)]

                if len(children_enumerations) == 0:
                    return ""

                match kind:
                    case QuantifierKind.FOR_ALL:
                        return f"And({', '.join(children_enumerations)})"
                    case QuantifierKind.EXISTS:
                        return f"Or({', '.join(children_enumerations)})"
            else:
                variable_names = [variable.token.source for variable in variables]
                ctx.dynamic_values.extend(zip(variable_names, variable_names))
                child = z3_expr(ctx, child)
                match kind:
                    case QuantifierKind.FOR_ALL:
                        result = f"forall({len(variables)}, lambda {', '.join(variable_names)}: {child})"
                    case QuantifierKind.EXISTS:
                        result = f"exists({len(variables)}, lambda {', '.join(variable_names)}: {child})"
                ctx.dynamic_values = ctx.dynamic_values[: -len(variables)]
                return result
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


def forall(sym_count: int, fn):
    return And([fn(*sym) for sym in itertools.product(all_symbols, repeat=sym_count)])


def exists(sym_count: int, fn):
    return Or([fn(*sym) for sym in itertools.product(all_symbols, repeat=sym_count)])

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
    for name, arity in ctx.ast.predicates:
        params = ", ".join(list(letters[:arity]))
        signature = f"def {name}({params}):"
        body = ident + f"return Bool('{name}(' + ', '.join([{params}]) + ')')"
        lines.extend([signature, body, ""])

    # generate symbols
    for i, symbol in enumerate(ctx.ast.symbols):
        lines.append(f"{symbol} = '{i}'")

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
        fn_body.append(f"solver.add({z3_expr(ctx, node)})")
    fn_body.append("")
    fn_body.append("solve(solver)")
    lines.extend([ident + line for line in fn_body])

    content = "\n".join(lines)
    result = z3_template.replace("%%%", content)
    return result.strip() + "\n"


def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <source_file>")
        sys.exit(1)

    source_filename = sys.argv[1]
    with open(source_filename, "r") as f:
        source = f.read()

    tokenizer = Tokenizer(source)

    tokens = []
    while (token := tokenizer.next_token()).kind != TokenKind.EOF:
        # print(token)
        if token.kind == TokenKind.COMMENT:
            continue
        tokens.append(token)

    parser = Parser(tokens)
    ast = parser.parse()
    # for node in ast.expressions:
    #     print(node)

    ctx = Context(ast, [], Flags())

    z3_code = z3_generate(ctx)
    print(z3_code)


if __name__ == "__main__":
    main()
