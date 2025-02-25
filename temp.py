import itertools
import sys
from z3 import And, Not, Bool, Implies, Or, Solver

def blue(a):
    return Bool('blue(' + ', '.join([a]) + ')')

def circle(a):
    return Bool('circle(' + ', '.join([a]) + ')')

def green(a):
    return Bool('green(' + ', '.join([a]) + ')')

def larger(a, b):
    return Bool('larger(' + ', '.join([a, b]) + ')')

def red(a):
    return Bool('red(' + ', '.join([a]) + ')')

def same_color(a, b):
    return Bool('same_color(' + ', '.join([a, b]) + ')')

def square(a):
    return Bool('square(' + ', '.join([a]) + ')')

def triangle(a):
    return Bool('triangle(' + ', '.join([a]) + ')')

A = '0'
B = '1'
C = '2'
D = '3'
all_symbols = set([C, B, D, A])
symbol_names = ['C', 'B', 'D', 'A']

def main():
    solver = Solver()
    
    solver.add(Not(Implies(red(B), Implies(square(B), triangle(C)))))
    solver.add(Not(And(blue(C), Not(triangle(C))) == Not(larger(D, B))))
    solver.add(Implies(Or(circle(C), circle(B)), And(larger(B, C), Not(green(D)))))
    solver.add(Implies(And(red(B), square(B)), blue(A) == blue(C)))
    solver.add(Implies(green(C), Not(larger(A, B))))
    solver.add(Or(Not(red(C)), green(C)))
    solver.add(Not(Or(larger(A, D), larger(D, A))) == larger(A, B))
    solver.add(Implies(Not(Or(triangle(B), triangle(C))), larger(A, B)))
    solver.add(Not(And(blue(C), square(C))))
    solver.add(Implies(Not(triangle(A)), Or(triangle(B), triangle(C))))
    solver.add(Implies(larger(A, C), And(square(D), Not(red(D)))))
    solver.add(forall(1, lambda x: And(Or(red(x), Or(green(x), blue(x))), And(Not(And(red(x), green(x))), And(Not(And(red(x), blue(x))), Not(And(green(x), blue(x))))))))
    solver.add(forall(1, lambda x: And(Or(triangle(x), Or(square(x), circle(x))), And(Not(And(triangle(x), square(x))), And(Not(And(triangle(x), circle(x))), Not(And(square(x), circle(x))))))))
    solver.add(And(forall(1, lambda x: forall(1, lambda y: Implies(larger(x, y), Not(larger(y, x))))), forall(1, lambda x: forall(1, lambda y: forall(1, lambda z: Implies(And(larger(x, y), larger(y, z)), larger(x, z)))))))
    solver.add(And(forall(1, lambda x: same_color(x, x)), And(forall(1, lambda x: forall(1, lambda y: Implies(same_color(x, y), same_color(y, x)))), And(forall(1, lambda x: forall(1, lambda y: forall(1, lambda z: Implies(And(same_color(x, y), same_color(y, z)), same_color(x, z))))), forall(1, lambda x: forall(1, lambda y: And(Or(Not(same_color(x, y)), Or(And(red(x), red(y)), Or(And(green(x), green(y)), Or(And(blue(x), blue(y)), Not(Or(red(x), Or(red(y), Or(green(x), Or(green(y), Or(blue(x), blue(y))))))))))), Or(same_color(x, y), And(Or(Not(red(x)), Not(red(y))), And(Or(Not(green(x)), Not(green(y))), Or(Not(blue(x)), Not(blue(y)))))))))))))
    
    solve(solver)

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

