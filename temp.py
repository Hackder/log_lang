from z3 import And, Bool, Goal, Implies, Not, Or, Tactic

x = Bool("x")  #  Note: Bool() rather than Boolean()
y = Bool("y")
z = Bool("z")

f = And(x, Or(x, y), And(x, z == Not(y), Implies(x, z)))

#  from https://stackoverflow.com/a/18003288/1911064

g = Goal()
g.add(f)

# use describe_tactics() to get to know the tactics available

t = Tactic("tseitin-cnf")
clauses = t(g)

for clause in clauses[0]:
    print(clause)
