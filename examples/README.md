# Logic Language Examples

This directory contains example `.logic` files demonstrating various features of the logic programming language.

## Available Commands

The logic language tool supports three main commands:

1. **transpile** - Converts .logic files to Python/Z3 code
   ```bash
   python3 main.py transpile <file.logic>
   ```

2. **solve** - Solves the logical formulas using Z3 SMT solver
   ```bash
   python3 main.py solve <file.logic>
   ```

3. **tableau** - Uses the tableau method for logical proofs
   ```bash
   python3 main.py tableau <file.logic>
   ```

## Example Files

### Propositional Logic

#### 01_basic_propositional.logic
Basic propositional logic demonstrating AND, OR, and NOT operations.
- **Works with**: tableau, solve
- **Status**: Unsatisfiable (contains contradictory assumptions)

#### 02_implications.logic
Demonstrates logical implication and equivalence operators.
- **Works with**: tableau, solve
- **Status**: Satisfiable

#### 05_classic_logic.logic
Classic logic principles including:
- Modus Ponens
- Modus Tollens
- Law of Excluded Middle
- Law of Non-Contradiction
- Double Negation
- De Morgan's Laws
- **Works with**: tableau, solve
- **Status**: Satisfiable

#### 11_satisfiable.logic
Simple satisfiable formulas for testing.
- **Works with**: tableau, solve
- **Status**: Satisfiable

#### 12_unsatisfiable.logic
Simple contradictory formulas.
- **Works with**: tableau, solve
- **Status**: Unsatisfiable

### Predicate Logic

#### 03_predicates.logic
Demonstrates predicates with arguments.
- **Works with**: tableau, solve
- **Status**: Unsatisfiable

#### 04_quantifiers.logic
First-order logic with universal (`@forall`) and existential (`@exists`) quantifiers.
- **Works with**: solve (tableau doesn't support quantifiers)
- **Status**: Satisfiable
- **Note**: Requires `#symbols` directive and parentheses around complex expressions

### Advanced Features

#### 06_ranges.logic
Using `#symbols(#range(n, m))` to define numeric ranges for quantification.
- **Works with**: solve
- **Status**: Satisfiable - demonstrates `Loves` predicate over range 1-3

#### 07_where_clause.logic
Using `@where` clauses to add constraints to quantifiers.
- **Works with**: solve (requires proper setup)
- **Status**: Partially working - numeric comparisons like `x > 2` don't work because #range generates string symbols

#### 08_knights_knaves.logic
Classic Knights and Knaves logic puzzle.
- **Works with**: tableau, solve
- **Status**: Open branch (puzzle has solutions)

#### 09_graph_theory.logic
Graph theory properties using predicate logic with quantifiers.
- **Works with**: solve (requires #symbols directive, not supported by tableau)
- **Status**: Satisfiable - demonstrates Edge and Path predicates
- **Note**: When quantifiers contain implications, use parentheses: `@forall x: (P(x) -> Q(x))`

#### 10_mathematical.logic
Mathematical properties like equality, commutativity, transitivity.
- **Works with**: solve
- **Status**: Satisfiable
- **Note**: Uses parentheses around implications with quantifiers

## Language Features

### Operators (by precedence, highest to lowest)
- `-` or `!` or `¬` or `~` - NOT (highest)
- `&` or `&&` - AND
- `|` or `||` - OR
- `->` or `=>`, `<-`, `<->` or `<=>` - Implication and equivalence (lowest)

**Note**: Quantifiers (`@forall`, `@exists`) bind tightly to their body. Use parentheses to extend their scope:
- `@forall x: P(x) & Q(x)` - quantifier only applies to `P(x)`
- `@forall x: (P(x) -> Q(x))` - quantifier applies to entire implication

### Quantifiers
- `@forall x` - Universal quantifier (for all x)
- `@exists x` - Existential quantifier (there exists x)

### Directives
- `#symbols(a, b, c)` - Define symbols for quantification
- `#symbols(#range(n, m))` - Define a range of numeric symbols

### Special Clauses
- `@where condition` - Add constraints to quantifiers
- `@forall x in #range(1, 5)` - Quantify over a range

### Comments
- `// This is a comment`

## Testing Examples

Run all examples with tableau:
```bash
for f in examples/*.logic; do
    echo "=== $f ==="
    python3 main.py tableau "$f"
done
```

Run all examples with solve:
```bash
for f in examples/*.logic; do
    echo "=== $f ==="
    python3 main.py transpile "$f" | grep -A 1000 "^import" > /tmp/test.py
    python3 /tmp/test.py
done
```

## Limitations

1. **Tableau method**: 
   - Does not support quantifiers
   - Does not support directives like `#symbols`
   - Best for propositional logic

2. **Solve command**:
   - Supports quantifiers via Z3
   - Supports symbols and ranges
   - More powerful but requires Z3 installation

3. **Where clause**: 
   - Currently has some parsing issues with complex expressions

## Expected Output

### Satisfiable formulas
Green "Satisfiable" message with model values, e.g.:
```
Satisfiable
A = False
B = True
```

### Unsatisfiable formulas
Red "Unsatisfiable" message:
```
Unsatisfiable
Total solutions: 0
```

### Tableau output
Shows proof tree with:
- `T` / `F` - True/False assumptions
- Alpha/Beta rules - Logical decomposition
- `Closed` - Branch closed (contradiction found)
- `Open` - Branch open (possible model)
