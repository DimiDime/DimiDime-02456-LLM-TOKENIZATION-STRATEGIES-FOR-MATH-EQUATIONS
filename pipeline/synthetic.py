#!/usr/bin/env python3
import csv
import random
import argparse
from typing import List, Dict, Tuple
import re


class EquationGenerator:
    """
    Generates synthetic equations (plain-text).
    """

    def __init__(self, seed: int = None):
        if seed is not None:
            random.seed(seed)

        # Variable pools
        self.variables = ['x', 'y', 'z', 'a', 'b', 'c', 'd', 'r', 's', 't', 'u', 'v', 'w']
        self.greek_vars = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'theta',
                           'lambda', 'mu', 'phi', 'psi', 'omega', 'sigma', 'tau']

        # Physics/Engineering variables
        self.physics_vars = {
            'F': 'force', 'm': 'mass', 'a': 'acceleration', 'v': 'velocity',
            'E': 'energy', 'P': 'power', 'W': 'work', 'p': 'momentum',
            'T': 'temperature', 'V': 'volume', 'n': 'amount', 'R': 'constant',
            'g': 'gravity', 'h': 'height', 'L': 'length', 'A': 'area',
            'I': 'current', 'Q': 'charge', 'B': 'magnetic field', 'C': 'capacitance'
        }

        # Constants (plain, non-LaTeX)
        self.constants = ['pi', 'e', 'c', 'h', 'G', 'k', 'R', 'N_A']
        self.numeric_constants = ['1', '2', '3', '4', '1/2', '1/3', '2/3', '4/3']

        # Subscripts and superscripts
        self.subscripts = ['0', '1', '2', 'i', 'f', 'max', 'min', 'avg', 'rms', 'net']
        self.superscripts = ['2', '3', '-1', 'n', '*', 'circ', '+', '-']

        # Operators (no LaTeX)
        self.binary_ops = ['+', '-', '*']
        self.functions = ['sin', 'cos', 'tan', 'ln', 'log', 'exp', 'sqrt']
        self.calculus_ops = ['int', 'sum', 'partial', 'd']

    def generate_scalar_term(self, complexity: int) -> List[str]:
        """Generate a list of scalar terms (variables/constants)."""
        scalars = []

        if complexity >= 1 and random.random() > 0.5:
            const = random.choice(self.numeric_constants)
            scalars.append(const)

        if complexity >= 2 and random.random() > 0.3:
            const = random.choice(self.constants)
            scalars.append(const)

        num_vars = min(complexity, random.randint(1, 3))
        for _ in range(num_vars):
            all_vars = list(self.physics_vars.keys()) + self.variables + self.greek_vars
            var = random.choice(all_vars)


            if random.random() > 0.6:
                sub = random.choice(self.subscripts)
                var = f"{var}_{sub}"

            if complexity >= 3 and random.random() > 0.8:
                sup = random.choice(self.superscripts)
                var = f"{var}^{sup}"

            scalars.append(var)

        return scalars

    def generate_simple_expression(self, complexity: int) -> str:
        """Generate simple algebraic expression."""
        scalars = self.generate_scalar_term(complexity)

        if len(scalars) == 1:
            expr = scalars[0]
        else:
            ops = [random.choice(self.binary_ops) for _ in range(len(scalars) - 1)]
            expr_parts = []
            for i, scalar in enumerate(scalars):
                expr_parts.append(scalar)
                if i < len(ops):
                    expr_parts.append(ops[i])
            expr = ''.join(expr_parts)

        return expr

    def generate_fraction_expression(self, complexity: int) -> str:
        """Generate fraction expression using '/'."""
        num_scalars = self.generate_scalar_term(max(1, complexity - 1))
        denom_scalars = self.generate_scalar_term(max(1, complexity - 1))

        if len(num_scalars) == 1:
            numerator = ''.join(num_scalars)
        else:
            numerator_parts = []
            for i, s in enumerate(num_scalars):
                if i == 0:
                    numerator_parts.append(s)
                else:
                    numerator_parts.append(random.choice(self.binary_ops))
                    numerator_parts.append(s)
            numerator = ''.join(numerator_parts)

        if len(denom_scalars) == 1:
            denominator = ''.join(denom_scalars)
        else:
            denominator_parts = []
            for i, s in enumerate(denom_scalars):
                if i == 0:
                    denominator_parts.append(s)
                else:
                    denominator_parts.append(random.choice(self.binary_ops))
                    denominator_parts.append(s)
            denominator = ''.join(denominator_parts)

        expr = f"({numerator})/({denominator})"
        return expr

    def generate_power_expression(self, complexity: int) -> str:
        """Generate expression with exponents."""
        base_scalars = self.generate_scalar_term(max(1, complexity - 1))
        base = ''.join(base_scalars) if len(base_scalars) == 1 else f"({'+'.join(base_scalars)})"

        if complexity >= 3 and random.random() > 0.5:
            exp_parts = random.choice(['2', 'n', '-1', 'n+1', '1/2'])
        else:
            exp_parts = random.choice(['2', '3', 'n'])

        expr = f"{base}^{exp_parts}"
        return expr

    def generate_function_expression(self, complexity: int) -> str:
        """Generate expression with mathematical functions."""
        func = random.choice(self.functions)

        if complexity >= 3 and random.random() > 0.5:
            inner_expr = self.generate_simple_expression(complexity - 1)
            arg = inner_expr
        else:
            arg_scalars = self.generate_scalar_term(max(1, complexity - 1))
            arg = ''.join(arg_scalars) if len(arg_scalars) == 1 else '+'.join(arg_scalars)

        if func == 'sqrt':
            expr = f"sqrt({arg})"
        else:
            expr = f"{func}({arg})"

        return expr

    def generate_calculus_expression(self, complexity: int) -> str:
        """Generate calculus-based expression (non-LaTeX)."""
        calc_op = random.choice(self.calculus_ops)

        if calc_op == 'int':
            integrand_scalars = self.generate_scalar_term(complexity)
            integrand = ''.join(integrand_scalars) if len(integrand_scalars) == 1 else '+'.join(integrand_scalars)
            var = random.choice(['x', 't', 'r', 'V'])

            if complexity >= 4 and random.random() > 0.5:
                bounds = [random.choice(['0', 'a', 'L']), random.choice(['L', 'b', 'inf'])]
                expr = f"int_{bounds[0]}^{bounds[1]}({integrand})d{var}"
            else:
                expr = f"int({integrand})d{var}"

            return expr

        elif calc_op == 'sum':
            term_scalars = self.generate_scalar_term(complexity)
            if len(term_scalars) == 1:
                term = term_scalars[0]
            else:
                term = '+'.join(term_scalars)
            index_var = random.choice(['i', 'j', 'k', 'n'])

            if complexity >= 3:
                bounds = [random.choice(['1', '0']), random.choice(['n', 'N', 'inf'])]
                expr = f"sum_{index_var}={bounds[0]}^{bounds[1]}({term})"
            else:
                expr = f"sum({term})"

            return expr

        elif calc_op == 'partial':
            func_var = random.choice(list(self.physics_vars.keys()))
            deriv_var = random.choice(['x', 'y', 't', 'r'])

            expr = f"partial{func_var}/partial{deriv_var}"
            return expr

        else:  # 'd'
            func_var = random.choice(list(self.physics_vars.keys()))
            deriv_var = random.choice(['x', 't'])

            expr = f"d{func_var}/d{deriv_var}"
            return expr

    def generate_composite_expression(self, complexity: int) -> str:
        """Generate composite expression combining multiple types."""
        components = []

        num_components = min(complexity, random.randint(2, 4))

        generators = [
            self.generate_simple_expression,
            self.generate_fraction_expression,
            self.generate_power_expression,
            self.generate_function_expression
        ]

        if complexity >= 4:
            generators.append(self.generate_calculus_expression)

        for _ in range(num_components):
            gen_func = random.choice(generators)
            expr = gen_func(max(1, complexity - 1))
            components.append(expr)

        ops = [random.choice(['+', '-']) for _ in range(len(components) - 1)]

        if components:
            full_expr = components[0]
            for i in range(1, len(components)):
                full_expr += f"{ops[i-1]}{components[i]}"
        else:
            full_expr = ""

        return full_expr

    def generate_equation(self, complexity: int) -> Dict:
        """
        Generate a complete equation (plain text).

        Only returns:
          - full_equation
          - complexity
        """
        lhs_var = random.choice(list(self.physics_vars.keys()) + self.variables[:5])

        if complexity >= 2 and random.random() > 0.7:
            sub = random.choice(self.subscripts)
            lhs_var = f"{lhs_var}_{sub}"

        if complexity == 1:
            rhs_expr = self.generate_simple_expression(complexity)
        elif complexity == 2:
            gen_func = random.choice([
                self.generate_fraction_expression,
                self.generate_power_expression
            ])
            rhs_expr = gen_func(complexity)
        elif complexity == 3:
            gen_func = random.choice([
                self.generate_function_expression
            ])
            rhs_expr = gen_func(complexity)
        elif complexity == 4:
            gen_func = random.choice([
                self.generate_calculus_expression,
            ])
            rhs_expr = gen_func(complexity)
        else:  # complexity >= 5
            rhs_expr = self.generate_composite_expression(complexity)

        full_equation = f"{lhs_var}={rhs_expr}".replace(" ", "")

        return {
            "full_equation": full_equation,
            "complexity": complexity,
        }

    def generate_dataset(self, num_equations: int, complexity_range: Tuple[int, int] = (1, 5)) -> List[Dict]:
        """
        Generate a dataset of equations.
        """
        dataset = []
        min_complexity, max_complexity = complexity_range

        for i in range(num_equations):
            if num_equations > (max_complexity - min_complexity):
                complexity = min_complexity + (i % (max_complexity - min_complexity + 1))
            else:
                complexity = random.randint(min_complexity, max_complexity)

            equation = self.generate_equation(complexity)
            equation["id"] = i + 1
            dataset.append(equation)

        return dataset

    def save_to_csv(self, dataset: List[Dict], filename: str):
        """Save dataset to CSV file (id, full_equation, complexity only)."""
        fieldnames = ["id", "full_equation", "complexity"]

        with open(filename, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            # Ensure we only write the 3 desired keys
            for row in dataset:
                writer.writerow({
                    "id": row["id"],
                    "full_equation": row["full_equation"],
                    "complexity": row["complexity"],
                })

        print(f"✓ Successfully generated {len(dataset)} equations")
        print(f"✓ Saved to: {filename}")


# ============================================================================
# DEFAULTS
# ============================================================================

DEFAULT_NUM_EQUATIONS = 10000
DEFAULT_MIN_COMPLEXITY = 1
DEFAULT_MAX_COMPLEXITY = 3
DEFAULT_OUTPUT_FILE = "HPC pipeline/data/equations.csv"
DEFAULT_RANDOM_SEED = None
DEFAULT_SHOW_PREVIEW = True
DEFAULT_NUM_PREVIEW = 5


def run_generator(
    num_equations: int = DEFAULT_NUM_EQUATIONS,
    min_complexity: int = DEFAULT_MIN_COMPLEXITY,
    max_complexity: int = DEFAULT_MAX_COMPLEXITY,
    output_file: str = DEFAULT_OUTPUT_FILE,
    random_seed: int = DEFAULT_RANDOM_SEED,
    show_preview: bool = DEFAULT_SHOW_PREVIEW,
    num_preview: int = DEFAULT_NUM_PREVIEW,
) -> None:
    print("=" * 70)
    print("Equation Generator (plain text, '/' fractions)")
    print("=" * 70)
    print(f"Generating {num_equations} equations...")
    print(f"Complexity range: {min_complexity} to {max_complexity}")
    if random_seed is not None:
        print(f"Random seed: {random_seed}")
    print()

    generator = EquationGenerator(seed=random_seed)
    dataset = generator.generate_dataset(
        num_equations=num_equations,
        complexity_range=(min_complexity, max_complexity),
    )

    generator.save_to_csv(dataset, output_file)

    if show_preview and num_preview > 0:
        print("\n" + "=" * 70)
        print(f"Preview of first {min(num_preview, len(dataset))} equations:")
        print("=" * 70)
        for i, eq in enumerate(dataset[:num_preview]):
            print(f"\n[{i+1}] id: {eq['id']}, complexity: {eq['complexity']}")
            print(f"    Equation: {eq['full_equation']}")

    print("\n" + "=" * 70)
    print("Generation complete!")
    print("=" * 70)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic equations (plain text, '/' for fractions) and save to CSV "
            "with columns: id, full_equation, complexity."
        )
    )
    parser.add_argument(
        "--num-equations", "-n",
        type=int,
        default=DEFAULT_NUM_EQUATIONS,
        help=f"Number of equations to generate (default: {DEFAULT_NUM_EQUATIONS}).",
    )
    parser.add_argument(
        "--min-complexity",
        type=int,
        default=DEFAULT_MIN_COMPLEXITY,
        help=f"Minimum complexity (1-5, default: {DEFAULT_MIN_COMPLEXITY}).",
    )
    parser.add_argument(
        "--max-complexity",
        type=int,
        default=DEFAULT_MAX_COMPLEXITY,
        help=f"Maximum complexity (1-5, default: {DEFAULT_MAX_COMPLEXITY}).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_FILE,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT_FILE}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed for reproducibility (default: None).",
    )
    parser.add_argument(
        "--no-preview",
        action="store_false",
        dest="show_preview",
        default=DEFAULT_SHOW_PREVIEW,
        help="Disable preview printing of generated equations.",
    )
    parser.add_argument(
        "--num-preview",
        type=int,
        default=DEFAULT_NUM_PREVIEW,
        help=f"How many equations to preview (default: {DEFAULT_NUM_PREVIEW}).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_generator(
        num_equations=args.num_equations,
        min_complexity=args.min_complexity,
        max_complexity=args.max_complexity,
        output_file=args.output,
        random_seed=args.seed,
        show_preview=args.show_preview,
        num_preview=args.num_preview,
    )
