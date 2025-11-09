"""Simple prescriptive analytics example using linear programming."""

from __future__ import annotations

import pandas as pd
from scipy.optimize import linprog


def production_planning_example() -> pd.Series:
    """Solve a basic production planning optimization problem.

    We maximize profit for two products given labor and material constraints.

    Returns:
        pandas.Series: Optimal units to produce of each product and the
        corresponding total profit.
    """
    # Profit per unit for products A and B
    profit = [-5.0, -4.0]  # negative for linprog minimization

    # Constraints: each row corresponds to a resource
    # Labor hours: 6 for A, 4 for B (<= 240 hours total)
    # Materials:   3 for A, 2 for B (<= 100 units total)
    # Minimum demand: at least 30 units of product B and 10 units of product A
    A_ub = [[6.0, 4.0], [3.0, 2.0], [0.0, -1.0], [-1.0, 0.0]]
    b_ub = [240.0, 100.0, -30.0, -10.0]

    bounds = [(0, None), (0, None)]

    result = linprog(
        c=profit,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        raise RuntimeError("Optimization failed")

    units_a, units_b = result.x
    max_profit = -result.fun
    return pd.Series(
        {
            "product_A": units_a,
            "product_B": units_b,
            "max_profit": max_profit,
        }
    )


if __name__ == "__main__":
    print(production_planning_example())
