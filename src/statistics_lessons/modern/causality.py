"""Causality analysis using graphical models."""

from __future__ import annotations

from typing import FrozenSet, Iterable, Tuple

try:
    from causalgraphicalmodels import CausalGraphicalModel
except Exception:  # pragma: no cover - handled in tests
    CausalGraphicalModel = None


def is_d_separated(
    edges: Iterable[Tuple[str, str]], x: str, y: str, z: Iterable[str]
) -> bool:
    """Check d-separation between two variables.

    Args:
        edges: Iterable of directed edges defining the DAG.
        x: First variable.
        y: Second variable.
        z: Conditioning set of variables.

    Returns:
        True if X and Y are d-separated given Z.

    Raises:
        ImportError: If causalgraphicalmodels is not installed.
    """
    if CausalGraphicalModel is None:  # pragma: no cover - dependency missing
        msg = "causalgraphicalmodels is required for this function"
        raise ImportError(msg)

    nodes = list({x, y, *z}.union({n for edge in edges for n in edge}))
    model = CausalGraphicalModel(
        nodes=nodes,
        edges=list(edges),
    )
    return model.is_d_separated(x, y, z)


def backdoor_adjustment_sets(
    edges: Iterable[Tuple[str, str]], x: str, y: str
) -> FrozenSet[FrozenSet[str]]:
    """Return valid backdoor adjustment sets for a causal query.

    Args:
        edges: Iterable of directed edges defining the DAG.
        x: Intervention variable.
        y: Outcome variable.

    Returns:
        All sets of variables that satisfy the backdoor criterion. Each inner
        set represents one valid adjustment set.

    Raises:
        ImportError: If causalgraphicalmodels is not installed.
    """
    if CausalGraphicalModel is None:  # pragma: no cover - dependency missing
        msg = "causalgraphicalmodels is required for this function"
        raise ImportError(msg)

    nodes = list({x, y}.union({n for edge in edges for n in edge}))
    model = CausalGraphicalModel(
        nodes=nodes,
        edges=list(edges),
    )
    return model.get_all_backdoor_adjustment_sets(x, y)
