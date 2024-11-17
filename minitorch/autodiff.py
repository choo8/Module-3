from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # TODO: Implement for Task 1.1.
    v1 = [v + (epsilon / 2.0) if idx == arg else v for idx, v in enumerate(vals)]
    v2 = [v - (epsilon / 2.0) if idx == arg else v for idx, v in enumerate(vals)]
    return (f(*v1) - f(*v2)) / epsilon


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # TODO: Implement for Task 1.4.
    vars, visited = [], set()

    def _dfs(_v: Variable) -> None:
        visited.add(_v.unique_id)

        if _v.history:
            for parent in _v.parents:
                if not _v.is_constant() and parent.unique_id not in visited:
                    _dfs(parent)

        if not _v.is_constant():
            vars.append(_v)

    _dfs(variable)

    return list(reversed(vars))


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    vars = topological_sort(variable)
    scalars = {v.unique_id: 0.0 for v in vars}
    scalars[variable.unique_id] = deriv  # Right-most variable
    for v in vars:
        if v.is_leaf():
            v.accumulate_derivative(scalars[v.unique_id])
        else:
            for _v, _deriv in v.chain_rule(scalars[v.unique_id]):
                if _v.unique_id in scalars:
                    scalars[_v.unique_id] += _deriv


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
