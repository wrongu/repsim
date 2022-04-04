## *Class* `OptimResult(enum.Enum):
    CONVERGED = 0
    MAX_STEPS_REACHED = 1
    CONDITIONS_VIOLATED = 2


def minimize(
    fun: Callable[[Point], Scalar],
    init: Point,
    space: Manifold,
    *,
    pt_tol: float = 1e-6,
    fn_tol: float = 1e-6,
    init_step_size: float = 0.1,
    wolfe_c1=1e-4,
    wolfe_c2=0.9,
    max_iter: int = 10000
) -> Tuple[Point, OptimResult]`


Minimize a function in a given space.

### Arguments
> - **Scalar])** (`None`: `None`): The function to minimize.
> - **init** (`Point`: `None`): The initial point.
> - **space** (`Manifold`: `None`): The manifold in which to minimize.
> - **pt_tol** (`float`: `None`): The tolerance for the point.
> - **fn_tol** (`float`: `None`): The tolerance for the function value.
> - **init_step_size** (`float`: `None`): The initial step size.
> - **wolfe_c1** (`float`: `None`): The Wolfe condition parameter c1.
> - **wolfe_c2** (`float`: `None`): The Wolfe condition parameter c2.
> - **max_iter** (`int`: `None`): The maximum number of iterations.

### Returns
> - **OptimResult]** (`None`: `None`): The optimized point and the result.

