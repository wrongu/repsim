### *Function* `path_length(pts: Iterable[Point], space: Manifold) -> Scalar ()`


Compute the length of a path in a manifold.

### Arguments
> - **pts** (`Iterable[Point]`: `None`): a sequence of points
> - **space** (`Manifold`: `None`): a manifold

### Returns
> - **Scalar** (`None`: `None`): the length of the path



### *Function* `subdivide_geodesic(
    pt_a: Point, pt_b: Point, space: Manifold, octaves: int = 5, **kwargs
) -> List[Point] ()`


Given two points on a geodesic, subdivide the geodesic into a list of points.

### Arguments
> - **pt_a** (`Point`: `None`): The first point of the geodesic.
> - **pt_b** (`Point`: `None`): The second point of the geodesic.
> - **space** (`Manifold`: `None`): The manifold in which the geodesic lives.
> - **octaves** (`int`: `None`): The number of subdivisions to make.

### Returns
> - **List[Point]** (`None`: `None`): A list of points along the geodesic.



### *Function* `project_along(
    pt_fro: Point, pt_to: Point, pt_a: Point, space: Manifold, tol=1e-6
) -> Tuple[Point, OptimResult] ()`


Find 'projection' of pt_a onto a geodesic that spans [pt_fro, pt_to]

### Arguments
> - **pt_fro** (`Point`: `None`): The first point of the geodesic.
> - **pt_to** (`Point`: `None`): The second point of the geodesic.
> - **pt_a** (`Point`: `None`): The point to project onto the geodesic.
> - **space** (`Manifold`: `None`): The manifold in which the geodesic lives.
> - **tol** (`float`: `None`): The tolerance for convergence, in units of `space.length`.

### Returns
> - **OptimResult]** (`None`: `None`): a point on the manifold that lies along a
        geodesic connecting [pt_fro, pt_to], such that the length from pt_a         to pt_x is minimized



### *Function* `point_along(
    pt_a: Point,
    pt_b: Point,
    space: Manifold,
    frac: float,
    guess: Union[Point, None] = None,
    **kwargs,
) -> Tuple[Point, OptimResult] ()`


Given ptA and ptB, return ptC along the geodesic between them, such that d(ptA,ptC) is frac percent of the total length ptA to ptB.

### Arguments
> - **pt_a** (`Point`: `None`): The first point of the geodesic.
> - **pt_b** (`Point`: `None`): The second point of the geodesic.
> - **space** (`Manifold`: `None`): The manifold in which the geodesic lives.
> - **frac** (`float`: `None`): The fraction of the total length of the geodesic to
        project along.
> - **None])** (`None`: `None`): A point to use as a guess for the result.
        If None, a midpoint is used.

### Returns
> - **OptimResult]** (`None`: `None`): a point on the manifold that lies along a
        geodesic connecting [pt_a, pt_b], such that the length from pt_a         to pt_x is minimized and d(ptA,ptC) is frac percent of the total         length ptA to ptB



### *Function* `_calc_error(pt_c):
        # Two sources of error: total length should be dist_ab, and dist_a/(dist_a+dist_b) should equal 'frac'
        dist_a, dist_b = space.length(pt_a, pt_c), space.length(pt_c, pt_b)
        total_length = dist_a + dist_b
        length_error = torch.clip(total_length - dist_ab, 0.0, None)
        frac_error = torch.abs(dist_a - frac * dist_ab)
        return length_error + frac_error

    return minimize(_calc_error, pt, space, **kwargs)


def midpoint(
    pt_a: Point, pt_b: Point, space: Manifold, **kwargs
) -> Tuple[Point, OptimResult] ()`


Compute the midpoint of the geodesic between pt_a and pt_b.

### Arguments
> - **pt_a** (`Point`: `None`): The first point of the geodesic.
> - **pt_b** (`Point`: `None`): The second point of the geodesic.
> - **space** (`Manifold`: `None`): The manifold in which the geodesic lives.

### Returns
> - **OptimResult]** (`None`: `None`): a midpoint on the manifold that lies along a
        geodesic connecting [pt_a, pt_b]

