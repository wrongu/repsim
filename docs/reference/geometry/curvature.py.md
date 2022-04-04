### *Function* `_bisector_length(x: Scalar, y: Scalar, z: Scalar) -> Scalar ()`


Given a triangle ABC with side lengths AB=x, BC=y, AC=z, returns the length of BD, where D is the midpoint of AC.

### Arguments
> - **x** (`Scalar`: `None`): The length of side AB.
> - **y** (`Scalar`: `None`): The length of side BC.
> - **z** (`Scalar`: `None`): The length of side AC.

### Returns
> - **Scalar** (`None`: `None`): The length of side BD.



### *Function* `alexandrov(pt_a: Point, pt_b: Point, pt_c: Point, space: Manifold) -> Scalar ()`


Compute the Alexandrov curvature of a triangle.

### Arguments
> - **pt_a** (`Point`: `None`): The first point of the triangle.
> - **pt_b** (`Point`: `None`): The second point of the triangle.
> - **pt_c** (`Point`: `None`): The third point of the triangle.
> - **space** (`Manifold`: `None`): The manifold in which the triangle lives.

### Returns
> - **Scalar** (`None`: `None`): The Alexandrov curvature of the triangle.

