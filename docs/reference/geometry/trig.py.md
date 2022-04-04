### *Function* `angle(pt_a: Point, pt_b: Point, pt_c: Point, space: Manifold, **kwargs) -> Scalar ()`


Angle B of triangle ABC, based on incident angle of geodesics AB and CB.

If B is along the geodesic from A to C, then the angle is pi (180 degrees). If A=C, then the angle is zero.

### Arguments
> - **pt_a** (`Point`: `None`): The first point of the triangle.
> - **pt_b** (`Point`: `None`): The second point of the triangle.
> - **pt_c** (`Point`: `None`): The third point of the triangle.
> - **space** (`Manifold`: `None`): The manifold in which the triangle lives.

### Returns
> - **Scalar** (`None`: `None`): The angle B of the triangle.

