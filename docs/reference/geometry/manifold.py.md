## *Class* `Manifold`


Base class for Manifolds with an associated length(). In other words, a Length Space


### *Function* `__init__ (self, *, dim: int, shape: tuple)`


Create a new Manifold with the given dimensionality and shape.

### Arguments
> - **dim** (`int`: `None`): The dimensionality of the manifold.
> - **shape** (`tuple`: `None`): The shape of the manifold.



### *Function* `project (self, pt: Point) -> Point: """ Project a point from the ambient space onto the manifold. Arguments: pt (Point): The point to project. Returns: Point: The projected point. """ return self._project(pt) if not self.contains(pt) else pt def _project(self, pt: Point) -> Point: """ Project without first checking contains() Arguments: pt (Point): The point to project. Returns: Point: The projected point. """ # Default behavior (to be overridden by subclasses): just return the # point. Equivalent to saying that the manifold is identical to the # ambient space. return pt def contains(self, pt: Point, atol: float = 1e-6) -> bool: """ Check whether the given point is within atol tolerance of the manifold. Manifold.contains checks match to ambient shape only. Further specialization done by subclasses. Arguments: pt (Point): The point to check. atol (float): The tolerance for containment. Returns: bool: Whether the point is within tolerance of the manifold. """ return pt.size()[-self.ambient :] == self.shape def length(self, pt_a: Point, pt_b: Point) -> Scalar: """ Compute the length of the geodesic between two points. Arguments: pt_a (Point): The first point of the geodesic. pt_b (Point): The second point of the geodesic. Returns: Scalar: The length of the geodesic. """ if not self.contains(pt_a): warnings.warn("pt_a is not on the manifold - trying to project") pt_a = self._project(pt_a) if not self.contains(pt_b): warnings.warn("pt_b is not on the manifold - trying to project") pt_b = self._project(pt_b) return self._length(pt_a, pt_b) def _length(self, pt_a: Point, pt_b: Point) -> Scalar: raise NotImplementedError("_length() must be implemented by a subclass") class HyperSphere(Manifold)`


Manifold of points on the surface of a dim-dimensional sphere (ambient dim+1).


### *Function* `__init__ (self, dim: int, radius: float = 1.0)`


Create a new Manifold of points on the surface of a dim-dimensional sphere.

### Arguments
> - **dim** (`int`: `None`): The dimensionality of the sphere.
> - **radius** (`float`: `None`): The radius of the sphere.



### *Function* `_project (self, pt: torch.Tensor) -> torch.Tensor: return self._radius * pt / torch.linalg.norm(pt, dim=-1) def contains(self, pt: torch.Tensor, atol: float = 1e-6) -> bool: """ Check whether the given point is within atol tolerance of the manifold. Arguments: pt (Point): The point to check. atol (float): The tolerance for containment. Returns: bool: Whether the point is within tolerance of the manifold. """ return torch.abs(torch.linalg.norm(pt, dim=-1) - self._radius) < atol def _length(self, pt_a: Point, pt_b: Point) -> Scalar: # Default length is arc-length along the surface of a hypersphere norm_dot = torch.sum(pt_a * pt_b, dim=-1) / torch.sqrt( torch.sum(pt_a * pt_a, dim=-1) * torch.sum(pt_b * pt_b, dim=-1) ) return torch.arccos(torch.clip(norm_dot, -1.0, +1.0)) class VectorSpace(Manifold)`


VectorSpace is a manifold of arbitrarily-sized vectors with the default metric being Euclidean.


### *Function* `_length (self, pt_a: Point, pt_b: Point) -> Scalar: return torch.linalg.norm(torch.reshape(pt_a - pt_b, (-1, self.ambient))) class Matrix(VectorSpace)`


Manifold of Matrices of size (rows, cols)


### *Function* `__init__ (self, rows: int, cols: int)`


Create a new Manifold of matrices of size (rows, cols).

### Arguments
> - **rows** (`int`: `None`): The number of rows in the matrices.
> - **cols** (`int`: `None`): The number of columns in the matrices.



## *Class* `SymmetricMatrix`


Manifold of Symmetric Matrices of size (rows, rows)


### *Function* `__init__ (self, rows: int)`


Create a new Manifold of symmetric matrices of size (rows, rows).

### Arguments
> - **rows** (`int`: `None`): The number of rows in the matrices.



### *Function* `_project (self, pt: torch.Tensor) -> torch.Tensor: return (pt + pt.transpose(-2, -1)) / 2 def contains(self, pt: torch.Tensor, atol: float = 1e-6) -> bool: return super(SymmetricMatrix, self).contains(pt, atol) and torch.allclose( pt, pt.transpose(-2, -1), atol=atol ) class SPDMatrix(SymmetricMatrix)`


Manifold of Symmetric Positive (Semi-)Definite Matrices


### *Function* `_project (self, pt: torch.Tensor, atol: float = 1e-6) -> torch.Tensor: # See https://github.com/geomstats/geomstats/blob/master/geomstats/geometry/spd_matrices.py#L59 sym_pt = super(SPDMatrix, self)._project(pt) s, v = torch.linalg.eigh(sym_pt) s = torch.clip(s, 0.0, None) # clip eigenvalues return torch.einsum("...id,...d,...jd->ij", v, s, v) def contains(self, pt: torch.Tensor, atol: float = 1e-6) -> bool: # Three checks: (1) is a matrix of the right (n,n) size, # (2) is symmetric, up to 'atol' tolerance # (3) has all non-negative eigenvalues (greater than -atol) # ...where (1) and (2) are handled by super() return super(SPDMatrix, self).contains(pt) and torch.all( torch.linalg.eigvalsh(pt) >= -atol ) class DistMatrix(SymmetricMatrix)`


Manifold of Pairwise-distance matrices, i.e. a SymmetricMatrix with zero diagonals and non-negative off-diagonals.
