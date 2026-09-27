"""
Helpers to run stencil computations with symjit kernels.

symjit kernels are point-wise: they are applied element by element to arrays of the
same shape. A stencil, however, reads neighbors, i.e., shifted slices of the arrays,
and slicing a multi-dimensional array (e.g., `a[1:-2, 2:-2, 1:]`) gives non-contiguous
views that symjit has to copy before calling the kernel.

The copies can be avoided by working on the flattened arrays. In row-major order the
neighbor of a grid point along the last axis is the next element, the neighbor along the
second axis is `nz` elements away, and along the first axis `ny * nz` elements away.
Therefore, shifting an operand is just a slice of the flat array, which is a contiguous
view. `Grid.apply` evaluates a kernel on the flat range spanning the output box (some of
the points in the range lie outside the box and are discarded) and returns the results as
3D views of the box.
"""

import numpy as np


class Grid:
    def __init__(self, shape):
        self.shape = tuple(shape)
        nx, ny, nz = self.shape
        self.size = nx * ny * nz
        self.sx = ny * nz  # flat offset of the neighbor along the first axis
        self.sy = nz  # ... along the second axis
        self.sz = 1  # ... along the third axis

    def flat(self, a):
        """The array as a contiguous 1D array (a view if `a` is already contiguous)"""
        return np.ascontiguousarray(a).reshape(-1)

    def broadcast(self, a):
        """A broadcastable array (e.g., a vector along an axis) expanded to the grid and flattened"""
        return self.flat(np.broadcast_to(a, self.shape))

    def apply(self, kernel, operands, box):
        """
        Evaluates a compiled kernel on shifted operands.

        operands: a list of `(flat_array, offset)`. The kernel argument for the grid point
            with flat index i is `flat_array[i + offset]`.
        box: `((i0, i1), (j0, j1), (k0, k1))`, the range of the output grid points.

        Returns a list with one 3D view of the box for each output of the kernel.
        """
        nx, ny, nz = self.shape
        (i0, i1), (j0, j1), (k0, k1) = box
        lo = (i0 * ny + j0) * nz + k0
        hi = ((i1 - 1) * ny + (j1 - 1)) * nz + k1
        assert all(lo + off >= 0 and hi + off <= self.size for _, off in operands)

        res = kernel(*[a[lo + off : hi + off] for a, off in operands])

        outs = []
        for r in res:
            buf = np.zeros(self.size)
            buf[lo:hi] = r
            outs.append(buf.reshape(self.shape)[i0:i1, j0:j1, k0:k1])
        return outs
