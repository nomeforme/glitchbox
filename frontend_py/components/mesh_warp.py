"""
Mesh-based warping for projection mapping.
Provides a grid of control points for fine-tuned surface warping.
"""

import numpy as np
import cv2
from typing import List, Tuple


class MeshWarp:
    """
    Manages a grid of control points for mesh-based warping.
    Each quad in the grid gets its own perspective transform.
    """

    def __init__(self, grid_size: int = 4):
        """
        Initialize mesh with an n×n grid of control points.

        Args:
            grid_size: Number of divisions per side (default 4 = 4×4 = 25 points)
        """
        self.grid_size = grid_size
        self.points = self._initialize_grid()

        # Cache for remap tables
        self.map_x = None
        self.map_y = None
        self.cached_size = None
        self.needs_update = False  # Default grid needs no warping

    def _initialize_grid(self) -> List[List[List[float]]]:
        """
        Initialize a regular grid of control points.
        Points are stored as percentages (0-100) for resolution independence.

        Returns:
            3D list: [row][col][x, y] where each point is [x_percent, y_percent]
        """
        points = []
        for row in range(self.grid_size + 1):
            row_points = []
            for col in range(self.grid_size + 1):
                x_percent = (col / self.grid_size) * 100
                y_percent = (row / self.grid_size) * 100
                row_points.append([x_percent, y_percent])
            points.append(row_points)
        return points

    def get_point(self, row: int, col: int) -> List[float]:
        """Get a control point at grid position [row][col]."""
        return self.points[row][col]

    def set_point(self, row: int, col: int, x_percent: float, y_percent: float):
        """Set a control point position."""
        self.points[row][col] = [x_percent, y_percent]
        self.needs_update = True

    def get_all_points_flat(self) -> List[Tuple[int, int, float, float]]:
        """
        Get all points as a flat list with indices.

        Returns:
            List of (row, col, x_percent, y_percent)
        """
        flat_points = []
        for row in range(self.grid_size + 1):
            for col in range(self.grid_size + 1):
                x, y = self.points[row][col]
                flat_points.append((row, col, x, y))
        return flat_points

    def reset_to_default(self):
        """Reset all points to regular grid positions."""
        self.points = self._initialize_grid()
        # Clear cached remap tables since we're back to default (no warp)
        self.map_x = None
        self.map_y = None
        self.needs_update = False

    def apply_warp(self, image: np.ndarray) -> np.ndarray:
        """
        Apply mesh-based warp to an image using pre-computed remap tables.
        Remap tables are cached and only recomputed when mesh changes.

        Args:
            image: Input image (RGB numpy array)

        Returns:
            Warped image
        """
        # If mesh is at default positions, just pass through
        if not self.needs_update and self.map_x is None:
            if self._is_default_grid():
                return image

        height, width = image.shape[:2]

        # Check if we need to rebuild remap tables
        if self.needs_update or self.cached_size != (height, width):
            # If still at default after update check, just pass through
            if self._is_default_grid():
                self.needs_update = False
                self.cached_size = (height, width)
                return image

            self._build_remap_tables(width, height)
            self.needs_update = False
            self.cached_size = (height, width)

        # Apply the warp using cached remap tables (very fast)
        if self.map_x is not None and self.map_y is not None:
            return cv2.remap(image, self.map_x, self.map_y, cv2.INTER_LINEAR)
        else:
            return image

    def _is_default_grid(self) -> bool:
        """Check if mesh is still at default (un-warped) positions."""
        for row in range(self.grid_size + 1):
            for col in range(self.grid_size + 1):
                expected_x = (col / self.grid_size) * 100
                expected_y = (row / self.grid_size) * 100
                actual = self.points[row][col]

                # Allow small tolerance for floating point errors
                if abs(actual[0] - expected_x) > 0.01 or abs(actual[1] - expected_y) > 0.01:
                    return False
        return True

    def _build_remap_tables(self, width: int, height: int):
        """
        Build remap lookup tables using proper bilinear mesh warping.
        Uses inverse mapping: for each destination pixel, find which quad it's in,
        calculate (u,v), then use same (u,v) to find source - ensures continuity.
        """
        try:
            print(f"[MeshWarp] Building remap tables for {width}x{height}...")

            # Initialize remap tables
            self.map_x = np.zeros((height, width), dtype=np.float32)
            self.map_y = np.zeros((height, width), dtype=np.float32)

            # Pre-compute destination quad info for faster lookup
            quad_info = []
            for row in range(self.grid_size):
                for col in range(self.grid_size):
                    # Destination quad corners (warped positions)
                    tl = self.points[row][col]
                    tr = self.points[row][col + 1]
                    br = self.points[row + 1][col + 1]
                    bl = self.points[row + 1][col]

                    dst_corners = np.array([
                        [tl[0] * width / 100, tl[1] * height / 100],
                        [tr[0] * width / 100, tr[1] * height / 100],
                        [br[0] * width / 100, br[1] * height / 100],
                        [bl[0] * width / 100, bl[1] * height / 100]
                    ])

                    # Source quad corners (regular grid)
                    x0 = (col / self.grid_size) * width
                    y0 = (row / self.grid_size) * height
                    x1 = ((col + 1) / self.grid_size) * width
                    y1 = ((row + 1) / self.grid_size) * height

                    src_corners = np.array([
                        [x0, y0],
                        [x1, y0],
                        [x1, y1],
                        [x0, y1]
                    ])

                    # Bounding box for quick rejection
                    bbox = [
                        int(np.floor(dst_corners[:, 0].min())),
                        int(np.floor(dst_corners[:, 1].min())),
                        int(np.ceil(dst_corners[:, 0].max())),
                        int(np.ceil(dst_corners[:, 1].max()))
                    ]

                    quad_info.append({
                        'dst': dst_corners,
                        'src': src_corners,
                        'bbox': bbox,
                        'row': row,
                        'col': col
                    })

            # Process each destination pixel
            print("[MeshWarp] Processing pixels...")
            for y in range(height):
                if y % 50 == 0:
                    print(f"[MeshWarp] Row {y}/{height}")

                for x in range(width):
                    # Guess which quad based on regular grid (optimization)
                    guess_row = min(int(y / height * self.grid_size), self.grid_size - 1)
                    guess_col = min(int(x / width * self.grid_size), self.grid_size - 1)
                    guess_idx = guess_row * self.grid_size + guess_col

                    # Check guessed quad first
                    quad = quad_info[guess_idx]
                    dst_corners = quad['dst']

                    found = False
                    if self._point_in_quad_fast(x, y, dst_corners):
                        found = True
                    else:
                        # Not in guessed quad, check neighbors
                        for quad in quad_info:
                            # Quick bbox check
                            if not (quad['bbox'][0] <= x <= quad['bbox'][2] and
                                    quad['bbox'][1] <= y <= quad['bbox'][3]):
                                continue

                            dst_corners = quad['dst']
                            if self._point_in_quad_fast(x, y, dst_corners):
                                found = True
                                break

                    if found:
                        # Solve for (u,v) in destination quad
                        u, v = self._inverse_bilinear_fast(x, y, dst_corners)

                        # Use same (u,v) to get source position
                        src_corners = quad['src']
                        src_x = ((1-u)*(1-v)*src_corners[0,0] + u*(1-v)*src_corners[1,0] +
                                u*v*src_corners[2,0] + (1-u)*v*src_corners[3,0])
                        src_y = ((1-u)*(1-v)*src_corners[0,1] + u*(1-v)*src_corners[1,1] +
                                u*v*src_corners[2,1] + (1-u)*v*src_corners[3,1])

                        self.map_x[y, x] = src_x
                        self.map_y[y, x] = src_y

            print(f"[MeshWarp] Remap tables built successfully")

        except Exception as e:
            print(f"[MeshWarp] Error building remap tables: {e}")
            import traceback
            traceback.print_exc()
            self.map_x = None
            self.map_y = None

    def _point_in_quad_fast(self, x, y, corners):
        """Fast point-in-quad test using cross products."""
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

        p = np.array([x, y])
        d1 = sign(p, corners[0], corners[1])
        d2 = sign(p, corners[1], corners[2])
        d3 = sign(p, corners[2], corners[3])
        d4 = sign(p, corners[3], corners[0])

        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0) or (d4 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0) or (d4 > 0)

        return not (has_neg and has_pos)

    def _inverse_bilinear_fast(self, x, y, corners):
        """
        Fast inverse bilinear using 2 Newton-Raphson iterations.
        Returns (u, v) coordinates for point (x, y) in quad.
        """
        p0, p1, p2, p3 = corners
        u, v = 0.5, 0.5

        for _ in range(2):  # Just 2 iterations for speed
            # Current residual
            fx = (1-u)*(1-v)*p0[0] + u*(1-v)*p1[0] + u*v*p2[0] + (1-u)*v*p3[0] - x
            fy = (1-u)*(1-v)*p0[1] + u*(1-v)*p1[1] + u*v*p2[1] + (1-u)*v*p3[1] - y

            # Jacobian
            dxdu = -(1-v)*p0[0] + (1-v)*p1[0] + v*p2[0] - v*p3[0]
            dxdv = -(1-u)*p0[0] - u*p1[0] + u*p2[0] + (1-u)*p3[0]
            dydu = -(1-v)*p0[1] + (1-v)*p1[1] + v*p2[1] - v*p3[1]
            dydv = -(1-u)*p0[1] - u*p1[1] + u*p2[1] + (1-u)*p3[1]

            det = dxdu * dydv - dxdv * dydu
            if abs(det) < 1e-10:
                break

            u -= (dydv * fx - dxdv * fy) / det
            v -= (dxdu * fy - dydu * fx) / det

        return np.clip(u, 0, 1), np.clip(v, 0, 1)

    def to_dict(self) -> dict:
        """
        Serialize mesh to dictionary for saving.

        Returns:
            Dictionary with grid_size and points
        """
        return {
            'grid_size': self.grid_size,
            'points': self.points
        }

    def from_dict(self, data: dict):
        """
        Load mesh from dictionary.

        Args:
            data: Dictionary with grid_size and points
        """
        self.grid_size = data.get('grid_size', 4)
        self.points = data.get('points', self._initialize_grid())

        # Only mark for update if not at default positions
        if not self._is_default_grid():
            self.needs_update = True
        else:
            self.needs_update = False
            self.map_x = None
            self.map_y = None

    def set_grid_size(self, new_size: int):
        """
        Change the grid resolution.
        Attempts to interpolate existing points to the new grid.

        Args:
            new_size: New grid size (n for n×n grid)
        """
        if new_size == self.grid_size:
            return

        old_points = self.points
        old_size = self.grid_size
        was_default = self._is_default_grid()

        # Create new grid
        self.grid_size = new_size
        self.points = self._initialize_grid()

        # Interpolate old points to new grid positions
        for new_row in range(new_size + 1):
            for new_col in range(new_size + 1):
                # Calculate corresponding position in old grid
                old_row_float = new_row * old_size / new_size
                old_col_float = new_col * old_size / new_size

                # Bilinear interpolation
                row0 = int(np.floor(old_row_float))
                row1 = min(row0 + 1, old_size)
                col0 = int(np.floor(old_col_float))
                col1 = min(col0 + 1, old_size)

                row_weight = old_row_float - row0
                col_weight = old_col_float - col0

                # Interpolate x coordinate
                x00 = old_points[row0][col0][0]
                x01 = old_points[row0][col1][0]
                x10 = old_points[row1][col0][0]
                x11 = old_points[row1][col1][0]

                x0 = x00 * (1 - col_weight) + x01 * col_weight
                x1 = x10 * (1 - col_weight) + x11 * col_weight
                x = x0 * (1 - row_weight) + x1 * row_weight

                # Interpolate y coordinate
                y00 = old_points[row0][col0][1]
                y01 = old_points[row0][col1][1]
                y10 = old_points[row1][col0][1]
                y11 = old_points[row1][col1][1]

                y0 = y00 * (1 - col_weight) + y01 * col_weight
                y1 = y10 * (1 - col_weight) + y11 * col_weight
                y = y0 * (1 - row_weight) + y1 * row_weight

                self.points[new_row][new_col] = [x, y]

        # Only mark for update if the grid was warped before
        if not was_default:
            self.needs_update = True
        else:
            # Still at default, no update needed
            self.needs_update = False
            self.map_x = None
            self.map_y = None
