"""Geometry helpers shared by the zone and line engines.

Everything here works in normalised coordinates (0.0-1.0) so a zone profile
is resolution independent. Pure functions, no state, trivially testable.
"""

from __future__ import annotations

import math
from typing import Sequence

Point = tuple[float, float]
Polygon = Sequence[Point]


def point_in_polygon(px: float, py: float, polygon: Polygon) -> bool:
    """Ray-casting point-in-polygon test.

    Counts how often a ray cast to the right from ``(px, py)`` crosses the
    polygon edges. An odd count means the point is inside. Works for convex
    and concave polygons alike.

    Points exactly on an edge are not guaranteed to fall on a particular
    side. That is acceptable here: a detection centre lands on a boundary for
    at most one frame, and the zone engine debounces state changes anyway.
    """
    n = len(polygon)
    if n < 3:
        return False

    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        # Does the edge straddle the horizontal ray?
        if (yi > py) != (yj > py):
            # x coordinate where the edge crosses the ray
            denominator = (yj - yi)
            if denominator != 0.0:
                x_cross = (xj - xi) * (py - yi) / denominator + xi
                if px < x_cross:
                    inside = not inside
        j = i
    return inside


def polygon_centroid(polygon: Polygon) -> Point:
    """Area centroid, falling back to the vertex mean for degenerate shapes."""
    n = len(polygon)
    if n == 0:
        return (0.0, 0.0)
    if n < 3:
        return (
            sum(p[0] for p in polygon) / n,
            sum(p[1] for p in polygon) / n,
        )

    area2 = 0.0
    cx = 0.0
    cy = 0.0
    for i in range(n):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        area2 += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross

    if abs(area2) < 1e-12:
        return (
            sum(p[0] for p in polygon) / n,
            sum(p[1] for p in polygon) / n,
        )
    factor = 1.0 / (3.0 * area2)
    return (cx * factor, cy * factor)


def polygon_area(polygon: Polygon) -> float:
    """Absolute shoelace area."""
    n = len(polygon)
    if n < 3:
        return 0.0
    total = 0.0
    for i in range(n):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % n]
        total += x0 * y1 - x1 * y0
    return abs(total) / 2.0


def line_side(px: float, py: float, start: Point, end: Point) -> float:
    """Signed cross product telling which side of a line a point is on.

    Positive and negative are opposite sides; near zero means on the line.
    The magnitude is proportional to the distance, so it can also be used as
    a cheap distance proxy.
    """
    return ((end[0] - start[0]) * (py - start[1])
            - (end[1] - start[1]) * (px - start[0]))


def segments_intersect(p1: Point, p2: Point, p3: Point, p4: Point) -> bool:
    """True when segment p1-p2 crosses segment p3-p4.

    Uses the standard orientation test. Collinear overlap is treated as an
    intersection, which is the behaviour we want for a boundary crossing.
    """
    def orientation(a: Point, b: Point, c: Point) -> int:
        val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
        if abs(val) < 1e-12:
            return 0
        return 1 if val > 0 else 2

    def on_segment(a: Point, b: Point, c: Point) -> bool:
        return (min(a[0], c[0]) - 1e-12 <= b[0] <= max(a[0], c[0]) + 1e-12
                and min(a[1], c[1]) - 1e-12 <= b[1] <= max(a[1], c[1]) + 1e-12)

    o1 = orientation(p1, p2, p3)
    o2 = orientation(p1, p2, p4)
    o3 = orientation(p3, p4, p1)
    o4 = orientation(p3, p4, p2)

    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and on_segment(p1, p3, p2):
        return True
    if o2 == 0 and on_segment(p1, p4, p2):
        return True
    if o3 == 0 and on_segment(p3, p1, p4):
        return True
    if o4 == 0 and on_segment(p3, p2, p4):
        return True
    return False


def point_segment_distance(px: float, py: float,
                           start: Point, end: Point) -> float:
    """Shortest distance from a point to a line SEGMENT, not an infinite line.

    The distinction matters for a boundary. An infinite line would report a
    person standing well past the end of a fence as being right on it, which
    is how a system ends up alarming about somebody walking through an open
    gateway twenty metres beyond where the fence stops.
    """
    ax, ay = start
    bx, by = end
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-12:
        return math.hypot(px - ax, py - ay)
    # Projection of the point onto the segment, clamped to its ends.
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def distance(a: Point, b: Point) -> float:
    """Euclidean distance."""
    return math.hypot(b[0] - a[0], b[1] - a[1])


def heading_degrees(a: Point, b: Point) -> float:
    """Compass-style heading from *a* to *b*, in degrees.

    0 is north (upward in image coordinates, where y decreases going up),
    90 is east, 180 south, 270 west. Returns 0.0 for zero movement.
    """
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    if abs(dx) < 1e-12 and abs(dy) < 1e-12:
        return 0.0
    # Image y grows downward, so negate dy to get a conventional bearing.
    angle = math.degrees(math.atan2(dx, -dy))
    return angle % 360.0


def angle_difference(a: float, b: float) -> float:
    """Smallest absolute difference between two bearings, 0-180."""
    diff = abs(a - b) % 360.0
    return diff if diff <= 180.0 else 360.0 - diff


def compass_direction(bearing: float) -> str:
    """Map a bearing to north / east / south / west."""
    bearing %= 360.0
    if bearing < 45.0 or bearing >= 315.0:
        return "north"
    if bearing < 135.0:
        return "east"
    if bearing < 225.0:
        return "south"
    return "west"


def bounding_box(polygon: Polygon) -> tuple[float, float, float, float]:
    """Axis-aligned bounds of a polygon as (x1, y1, x2, y2)."""
    if not polygon:
        return (0.0, 0.0, 0.0, 0.0)
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return (min(xs), min(ys), max(xs), max(ys))


def iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Intersection over union of two [x1, y1, x2, y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
