"""Draw-time legibility check for matplotlib figures.

Figures for posts and reports are saved through :func:`save_checked`. It draws
the figure first and refuses to save it when any text overlaps other text, a
data mark or the legend frame, when two scatter points overlap, when text
leaves the canvas, when a font renders below ``min_points`` at the width the
figure is shown at, or when a legend has more entries than a reader can hold.
Overlap can be measured reliably only from matplotlib's own layout, not from
the PNG, which is why the check runs where the figure is built.

Fill areas (``fill_between``) and grid lines are not treated as marks: text may
sit on a light background, but never on a point, line, error bar or bar. A
whisker or reference line may cross a point.

Copy this file into the project (for example ``harness/figure_check.py``) and
save every figure with ``save_checked(figure, path)``.
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.transforms import Bbox

#: Gap in points that two shapes must keep to count as separate.
PAD_POINTS = 1.0
#: Width in inches at which figures are shown (a Hugging Face blog column).
DISPLAY_WIDTH_IN = 8.0


class FigureLegibilityError(ValueError):
    """Raised when a figure has overlapping, clipped or too-small text."""


def _is_marker(marker: object) -> bool:
    return marker not in (None, "None", "none", "", " ")


def _drawn_texts(figure: Figure) -> list[Text]:
    """Return every visible, non-empty text, skipping tick labels outside the view."""
    hidden_ticks: set[int] = set()
    for axes in figure.axes:
        for axis in (axes.xaxis, axes.yaxis):
            low, high = sorted(axis.get_view_interval())
            eps = 1e-9 * max(abs(low), abs(high), 1.0)
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if not low - eps <= tick.get_loc() <= high + eps:
                    hidden_ticks.update({id(tick.label1), id(tick.label2)})
    texts = []
    for text in figure.findobj(Text):
        if id(text) in hidden_ticks or not text.get_visible() or not text.get_text().strip():
            continue
        if text.get_alpha() == 0:
            continue
        texts.append(text)
    return texts


def _marks(figure: Figure) -> list[tuple[str, np.ndarray, np.ndarray, list[Bbox]]]:
    """Return the data marks of every axes as display-space shapes.

    Each entry is ``(description, circles, segments, boxes)``: circles as an
    ``(n, 3)`` array of x, y and radius, segments as ``(m, 2, 3)`` pairs of
    points with their half line width, and rectangles as boxes.
    """
    to_px = figure.dpi / 72
    marks = []
    for axes in figure.axes:
        for line in axes.lines:
            if not line.get_visible() or line.get_alpha() == 0:
                continue
            points = line.get_transform().transform(line.get_xydata())
            circles = np.empty((0, 3))
            segments = np.empty((0, 2, 3))
            if _is_marker(line.get_marker()):
                radius = (line.get_markersize() / 2 + line.get_markeredgewidth() / 2) * to_px
                circles = np.column_stack([points, np.full(len(points), radius)])
            if line.get_linestyle() not in ("None", "none", "", " ") and line.get_linewidth() > 0:
                half = line.get_linewidth() / 2 * to_px
                pairs = np.stack([points[:-1], points[1:]], axis=1)
                segments = np.concatenate([pairs, np.full((len(pairs), 2, 1), half)], axis=2)
            marks.append((f"line {line.get_label()!r}", circles, segments, []))
        for collection in axes.collections:
            if not collection.get_visible() or collection.get_alpha() == 0:
                continue
            if isinstance(collection, PathCollection):
                offsets = collection.get_offset_transform().transform(collection.get_offsets())
                sizes = np.broadcast_to(collection.get_sizes(), (len(offsets),))
                edge = float(np.max(collection.get_linewidths(), initial=0)) / 2
                # 1.2 covers square and diamond markers, whose corners reach past sqrt(s)/2.
                radius = (1.2 * np.sqrt(sizes) / 2 + edge) * to_px
                circles = np.column_stack([offsets, radius])
                marks.append(
                    (f"scatter {collection.get_label()!r}", circles, np.empty((0, 2, 3)), [])
                )
            elif isinstance(collection, LineCollection):
                transform = collection.get_transform()
                half = float(np.max(collection.get_linewidths(), initial=0)) / 2 * to_px
                rows = []
                for segment in collection.get_segments():
                    points = transform.transform(np.asarray(segment))
                    for a, b in itertools.pairwise(points):
                        rows.append([[*a, half], [*b, half]])
                segments = np.array(rows).reshape(-1, 2, 3)
                marks.append((f"lines {collection.get_label()!r}", np.empty((0, 3)), segments, []))
        boxes = [
            patch.get_window_extent()
            for patch in axes.patches
            if isinstance(patch, Rectangle) and patch.get_visible() and patch.get_alpha() != 0
        ]
        if boxes:
            marks.append(("bars", np.empty((0, 3)), np.empty((0, 2, 3)), boxes))
    return marks


def _box_hits_circles(box: Bbox, circles: np.ndarray, pad: float) -> bool:
    if not len(circles):
        return False
    nearest_x = np.clip(circles[:, 0], box.x0, box.x1)
    nearest_y = np.clip(circles[:, 1], box.y0, box.y1)
    distance = np.hypot(circles[:, 0] - nearest_x, circles[:, 1] - nearest_y)
    return bool(np.any(distance < circles[:, 2] + pad))


def _box_hits_segments(box: Bbox, segments: np.ndarray, pad: float) -> bool:
    for (x0, y0, half), (x1, y1, _) in segments:
        steps = max(2, math.ceil(math.hypot(x1 - x0, y1 - y0)) + 1)
        xs, ys = np.linspace(x0, x1, steps), np.linspace(y0, y1, steps)
        reach = half + pad
        inside = (
            (xs > box.x0 - reach)
            & (xs < box.x1 + reach)
            & (ys > box.y0 - reach)
            & (ys < box.y1 + reach)
        )
        if inside.any():
            return True
    return False


def _overlap(a: Bbox, b: Bbox, pad: float) -> bool:
    return min(a.x1, b.x1) - max(a.x0, b.x0) > -pad and min(a.y1, b.y1) - max(a.y0, b.y0) > -pad


def _hits(box: Bbox, mark: tuple[str, np.ndarray, np.ndarray, list[Bbox]], pad: float) -> bool:
    _, circles, segments, rectangles = mark
    return (
        _box_hits_circles(box, circles, pad)
        or _box_hits_segments(box, segments, pad)
        or any(_overlap(box, rectangle, pad) for rectangle in rectangles)
    )


def check_legibility(
    figure: Figure,
    *,
    min_points: float = 9.0,
    display_width_in: float = DISPLAY_WIDTH_IN,
    max_legend_entries: int = 8,
) -> None:
    """Raise :class:`FigureLegibilityError` listing every legibility problem of *figure*.

    Args:
        figure: A fully built figure.
        min_points: Smallest font size allowed once the figure is scaled to
            ``display_width_in``.
        display_width_in: Width at which readers see the figure.
        max_legend_entries: Most entries one legend may have.
    """
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    texts = _drawn_texts(figure)
    boxes = {id(text): text.get_window_extent(renderer) for text in texts}
    scale = display_width_in / figure.get_figwidth()
    canvas = figure.bbox
    pad = PAD_POINTS * figure.dpi / 72
    problems = []

    for text in texts:
        shown = text.get_fontsize() * scale
        if shown < min_points:
            problems.append(f"{text.get_text()!r} renders at {shown:.1f} pt (< {min_points:g})")
        box = boxes[id(text)]
        if (
            box.x0 < canvas.x0 - 0.5
            or box.y0 < canvas.y0 - 0.5
            or box.x1 > canvas.x1 + 0.5
            or box.y1 > canvas.y1 + 0.5
        ):
            problems.append(f"{text.get_text()!r} runs off the canvas")

    for a, b in itertools.combinations(texts, 2):
        if _overlap(boxes[id(a)], boxes[id(b)], pad):
            problems.append(f"{a.get_text()!r} overlaps {b.get_text()!r}")

    marks = _marks(figure)
    for text in texts:
        for mark in marks:
            if _hits(boxes[id(text)], mark, pad):
                problems.append(f"{text.get_text()!r} overlaps {mark[0]}")

    legends = [axes.get_legend() for axes in figure.axes] + list(figure.legends)
    for legend in filter(None, legends):
        frame = legend.get_window_extent(renderer)
        entries = len([t for t in legend.get_texts() if t.get_text().strip()])
        if entries > max_legend_entries:
            problems.append(f"a legend has {entries} entries (> {max_legend_entries})")
        for mark in marks:
            if _hits(frame, mark, pad):
                problems.append(f"the legend overlaps {mark[0]}")
        # Any other text under the legend's box (a title, an annotation): its swatches cover it.
        own = {id(t) for t in legend.findobj(Text)}
        for text in texts:
            if id(text) not in own and _overlap(frame, boxes[id(text)], pad):
                problems.append(f"the legend overlaps {text.get_text()!r}")

    points = [
        (description, circle)
        for description, circles, _, _ in marks
        if description.startswith("scatter")
        for circle in circles
    ]
    for (name_a, a), (name_b, b) in itertools.combinations(points, 2):
        # Undo the 1.2 corner allowance: neighbours may touch at their corners.
        if math.hypot(a[0] - b[0], a[1] - b[1]) < (a[2] + b[2]) / 1.2:
            problems.append(f"two points overlap ({name_a} and {name_b})")

    if problems:
        raise FigureLegibilityError("figure is not legible:\n- " + "\n- ".join(problems))


def save_checked(figure: Figure, path: Path, *, dpi: int = 200, **check: float) -> Path:
    """Check *figure* with :func:`check_legibility`, then save it to *path* on white."""
    check_legibility(figure, **check)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, facecolor="white")
    return path
