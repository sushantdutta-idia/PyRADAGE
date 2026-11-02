"""Shared typography and publication exports, without global rc changes.

Callers own staging, collision checks and final publication. Scientific arrays
and model assumptions are never modified by these rendering helpers.
"""
from pathlib import Path

FONT_SIZE = 18
AXIS_LABEL_SIZE = 20
TITLE_SIZE = 20
TICK_SIZE = 17
LEGEND_SIZE = 16
SUPTITLE_SIZE = 21
RASTER_DPI = 600
EXPORT_FORMATS = ("png", "pdf", "svg")

PUBLICATION_STYLE = {
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": FONT_SIZE, "axes.labelsize": AXIS_LABEL_SIZE,
    "axes.titlesize": TITLE_SIZE, "axes.titlepad": 12., "axes.labelpad": 8.,
    "axes.linewidth": 1.3, "axes.formatter.use_mathtext": True,
    "xtick.labelsize": TICK_SIZE, "ytick.labelsize": TICK_SIZE,
    "xtick.major.size": 7., "ytick.major.size": 7.,
    "xtick.minor.size": 4., "ytick.minor.size": 4.,
    "xtick.major.width": 1.2, "ytick.major.width": 1.2,
    "xtick.minor.width": 1., "ytick.minor.width": 1.,
    "legend.fontsize": LEGEND_SIZE, "legend.title_fontsize": LEGEND_SIZE,
    "figure.titlesize": SUPTITLE_SIZE, "lines.linewidth": 2.,
    "lines.markersize": 6., "errorbar.capsize": 3.,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "savefig.dpi": RASTER_DPI,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "text.usetex": False, "mathtext.fontset": "dejavusans",
}


def publication_figure(*, figsize=(12, 9), layout="constrained"):
    """Create a figure; final styling also reaches axes added subsequently."""
    from matplotlib import rc_context
    from matplotlib.figure import Figure
    with rc_context(PUBLICATION_STYLE):
        return Figure(figsize=figsize, layout=layout)


def _enlarge(text, minimum):
    text.set_fontsize(max(text.get_fontsize(), minimum))
    text.set_fontfamily("DejaVu Sans")


def style_publication_figure(figure, *, show_titles=False):
    """Apply readable typography and omit top titles by default.

    Title artists retain their text for programmatic provenance, but are
    hidden and excluded from layout for a clean publication export. Captions
    and scientific scope belong in the accompanying report; labels, legends,
    annotations and colour bars remain visible. ``show_titles`` is an explicit
    low-level opt-in, never the default workflow behaviour.
    """
    from matplotlib.text import Text
    for text in figure.findobj(match=Text):
        _enlarge(text, LEGEND_SIZE)
    for axis in figure.axes:
        axis.tick_params(axis="both", which="major", length=7., width=1.2)
        # WCSAxes permits only length for minor ticks; major tick styling
        # already supplies the visible sky-coordinate width.
        if hasattr(axis, "coords"):
            axis.tick_params(axis="both", which="minor", length=4.)
        else:
            axis.tick_params(axis="both", which="minor", length=4., width=1.)
        for spine in axis.spines.values():
            spine.set_linewidth(1.3)
        axis.xaxis.labelpad = max(axis.xaxis.labelpad, 8.)
        axis.yaxis.labelpad = max(axis.yaxis.labelpad, 8.)
        for text in (axis.xaxis.label, axis.yaxis.label):
            _enlarge(text, AXIS_LABEL_SIZE)
        for title in (axis.title, axis._left_title, axis._right_title):
            _enlarge(title, TITLE_SIZE)
        if axis.title.get_text():
            axis.set_title(axis.title.get_text(), fontsize=axis.title.get_fontsize(),
                           fontfamily="DejaVu Sans", pad=14.)
        for title in (axis.title, axis._left_title, axis._right_title):
            title.set_visible(bool(show_titles))
            title.set_in_layout(bool(show_titles))
        for text in (*axis.get_xticklabels(which="both"), *axis.get_yticklabels(which="both"),
                     axis.xaxis.get_offset_text(), axis.yaxis.get_offset_text()):
            _enlarge(text, TICK_SIZE)
        for text in axis.texts:
            _enlarge(text, FONT_SIZE)
        legend = axis.get_legend()
        if legend is not None:
            for text in (*legend.get_texts(), legend.get_title()):
                _enlarge(text, LEGEND_SIZE)
        for line in axis.lines:
            if line.get_linewidth() > 0:
                line.set_linewidth(max(line.get_linewidth(), 1.5))
    for legend in figure.legends:
        for text in (*legend.get_texts(), legend.get_title()):
            _enlarge(text, LEGEND_SIZE)
    for text in figure.texts:
        _enlarge(text, SUPTITLE_SIZE if text is figure._suptitle else FONT_SIZE)
    if figure._suptitle is not None:
        figure._suptitle.set_visible(bool(show_titles))
        figure._suptitle.set_in_layout(bool(show_titles))
    return figure


def save_publication_figure(figure, path, *, formats=EXPORT_FORMATS, clear=False, show_titles=False):
    """Render title-free large-font PNG (600 dpi), editable PDF and SVG.

    Each requested format replaces the path suffix. A single ``pdf`` format
    and a path already ending in .pdf therefore render to that exact path.
    Existing destinations are caller-managed, allowing zero-byte mkstemp
    files. Public writers must preflight and atomically publish these files.
    """
    from matplotlib import rc_context
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    formats = tuple(formats)
    if not formats or len(set(formats)) != len(formats) or any(fmt not in EXPORT_FORMATS for fmt in formats):
        raise ValueError("publication formats must be distinct png, pdf or svg")
    path = Path(path)
    targets = [path.with_suffix("."+fmt) for fmt in formats]
    FigureCanvasAgg(figure)
    try:
        with rc_context(PUBLICATION_STYLE):
            style_publication_figure(figure, show_titles=show_titles)
            for fmt, target in zip(formats, targets):
                figure.savefig(target, format=fmt, dpi=RASTER_DPI,
                               bbox_inches="tight", pad_inches=.12)
        return targets
    finally:
        if clear:
            figure.clear()
