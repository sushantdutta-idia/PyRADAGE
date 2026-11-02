"""Shared publication exports preserve data and usable text across formats."""
from __future__ import annotations

import struct
from xml.etree import ElementTree

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
from matplotlib import rc_context, rcParams
from matplotlib.text import Text

from fr2dynage.plotting import (
    AXIS_LABEL_SIZE, EXPORT_FORMATS, LEGEND_SIZE, PUBLICATION_STYLE,
    RASTER_DPI, SUPTITLE_SIZE, TICK_SIZE, TITLE_SIZE,
    publication_figure, save_publication_figure, style_publication_figure,
)


def _png_resolution(path):
    """Read standard PNG physical-density metadata without extra test deps."""
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    offset = 8
    while offset < len(data):
        size = struct.unpack(">I", data[offset:offset+4])[0]
        name = data[offset+4:offset+8]
        if name == b"pHYs":
            x, y, unit = struct.unpack(">IIB", data[offset+8:offset+8+size])
            assert unit == 1
            return x*.0254, y*.0254
        offset += size+12
    pytest.fail("PNG has no physical-resolution metadata")


def test_exports_are_large_type_high_resolution_and_editable(tmp_path):
    # A small figure keeps raster memory bounded while exercising all output
    # backends and pre-existing small labels, rather than default-only fonts.
    with rc_context({"font.size": 8., "pdf.fonttype": 3, "svg.fonttype": "path"}):
        before = {key: rcParams[key] for key in PUBLICATION_STYLE}
        figure = publication_figure(figsize=(3.4, 2.8))
        axis = figure.add_subplot()
        x, y = np.array([1., 2., 3.]), np.array([2., 3., 2.5])
        line, = axis.plot(x, y, label="Measured")
        axis.set_xlabel("Frequency", fontsize=7)
        axis.set_ylabel("Intensity", fontsize=7)
        axis.set_title("Core spectrum", fontsize=7)
        axis.set_title("Left title", loc="left", fontsize=7)
        axis.set_title("Right title", loc="right", fontsize=7)
        figure.suptitle("Source title", fontsize=7)
        axis.tick_params(labelsize=6)
        axis.legend(fontsize=6)
        files = save_publication_figure(figure, tmp_path/"spectrum.png")
        assert files == [tmp_path/f"spectrum.{fmt}" for fmt in EXPORT_FORMATS]
        assert axis.xaxis.label.get_fontsize() >= AXIS_LABEL_SIZE
        assert axis.yaxis.label.get_fontsize() >= AXIS_LABEL_SIZE
        assert axis.title.get_fontsize() >= TITLE_SIZE
        assert all(text.get_fontsize() >= TICK_SIZE for text in axis.get_xticklabels()+axis.get_yticklabels())
        assert all(text.get_fontsize() >= LEGEND_SIZE for text in axis.get_legend().get_texts())
        assert all(text.get_fontsize() >= LEGEND_SIZE for text in figure.findobj(match=Text) if text.get_text())
        np.testing.assert_array_equal(line.get_xdata(), x)
        np.testing.assert_array_equal(line.get_ydata(), y)
        assert _png_resolution(tmp_path/"spectrum.png") == pytest.approx((RASTER_DPI, RASTER_DPI), abs=.1)
        pdf = (tmp_path/"spectrum.pdf").read_bytes()
        assert pdf.startswith(b"%PDF-")
        assert b"/FontFile2" in pdf
        assert b"/CIDFontType2" in pdf
        assert b"/Subtype /Type3" not in pdf
        svg = ElementTree.parse(tmp_path/"spectrum.svg")
        text_nodes = svg.findall(".//{http://www.w3.org/2000/svg}text")
        assert text_nodes
        exported_text = ' '.join(''.join(node.itertext()) for node in text_nodes)
        assert 'Frequency' in exported_text and 'Intensity' in exported_text
        for title in ['Core spectrum', 'Left title', 'Right title', 'Source title']:
            assert title not in exported_text
        assert all(not title.get_visible() and not title.get_in_layout()
                   for title in [axis.title,axis._left_title,axis._right_title,figure._suptitle])
        assert {key: rcParams[key] for key in PUBLICATION_STYLE} == before
        figure.clear()


def test_finalizer_enlarges_annotations_and_preserves_larger_labels():
    figure = publication_figure(figsize=(3.4, 2.8))
    axis = figure.add_subplot()
    axis.plot([1., 2.], [2., 1.])
    axis.set_xlabel("Large label", fontsize=28)
    axis.tick_params(labelsize=26)
    annotation = axis.annotate("Core", (1., 2.), fontsize=5)
    figure.suptitle("Source", fontsize=7)
    style_publication_figure(figure)
    assert axis.xaxis.label.get_fontsize() == 28
    assert annotation.get_fontsize() >= 18
    assert figure._suptitle.get_fontsize() >= SUPTITLE_SIZE
    assert not figure._suptitle.get_visible()
    assert all(text.get_fontsize() >= 26 for text in axis.get_xticklabels()+axis.get_yticklabels())
    figure.clear()


def test_top_titles_are_an_explicit_low_level_opt_in(tmp_path):
    figure=publication_figure(figsize=(3.4,2.8))
    axis=figure.add_subplot();axis.plot([0.,1.],[1.,2.])
    axis.set_title('Explicit title');figure.suptitle('Explicit source')
    save_publication_figure(figure,tmp_path/'titles.svg',formats=('svg',),show_titles=True)
    exported=(tmp_path/'titles.svg').read_text()
    assert 'Explicit title' in exported and 'Explicit source' in exported
    assert axis.title.get_visible() and figure._suptitle.get_visible()
    figure.clear()


@pytest.mark.parametrize("formats", [(), ("png", "png"), ("eps",), ("jpg", "pdf")])
def test_invalid_export_formats_do_not_publish(tmp_path, formats):
    figure = publication_figure(figsize=(2., 2.))
    with pytest.raises(ValueError, match="publication formats"):
        save_publication_figure(figure, tmp_path/"bad.png", formats=formats)
    assert list(tmp_path.iterdir()) == []
    figure.clear()


def test_single_format_targets_exact_suffix_and_optional_clear(tmp_path):
    figure = publication_figure(figsize=(2.5, 2.))
    axis = figure.add_subplot()
    axis.text(.5, .5, "Radio", ha="center")
    targets = save_publication_figure(figure, tmp_path/"only.svg", formats=("svg",), clear=True)
    assert targets == [tmp_path/"only.svg"]
    assert {p.name for p in tmp_path.iterdir()} == {"only.svg"}
    assert not figure.axes
