"""Every layout label this package names is one Surya can return.

A label is a string, and a string that matches nothing fails silently. Two
did. A block Surya labels Diagram comes back unread, like a Picture, and
FIGURE_LABELS did not hold it: the block left no figure, no text and no
trace in the cache. And display equations were looked for under "Formula",
headings under "Title"; Surya's words are Equation and SectionHeader, so
neither test ever passed.

The vocabulary is read from the installed package's own files, without
importing it (importing surya loads torch).
"""
import ast
import importlib.util
import pathlib

import pytest

from codicology import pipeline as vtb

SRC = pathlib.Path(vtb.__file__).parent


def _surya_source(*parts):
    spec = importlib.util.find_spec("surya")
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("surya is not installed")
    return pathlib.Path(list(spec.submodule_search_locations)[0],
                        *parts).read_text(encoding="utf-8")


def _assigned(source, name):
    """The literal a module assigns to `name`."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} is no longer assigned as a literal")


@pytest.fixture(scope="module")
def relabel():
    """Raw model label -> the public name a block carries."""
    return _assigned(_surya_source("layout", "label.py"), "LAYOUT_PRED_RELABEL")


@pytest.fixture(scope="module")
def unread(relabel):
    """The public names of the blocks Surya returns without reading."""
    raw = _assigned(_surya_source("inference", "prompts.py"), "SKIP_OCR_LABELS")
    return {relabel.get(r, r) for r in raw}


def _strings(node):
    return {n.value for n in ast.walk(node)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)}


def _is_label(node):
    return (isinstance(node, ast.Attribute) and node.attr == "label") or (
        isinstance(node, ast.Name) and node.id in ("label", "lab"))


def _labels_named(path):
    """Every string a module compares a block's label with, and every
    string in a set it calls *_LABELS."""
    named = {}
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        found = set()
        if isinstance(node, ast.Compare):
            sides = [node.left, *node.comparators]
            if any(_is_label(s) for s in sides):
                for s in sides:
                    if not _is_label(s):
                        found |= _strings(s)
        elif isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id.endswith("_LABELS")
                for t in node.targets):
            found = _strings(node.value)
        for label in found:
            named.setdefault(label, node.lineno)
    return named


def test_the_scan_sees_the_labels_the_pipeline_acts_on():
    """Guard the guard: if the source stops comparing labels this way, the
    test below would pass by finding nothing."""
    named = set(_labels_named(SRC / "pipeline.py"))
    assert {"Picture", "Figure", "Diagram", "Caption", "PageHeader",
            "PageFooter", "SectionHeader", "Footnote", "Form",
            "TableOfContents"} <= named


@pytest.mark.parametrize("module", sorted(p.name for p in SRC.glob("*.py")))
def test_every_label_named_is_one_surya_returns(module, relabel):
    vocabulary = set(relabel.values())
    strangers = {label: line
                 for label, line in _labels_named(SRC / module).items()
                 if label not in vocabulary}
    assert not strangers, (
        f"{module} tests a block's label against names Surya does not "
        f"return (name: line) {strangers}; its vocabulary is "
        f"{sorted(vocabulary)}")


def test_every_block_surya_leaves_unread_is_kept_as_a_picture(unread):
    """An unread block has no HTML: unless its pixels are kept it is gone."""
    assert unread - {"BlankPage"} == set(vtb.FIGURE_LABELS)


# ── a Diagram block through the backend ──────────────────────────────────────

class _Block:
    def __init__(self, label, box, html="", skipped=False, order=0):
        x0, y0, x1, y1 = box
        self.polygon = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        self.label, self.html, self.skipped = label, html, skipped
        self.error, self.reading_order, self.confidence = False, order, 1.0


class _Page:
    def __init__(self, blocks, size):
        self.blocks, self.image_bbox = blocks, [0, 0, *size]


def _backend(pages):
    backend = object.__new__(vtb.SuryaBackend)      # no server, no model
    backend.langs, backend._shape = ["en"], "predictor"
    backend._rec = lambda images: pages
    return backend


def test_a_diagram_block_is_kept_as_a_figure():
    from PIL import Image, ImageDraw
    page = Image.new("RGB", (800, 1000), "white")
    ImageDraw.Draw(page).ellipse((200, 300, 600, 700), fill="black")
    blocks = [_Block("Text", (80, 80, 720, 200), "<p>As the diagram shows,</p>"),
              _Block("Diagram", (180, 280, 620, 720), skipped=True, order=1),
              _Block("Caption", (180, 740, 620, 780), "<p>Fig. 3</p>", order=2)]
    (items,) = _backend([_Page(blocks, page.size)]).run_items([page])

    assert [it.label for it in items] == ["Text", "Diagram", "Caption"]
    diagram = items[1]
    assert diagram.figure is not None and diagram.figure.size == (440, 440)
    assert diagram.figure.getpixel((220, 220)) == (0, 0, 0)    # the ink itself
    assert not diagram.html


def test_a_blank_page_block_leaves_nothing():
    from PIL import Image
    page = Image.new("RGB", (800, 1000), "white")
    blocks = [_Block("BlankPage", (0, 0, 800, 1000), skipped=True)]
    (items,) = _backend([_Page(blocks, page.size)]).run_items([page])
    assert items == []
