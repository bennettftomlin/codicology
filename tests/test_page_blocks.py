"""A page as the assembly half receives it: its blocks, before they are
joined (model.Page). Phase 2 of the plan: the reading half hands over
pages of blocks, and the strings every later pass reads are rendered from
them, so nothing downstream changes."""
from ebooklib import epub
from PIL import Image, ImageDraw

from codicology import pipeline as P
from codicology.model import Block, Page, PageItem, kind_of


def _drawing():
    im = Image.new("RGB", (300, 200), "white")
    d = ImageDraw.Draw(im)
    for k in range(0, 200, 20):
        d.line((0, k, 300, 200 - k), fill="black", width=3)
    d.ellipse((80, 40, 220, 160), outline="black", width=4)
    return im


class Reader:
    batch_size = 4
    name = "fake"
    langs = ["en"]

    def __init__(self, pages):
        self.pages = pages

    def run_items(self, images):
        return [self.pages[k] for k in range(len(images))]


def _pages(tmp_path, n):
    paths = []
    for k in range(n):
        im = Image.new("RGB", (600, 800), "white")
        ImageDraw.Draw(im).rectangle((60, 80, 540, 600), fill=(40, 40, 40))
        p = tmp_path / f"page_{k:04d}.png"
        im.save(p)
        paths.append(str(p))
    return paths


def test_the_pages_are_the_blocks_and_the_bodies_their_rendering(tmp_path, monkeypatch):
    first = [
        PageItem(html="<p>12</p>", is_furniture=True, label="PageHeader", box=(0.4, 0.02, 0.6, 0.05)),
        PageItem(html="<h2>The Firm</h2>", label="SectionHeader", box=(0.1, 0.1, 0.9, 0.15), conf=0.99),
        PageItem(html="<p>Its books were kept in a fair hand.<sup>1</sup></p>", label="Text",
                 box=(0.1, 0.2, 0.9, 0.5), conf=0.97),
        PageItem(html="<p>1 So the clerk said.</p>", label="Footnote", box=(0.1, 0.9, 0.9, 0.95)),
    ]
    second = [
        PageItem(figure=_drawing(), label="Picture", box=(0.1, 0.1, 0.9, 0.5)),
        PageItem(html="Figure 1. The counting house.", is_caption=True, label="Caption",
                 box=(0.1, 0.52, 0.9, 0.56)),
        PageItem(html="<p>It stood by the river.</p>", label="Text", box=(0.1, 0.6, 0.9, 0.8)),
    ]
    paths = _pages(tmp_path, 2)
    monkeypatch.setattr(P, "better_figure", lambda path, box, crop: crop)
    rp = P._read_pages(paths, Reader([first, second]), None, epub.EpubBook(), False)

    assert len(rp.pages) == 2 and all(isinstance(p, Page) for p in rp.pages)
    assert rp.bodies == [p.render() for p in rp.pages]

    one = rp.pages[0]
    assert [b.kind for b in one.blocks] == ["heading", "text", "rule", "footnote"]
    assert one.blocks[0].label == "SectionHeader" and one.blocks[0].conf == 0.99
    assert one.blocks[1].box == (0.1, 0.2, 0.9, 0.5)
    assert one.blocks[2].html == "<hr/>" and one.blocks[2].label is None
    assert [(b.kind, b.html) for b in one.furniture] == [("furniture", "<p>12</p>")]
    assert rp.page_furniture[0] == ["12"]

    two = rp.pages[1]
    assert [b.kind for b in two.blocks] == ["figure", "text"]
    fig = two.blocks[0]
    assert "Figure 1. The counting house." in fig.caption and fig.label == "Picture"
    # the figure's file name is settled in the block as in the body
    assert 'src="images/fig_0000.' in fig.html and ".encoding" not in fig.html
    assert rp.page_figures[1] and rp.page_figures[1][0] in fig.html


def test_a_kind_is_the_label_in_the_assembly_half_s_words():
    assert kind_of(PageItem(html="<p>x</p>", label="Footnote")) == "footnote"
    assert kind_of(PageItem(html="<p>x</p>", label="Formula")) == "equation"
    assert kind_of(PageItem(html="<p>x</p>", label=None)) == "text"
    assert kind_of(PageItem(html="<p>x</p>", label="Something new")) == "text"
    assert kind_of(PageItem(html="<p>12</p>", label="Text", is_furniture=True)) == "furniture"
    assert kind_of(PageItem(whole_page=True)) == "figure"
    b = Block.of(PageItem(html="<p>a</p>", label="ListItem", box=(0, 0, 1, 1), conf=0.5))
    assert (b.kind, b.label, b.box, b.conf) == ("list-item", "ListItem", (0, 0, 1, 1), 0.5)
