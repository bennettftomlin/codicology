"""A line the layout called furniture that the book never repeats.

The reading trusts the layout's PageHeader and PageFooter labels and sets
those blocks aside. What makes a line furniture is that it recurs; a
plate's caption, an author's note at the foot of an article or a printer's
imprint sits at a page's edge once, and was lost from the book.
"""
from codicology.furniture import restore_lone_furniture
from codicology.model import Block, Page


def _page(i, body_blocks, furniture):
    return Page(index=i, path=f"page_{i:04d}.png",
                blocks=[Block("text", h) for h in body_blocks],
                furniture=[Block("furniture", h, label=lab, box=box)
                           for h, lab, box in furniture])


HEAD = ("<p>THE CLASS STRUGGLE IN THE ANCIENT GREEK WORLD AND ITS LATER "
        "HISTORY</p>", "PageHeader", (0.1, 0.02, 0.9, 0.05))


def _book(extra_furniture, page=3, n=8):
    pages = []
    for i in range(n):
        furn = [HEAD] + ([(f"<p>{i + 1}</p>", "PageFooter", (0.45, 0.95, 0.55, 0.97))])
        if i == page:
            furn += extra_furniture
        pages.append(_page(i, [f"<p>Prose of page {i + 1}.</p>"], furn))
    return pages, [p.render() for p in pages]


def test_an_authors_note_at_the_foot_goes_back_to_the_end_of_its_page():
    note = ("<p>This is an updated version of an article printed in Green Left "
            "Weekly in March 2019.</p>", "PageFooter", (0.1, 0.9, 0.9, 0.94))
    pages, bodies = _book([note])
    restored = restore_lone_furniture(pages, bodies)
    assert restored == [(3, "This is an updated version of an article printed "
                            "in Green Left Weekly in March 2019.")]
    assert bodies[3] == "<p>Prose of page 4.</p>" + note[0]
    assert all("Green Left" not in b.html for b in pages[3].furniture)


def test_a_line_at_the_head_goes_back_to_the_start():
    line = ("<p>HEADQUARTERS DEPARTMENT OF THE ARMY Washington, DC, 1 June "
            "2003</p>", "PageHeader", (0.1, 0.05, 0.9, 0.08))
    pages, bodies = _book([line])
    restore_lone_furniture(pages, bodies)
    assert bodies[3].startswith(line[0])


def test_a_running_head_is_not_put_back():
    pages, bodies = _book([])
    assert restore_lone_furniture(pages, bodies) == []
    assert bodies[3] == "<p>Prose of page 4.</p>"


def test_a_head_seen_once_that_carries_the_page_number_stays_out():
    head = ("<p>VII. The Class Struggle on the Ideological Plane (iv) 427</p>",
            "PageHeader", (0.1, 0.02, 0.9, 0.05))
    pages, bodies = _book([head])
    assert restore_lone_furniture(pages, bodies) == []


def test_a_head_seen_once_but_like_another_pages_stays_out():
    a = ("<p>VII. The Class Struggle on the Ideological Plane (ii) and its "
         "sequel</p>", "PageHeader", (0.1, 0.02, 0.9, 0.05))
    b = ("<p>VII. The Class Struggle on the Ideological Plane (iv) and its "
         "sequel</p>", "PageHeader", (0.1, 0.02, 0.9, 0.05))
    pages, bodies = _book([a], page=3)
    pages[5].furniture.append(Block("furniture", b[0], label=b[1], box=b[2]))
    assert restore_lone_furniture(pages, bodies) == []


def test_a_short_line_is_left_to_the_folio_audit():
    pages, bodies = _book([("<p>Chapter Four</p>", "PageFooter", (0.1, 0.9, 0.9, 0.94))])
    assert restore_lone_furniture(pages, bodies) == []


def test_a_book_with_pages_cached_before_furniture_was_kept_is_left_alone():
    note = ("<p>This is an updated version of an article printed in Green Left "
            "Weekly in March 2019.</p>", "PageFooter", (0.1, 0.9, 0.9, 0.94))
    pages, bodies = _book([note])
    pages[6].furniture_known = False
    assert restore_lone_furniture(pages, bodies) == []


def test_an_articles_title_run_once_as_its_second_pages_head_stays_out():
    # Ideas for the Struggle: a two-page article shows its running head once
    head = ("<p>Reasons for Popular Scepticism Towards Politics &amp; "
            "Politicians 23</p>", "PageHeader", (0.1, 0.02, 0.9, 0.05))
    pages, bodies = _book([head], page=3)
    bodies[2] = ("<h1>7. Reasons for Popular Scepticism Towards Politics "
                 "&amp; Politicians</h1>") + bodies[2]
    assert restore_lone_furniture(pages, bodies) == []
    assert any("Scepticism" in b.html for b in pages[3].furniture)


def test_nothing_is_put_back_on_a_page_with_nothing_else_on_it():
    # The History of Lloyd's p433: a citation "read" off a blank leaf
    line = ("<p>Journal of Professional Business Ethics (2018) 13:1-12 | "
            "doi:10.1007/s10547-017-0009-0</p>", "PageHeader",
            (0.1, 0.02, 0.9, 0.05))
    pages, bodies = _book([line], page=3)
    bodies[3] = "\n"
    assert restore_lone_furniture(pages, bodies) == []
    assert bodies[3] == "\n"


def test_a_plates_caption_goes_back_under_its_picture():
    caption = ("<p>VII. JOURNAL OF SIR THOMAS GRESHAM, 1546: OPENING ENTRIES "
               "(see p. 190)</p>", "PageFooter", (0.1, 0.9, 0.9, 0.94))
    pages, bodies = _book([caption], page=3)
    bodies[3] = '<figure><img src="picture:abc" alt=""/></figure>'
    assert len(restore_lone_furniture(pages, bodies)) == 1
    assert bodies[3].endswith(caption[0])
