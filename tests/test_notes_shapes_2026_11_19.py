"""Three shapes of note the linkers read past (shelf, 19 November).

Behind the Cold War on China heads each article's notes with its title, and
every note is a bare web address, "1 https://…": the title head was known
by the "1" under it only when a word followed. Disturbing Times and
Anti-Racism Inc. run a note over the page and finish it under the next
page's rule, where the page before shows no rule to say a note was left
unfinished; and Disturbing Times prints feet with plain numbers under no
rule at all.
"""
from codicology.notes import (_bare_feet_without_a_rule, _HEAD_THEN_FIRST,
                              link_footnotes)

PROSE = ("<p>" + "The art of Makuria is challenging to define, and its links "
         "with Byzantium run through all of it. " * 3)


def test_a_title_head_over_a_linked_first_note_opens_a_group():
    page = ('<h2>Putting Australia\'s Anti-China Campaign in Perspective</h2>'
            '<ol style="list-style-type: none;"><li>1 <a href="https://x.org/a">'
            'https://x.org/a</a></li>')
    assert _HEAD_THEN_FIRST.search(page)


def test_a_note_finished_under_the_rule_links_on_the_layouts_word():
    """Disturbing Times p30: under the rule, the end of note 2 and then
    notes 3 and 4; the page before set its feet as a list, with no rule."""
    before = "<p>Prose.</p><ol><li>1 My thanks go out to</li><li>2 See “In memoriam</li></ol>"
    body = (PROSE + "polymath.<sup>3</sup> Latin.<sup>4</sup></p><hr/>"
            "<p>(1943–2004),” <i>BASP</i> 41 (2004): 13–37.</p>"
            "<p>3 Ca. 700–1450 CE. See Obłuski.</p><p>4 Stephen M. Bay.</p>")
    bodies = [before, body]
    assert link_footnotes(list(bodies), allow_numbered=True)["linked"] == 0
    stats = link_footnotes(bodies, allow_numbered=True, foot_labelled=[0.0, 1.0])
    assert stats["linked"] == 2 and 'id="fn-p1-n3"' in bodies[1]
    assert "(1943–2004),” <i>BASP</i>" in bodies[1]


def test_plainly_numbered_feet_under_no_rule_link():
    """Disturbing Times p83: notes 20 to 23, "20 Kurt W. Weitzmann…"."""
    body = (PROSE + "define.<sup>20</sup> Arabic.<sup>21</sup> studies.<sup>22</sup></p>"
            "<p>20 Kurt W. Weitzmann was one of the earliest.</p>"
            "<p>21 Grzegorz Ochała, “Multilingualism.”</p>"
            "<p>22 Ibid., 3.</p>")
    assert _bare_feet_without_a_rule(body) == body.index("<p>20")
    bodies = [body]
    assert link_footnotes(bodies, allow_numbered=True)["linked"] == 3


def test_a_numbered_list_closing_a_page_is_not_its_feet():
    body = (PROSE + "The steps are these:</p><p>1 Mix the clay.</p><p>2 Shape it.</p>")
    assert _bare_feet_without_a_rule(body) == 0
    raised_twice = (PROSE + "<sup>2</sup> and <sup>2</sup>.</p><p>2 Ibid.</p><p>3 Ibid.</p>")
    assert _bare_feet_without_a_rule(raised_twice) == 0


def test_numbers_that_do_not_climb_are_not_a_foot_run():
    body = (PROSE + "x.<sup>4</sup> y.<sup>9</sup></p><p>4 One.</p><p>9 Two.</p>")
    assert _bare_feet_without_a_rule(body) == 0
