"""The words of a fragment with its markup removed: only what is shaped like
a tag goes."""
from codicology.markup import _strip_tags


def test_a_bare_less_than_keeps_the_words_after_it():
    """A reading that left "<" unescaped lost everything up to the next ">"."""
    assert _strip_tags("<p>if a < b and c > d</p>") == "if a < b and c > d"
    assert _strip_tags("<p>a<b and c>d</p>") == "a<b and c>d"


def test_tags_with_attributes_still_go():
    assert _strip_tags('<p class="x">Chapter 5</p><h2>Elasticity</h2>') \
        == "Chapter 5 Elasticity"
    assert _strip_tags('<img src="a.png" alt="Figure"/>tail') == "tail"
    assert _strip_tags('<a href="http://a.b/c?d=1&amp;e=2">link</a>') == "link"
    assert _strip_tags('<td colspan=2>cell</td>') == "cell"


def test_comments_go_and_escaped_brackets_come_back():
    assert _strip_tags("x <!-- a > b --> y") == "x  y"
    assert _strip_tags("<p>&lt;&lt;And so&gt;&gt;</p>") == "<<And so>>"
