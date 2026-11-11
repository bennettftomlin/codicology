"""An address the recogniser opened and never closed.

It writes a printed web address as a link. Left without its closing
quotation mark, the address runs on to the next quotation mark on the page:
the text between is inside an attribute, where no reader sees it, and the
words of the next address become attributes named "04" and "2016", which
make the page a file a strict reading system refuses. Five on one shelf of
152 books, one of them a whole entry of a bibliography.
"""
import xml.etree.ElementTree as ET

from codicology import pipeline as vtb


def _mend(body):
    bodies = [body]
    return vtb.mend_swallowed_links(bodies), bodies[0]


def _well_formed(body):
    ET.fromstring(f"<body>{body}</body>".encode("utf-8"))
    return True


# as the parser left them, in the caches of four books
TWO_CITATIONS = (
    '<p>18 Scott Jaschik, “Nonscholarly Litmus Tests,” <i>Inside Higher Ed</i>, March 1, 2016, '
    '<a 04="" 12="" 2016="" href="https://www.insidehighered.com/news/2016/03/01/scholars-india; '
    'Elizabeth Redden, “The Religious War against American Scholars of India,” '
    '&lt;i&gt;Inside Higher Ed&lt;/i&gt;, April 12, 2016, &lt;a href=" https:="" news="" '
    'scholars-who-study-hinduism"="" www.insidehighered.com="">'
    'https://www.insidehighered.com/news/2016/04/12/scholars-who-study-hinduism</a>.</p>')

A_WHOLE_ENTRY = (
    '<p id="bib-0021-19">Green, Joshua. 2017. “Does Stephen Miller Speak for Trump?” '
    '<i>Bloomberg Businessweek</i>. February 28. <a <="" href="https://www.bloomberg.com/news/'
    'vice-versa.Shantz, Jeff. 2016. Crisis States: Governance, Resistance, and Precarious '
    'Capitalism. Earth: punctum books." p=""/></p>')

INTO_THE_NEXT_BLOCK = (
    '<p>See <a 144="" 181="" 905="" 923"="" href="https://ballotpedia.org/California_Proposi-'
    '&lt;/p&gt;&lt;/div&gt;&lt;div data-bbox="><p>114</p></a></p>')


def test_what_the_address_swallowed_is_text_again():
    n, body = _mend(TWO_CITATIONS)
    assert n == 1 and _well_formed(body)
    assert ("March 1, 2016, https://www.insidehighered.com/news/2016/03/01/scholars-india; "
            "Elizabeth Redden, “The Religious War against American Scholars of India,” "
            "<i>Inside Higher Ed</i>, April 12, 2016, "
            "https://www.insidehighered.com/news/2016/04/12/scholars-who-study-hinduism.</p>") in body
    assert "<a" not in body and "</a>" not in body


def test_an_entry_of_a_bibliography_is_given_back():
    n, body = _mend(A_WHOLE_ENTRY)
    assert n == 1 and _well_formed(body)
    assert "Shantz, Jeff. 2016. Crisis States: Governance, Resistance, and Precarious Capitalism." in \
        "".join(ET.fromstring(f"<body>{body}</body>").itertext())
    assert 'id="bib-0021-19"' in body and "<i>Bloomberg Businessweek</i>" in body


def test_the_opening_of_the_tag_it_ran_into_is_not_text():
    n, body = _mend(INTO_THE_NEXT_BLOCK)
    assert n == 1 and _well_formed(body)
    assert "See https://ballotpedia.org/California_Proposi-" in body
    assert "data-bbox" not in body and "div" not in body


def test_a_link_that_is_only_a_link_is_left_alone():
    for body in (
            '<p>See <a href="https://example.org/a/b?c=d&amp;e=f">example.org</a>.</p>',
            # an address the printer broke over a line, read with the space
            '<p><a href="http://www.kollektivhus.nu/pdf/Swedish Cohousing14.pdf">kollektivhus</a></p>',
            '<p><sup><a epub:type="noteref" id="ref-g5-6" href="page_0186.xhtml#note-g5-6">6</a></sup></p>',
            '<p><a id="x"></a>No address at all.</p>'):
        assert _mend(body) == (0, body)


def test_an_address_with_spaces_in_it_is_still_an_address():
    """Printers break long addresses over lines, and the recogniser reads
    the break as a space. That is not a sign of anything swallowed, whatever
    the link's own words say."""
    for body in (
            '<p>See: <a href="http://www.ons.gov.uk/people/personal and household finances/income and '
            'wealth/yearending2018">www.ons.gov.uk/people/personal and household finances/income and '
            'wealth/yearending2018</a></p>',
            '<p><a href="http://example.com/search?q=big red dog house">site</a></p>',
            '<p><a href="http://a.com/very long address with spaces more">other text</a></p>'):
        assert _mend(body) == (0, body)


def test_what_the_link_held_is_not_a_paragraph_inside_a_paragraph():
    n, body = _mend(INTO_THE_NEXT_BLOCK)
    assert n == 1 and "<p>114" not in body and body.count("<p>") == body.count("</p>") == 1
    assert "California_Proposi- 114" in " ".join(body.split())


def test_an_address_the_links_own_words_repeat_lost_nothing():
    # the recogniser wrote the next entry into the address and into the link:
    # it is printed once, and is not a link to anywhere
    n, body = _mend('<p>119–139. <a href="https://doi.org/10.1111/1467.Jackson, W., \'Researching the '
                    'Policed\', &lt;i&gt;Policing and Society&lt;/i&gt; 30/2 (2020).">https://doi.org/'
                    '10.1111/1467.Jackson, W., ‘Researching the Policed’, <i>Policing and Society</i> '
                    '30/2 (2020).</a></p>')
    assert n == 1 and body == ('<p>119–139. https://doi.org/10.1111/1467.Jackson, W., ‘Researching the '
                               'Policed’, <i>Policing and Society</i> 30/2 (2020).</p>')


def test_only_the_broken_link_is_taken_off():
    good = '<a href="https://example.org">example.org</a>'
    n, body = _mend(f"<p>{good} and then</p>" + A_WHOLE_ENTRY + f"<p>{good}</p>")
    assert n == 1 and body.count(good) == 2 and _well_formed(body)
