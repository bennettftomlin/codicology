"""codicology — one command, three verbs.

    codicology convert …    build the PDF/EPUB (the pipeline; see convert -h)
    codicology verify EPUB PDF
                            the after-every-build check: is anything missing?
    codicology compare EPUB PDF
                            word-by-word disagreement with the source's layer

The check subcommands take the book first and the source second, in both
cases — the old standalone scripts disagreed with each other about the
order, and one of the two had to move.
"""
import argparse
import sys

from . import __version__


def main(argv: "list[str] | None" = None) -> "int | None":
    argv = list(sys.argv[1:] if argv is None else argv)

    # `convert` owns a large parser of its own, built where the pipeline
    # lives; hand the rest of the line over rather than mirroring it here.
    if argv and argv[0] == "convert":
        from . import pipeline
        return pipeline.main(argv[1:])

    parser = argparse.ArgumentParser(
        prog="codicology",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version",
                        version=f"codicology {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser(
        "convert", add_help=False,
        help="build a clean PDF and an OCR'd EPUB from a video, photographs, "
             "or a PDF (codicology convert -h for the pipeline's own help)")

    p_verify = sub.add_parser(
        "verify",
        help="check a built EPUB against the source's own text: is anything "
             "missing?")
    p_verify.add_argument("epub", help="the EPUB this pipeline built")
    p_verify.add_argument("pdf", help="the source PDF it was built from")

    p_compare = sub.add_parser(
        "compare",
        help="word-by-word disagreement between the EPUB and the source's "
             "text layer")
    p_compare.add_argument("epub", help="the EPUB this pipeline built")
    p_compare.add_argument("pdf", help="the source PDF it was built from")

    args = parser.parse_args(argv)
    if args.command == "verify":
        from . import verify
        return verify.main(args.epub, args.pdf)
    if args.command == "compare":
        from . import compare
        return compare.main(args.pdf, args.epub)
    return None


if __name__ == "__main__":
    sys.exit(main())
