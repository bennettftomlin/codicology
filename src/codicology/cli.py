"""codicology — one command, five verbs.

    codicology convert …    build the PDF/EPUB (the pipeline; see convert -h)
    codicology verify EPUB PDF
                            the after-every-build check: is anything missing?
    codicology compare EPUB PDF
                            word-by-word disagreement with the source's layer
    codicology doctor       can this environment run a build? --full reads
                            one synthetic page through the real backend

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

    p_doctor = sub.add_parser(
        "doctor",
        help="report whether this environment can run a build: binaries, "
             "packages, tesseract's languages. --full also reads one "
             "synthetic page, which is the only check that catches a surya "
             "that imports cleanly but cannot serve inference")
    p_doctor.add_argument("--full", action="store_true",
                          help="also OCR a synthetic test page through the "
                               "real backend (the first run may download "
                               "model weights)")
    p_doctor.add_argument("--ocr", default="surya",
                          help="backend for the --full smoke test "
                               "(default: surya)")
    p_doctor.add_argument("--lang", default="en",
                          help="comma-separated language codes for the "
                               "smoke test (default: en)")
    p_doctor.add_argument("--json", action="store_true", dest="as_json",
                          help="one JSON object on stdout, for a program "
                               "driving this pipeline")

    p_adj = sub.add_parser(
        "adjudicate",
        help="where the readers disagree: re-read the book with the "
             "witnesses and emit the dispute record — every word the "
             "engines could not agree on and which rung of the ladder "
             "settled it. Never changes the book")
    p_adj.add_argument("epub", help="the EPUB this pipeline built")
    p_adj.add_argument("pdf", help="the source it was built from")
    p_adj.add_argument("--report", metavar="JSON",
                       help="also write the full record as JSON")
    p_adj.add_argument("--limit", type=int, default=None,
                       help="examine only the first N pages")
    p_adj.add_argument("--calibrate", action="store_true",
                       help="score the witness engines AND the ladder's own "
                            "verdicts against this born-digital book's text "
                            "instead of emitting a dispute record")

    p_rev = sub.add_parser(
        "review",
        help="render a dispute record into a self-contained review sheet: "
             "the ink itself beside every disputed word, ranked by how "
             "likely the shipped reading is wrong, with a field for the "
             "human's own reading — which outranks every rung. Exports "
             "decisions for apply or the next rebuild")
    p_rev.add_argument("report", help="dispute record JSON from adjudicate")
    p_rev.add_argument("--out", metavar="HTML",
                       help="where to write the sheet (default: beside the "
                            "report)")
    p_rev.add_argument("--dpi", type=int, default=200,
                       help="render resolution for ink crops (default: 200)")
    p_rev.add_argument("--no-crops", action="store_true",
                       help="text-only sheet, no page rendering")

    args = parser.parse_args(argv)
    if args.command == "verify":
        from . import verify
        return verify.main(args.epub, args.pdf)
    if args.command == "compare":
        from . import compare
        return compare.main(args.pdf, args.epub)
    if args.command == "doctor":
        from . import doctor
        return doctor.main(full=args.full, ocr=args.ocr, lang=args.lang,
                           as_json=args.as_json)
    if args.command == "adjudicate":
        from . import adjudicate
        if args.calibrate:
            adjudicate.calibrate(args.pdf, epub=args.epub, limit=args.limit)
            return 0
        return adjudicate.main(args.epub, args.pdf, report=args.report,
                               limit=args.limit)
    if args.command == "review":
        from . import review
        return review.main(args.report, out=args.out, dpi=args.dpi,
                           crops=not args.no_crops)
    return None


if __name__ == "__main__":
    sys.exit(main())
