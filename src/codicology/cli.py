"""codicology — one command, and its verbs.

    codicology convert …    build the PDF/EPUB (the pipeline; see convert -h)
    codicology verify EPUB PDF
                            the after-every-build check: is anything missing?
    codicology compare EPUB PDF
                            word-by-word disagreement with the source's layer
    codicology doctor       can this environment run a build? --full reads
                            one synthetic page through the real backend
    codicology adjudicate EPUB PDF
                            re-read the book with the witness engines and
                            emit the dispute record — every word the readers
                            disagreed on and which rung of the ladder settled
                            it. Never changes the book
    codicology review REPORT
                            render the dispute record as a self-contained
                            sheet: the ink beside every disputed word, ranked
                            by how likely the shipped reading is wrong, with
                            a field for the human's own reading — which
                            outranks every rung
    codicology apply EPUB DECISIONS
                            feed the sheet's exported decisions back into the
                            book. The same file, left beside the OCR cache,
                            is picked up by the next rebuild via
                            convert --apply-decisions
    codicology decisions list|retire|file …
                            look after decisions files: list one, take a
                            row out of use, or file a newly exported one as
                            the next round without losing the rounds before
    codicology relabel CACHE PDF
                            backfill layout labels into an OCR cache written
                            before they were kept, without re-reading a word
    codicology rekey CACHE PDF
                            refile an OCR cache's readings under the names
                            their pages have now. A build does it for itself
                            on first meeting an older cache; this does a
                            whole cache ahead of time
    codicology shelf build SHELF OUT | compare BASE NEW | installed SHELF OUT
                            build every book a shelf file names from its
                            cache, reading nothing, several at once; and
                            say what differs between two such runs, book
                            by book. How a change to the pipeline is judged

The check subcommands take the book first and the source second, in every
case — the old standalone scripts disagreed with each other about the
order, and one of the two had to move.
"""
import argparse
import signal
import sys

from . import __version__


def _exit_cleanly_on_signal() -> None:
    """Turn a termination signal into an ordinary exit.

    The inference server is spawned into its own session — deliberately, so
    that it outlives one page's work and can be attached to again — which
    puts it beyond the reach of any process-group kill aimed at us. Nothing
    stops it except the `atexit` handler that the spawner registers, and
    Python does not run `atexit` for a signal it never handled: the default
    disposition terminates immediately.

    So a stopped job left the server running. Measured on this machine: a
    2 GB llama-server holding its port with its parent gone, and beside it
    a detection server orphaned the same way eight days earlier. Converting
    the signal to SystemExit runs the handlers, and the server goes with us.

    Interactive Ctrl-C already unwinds through KeyboardInterrupt; this is
    for SIGTERM, and for SIGHUP when the terminal goes away.
    """
    def bow_out(signum, _frame):
        # 128+n is the shell's own convention for "died of signal n", so a
        # caller reading the status still learns which one it was.
        sys.exit(128 + signum)

    for name in ("SIGTERM", "SIGHUP"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue                     # SIGHUP does not exist on Windows
        try:
            signal.signal(sig, bow_out)
        except (ValueError, OSError):
            # not the main thread, or the platform refuses: the build still
            # runs, it just cleans up less well on an abort
            pass


def main(argv: "list[str] | None" = None) -> "int | None":
    _exit_cleanly_on_signal()
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
    p_adj.add_argument("--second-reader", action="store_true",
                       help="also put the disputes the ladder leaves open to "
                            "the second reader (second.py), one reading of "
                            "each page that has one. It and Apple Vision "
                            "then settle a dispute as a count of readers: "
                            "one with a view, or both agreeing, and never "
                            "the two of them against each other. Its "
                            "readings are kept beside --report")
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

    p_rel = sub.add_parser(
        "relabel",
        help="backfill layout labels into an existing OCR cache without "
             "re-reading a word: pages regenerate through the same "
             "extraction path, a small local layout model names the "
             "blocks, and labels attach by geometry. Text stays "
             "byte-identical, so dispute records and decisions survive")
    p_rel.add_argument("cache", help="the .ocr.gz cache to upgrade")
    p_rel.add_argument("pdf", help="the source PDF its pages came from")
    p_rel.add_argument("--ocr", default="surya",
                       help="backend tag the cache was written under "
                            "(default: surya)")
    p_rel.add_argument("--lang", default="en",
                       help="language tag the cache was written under "
                            "(default: en)")

    p_rekey = sub.add_parser(
        "rekey",
        help="refile an OCR cache's readings under the names their pages "
             "have now, without building the book or reading a word. A "
             "build does this for itself the first time it meets an older "
             "cache; this brings one over ahead of time, while the encoder "
             "its old names depend on is still installed")
    p_rekey.add_argument("cache", help="the .ocr.gz cache to bring over")
    p_rekey.add_argument("pdf", help="the source PDF its pages came from")
    p_rekey.add_argument("--ocr", default="surya",
                         help="backend the cache was written by "
                              "(default: surya)")

    p_apply = sub.add_parser(
        "apply",
        help="feed a review sheet's exported decisions back into the EPUB: "
             "each correction lands at its recorded occurrence in text "
             "nodes only, the original is kept beside the book as "
             ".preapply, and decisions whose site no longer exists are "
             "reported stale, never guessed at")
    p_apply.add_argument("epub", help="the EPUB this pipeline built")
    p_apply.add_argument("decisions",
                         help="decisions JSON exported from the review sheet")
    p_apply.add_argument("--out", metavar="EPUB",
                         help="write here instead of correcting in place")

    p_bind = sub.add_parser(
        "bind",
        help="bind the pages of a built EPUB into chapters, as convert "
             "now does by default: a paragraph that runs over the page "
             "becomes one paragraph and a reader's screen turns where the "
             "text does. Nothing is read again and no word changes; every "
             "page's start and printed number stay marked, and verify, "
             "adjudicate and apply go on working by page")
    p_bind.add_argument("epub", help="an EPUB this pipeline built, one "
                                     "file to a page")
    p_bind.add_argument("--out", metavar="EPUB",
                        help="write here instead of binding in place")

    p_rebind = sub.add_parser(
        "rebind",
        help="bind a bound EPUB again by the rules as they now are (where "
             "a page's mark stands when the page turned inside a word, "
             "say). Its files, links and pages stay what they were")
    p_rebind.add_argument("epub", help="an EPUB bound into chapters")
    p_rebind.add_argument("--out", metavar="EPUB",
                          help="write here instead of in place")

    p_unbind = sub.add_parser(
        "unbind",
        help="take a bound EPUB apart into one file to a printed page "
             "again")
    p_unbind.add_argument("epub", help="an EPUB bound into chapters")
    p_unbind.add_argument("--out", metavar="EPUB",
                          help="write here instead of unbinding in place")

    p_dec = sub.add_parser(
        "decisions",
        help="look after a book's decisions files: list one, take a row "
             "out of use, or file a newly exported one as the next round "
             "of review without losing the rounds before it")
    dsub = p_dec.add_subparsers(dest="what", required=True)
    d_list = dsub.add_parser(
        "list", help="print a decisions file a row to a line, numbered")
    d_list.add_argument("file", help="a decisions JSON")
    d_retire = dsub.add_parser(
        "retire",
        help="take rows out of use by their number in `decisions list`. "
             "They stay in the file as retired, so a later round cannot "
             "bring them back unnoticed")
    d_retire.add_argument("file", help="a decisions JSON")
    d_retire.add_argument("rows", nargs="+", type=int,
                          help="row numbers, as `decisions list` prints them")
    d_file = dsub.add_parser(
        "file",
        help="put an exported decisions file where a book's rebuilds find "
             "it. The same sheet exported again replaces the round in "
             "hand; a file from another sheet becomes a new round, and "
             "the one in hand is kept beside it as an earlier one")
    d_file.add_argument("exported", help="the file the review sheet saved")
    d_file.add_argument("target",
                        help="where the book's decisions live, for example "
                             "book-disputes.decisions.json beside its cache")
    d_file.add_argument("--json", action="store_true", dest="as_json",
                        help="print what was done as one JSON object")

    p_shelf = sub.add_parser(
        "shelf",
        help="build every book a shelf file names from its OCR cache, "
             "reading nothing, and say what differs between two such "
             "runs: how a change to the pipeline is judged on the books "
             "it has already built")
    ssub = p_shelf.add_subparsers(dest="what", required=True)
    s_build = ssub.add_parser(
        "build",
        help="build the shelf's books from their caches with "
             "--cache-only, several at once, and keep a print of each")
    s_build.add_argument("shelf", help="the shelf file: a JSON list of "
                                       "books, each with id, pdf, cache, "
                                       "and the options it is built with")
    s_build.add_argument("out", help="the folder this run is kept in")
    s_build.add_argument("--jobs", type=int, default=None,
                         help="books built at once (default: 4)")
    s_build.add_argument("--only", nargs="+", metavar="ID",
                         help="build only these books")
    s_build.add_argument("--code", metavar="DIR",
                         help="build with the codicology package under "
                              "this folder (a src folder of another "
                              "checkout, say), not the one this command "
                              "runs from: how a base run is made of a "
                              "commit before a change")
    s_build.add_argument("--verify", action="store_true",
                         help="also run verify on each book against its "
                              "PDF, and keep what it said")
    s_build.add_argument("--resume", action="store_true",
                         help="leave the books this folder already holds "
                              "a good print of, and build the rest")
    s_build.add_argument("--read-missing", action="store_true",
                         help="when the rest are done, build the books "
                              "--cache-only refused for a page with no "
                              "reading the ordinary way, one at a time: "
                              "the cached pages from the cache, the "
                              "missing ones read. This starts the "
                              "recogniser")
    s_build.add_argument("--prints-only", action="store_true",
                         help="keep each book's print, log and build "
                              "record and not the book itself: enough to "
                              "compare against, in a hundredth of the room")
    s_build.add_argument("--against", metavar="BASE",
                         help="when the run is done, compare it with this "
                              "earlier run")
    s_build.add_argument("--json", metavar="FILE", dest="report",
                         help="with --against: write the whole comparison "
                              "here")
    s_build.add_argument("--show", type=int, default=None,
                         help="changes shown for each book (default: 3)")
    s_cmp = ssub.add_parser(
        "compare",
        help="what differs between two runs, book by book: words, markup "
             "alone, the binding, pictures, contents. Exit status 1 if "
             "anything does")
    s_cmp.add_argument("base", help="the earlier run's folder")
    s_cmp.add_argument("new", help="the later run's folder")
    s_cmp.add_argument("--only", nargs="+", metavar="ID",
                       help="compare only these books")
    s_cmp.add_argument("--json", metavar="FILE", dest="report",
                       help="write the whole comparison here")
    s_cmp.add_argument("--show", type=int, default=None,
                       help="changes shown for each book (default: 3)")
    s_list = ssub.add_parser("list", help="print the books a shelf names")
    s_list.add_argument("shelf", help="the shelf file")
    s_inst = ssub.add_parser(
        "installed",
        help="take prints of the copies the shelf file says are installed, "
             "building nothing, so a run can be compared with the shelf "
             "as it stands: which books a rebuild would change")
    s_inst.add_argument("shelf", help="the shelf file (each book's "
                                      "\"installed\" names its copy)")
    s_inst.add_argument("out", help="the folder the prints are kept in")
    s_inst.add_argument("--only", nargs="+", metavar="ID",
                        help="only these books")

    args = parser.parse_args(argv)
    if args.command == "shelf":
        import json
        from . import shelf
        try:
            if args.what == "list":
                shelf.listing(args.shelf)
                return 0
            if args.what == "installed":
                record = shelf.installed(args.shelf, args.out,
                                         only=args.only)
                return 1 if record["failed"] else 0
            status = 0
            if args.what == "build":
                record = shelf.build(
                    args.shelf, args.out, jobs=args.jobs or shelf.JOBS,
                    only=args.only, code=args.code, verify=args.verify,
                    resume=args.resume, prints_only=args.prints_only,
                    read_missing=args.read_missing)
                status = 1 if record["failed"] else 0
                if not args.against:
                    return status
                base, new = args.against, args.out
                print()
            else:
                base, new = args.base, args.new
            result = shelf.compare(base, new, only=args.only)
            if args.report:
                with open(args.report, "w", encoding="utf-8") as fh:
                    json.dump(result, fh, ensure_ascii=False, indent=1)
            differ = shelf.tell(result, show=shelf.SHOW if args.show is None
                                else args.show)
            return 1 if differ or status else 0
        except shelf.ShelfError as exc:
            print(f"[!] {exc}")
            return 2
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
        second = None
        if args.second_reader:
            second = adjudicate.second_words_for(args.report)
            if second is None:
                from . import second as _second
                print("[!] --second-reader was asked for and there is none "
                      f"to ask: put the model's two .gguf files in "
                      f"{_second.DEFAULT_DIR} and have llama-server on the "
                      "PATH")
                return 2
        if args.calibrate:
            adjudicate.calibrate(args.pdf, epub=args.epub, limit=args.limit,
                                 second=second)
            return 0
        return adjudicate.main(args.epub, args.pdf, report=args.report,
                               limit=args.limit, second=second)
    if args.command == "review":
        from . import review
        return review.main(args.report, out=args.out, dpi=args.dpi,
                           crops=not args.no_crops)
    if args.command == "apply":
        from . import review
        review.apply_decisions(args.epub, args.decisions, out=args.out)
        return 0
    if args.command == "rebind":
        from . import binding
        try:
            st = binding.rebind_epub(args.epub, out=args.out)
        except binding.CannotBind as exc:
            print(f"not bound again: {exc}")
            return 1
        print(f"{st['changed']} of {st['files']} file(s) bound again "
              f"differently ({st['pages']} pages, all as they were): "
              f"{args.out or args.epub}")
        return 0
    if args.command in ("bind", "unbind"):
        from . import binding
        try:
            st = (binding.bind_epub if args.command == "bind"
                  else binding.unbind_epub)(args.epub, out=args.out)
        except binding.CannotBind as exc:
            print(f"not {'bound' if args.command == 'bind' else 'unbound'}: "
                  f"{exc}")
            return 1
        if args.command == "bind":
            print(f"bound {st['pages']} page(s) into {st['files']} file(s): "
                  f"{st['joined']} paragraph(s) that ran over a page are "
                  f"one paragraph"
                  + (f", {st['words']} of them broken inside a word"
                     if st.get("words") else "")
                  + (f"; what stood at the foot of {st['held']} page(s) "
                     f"waits for the paragraph to end" if st["held"] else "")
                  + f": {args.out or args.epub}")
        else:
            print(f"{st['files']} file(s) taken apart into {st['pages']} "
                  f"page(s): {args.out or args.epub}")
        return 0
    if args.command == "relabel":
        from . import pipeline
        pipeline.relabel_cache(args.cache, args.pdf, backend_name=args.ocr,
                               langs=args.lang.split(","))
        return 0
    if args.command == "decisions":
        import json
        from . import review
        try:
            if args.what == "list":
                review.list_decisions(args.file)
            elif args.what == "retire":
                for d in review.retire_decisions(args.file, args.rows):
                    print(f"retired: p{d.get('page')} {d.get('old')!r} -> "
                          f"{d.get('new')!r}")
            else:
                done = review.file_decisions(args.exported, args.target)
                if args.as_json:
                    print(json.dumps(done))
                else:
                    print(f"{done['rows']} decision(s) filed as round "
                          f"{done['round']}"
                          + (", in place of the same sheet's earlier export"
                             if done["replaced"] else "")
                          + (f"; {done['dropped_retired']} left out because "
                             f"an earlier round retired them"
                             if done["dropped_retired"] else "")
                          + f": {args.target}")
        except (OSError, ValueError) as exc:
            print(f"[!] {exc}")
            return 2
        return 0
    if args.command == "rekey":
        from . import pipeline
        pipeline.rekey_cache(args.cache, args.pdf, backend_name=args.ocr)
        return 0
    return None


if __name__ == "__main__":
    sys.exit(main())
