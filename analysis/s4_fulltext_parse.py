"""
Step 4. Measure every retained full-text PDF from scratch.

Inputs: analysis/data/fulltext/v2_target/pdf (automated downloads, mapped to DOIs by
the fetch_status*.csv files) and manual_pdf (the author's browser downloads, mapped
by the DOI printed in the file or encoded in its name). Files listed in
nonarticle_files.csv are skipped, as are files modified in the last 15 minutes
(another process may still be writing them).

Every file passes four checks before it is measured; a file that fails one is kept
in the output with its status and is not analysed:

  identity       the corpus title of the DOI must appear in the first two pages:
                 at least 30 percent of the title's words of four or more letters
                 (status "title_mismatch"; titles with fewer than three such words
                 are not checked)
  supplementary  supporting-information files are rejected: the document opens
                 with "Supporting/Supplementary Information/Material" (within its
                 first 200 characters; ACS articles print a "Supporting Information"
                 link later on their first page, which does not count) and has at
                 least three "Figure S1"-style labels (status "supplementary")
  readable       at least half the words of three or more letters must be ordinary
                 English words, from a vocabulary built from the corpus abstracts;
                 PDFs whose fonts extract as scrambled characters fail
                 (status "unreadable_text")
  an article     body of at least 500 words; a front-matter heading (editorial
                 board, table of contents, calendar, erratum, ...) rejects a file only
                 when its body is short, because the same words open real articles
                 ("Table of Contents graphic", "Calendar aging of ...")
                 (status "not_article_or_short")

Measures, all deterministic:
  body            text before the reference-list heading (if found after 30% of the text)
  marker/control  listed forms of the marker and control families (lexicon.py)
  hedge/booster   as for abstracts
  figures/tables  distinct caption numbers at line starts ("Figure 3", "Table 2")
  equations       distinct right-margin equation numbers "(12)" in the body
  ai_mention      the body names a generative language model or the technology
  ai_disclosure   the document states that such a tool was used in preparing the
                  manuscript. Three forms count: (1) the standard declaration
                  "During the preparation of this work the author(s) used [tool] in
                  order to [purpose]", when a use verb follows the phrase in the same
                  sentence together with a named tool (a wider list that includes
                  writing assistants such as Grammarly, DeepL and QuillBot, and
                  line-break hyphens such as "Chat- GPT") or a writing purpose; the
                  phrase alone does not count, because it also opens ordinary
                  acknowledgements; (2) a declaration heading ("AI-assisted
                  technologies in the writing process") followed by a use verb and
                  a tool or writing purpose; (3) a use stated as done ("used",
                  "was employed", "acknowledge the use of", "the use of ... is
                  acknowledged"), a listed tool and a writing purpose within one
                  sentence. A writing purpose must name
                  the text ("improve the language", "proofreading", "readability"):
                  broad stems also match research about language models. Statements
                  of non-use ("did not use") and quoted policy ("should only be used") are
                  not counted. Searched outside the
                  reference list

measure_text() is also used by s4b_fetch_bronze.py for articles read in memory.
Rerun with --all to re-measure every file; otherwise files already measured with the
current PARSER_VERSION are skipped.
"""
import os, re, sys, time, glob
import settings as P
from collections import Counter
from multiprocessing import Pool
import pandas as pd
import lexicon as L

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = P.OUT
V2 = P.FULLTEXT
CORPUS = P.CORPUS
OUTFILE = os.path.join(OUT, "fulltext_articles.parquet")
VOCAB = os.path.join(OUT, "english_vocab.txt")
PARSER_VERSION = 4
MIN_BODY = 500
SHORT_BODY = 1500

FIG = re.compile(r"(?im)^\s*(?:fig(?:ure)?\.?)\s*(\d{1,3})\s*(?:[.:)]|$|\s+\S)")
TAB = re.compile(r"(?im)^\s*(?:table|tab\.)\s*(\d{1,3})\s*(?:[.:)]|$|\s+\S)")
EQN = re.compile(r"(?m)\(\s*(\d{1,3}[a-z]?)\s*\)\s*$")
REFS = re.compile(r"(?im)^\s*(?:\d+\.?\s*|[ivxlc]+\.?\s*)?(references?|bibliography|literature\s+cited|works\s+cited|r\s?e\s?f\s?e\s?r\s?e\s?n\s?c\s?e\s?s)\s*:?\s*$")
FRONT = re.compile(r"(?im)^\s*(editorial\s+board|table\s+of\s+contents|author\s+index|subject\s+index|calendar|call\s+for\s+papers|instructions?\s+(for|to)\s+authors|list\s+of\s+reviewers|masthead|front\s+matter|erratum|corrigendum|retraction\s+notice|book\s+review|obituary|in\s+memoriam)\b")
SUPP_HEAD = re.compile(r"(?i)\b(supporting|supplementary|supplemental)\s+(information|material|materials|data|file)\b")
SUPP_LABEL = re.compile(r"(?i)\b(?:fig(?:ure)?\.?|table)\s*S\d{1,2}\b")
TOOL = r"(?:chat\s?-?gpt|gpt-?(?:3(?:\.5)?|4o?|4\.\d|5)|large\s+language\s+models?|llms?|generative\s+(?:ai|artificial\s+intelligence)|openai|google\s+bard|claude|gemini|deepseek|copilot|qwen|ai[-\s](?:assisted|based|powered)\s+(?:writing|language|tool)s?|ai\s+(?:writing\s+)?(?:tool|assistant)s?)"
MENTION = re.compile(r"(?i)\b(?:chat\s?-?gpt|gpt-?(?:3(?:\.5)?|4o?)|large\s+language\s+models?|llms?|generative\s+(?:ai|artificial\s+intelligence)|openai)\b")
USE = r"(?:used|use|using|utili[sz]ed|employed|assisted|with\s+the\s+(?:help|assistance|aid)\s+of|aided)"
PURPOSE = r"(?:improv|enhanc|polish|edit|proofread|refin|languag|readab|grammar|spelling|writing|wording|translat|paraphras|rephras|clarity)"
# The standard declaration (Elsevier wording, used by other publishers too). Inside it a
# wider list of tools applies, including writing assistants and line-break hyphens.
DECL = r"during\s+the\s+(?:preparation|writing)\s+of\s+this\s+(?:work|manuscript|paper|article|study|review)"
TOOL_DECL = (rf"(?:{TOOL}|chat\s?-?\s?g[pt]{{2}}|gram-?\s?marly|deep\s?l(?:\s+write)?|quill\s?-?\s?bo\s?t|"
             r"mistral(?:\s?ai)?|scispace|writefull|paperpal|trinka|perplexity|"
             r"ai[-\s]+assisted\s+technolog(?:y|ies)|artificial\s+intelligence)")
USE_DECL = r"(?:used|utili[sz]ed|employed|leveraged|relied\s+on|made\s+use\s+of)"
NEGATION = re.compile(r"(?i)\b(?:not|never|no)\s+(?:\w+\s+){0,2}(?:use|used|using|utili[sz]ed|employed|generative|ai|artificial)\b")
# Policy text quoted in an article ("these technologies should only be used to ...").
PRESCRIPTIVE = re.compile(r"(?i)\b(?:should|must|shall)\b|\bwhere\s+authors\s+use\b")
# A writing purpose aimed at the text itself. Broad stems such as "language" or "improve"
# alone also match research about language models ("natural language", "improve the
# model"), so the purpose must name the text, its language or its readability.
OBJ = r"(?:language|text|manuscript|paper|article|writing|english|grammar|readability|clarity|wording|style|phrasing|translation|spelling)"
WPURP = (r"(?:proofread\w*|grammar\w*|grammatical\w*|readability|spelling|wording|"
         rf"(?:language|text|manuscript|writing|copy)[-\s]+(?:edit\w*|polish\w*|refine\w*|check\w*)|"
         rf"(?:improv|enhanc|refin|polish|edit|check|correct|revis)\w*\s+(?:and\s+\w+\s+)?(?:the\s+|its\s+|our\s+|their\s+)?(?:\w+\s+){{0,2}}{OBJ}\w*)")
# Use stated as done: past forms and acknowledgements. Present-tense "use" and "using"
# also describe what tools could do, in articles about them.
USE_PAST = (r"(?:used|utili[sz]ed|employed|aided|assisted|with\s+the\s+(?:help|assistance|aid)\s+of|"
            r"(?:acknowledge\w*|declare\w*|disclose\w*)\s+(?:the\s+use\s+of|using))")
# The declaration headings some publishers print above the statement.
HEAD = (r"(?:(?:ai|artificial\s+intelligence)[-\s]+assisted\s+(?:technolog(?:y|ies)|tools?)\s+(?:in\s+the\s+writing\s+process|usage)"
        r"|declaration\s+of\s+generative\s+ai\s+in\s+scientific\s+writing)")
DISCLOSE = [re.compile(rf"(?is)\b{DECL}[^.]{{0,250}}?\b{USE_DECL}\b[^.]{{0,250}}?(?:\b{TOOL_DECL}\b|\b{WPURP})"),
            re.compile(rf"(?is)\b{HEAD}.{{0,150}}?\b(?:{USE_DECL}|with\s+the\s+(?:help|assistance|aid)\s+of|assistance\s+of)\b[^.]{{0,200}}?(?:\b{TOOL_DECL}\b|\b{WPURP})"),
            re.compile(rf"(?is)\b{USE_PAST}\b[^.;]{{0,200}}\b{TOOL}\b[^.;]{{0,200}}\b{WPURP}"),
            re.compile(rf"(?is)\b{TOOL}\b[^.;]{{0,120}}\b(?:was|were|has\s+been|have\s+been)\s+{USE}\b[^.;]{{0,200}}\b{WPURP}"),
            re.compile(rf"(?is)\bthe\s+use\s+of\s+{TOOL}\b[^.;]{{0,200}}\b{WPURP}[^.;]{{0,120}}\b(?:is|was|are|were)\s+(?:hereby\s+)?(?:acknowledged|declared|disclosed)")]
DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s\"'<>)\]}]+", re.I)


def discloses(text):
    """True if some sentence states that an AI tool was used in writing (not its non-use)."""
    for p in DISCLOSE:
        for m in p.finditer(text):
            lead = text[max(0, m.start() - 40):m.start()].rsplit(".", 1)[-1]   # same sentence only
            if not (NEGATION.search(lead + m.group(0)) or PRESCRIPTIVE.search(lead + m.group(0))):
                return True
    return False
TITLE_WORD = re.compile(r"[a-z]{4,}")
WORD3 = re.compile(r"[a-z]{3,}")

KNOWN, TITLES, ENGLISH = set(), {}, set()


def init(known, titles, english):
    global KNOWN, TITLES, ENGLISH
    KNOWN, TITLES, ENGLISH = known, titles, english


def read(path):
    import pymupdf
    try:
        with pymupdf.open(path) as doc:
            pages = [p.get_text("text") for p in doc]
        return "\n".join(pages), len(pages), "\n".join(pages[:2])
    except Exception:
        return None, 0, ""


def title_overlap(title, head):
    words = set(TITLE_WORD.findall(str(title or "").lower()))
    if len(words) < 3:
        return None
    return len(words & set(TITLE_WORD.findall(head.lower()))) / len(words)


def english_share(text):
    words = WORD3.findall(text.lower())
    if not words:
        return 0.0
    return sum(w in ENGLISH for w in words) / len(words)


def measure_text(text, head, title):
    """Quality checks and measures for one document. Returns a dict with a status."""
    out = {"title_overlap": title_overlap(title, head), "english_share": english_share(text[:60000])}
    if out["title_overlap"] is not None and out["title_overlap"] < 0.3:
        return {**out, "status": "title_mismatch"}
    if SUPP_HEAD.search(text.lstrip()[:200]) and len(SUPP_LABEL.findall(text)) >= 3:
        return {**out, "status": "supplementary"}
    if out["english_share"] < 0.5:
        return {**out, "status": "unreadable_text"}
    m = REFS.search(text)
    if m and m.start() > 0.3 * len(text):
        body, refs = text[:m.start()], text[m.start():]
    else:
        body, refs = text, ""
    toks = L.tokenize(body)
    n = len(toks)
    if n < MIN_BODY or (n < SHORT_BODY and FRONT.search(text[:1500])):
        return {**out, "status": "not_article_or_short", "body_words": n}
    c = Counter(toks)
    fam = Counter()
    for w, k in c.items():
        f = L.FORM_TO_FAMILY.get(w)
        if f:
            fam[f] += k
    nonref = body + "\n" + text[len(body) + len(refs):]
    return dict(
        out, status="ok", body_words=n,
        marker=sum(fam[f] for f in L.MARKER_FAMILIES),
        control=sum(fam[f] for f in L.CONTROL_FAMILIES),
        hedge=sum(k for w, k in c.items() if w in L.HEDGES),
        booster=sum(k for w, k in c.items() if w in L.BOOSTERS),
        long_words=sum(k for w, k in c.items() if len(w) >= 9),
        ttr100=L.ttr_window(toks),
        n_figures=len({x.group(1) for x in FIG.finditer(text)}),
        n_tables=len({x.group(1) for x in TAB.finditer(text)}),
        n_equations=len({x.group(1) for x in EQN.finditer(body)}),
        ai_mention=bool(MENTION.search(body)),
        ai_disclosure=discloses(nonref),
        **{"fam_" + f: fam[f] for f in L.ALL_FAMILIES})


def find_doi(text, name):
    for x in DOI_RE.finditer(text):
        cand = x.group(0).rstrip(".,;)]").lower()
        if cand in KNOWN:
            return cand
    stem = re.sub(r"^\d{4}[_-]", "", os.path.splitext(name)[0]).lower()
    for g in (stem.replace("_", "/"), stem.replace("-", "/", 1)):
        if g in KNOWN:
            return g
    return None


def job(args):
    path, doi, folder = args
    name = os.path.basename(path)
    text, pages, head = read(path)
    if not text:
        return dict(file=name, folder=folder, doi=doi, status="unreadable", parser=PARSER_VERSION)
    if doi is None:
        doi = find_doi(text, name)
    base = dict(file=name, folder=folder, doi=doi, pages=pages, parser=PARSER_VERSION)
    if doi is None:
        return {**base, "status": "no_doi"}
    return {**base, **measure_text(text, head, TITLES.get(doi))}


def english_vocabulary():
    """Words used at least five times in a sample of 200,000 corpus abstracts."""
    if os.path.exists(VOCAB):
        return set(open(VOCAB, encoding="utf-8").read().split())
    ab = pd.read_parquet(CORPUS, columns=["abstract"]).abstract.dropna()
    ab = ab.sample(min(200000, len(ab)), random_state=11)
    c = Counter(w for t in ab for w in WORD3.findall(t.lower()))
    vocab = sorted(w for w, k in c.items() if k >= 5)
    open(VOCAB, "w", encoding="utf-8").write("\n".join(vocab))
    return set(vocab)


def lookups():
    c = pd.read_parquet(CORPUS, columns=["doi", "title"])
    c = c[c.doi.notna()].assign(doi=lambda d: d.doi.str.lower()).drop_duplicates("doi")
    return set(c.doi), dict(zip(c.doi, c.title)), english_vocabulary()


def main():
    t = time.time()
    redo = "--all" in sys.argv
    st = pd.concat([pd.read_csv(p, low_memory=False) for p in glob.glob(os.path.join(V2, "fetch_status*.csv"))])
    st = st[st.file.notna()].drop_duplicates("file")
    fmap = dict(zip(st.file, st.doi.str.lower()))
    known, titles, english = lookups()
    skip = set(pd.read_csv(os.path.join(V2, "nonarticle_files.csv")).name)
    done = pd.read_parquet(OUTFILE) if (os.path.exists(OUTFILE) and not redo) else pd.DataFrame()
    if len(done) and "parser" in done:
        done = done[done.parser == PARSER_VERSION]
    else:
        done = pd.DataFrame(columns=["file", "folder"])
    seen = set(zip(done.file, done.folder))
    now = time.time(); jobs = []
    for folder in ("pdf", "manual_pdf"):
        for p in glob.glob(os.path.join(V2, folder, "*.pdf")):
            b = os.path.basename(p)
            p = os.path.abspath(p)
            if len(p) > 250:
                p = "\\\\?\\" + p          # Windows long-path prefix
            if b in skip or (b, folder) in seen or now - os.path.getmtime(p) < 900:
                continue
            jobs.append((p, fmap.get(b) if folder == "pdf" else None, folder))
    print("to parse", len(jobs), "already", len(done), flush=True)
    rows = run_jobs(jobs, (known, titles, english), done, t)
    allr = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
    allr.to_parquet(OUTFILE, index=False)
    print(allr.groupby(["folder", "status"]).size().to_string(), "\ntime", round(time.time() - t), flush=True)


def run_jobs(jobs, initargs, done, t0, timeout=300):
    """Parse in a process pool, robust to a worker that dies or a PDF that hangs.

    A worker that crashes loses its task and would leave the pool waiting forever, so
    results are awaited with a timeout. When the wait times out the pool is replaced
    and the unfinished files are tried again; a file still unfinished on its own in a
    single-worker pool gets status "parse_timeout". Progress is saved every 1,000 files.
    """
    import multiprocessing as mp
    rows, remaining, rounds = [], list(jobs), 0
    key = lambda j: (os.path.basename(j[0]), j[2])
    while remaining:
        rounds += 1
        workers = 12 if rounds <= 2 else 1
        batch = remaining if workers > 1 else remaining[:1]
        got = set()
        pool = Pool(workers, initializer=init, initargs=initargs)
        it = pool.imap_unordered(job, batch, chunksize=1)
        try:
            for _ in range(len(batch)):
                r = it.next(timeout=timeout)
                rows.append(r)
                got.add((r["file"], r["folder"]))
                if len(rows) % 1000 == 0:
                    pd.concat([done, pd.DataFrame(rows)], ignore_index=True).to_parquet(OUTFILE, index=False)
                    print(len(rows), round(time.time() - t0), flush=True)
            pool.close()
        except mp.TimeoutError:
            print(f"timeout after {len(got)} of {len(batch)}; replacing the pool", flush=True)
        finally:
            pool.terminate()
        if workers == 1 and not got:
            path, doi, folder = batch[0]
            rows.append(dict(file=os.path.basename(path), folder=folder, doi=doi,
                             status="parse_timeout", parser=PARSER_VERSION))
            got.add(key(batch[0]))
        remaining = [j for j in remaining if key(j) not in got]
    return rows


if __name__ == "__main__":
    main()
