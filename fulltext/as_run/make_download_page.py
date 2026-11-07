"""
Build a clickable page for the articles still to be downloaded by hand.

Copying DOIs out of a spreadsheet one at a time is the slowest possible way to do
this, and the author has had to do it 129 times. A page of links removes the
copying: a direct PDF link downloads on click where one is known, and a DOI link
opens the article page where it is not.

Two things determine the order. Articles are grouped by journal, because a reader
already authenticated at a publisher can work down that publisher's entries without
signing in again. Within the page, open-access articles come first, since those
succeed from any network and the subscription ones need the campus connection.

Progress is read from the files already downloaded rather than from a position in
the list, because a listed article can be unobtainable and skipping it is normal.
Run manual_progress.py first to refresh that, then run this.
"""

import html
import os
import sys

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
FT = os.path.join(HERE, "data", "fulltext")
MANUAL = os.path.join(FT, "manual")
V2 = os.path.join(FT, "v2_target")

LIST_CSV = os.path.join(MANUAL, "download_list.csv")
HAVE_CSV = os.path.join(MANUAL, "collected_dois.csv")
SAVE_TO = os.path.join(V2, "manual_pdf")

OUT_HTML = os.path.join(MANUAL, "download_page.html")
OUT_CSV = os.path.join(MANUAL, "still_needed.csv")

RESOLVERS = [
    os.path.join(V2, "resolved.csv"),
    os.path.join(V2, "alternates.csv"),
    os.path.join(V2, "resolved_v3.csv"),
]

# Publisher PDF links are kept, deliberately.
#
# The fetcher treats these hosts as blocked, and for a script they are: MDPI,
# Wiley and IEEE answer an unattended request with a Cloudflare challenge. That
# is bot detection, not an entitlement check, and a person clicking in a signed-in
# browser passes it. Filtering them out on the strength of the script's experience
# removed 496 working links from this page, most of the MDPI set among them, and
# made the page markedly worse for its actual user.
#
# Nothing here evades anything. The links are ordinary publisher URLs, opened by
# hand, by a reader with access.
def usable(url):
    """True when a link points at a file rather than a landing page."""
    if not isinstance(url, str) or not url.startswith("http"):
        return False
    lowered = url.lower()
    # A DOI resolver or bare handle is a redirect to a landing page, so it offers
    # nothing the Open button does not already do.
    return not any(h in lowered for h in
                   ("doi.org/", "hdl.handle.net", "dx.doi.org"))


def direct_links():
    """Best known direct PDF link per DOI, preferring earlier resolvers."""
    links = {}
    for path in RESOLVERS:
        if not os.path.exists(path):
            continue
        frame = pd.read_csv(path, usecols=lambda c: c in
                            ("doi", "pdf_url", "host_type"))
        frame = frame[frame["pdf_url"].notna()]
        for row in frame.itertuples():
            key = str(row.doi).lower().strip()
            if key not in links:
                links[key] = str(row.pdf_url)
    return links


def main():
    listed = pd.read_csv(LIST_CSV)
    listed["key"] = listed["doi"].astype(str).str.lower().str.strip()

    have = set()
    if os.path.exists(HAVE_CSV):
        have = {str(d).lower().strip()
                for d in pd.read_csv(HAVE_CSV)["doi"].dropna()}
    else:
        print(f"no {os.path.basename(HAVE_CSV)}; run manual_progress.py first")

    listed["collected"] = listed["key"].isin(have)
    done = int(listed["collected"].sum())
    total = len(listed)

    remaining = listed[~listed["collected"]].copy()
    links = direct_links()
    remaining["pdf_url"] = remaining["key"].map(links)
    remaining.loc[~remaining["pdf_url"].map(usable), "pdf_url"] = pd.NA

    # A missing publisher must not remove the article from the page. Grouping on
    # a column containing nulls silently drops those rows, which cost 59 entries
    # before it was caught.
    for column in ("publisher", "venue"):
        remaining[column] = remaining[column].fillna("Unidentified")

    remaining.to_csv(OUT_CSV, index=False)

    # Open access first, then by journal so one login serves many articles.
    remaining["oa"] = (remaining["access"] == "open access").map(
        {True: 0, False: 1})
    remaining = remaining.sort_values(
        ["oa", "publisher", "venue", "priority"], kind="stable")

    blocks = []
    for (access, publisher, venue), group in remaining.groupby(
            ["access", "publisher", "venue"], sort=False):
        rows = []
        for item in group.itertuples():
            # Titles carry apostrophes, so attributes are double quoted and every
            # value is escaped with quote=True.
            doi = html.escape(str(item.doi), quote=True)
            title = html.escape(str(item.title)[:150], quote=True)
            year = int(item.publication_year)
            pdf = item.pdf_url if isinstance(item.pdf_url, str) else ""
            if pdf:
                button = (f'<a class="btn direct" href="'
                          f'{html.escape(pdf, quote=True)}" '
                          f'target="_blank" rel="noopener">PDF</a>')
            else:
                button = (f'<a class="btn" href="https://doi.org/{doi}" '
                          f'target="_blank" rel="noopener">Open</a>')
            rows.append(
                f'<tr data-doi="{doi}">'
                f'<td class="chk"><input type="checkbox"></td>'
                f'<td class="yr">{year}</td>'
                f'<td class="ti">{title}</td>'
                f'<td class="ac">{button}</td></tr>')

        tag = "oa" if access == "open access" else "sub"
        label = "open access" if access == "open access" else "subscription"
        blocks.append(
            f'<section class="jr {tag}" data-access="{tag}">'
            f'<h3>{html.escape(str(venue), quote=True)}'
            f'<span class="meta">{html.escape(str(publisher), quote=True)}'
            f' &middot; <span class="badge {tag}">{label}</span>'
            f' &middot; {len(group)} article{"s" if len(group) != 1 else ""}'
            f'</span></h3>'
            f'<table>{"".join(rows)}</table></section>')

    n_oa = int((remaining["access"] == "open access").sum())
    n_sub = len(remaining) - n_oa
    n_direct = int(remaining["pdf_url"].notna().sum())
    save_path = SAVE_TO.replace("\\", "/")

    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Articles still needed</title>
<style>
 :root {{ --line:#dcdcdc; --ink:#222; --mut:#666; }}
 * {{ box-sizing: border-box; }}
 body {{ font-family: system-ui, -apple-system, Segoe UI, sans-serif;
        margin: 0; padding: 0 0 4rem; color: var(--ink); background: #fff; }}
 header {{ position: sticky; top: 0; background: #fff; z-index: 10;
           border-bottom: 1px solid var(--line); padding: 1rem 1.5rem; }}
 h1 {{ font-size: 1.15rem; margin: 0 0 .5rem; }}
 .bar {{ display: flex; gap: 1.2rem; flex-wrap: wrap; align-items: center;
         font-size: .85rem; color: var(--mut); }}
 .bar strong {{ color: var(--ink); }}
 .prog {{ height: 5px; background: #eee; border-radius: 3px; overflow: hidden;
          margin-top: .6rem; }}
 .prog i {{ display: block; height: 100%; width: {100 * done / total:.1f}%;
            background: #2a7; }}
 main {{ padding: 1.25rem 1.5rem; max-width: 1000px; }}
 .note {{ background: #f7f9fb; border-left: 3px solid #89a; padding: .8rem 1rem;
          font-size: .85rem; line-height: 1.55; margin-bottom: 1.5rem; }}
 .note code {{ background: #eef1f4; padding: .1rem .3rem; border-radius: 3px;
               font-size: .95em; }}
 section.jr {{ margin-bottom: 1.5rem; border: 1px solid var(--line);
               border-radius: 6px; overflow: hidden; }}
 section.jr h3 {{ margin: 0; padding: .6rem .9rem; background: #fafafa;
                  border-bottom: 1px solid var(--line); font-size: .92rem; }}
 .meta {{ display: block; font-weight: 400; color: var(--mut);
          font-size: .78rem; margin-top: .15rem; }}
 .badge {{ padding: .05rem .4rem; border-radius: 3px; font-size: .95em; }}
 .badge.oa {{ background: #e3f3e8; color: #1d6b3a; }}
 .badge.sub {{ background: #fdeedf; color: #8a4b12; }}
 table {{ width: 100%; border-collapse: collapse; }}
 td {{ padding: .38rem .6rem; border-bottom: 1px solid #f0f0f0;
       font-size: .84rem; vertical-align: middle; }}
 tr:last-child td {{ border-bottom: 0; }}
 tr.got {{ background: #f4faf6; }}
 tr.got .ti {{ color: #8a8a8a; text-decoration: line-through; }}
 td.chk {{ width: 30px; }} td.yr {{ width: 46px; color: var(--mut); }}
 td.ac {{ width: 82px; text-align: right; }}
 .btn {{ display: inline-block; padding: .2rem .6rem; border-radius: 4px;
         text-decoration: none; font-size: .78rem; border: 1px solid #c3cdd6;
         color: #2b5f88; background: #fff; }}
 .btn:hover {{ background: #eef4f9; }}
 .btn.direct {{ background: #2b6cb0; border-color: #2b6cb0; color: #fff; }}
 .btn.direct:hover {{ background: #24578c; }}
 button.f {{ font: inherit; font-size: .8rem; padding: .25rem .7rem;
             border: 1px solid #c3cdd6; background: #fff; border-radius: 4px;
             cursor: pointer; }}
 button.f[aria-pressed="true"] {{ background: #2b6cb0; color: #fff;
                                  border-color: #2b6cb0; }}
 .hide {{ display: none !important; }}
</style>

<header>
  <h1>Articles still needed &mdash; {len(remaining):,} of {total:,}</h1>
  <div class="bar">
    <span><strong>{done}</strong> collected</span>
    <span><strong>{n_oa:,}</strong> open access</span>
    <span><strong>{n_sub:,}</strong> subscription</span>
    <span><strong>{n_direct:,}</strong> direct PDF links</span>
    <span style="margin-left:auto">
      <button class="f" id="f-all" aria-pressed="true">All</button>
      <button class="f" id="f-oa" aria-pressed="false">Open access only</button>
      <button class="f" id="f-todo" aria-pressed="false">Hide checked</button>
    </span>
  </div>
  <div class="prog"><i></i></div>
</header>

<main>
<div class="note">
<p><strong>Save everything to</strong> <code>{save_path}</code>.
Filenames do not matter; the DOI printed inside each file is what identifies it, so
accept whatever name the publisher gives.</p>
<p>A blue <strong>PDF</strong> button is a direct link and should download on click.
A plain <strong>Open</strong> button goes to the article page, where the PDF link is
usually top right. Subscription articles need the campus or VPN connection; open
access ones work anywhere.</p>
<p>Checkboxes are only a visual reminder and are remembered in this browser. They do
not record anything: progress is measured by reading the files you saved. When you
have a batch, run <code>manual_progress.py</code> and then this script again, and the
page will rebuild with the collected articles removed.</p>
<p>Articles are grouped by journal so that one sign-in covers a run of them, and
open access is listed first. Skip anything that will not come; 25 of the first 129
could not be obtained either, which is expected and already accounted for.</p>
</div>

{''.join(blocks)}
</main>

<script>
 const KEY = "dl-checked-v1";
 const saved = new Set(JSON.parse(localStorage.getItem(KEY) || "[]"));

 document.querySelectorAll("tr[data-doi]").forEach(tr => {{
   const doi = tr.dataset.doi;
   const box = tr.querySelector("input");
   if (saved.has(doi)) {{ box.checked = true; tr.classList.add("got"); }}
   box.addEventListener("change", () => {{
     tr.classList.toggle("got", box.checked);
     box.checked ? saved.add(doi) : saved.delete(doi);
     localStorage.setItem(KEY, JSON.stringify([...saved]));
     applyFilters();
   }});
 }});

 // Clicking the link is the usual signal that an article has been dealt with.
 document.querySelectorAll("a.btn").forEach(a => {{
   a.addEventListener("click", () => {{
     const tr = a.closest("tr");
     const box = tr.querySelector("input");
     if (!box.checked) {{
       box.checked = true;
       box.dispatchEvent(new Event("change"));
     }}
   }});
 }});

 const btns = {{ all: document.getElementById("f-all"),
                oa: document.getElementById("f-oa"),
                todo: document.getElementById("f-todo") }};
 let oaOnly = false, hideDone = false;

 function applyFilters() {{
   document.querySelectorAll("section.jr").forEach(sec => {{
     const isOA = sec.dataset.access === "oa";
     let shown = 0;
     sec.querySelectorAll("tr[data-doi]").forEach(tr => {{
       const hide = (hideDone && tr.classList.contains("got"));
       tr.classList.toggle("hide", hide);
       if (!hide) shown++;
     }});
     sec.classList.toggle("hide", (oaOnly && !isOA) || shown === 0);
   }});
 }}

 btns.oa.addEventListener("click", () => {{
   oaOnly = !oaOnly;
   btns.oa.setAttribute("aria-pressed", oaOnly);
   btns.all.setAttribute("aria-pressed", !oaOnly && !hideDone);
   applyFilters();
 }});
 btns.todo.addEventListener("click", () => {{
   hideDone = !hideDone;
   btns.todo.setAttribute("aria-pressed", hideDone);
   btns.all.setAttribute("aria-pressed", !oaOnly && !hideDone);
   applyFilters();
 }});
 btns.all.addEventListener("click", () => {{
   oaOnly = hideDone = false;
   btns.oa.setAttribute("aria-pressed", false);
   btns.todo.setAttribute("aria-pressed", false);
   btns.all.setAttribute("aria-pressed", true);
   applyFilters();
 }});

 applyFilters();
</script>
</html>
"""

    with open(OUT_HTML, "w", encoding="utf-8") as fh:
        fh.write(page)

    print(f"collected so far : {done} of {total}")
    print(f"still needed     : {len(remaining):,} "
          f"({n_oa:,} open access, {n_sub:,} subscription)")
    print(f"direct PDF links : {n_direct:,}")
    print()
    print(f"page : {OUT_HTML}")
    print(f"csv  : {OUT_CSV}")


if __name__ == "__main__":
    main()
