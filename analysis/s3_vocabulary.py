"""
Step 3. Full-vocabulary excess analysis and word-list sensitivity.

Every word is grouped with its inflected forms (Snowball stem). Marker and control
families are counted from their listed forms only, exactly as in the headline
measure, and their stems are excluded from the background vocabulary so no family
appears twice.

Annual frequencies use a journal-balanced sample: at most 400 abstracts per journal
per year (seed 17), core journals, abstracts of at least 40 words. Each word's
2015-2022 annual frequencies get a straight line; the line's mean over 2023-2025 is
the projection; ratio = observed mean / projected mean (reported when the projection
is at least one per million words).

Data-driven lists. Candidates: not function words, at least four letters, at least
10 per million both projected and observed. Breadth: the word must already be in use
(at least 5 per million before 2023) in every one of the 14 subject categories, and
must exceed its own projection in at least 12 of them. This keeps out topical words
that are new to a field or rose with one field's research agenda. The 40 candidates with the highest
ratio form the list. The held-out list uses only abstracts published before April
2024 and is evaluated only on later months; the disjoint list excludes every marker
and control family.
"""
import os, sys, json
import settings as P
from collections import Counter
from multiprocessing import Pool
import numpy as np, pandas as pd
import lexicon as L, its_model as M

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = P.OUT
CORPUS = P.CORPUS
MIN_COUNT = 400
SPLIT = pd.Timestamp("2024-04-01")
LIST_SIZE = 40
RESERVED = None
PRE = list(range(2015, 2023))


def init():
    global RESERVED
    RESERVED = L.reserved_stems()


def keyed(text):
    c = Counter(); toks = L.tokenize(text)
    for w in toks:
        f = L.FORM_TO_FAMILY.get(w)
        if f:
            c["F:" + f] += 1
        else:
            s = L.stem(w)
            if s not in RESERVED:
                c[s] += 1
    return c, len(toks)


def count_chunk(args):
    texts, keys = args
    tot = {}
    for t, k in zip(texts, keys):
        c, n = keyed(t)
        if k not in tot:
            tot[k] = [Counter(), 0]
        tot[k][0].update(c); tot[k][1] += n
    return tot


def surface_chunk(texts):
    sc = {}
    for t in texts:
        for w in L.tokenize(t):
            if w in L.FORM_TO_FAMILY:
                continue
            d = sc.setdefault(L.stem(w), Counter()); d[w] += 1
    return {s: d.most_common(1)[0] for s, d in sc.items()}


def counts_by(texts, keys, pool):
    chunks = [(texts[i:i + 4000], keys[i:i + 4000]) for i in range(0, len(texts), 4000)]
    agg = {}
    for part in pool.imap(count_chunk, chunks):
        for k, (c, n) in part.items():
            if k not in agg:
                agg[k] = [Counter(), 0]
            agg[k][0].update(c); agg[k][1] += n
    return agg


def project(f_pre, post_years):
    b, a = np.polyfit(np.array(PRE, float), f_pre, 1)
    return float(np.mean(a + b * np.array(post_years, float)))


def excess(yc, post_years, words=None, min_count=MIN_COUNT):
    overall = Counter()
    for y in yc:
        overall.update(yc[y][0])
    vocab = words if words is not None else [w for w, c in overall.items() if c >= min_count or w.startswith("F:")]
    rows = []
    for w in vocab:
        f_pre = np.array([1e6 * yc[y][0][w] / yc[y][1] for y in PRE])
        f_post = np.array([1e6 * yc[y][0][w] / yc[y][1] for y in post_years])
        cf = project(f_pre, post_years); obs = float(f_post.mean())
        rows.append(dict(key=w, total=overall[w], pre=f_pre.mean(), cf=cf, post=obs,
                         gap=obs - cf, ratio=obs / cf if cf >= 1 else np.nan))
    return pd.DataFrame(rows)


def year_key(d, last_post):
    k = np.where(d.month >= pd.Timestamp("2023-01-01"), d.month.dt.year, d.year)
    return np.minimum(k, last_post)


def main():
    art = pd.read_parquet(os.path.join(OUT, "articles.parquet"),
                          columns=["openalex_id", "cohort", "has_text", "jan1", "month", "year",
                                   "source_id", "scholar_category"])
    art = art[art.year.between(2015, 2025) & (art.cohort == "core") & art.has_text]
    ab = pd.read_parquet(CORPUS, columns=["openalex_id", "abstract"])
    art = art.merge(ab, on="openalex_id", how="left")
    rng = np.random.default_rng(17)
    keep = []
    for _, g in art.groupby(["source_id", "year"]):
        keep.append(g if len(g) <= 400 else g.loc[rng.choice(g.index.values, 400, replace=False)])
    bal = pd.concat(keep)
    print("balanced sample", len(bal), flush=True)
    with Pool(12, initializer=init) as pool:
        texts = bal.abstract.tolist()
        lab = {}
        for part in pool.imap(surface_chunk, [texts[i:i + 4000] for i in range(0, len(texts), 4000)]):
            for s, (w, n) in part.items():
                if s not in lab or n > lab[s][1]:
                    lab[s] = (w, n)
        label = lambda k: k[2:] if k.startswith("F:") else lab.get(k, (k, 0))[0]
        yc = counts_by(texts, bal.year.tolist(), pool)
        ex = excess(yc, [2023, 2024, 2025])
        ex["label"] = [label(k) for k in ex.key]
        ex["kind"] = ["control" if k[2:] in L.CONTROL_FAMILIES else "marker" if k.startswith("F:") else "other" for k in ex.key]
        ex.to_csv(os.path.join(OUT, "excess_vocabulary.csv"), index=False)
        fam = ex[ex.kind != "other"].sort_values("ratio", ascending=False)
        fam.to_csv(os.path.join(OUT, "family_table.csv"), index=False)
        print(fam[["label", "kind", "pre", "cf", "post", "ratio"]].round(2).to_string(), flush=True)

        def select(data, post_years, excl):
            yk = year_key(data, max(post_years))
            e = excess(counts_by(data.abstract.tolist(), list(yk), pool), post_years)
            e["label"] = [label(k) for k in e.key]
            cand = e[(e.cf >= 10) & (e.post >= 10) & (~e.key.isin(excl))
                     & (~e.label.isin(L.FUNCTION_WORDS)) & (e.label.str.len() >= 4)]
            cand = cand.nlargest(400, "ratio")
            ccd = counts_by(data.abstract.tolist(), list(zip(data.scholar_category, yk)), pool)
            wins = Counter(); present = Counter()
            for c in sorted(set(data.scholar_category)):
                sub = {y: ccd[(c, y)] for y in PRE + post_years if (c, y) in ccd}
                if len(sub) < len(PRE) + len(post_years):
                    continue
                ec = excess(sub, post_years, words=list(cand.key))
                for w, o, p, pre in zip(ec.key, ec.post, ec.cf, ec.pre):
                    if pre >= 5:
                        present[w] += 1
                    if o > p:
                        wins[w] += 1
            ncat = len(set(data.scholar_category))
            cand = cand[[present[w] == ncat and wins[w] >= ncat - 2 for w in cand.key]]
            chosen = cand.nlargest(LIST_SIZE, "ratio")
            return chosen.key.tolist(), chosen.label.tolist()

        held, held_lab = select(bal[bal.month < SPLIT], [2023], [])
        disjoint, dis_lab = select(bal, [2023, 2024, 2025], [k for k in ex.key if k.startswith("F:")])
        json.dump({"held_out": held_lab, "disjoint": dis_lab}, open(os.path.join(OUT, "alt_lists.json"), "w"), indent=1)
        print("held-out:", held_lab); print("disjoint:", dis_lab, flush=True)
        prim = art[~art.jan1]
        mc = counts_by(prim.abstract.tolist(), list(prim.month), pool)
    rows = []
    for name, words, keepm in [("Held-out list", held, lambda m: (m <= M.T0) | (m >= SPLIT)),
                               ("Disjoint list", disjoint, lambda m: m == m)]:
        s = pd.DataFrame([{"month": m, "value": 1e4 * sum(mc[m][0][w] for w in words) / mc[m][1]} for m in sorted(mc)])
        s = s[keepm(s.month)]
        res, _ = M.fit(s)
        rows.append({"variant": name, "n_words": len(words), **res})
        print(name, len(words), round(res["departure_pct"], 1), flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "list_sensitivity.csv"), index=False)


if __name__ == "__main__":
    main()
