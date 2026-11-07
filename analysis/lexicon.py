"""
Word lists and tokenization shared by every step of the clean pipeline.

The marker list is the predefined list of the study: 46 written forms, fixed before
estimation. Here every form belongs to exactly one word family, and every count,
table and figure is reported per family, never per form, so that inflections such
as underscore / underscores / underscoring are always one word.

The four negative controls are kept in a separate family set. They are NOT part of
the marker aggregate.
"""
import re

TOKEN_RE = re.compile(r"[a-z][a-z\-']+")


def tokenize(text):
    return TOKEN_RE.findall(str(text).lower())


# family -> written forms (the 46 predefined forms, unchanged)
MARKER_FAMILIES = {
    "additionally": ["additionally"],
    "align": ["align", "aligning"],
    "commendable": ["commendable"],
    "comprehensive": ["comprehensive"],
    "consequently": ["consequently"],
    "crucial": ["crucial"],
    "cutting-edge": ["cutting-edge"],
    "delve": ["delve"],
    "elevate": ["elevate"],
    "foster": ["foster", "fostering"],
    "furthermore": ["furthermore"],
    "harness": ["harness", "harnessing"],
    "innovative": ["innovative"],
    "intricate": ["intricate", "intricacies"],
    "landscape": ["landscape"],
    "leverage": ["leverage", "leveraging"],
    "meticulous": ["meticulous", "meticulously"],
    "multifaceted": ["multifaceted"],
    "notably": ["notably"],
    "noteworthy": ["noteworthy"],
    "nuanced": ["nuanced"],
    "paradigm": ["paradigm"],
    "pivotal": ["pivotal"],
    "realm": ["realm"],
    "robust": ["robust"],
    "seamless": ["seamless", "seamlessly"],
    "showcase": ["showcase", "showcasing"],
    "significant": ["significant"],
    "tapestry": ["tapestry"],
    "underscore": ["underscore", "underscores", "underscoring"],
    "unveil": ["unveil", "unveiling"],
}

CONTROL_FAMILIES = {
    "holistic": ["holistic"],
    "moreover": ["moreover"],
    "state-of-the-art": ["state-of-the-art"],
    "vital": ["vital"],
}

# Seven families that appear in neither published source (Kobak et al. 2025;
# Liang et al. 2024, peer reviews); added by the author.
ADDED_FAMILIES = ["cutting-edge", "furthermore", "landscape", "paradigm",
                  "robust", "significant", "tapestry"]

FORM_TO_FAMILY = {}
for fam, forms in {**MARKER_FAMILIES, **CONTROL_FAMILIES}.items():
    for f in forms:
        FORM_TO_FAMILY[f] = fam
MARKER_FORMS = {f for forms in MARKER_FAMILIES.values() for f in forms}
CONTROL_FORMS = {f for forms in CONTROL_FAMILIES.values() for f in forms}
ALL_FAMILIES = list(MARKER_FAMILIES) + list(CONTROL_FAMILIES)

HEDGES = {
    "may", "might", "could", "possibly", "potentially", "suggest", "suggests",
    "appear", "appears", "seem", "seems", "likely", "unlikely", "probable",
    "presumably", "arguably", "relatively", "somewhat", "generally", "typically",
}
BOOSTERS = {
    "clearly", "obviously", "definitely", "certainly", "undoubtedly", "strongly",
    "highly", "extremely", "substantially", "considerably", "remarkably",
    "dramatically", "significantly", "greatly", "vastly",
}

STOPWORDS = {
    "the", "and", "for", "are", "was", "were", "this", "that", "with", "from",
    "have", "has", "had", "not", "but", "its", "their", "which", "these", "those",
    "can", "been", "being", "than", "then", "when", "where", "into", "also",
    "such", "each", "more", "most", "other", "some", "all", "both", "our", "we",
}


def syllables(word):
    word = re.sub(r"[^a-z]", "", word)
    if not word:
        return 0
    count = len(re.findall(r"[aeiouy]+", word))
    if word.endswith("e") and count > 1 and not word.endswith(("le", "ee")):
        count -= 1
    return max(count, 1)


def ttr_window(tokens, window=100):
    if len(tokens) < window:
        return len(set(tokens)) / len(tokens) if tokens else float("nan")
    r = [len(set(tokens[i:i + window])) / window
         for i in range(0, len(tokens) - window + 1, window)]
    return sum(r) / len(r)


_stem_cache = {}
_stemmer = None


def family_key(word):
    """Word-family key for the full-vocabulary analysis.

    Marker and control forms map to their family. Every other word maps to its
    Snowball (Porter2) stem, so inflected forms of any word are counted together.
    """
    global _stemmer
    k = _stem_cache.get(word)
    if k is None:
        if _stemmer is None:
            from nltk.stem.snowball import SnowballStemmer
            _stemmer = SnowballStemmer("english")
        k = _stemmer.stem(word)
        _stem_cache[word] = k
    return k


def stem(word):
    """Snowball (Porter2) stem, memoized. Used for the full-vocabulary analysis,
    where every word is grouped with its inflected forms."""
    return family_key(word)


def reserved_stems():
    """Stems belonging to marker or control families. The full-vocabulary cloud
    excludes these so that each marker family appears once, computed from its
    listed forms exactly as in the tables."""
    out = set()
    for fam, forms in {**MARKER_FAMILIES, **CONTROL_FAMILIES}.items():
        for f in forms + [fam]:
            out.add(stem(f))
    return out


# NLTK English stopword list (179 words), plus "also", used only to keep function
# words out of the data-driven alternative word lists.
FUNCTION_WORDS = set("""i me my myself we our ours ourselves you you're you've you'll you'd
your yours yourself yourselves he him his himself she she's her hers herself it it's its
itself they them their theirs themselves what which who whom this that that'll these those
am is are was were be been being have has had having do does did doing a an the and but if
or because as until while of at by for with about against between into through during
before after above below to from up down in out on off over under again further then once
here there when where why how all any both each few more most other some such no nor not
only own same so than too very s t can will just don don't should should've now d ll m o re
ve y ain aren aren't couldn couldn't didn didn't doesn doesn't hadn hadn't hasn hasn't haven
haven't isn isn't ma mightn mightn't mustn mustn't needn needn't shan shan't shouldn
shouldn't wasn wasn't weren weren't won won't wouldn wouldn't also""".split())
