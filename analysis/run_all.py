"""
Run the analysis from the corpus to the figures, in order.

    python run_all.py            abstract and bibliographic analysis, then full text
    python run_all.py --no-pdf   skip steps 4, 4b and 4c (they need the PDFs, which are
                                 not distributed); step 5 then uses the shipped record,
                                 fulltext/fulltext_sample.csv.gz, copied to out/

Steps:
  s1_abstract_metrics.py   per-article measures from abstracts and records
  s2_series.py             monthly series, segmented regressions, robustness
  s3_vocabulary.py         full-vocabulary excess ratios and data-chosen word lists
  s4_fulltext_parse.py     measure the stored full-text PDFs
  s4b_fetch_bronze.py      measure free-to-read articles in memory (network)
  s4c_fulltext_record.py   write the record of the full-text sample
  s5_fulltext_results.py   before/after comparison of the full-text sample
  s6_figures.py            figures
  s7_extra_numbers.py      descriptive numbers quoted in the text
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = ["s1_abstract_metrics.py", "s2_series.py", "s3_vocabulary.py",
         "s4_fulltext_parse.py", "s4b_fetch_bronze.py", "s4c_fulltext_record.py",
         "s5_fulltext_results.py",
         "s6_figures.py", "s7_extra_numbers.py"]


def main():
    skip_pdf = "--no-pdf" in sys.argv
    if skip_pdf:
        import shutil
        import settings as P
        shutil.copyfile(os.path.join(P.ROOT, "fulltext", "fulltext_sample.csv.gz"),
                        os.path.join(P.OUT, "fulltext_sample.csv.gz"))
    for step in STEPS:
        if skip_pdf and step.startswith("s4"):
            continue
        print(f"\n=== {step}", flush=True)
        if subprocess.run([sys.executable, "-u", step], cwd=HERE).returncode:
            sys.exit(f"{step} failed")


if __name__ == "__main__":
    main()
