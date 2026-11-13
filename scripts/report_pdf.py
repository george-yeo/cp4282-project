"""Build REPORT_BETA.pdf from REPORT_BETA.md.

The Markdown is converted to HTML (python-markdown) and printed to PDF by a headless Chrome or
Edge, so the figures, tables and Unicode maths come out as they do on GitHub.

    python -m pip install markdown
    python scripts/report_pdf.py                  # writes REPORT_BETA.pdf
    python scripts/report_pdf.py --browser /path/to/chrome
"""

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
BROWSERS = (
    "C:/Program Files/Google/Chrome/Application/chrome.exe",
    "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
    "C:/Program Files/Microsoft/Edge/Application/msedge.exe",
    "google-chrome", "chromium", "chromium-browser", "microsoft-edge",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
)
CSS = """
@page { size: A4; margin: 20mm 19mm 20mm 19mm;
        @bottom-center { content: counter(page); font: 9pt 'Segoe UI', Arial, sans-serif; color: #666; } }
html { font: 10.2pt/1.42 'Segoe UI', 'Helvetica Neue', Arial, 'Cambria Math', sans-serif; color: #111; }
body { margin: 0; }
h1 { font-size: 19pt; line-height: 1.2; margin: 0 0 4pt; }
h1 + p { color: #444; margin: 0 0 14pt; }
h2 { font-size: 13.5pt; margin: 18pt 0 6pt; padding-top: 6pt; border-top: 0.6pt solid #bbb; }
h3 { font-size: 11.2pt; margin: 13pt 0 4pt; }
h2, h3 { break-after: avoid; }
p { margin: 0 0 7pt; }
ul, ol { margin: 0 0 7pt; padding-left: 16pt; }
li { margin: 0 0 3pt; }
img { display: block; max-width: 80%; max-height: 80mm; margin: 6pt auto 3pt; }
sup, sub { font-size: 0.72em; line-height: 0; }
p:has(> img) { break-inside: avoid; break-after: avoid; }
p:has(> img) + p em, p > em:only-child { color: #444; font-size: 9.4pt; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 8pt; font-size: 9pt; line-height: 1.3;
        break-inside: avoid; }
th, td { border: 0.5pt solid #bdbdbd; padding: 3pt 5pt; text-align: left; vertical-align: top; }
th { background: #f1f1ee; font-weight: 600; }
code { font: 9pt Consolas, 'Courier New', monospace; background: #f3f3f0; padding: 0 2pt;
       overflow-wrap: anywhere; }
a { color: #1c5cab; text-decoration: none; overflow-wrap: anywhere; }
strong { font-weight: 600; }
"""


def find_browser(explicit):
    for candidate in ([explicit] if explicit else BROWSERS):
        path = candidate if Path(candidate).exists() else shutil.which(candidate)
        if path:
            return str(path)
    raise SystemExit("No Chrome or Edge found; pass --browser.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=str(ROOT / "REPORT_BETA.md"))
    ap.add_argument("--out", default=str(ROOT / "REPORT_BETA.pdf"))
    ap.add_argument("--browser", default=None)
    args = ap.parse_args()

    text = Path(args.source).read_text(encoding="utf-8")
    body = markdown.markdown(text, extensions=["tables", "sane_lists"])
    body = re.sub(r'(?<![">])(https?://[^\s<)]+)', r'<a href="\1">\1</a>', body)
    # The Markdown writes powers and one subscript in plain text; typeset them in the PDF.
    body = body.replace("e^b", "e<sup>b</sup>").replace("α_min", "α<sub>min</sub>")
    body = re.sub(r"\^\(([^()]*)\)", r"<sup>\1</sup>", body)
    title = text.splitlines()[0].lstrip("# ").strip()
    html = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{title}</title>'
            f'<base href="{ROOT.as_uri()}/"><style>{CSS}</style></head><body>{body}</body></html>')
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "report.html"
        page.write_text(html, encoding="utf-8")
        subprocess.run([find_browser(args.browser), "--headless=new", "--disable-gpu",
                        "--no-pdf-header-footer", "--allow-file-access-from-files",
                        f"--user-data-dir={Path(tmp) / 'profile'}",
                        f"--print-to-pdf={Path(args.out).resolve()}", page.as_uri()],
                       check=True, capture_output=True, timeout=180)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
