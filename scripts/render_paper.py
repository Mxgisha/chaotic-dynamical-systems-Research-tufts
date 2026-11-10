"""Render a markdown file to a self-contained HTML file and a PDF.

Figures are inlined as data URIs so both outputs stand alone, and DejaVu fonts
are embedded because the built-in PDF fonts lack the maths glyphs (using <=, =>,
lambda, nabla) that the paper needs.

    python scripts/render_paper.py

Writes <name>.html and <name>.pdf next to the input, e.g.

    python scripts/render_paper.py            # PAPER.md -> PAPER.html, PAPER.pdf
    python scripts/render_paper.py PIPELINE.md
"""
import base64, pathlib, re, sys
import markdown

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import sys
MD = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'PAPER.md'
MD = MD if MD.is_absolute() else ROOT / MD
md_text = MD.read_text()

# Rewrite image links to absolute paths so the PDF renderer can find them, and
# inline them as data URIs so the HTML is self-contained.
def inline(match):
    alt, src = match.group(1), match.group(2)
    p = ROOT / src
    if not p.exists():
        print(f'  WARNING missing figure: {src}', file=sys.stderr)
        return match.group(0)
    b64 = base64.b64encode(p.read_bytes()).decode()
    return f'![{alt}](data:image/png;base64,{b64})'

md_text = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', inline, md_text)

html_body = markdown.markdown(
    md_text, extensions=['tables', 'fenced_code', 'toc', 'attr_list', 'sane_lists']
)

# xhtml2pdf needs real font files to render the maths glyphs (<=, =>, lambda,
# nabla) that the built-in PDF fonts lack.  Borrow matplotlib's bundled DejaVu,
# which covers them.  A placeholder avoids f-string brace escaping entirely.
import matplotlib

FONTDIR = matplotlib.get_data_path() + '/fonts/ttf'

CSS = """
@font-face { font-family: DejaVuSerif; src: url("__FD__/DejaVuSerif.ttf"); }
@font-face { font-family: DejaVuSerif; font-weight: bold;
             src: url("__FD__/DejaVuSerif-Bold.ttf"); }
@font-face { font-family: DejaVuSerif; font-style: italic;
             src: url("__FD__/DejaVuSerif-Italic.ttf"); }
@font-face { font-family: DejaVuSansMono; src: url("__FD__/DejaVuSansMono.ttf"); }

@page { size: A4; margin: 20mm 18mm; }
body { font-family: DejaVuSerif, serif; font-size: 10pt; line-height: 1.5;
       color: #111; }
h1 { font-size: 19pt; line-height: 1.2; margin: 0 0 4pt 0; }
h1 + h3 { margin-top: 0; color: #444; font-weight: normal; font-size: 12.5pt; }
h2 { font-size: 13.5pt; margin-top: 20pt; border-bottom: 1px solid #ccc;
     padding-bottom: 2pt; page-break-after: avoid; }
h3 { font-size: 11.5pt; margin-top: 14pt; page-break-after: avoid; }
h4 { font-size: 10.5pt; margin-top: 12pt; page-break-after: avoid; }
p, li { text-align: justify; }
code { font-family: DejaVuSansMono, monospace; font-size: 8.5pt;
       background: #f4f4f4; padding: 0 2px; }
pre { background: #f7f7f7; border-left: 3px solid #bbb; padding: 6pt 8pt;
      font-size: 8pt; line-height: 1.35; page-break-inside: avoid; }
pre code { background: none; }
table { width: 100%; margin: 8pt 0; font-size: 8.5pt;
        page-break-inside: avoid; }
th, td { border: 1px solid #ccc; padding: 3pt 5pt; text-align: left; }
th { background: #eee; }
img { max-width: 100%; height: auto; margin: 8pt 0;
      page-break-inside: avoid; }
blockquote { border-left: 3px solid #666; margin: 10pt 0; padding: 2pt 12pt;
             background: #fafafa; }
blockquote p { text-align: left; }
hr { border: none; border-top: 1px solid #ddd; margin: 16pt 0; }
a { color: #0645ad; text-decoration: none; }
""".replace("__FD__", FONTDIR)

html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Predicting the Wrong Thing Correctly</title>
<style>{CSS}</style></head>
<body>{html_body}</body></html>"""

html_out = MD.with_suffix('.html')
html_out.write_text(html)
print('wrote', html_out.name)

from xhtml2pdf import pisa
pdf_out = MD.with_suffix('.pdf')
try:
    # WeasyPrint handles multiple embedded images correctly; xhtml2pdf silently
    # reuses the first image object for every <img> in the document.
    from weasyprint import HTML
    HTML(string=html, base_url=str(ROOT)).write_pdf(pdf_out)
    print('wrote', pdf_out.name, '| weasyprint')
except Exception as exc:                                   # pragma: no cover
    from xhtml2pdf import pisa
    with open(pdf_out, 'wb') as out:
        status = pisa.CreatePDF(html, dest=out)
    print('wrote', pdf_out.name, '| xhtml2pdf fallback:', exc)
