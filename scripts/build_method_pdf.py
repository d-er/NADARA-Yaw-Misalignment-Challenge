"""Method description as PDF: renders the method part of README.md (everything before
"## 9. Repository layout") with pandoc and pdfLaTeX.
Reads README.md and the figures it references; writes docs/T0_method_33.pdf.
Needs pypandoc_binary and a system pdflatex. Run after the figure scripts.
"""
import tempfile
from pathlib import Path
import pypandoc

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "T0_method_33.pdf"

CUT = "\n## 9. Repository layout"

# characters used in the README that pdfLaTeX does not know; inline code is set
# with listings so that long paths break instead of running off the page
HEADER = r"""\DeclareUnicodeCharacter{2212}{\ensuremath{-}}
\DeclareUnicodeCharacter{00B0}{\ensuremath{^\circ}}
\DeclareUnicodeCharacter{2026}{\ensuremath{\ldots}}
\lstset{basicstyle=\ttfamily,breaklines=true,columns=fullflexible,keepspaces=true,
  literate={−}{{$-$}}1 {°}{{$^\circ$}}1 {…}{{\ldots}}1 {§}{{\S}}1 {±}{{$\pm$}}1}
\emergencystretch=5em
"""

text = (ROOT / "README.md").read_text()
assert CUT in text, "README.md has no section 9 heading to cut at"
method = text.split(CUT)[0]

with tempfile.TemporaryDirectory() as tmp:
    hdr = Path(tmp) / "header.tex"
    hdr.write_text(HEADER)
    pypandoc.convert_text(
        method, "pdf", format="markdown-implicit_figures", outputfile=str(OUT),
        extra_args=["--pdf-engine=pdflatex", "--syntax-highlighting=idiomatic", f"--resource-path={ROOT}",
                    "-V", "geometry:margin=2.2cm", "-V", "fontsize=10pt", "-V", "colorlinks=true", "-H", str(hdr)])
print(f"wrote {OUT.relative_to(ROOT)}")
