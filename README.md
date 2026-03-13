# Handwriting → LaTeX PDF

Upload a photo of handwritten math or text and get a typeset document back. Uses GPT-4o vision to convert handwriting to LaTeX, then compiles it with pdflatex.

![UI Demo](https://github.com/sahajkhandelwal1/neural-handwriting-latex/releases/download/v0.1.0/ReadmeDemo.png)

![Comparison](https://github.com/sahajkhandelwal1/neural-handwriting-latex/releases/download/v0.1.0/Comparision.png)

## How It Works

1. Upload a JPEG, PNG, WebP, or GIF of handwritten content
2. GPT-4o (high-detail vision) transcribes it to LaTeX
3. pdflatex compiles the LaTeX to a PDF
4. A rendered preview of the output is shown in the browser
5. Export as **PDF** (recommended), LaTeX source (`.tex`), or **PNG**

Past conversions are saved in the History panel with all three export formats available.

## Setup

**Prerequisites:**
- Python 3.9+
- MacTeX (for pdflatex): `brew install --cask mactex-no-gui`
- OpenAI API key

**Install:**
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

**Configure:**
```bash
echo "OPENAI_API_KEY=sk-..." > .env
```

**Run:**
```bash
eval "$(/usr/libexec/path_helper)"   # add pdflatex to PATH (Mac)
uvicorn main:app --reload --port 8000
```

Then open http://localhost:8000.

## Stack

- **Backend:** FastAPI + Python
- **AI:** OpenAI GPT-4o vision API
- **PDF:** pdflatex (MacTeX)
- **PNG rendering:** Ghostscript
- **Frontend:** Vanilla HTML/CSS/JS
