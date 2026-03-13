# Handwriting → LaTeX PDF

Upload a photo of handwritten math or text and get a typeset PDF back. Uses GPT-4o vision to convert handwriting to LaTeX, then compiles it with pdflatex.

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

## How It Works

1. Upload a JPEG, PNG, WebP, or GIF of handwritten content
2. GPT-4o (high-detail vision) transcribes it to LaTeX
3. pdflatex compiles the LaTeX to PDF
4. PDF downloads automatically in the browser

## Stack

- **Backend:** FastAPI + Python
- **AI:** OpenAI GPT-4o vision API
- **PDF:** pdflatex (MacTeX)
- **Frontend:** Vanilla HTML/CSS/JS
