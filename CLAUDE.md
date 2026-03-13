# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the App

pdflatex must be on PATH before starting. If it isn't:
```bash
eval "$(/usr/libexec/path_helper)"
```

Start the server:
```bash
source venv/bin/activate
uvicorn main:app --reload --port 8000
```

Install dependencies:
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Architecture

Single-file FastAPI backend (`main.py`) with a vanilla JS frontend (`static/index.html`).

**Request pipeline (`POST /convert`):**
1. Validate image MIME type
2. Write upload to `tmp/{job_id}_input{ext}`
3. Base64-encode bytes → send to OpenAI `gpt-4o` vision API (run in thread executor since SDK is sync)
4. Strip any markdown fences from response
5. Wrap LaTeX body in `LATEX_TEMPLATE` → write `tmp/{job_id}.tex`
6. Run `pdflatex -interaction=nonstopmode -halt-on-error` via subprocess (30s timeout)
7. Return `FileResponse` for the PDF; schedule `cleanup_job` to delete all `tmp/{job_id}*` files after 30s

**Static files** are mounted last via `StaticFiles(directory="static", html=True)` — this must stay last or it will intercept API routes.

**`tmp/` directory** is created at startup and never checked in. All per-request files are namespaced by `job_id = uuid.uuid4().hex`.

## Key Details

- OpenAI model: `gpt-4o` with `"detail": "high"` image encoding
- LaTeX packages in template: `amsmath`, `amssymb`, `amsfonts`, `fontenc`, `lmodern`, `geometry` (1in margins), `parskip`, `microtype`
- pdflatex errors are extracted from the `.log` file (lines starting with `!`) and returned as HTTP 422
- The OpenAI SDK is synchronous — always call it via `loop.run_in_executor(None, ...)`
- `.env` must contain `OPENAI_API_KEY`
