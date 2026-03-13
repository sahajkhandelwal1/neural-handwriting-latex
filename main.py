import asyncio
import base64
import json
import os
import re
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

import aiofiles
import openai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

# Fail fast if pdflatex not installed
if not shutil.which("pdflatex"):
    raise RuntimeError(
        "pdflatex not found. Install MacTeX: brew install --cask mactex-no-gui\n"
        "Then add /Library/TeX/texbin to PATH and restart."
    )

TMP = Path("tmp")
TMP.mkdir(exist_ok=True)

HISTORY_DIR = Path("history")
HISTORY_DIR.mkdir(exist_ok=True)

HISTORY_FILE = HISTORY_DIR / "index.json"

client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp", "image/gif"}
EXT_MAP = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
}

LATEX_TEMPLATE = r"""\documentclass{{article}}
\usepackage{{amsmath}}
\usepackage{{amssymb}}
\usepackage{{amsfonts}}
\usepackage[T1]{{fontenc}}
\usepackage{{lmodern}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{parskip}}
\usepackage{{microtype}}
\begin{{document}}
{body}
\end{{document}}
"""

SYSTEM_PROMPT = """You are a LaTeX transcription assistant. Convert the handwritten content in images to LaTeX.

Rules:
- Output ONLY the LaTeX body — no \\documentclass, no \\begin{document}, no \\end{document}
- Use $...$ for inline math
- Use \\[...\\] for display equations
- Use align* environment for multi-line derivations
- Do NOT wrap your response in markdown code fences (no ```latex or ```)
- Make the most mathematically reasonable interpretation of ambiguous handwriting
- Preserve the logical structure and flow of the content"""


# --- History helpers ---

def load_history() -> list:
    if not HISTORY_FILE.exists():
        return []
    try:
        return json.loads(HISTORY_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def save_history(entries: list) -> None:
    HISTORY_FILE.write_text(json.dumps(entries, indent=2))


# --- AI + compilation ---

def call_openai(image_b64: str, media_type: str) -> str:
    message = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=4096,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{image_b64}",
                            "detail": "high",
                        },
                    },
                    {"type": "text", "text": "Convert this handwritten content to LaTeX."},
                ],
            },
        ],
    )
    return message.choices[0].message.content


def strip_fences(text: str) -> str:
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text.strip())
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


def extract_log_errors(log_path: Path) -> str:
    if not log_path.exists():
        return "Unknown compilation error"
    lines = log_path.read_text(errors="replace").splitlines()
    errors = [l for l in lines if l.startswith("!")][:5]
    return "\n".join(errors) if errors else "Unknown compilation error"


async def cleanup_job(job_id: str, delay: int = 30):
    await asyncio.sleep(delay)
    for f in TMP.glob(f"{job_id}*"):
        try:
            f.unlink()
        except OSError:
            pass


# --- App ---

app = FastAPI(title="Neural Handwriting LaTeX")


@app.post("/convert")
async def convert(file: UploadFile):
    content_type = file.content_type or ""
    if content_type not in ALLOWED_MIME:
        raise HTTPException(status_code=415, detail="File must be an image")

    job_id = uuid.uuid4().hex
    ext = EXT_MAP[content_type]
    original_name = file.filename or f"upload{ext}"
    input_path = TMP / f"{job_id}_input{ext}"
    tex_path = TMP / f"{job_id}.tex"
    pdf_path = TMP / f"{job_id}.pdf"
    log_path = TMP / f"{job_id}.log"
    hist_pdf = HISTORY_DIR / f"{job_id}.pdf"

    try:
        async with aiofiles.open(input_path, "wb") as f:
            contents = await file.read()
            await f.write(contents)

        image_b64 = base64.standard_b64encode(contents).decode("utf-8")

        try:
            loop = asyncio.get_event_loop()
            latex_body = await loop.run_in_executor(
                None, call_openai, image_b64, content_type
            )
        except openai.APIError as e:
            raise HTTPException(status_code=502, detail=f"AI service error: {e}")

        latex_body = strip_fences(latex_body)
        tex_content = LATEX_TEMPLATE.format(body=latex_body)

        async with aiofiles.open(tex_path, "w") as f:
            await f.write(tex_content)

        try:
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: subprocess.run(
                        [
                            "pdflatex",
                            "-interaction=nonstopmode",
                            "-halt-on-error",
                            "-output-directory", str(TMP),
                            str(tex_path),
                        ],
                        capture_output=True,
                        text=True,
                    ),
                ),
                timeout=30,
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="PDF generation timed out")

        if result.returncode != 0:
            errors = extract_log_errors(log_path)
            raise HTTPException(status_code=422, detail=f"LaTeX compilation failed: {errors}")

        if not pdf_path.exists():
            raise HTTPException(status_code=422, detail="PDF was not generated")

        # Save to history
        shutil.copy2(pdf_path, hist_pdf)
        entries = load_history()
        entries.insert(0, {
            "id": job_id,
            "name": original_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        save_history(entries)

        asyncio.create_task(cleanup_job(job_id, delay=30))
        return FileResponse(
            path=str(pdf_path),
            media_type="application/pdf",
            filename="handwriting.pdf",
        )

    except HTTPException:
        asyncio.create_task(cleanup_job(job_id, delay=5))
        raise
    except Exception as e:
        asyncio.create_task(cleanup_job(job_id, delay=5))
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


@app.get("/history")
async def get_history():
    return JSONResponse(load_history())


@app.get("/history/{job_id}/pdf")
async def download_history_pdf(job_id: str):
    # Sanitize — only hex chars allowed
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise HTTPException(status_code=400, detail="Invalid job ID")
    pdf_path = HISTORY_DIR / f"{job_id}.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF not found")
    return FileResponse(path=str(pdf_path), media_type="application/pdf", filename="handwriting.pdf")


@app.delete("/history/{job_id}")
async def delete_history_entry(job_id: str):
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise HTTPException(status_code=400, detail="Invalid job ID")
    pdf_path = HISTORY_DIR / f"{job_id}.pdf"
    try:
        pdf_path.unlink(missing_ok=True)
    except OSError:
        pass
    entries = [e for e in load_history() if e["id"] != job_id]
    save_history(entries)
    return {"ok": True}


# Mount static files LAST
app.mount("/", StaticFiles(directory="static", html=True), name="static")
