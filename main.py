import asyncio
import base64
import os
import re
import shutil
import uuid
from pathlib import Path

import aiofiles
import openai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

# Fail fast if pdflatex not installed
if not shutil.which("pdflatex"):
    raise RuntimeError(
        "pdflatex not found. Install MacTeX: brew install --cask mactex\n"
        "Then add /Library/TeX/texbin to PATH and restart."
    )

TMP = Path("tmp")
TMP.mkdir(exist_ok=True)

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
    """Remove markdown code fences if the model added them despite instructions."""
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


app = FastAPI(title="Handwriting to LaTeX PDF")


@app.post("/convert")
async def convert(file: UploadFile):
    # Validate MIME type
    content_type = file.content_type or ""
    if content_type not in ALLOWED_MIME:
        raise HTTPException(status_code=415, detail="File must be an image")

    job_id = uuid.uuid4().hex
    ext = EXT_MAP[content_type]
    input_path = TMP / f"{job_id}_input{ext}"
    tex_path = TMP / f"{job_id}.tex"
    pdf_path = TMP / f"{job_id}.pdf"
    log_path = TMP / f"{job_id}.log"

    try:
        # Write upload to disk
        async with aiofiles.open(input_path, "wb") as f:
            contents = await file.read()
            await f.write(contents)

        # Base64 encode for OpenAI API
        image_b64 = base64.standard_b64encode(contents).decode("utf-8")

        # Call OpenAI in thread executor (SDK is synchronous)
        try:
            loop = asyncio.get_event_loop()
            latex_body = await loop.run_in_executor(
                None, call_openai, image_b64, content_type
            )
        except openai.APIError as e:
            raise HTTPException(status_code=502, detail=f"AI service error: {e}")

        latex_body = strip_fences(latex_body)

        # Write .tex file
        tex_content = LATEX_TEMPLATE.format(body=latex_body)
        async with aiofiles.open(tex_path, "w") as f:
            await f.write(tex_content)

        # Run pdflatex
        try:
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: __import__("subprocess").run(
                        [
                            "pdflatex",
                            "-interaction=nonstopmode",
                            "-halt-on-error",
                            "-output-directory",
                            str(TMP),
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
            raise HTTPException(
                status_code=422, detail=f"LaTeX compilation failed: {errors}"
            )

        if not pdf_path.exists():
            raise HTTPException(status_code=422, detail="PDF was not generated")

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


# Mount static files LAST (catches all unmatched routes)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
