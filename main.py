from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil
import os
import uuid
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from classifier import classify_text_and_files

app = FastAPI()

# Optional: Allow CORS from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000"] etc.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
# Serve the frontend HTML
@app.get("/")
def read_root():
    return FileResponse("index.html")
@app.post("/api/classify")
async def classify(
    task_name: str = Form(...),
    text_input: str = Form(""),
    files: list[UploadFile] = File(default=[])
):
    # Limit to 4 files
    if len(files) > 4:
        return JSONResponse(status_code=400, content={"error": "Maximum 4 files allowed."})

    file_paths = []
    try:
        # Save uploaded files to disk temporarily
        for file in files:
            suffix = os.path.splitext(file.filename)[-1]
            temp_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}{suffix}")
            with open(temp_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            file_paths.append(temp_path)

        # Call classifier
        df = classify_text_and_files(text_input, file_paths, task_name)
        return df.to_dict(orient="records")

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        # Clean up uploaded files
        for path in file_paths:
            if os.path.exists(path):
                os.remove(path)
