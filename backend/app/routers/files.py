from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import os
import uuid

router = APIRouter(
    prefix="/api/v1/files",
    tags=["files"]
)

UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a file and return its generated ID.
    """
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, file_id)

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    return {"file_id": file_id, "filename": file.filename}


@router.get("/download/{file_id}")
def download_file(file_id: str):
    """
    Download a previously uploaded file by its ID.
    """
    file_path = os.path.join(UPLOAD_DIR, file_id)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path=file_path,
        filename=f"{file_id}.bin",
        media_type="application/octet-stream"
    )
