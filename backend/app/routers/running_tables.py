from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from app.utils.mockdata import get_test_running_table

router = APIRouter(
    prefix="/api/v1/conversations",
    tags=["running-tables"],
)

class RunningTable(BaseModel):
    id: str
    name: str
    status: str

@router.get("/{conversation_id}/running-tables/{table_id}", response_model=List[RunningTable])
def get_running_table(conversation_id: str, table_id: str):
    # if conversation_id == "test" and table_id == "test":
    #     return get_test_running_table()
    # You can also return dummy data for any other ID:
    return [
        {"id": table_id, "name": f"Running Table {table_id}", "status": "running"},
        {"id": table_id+"1", "name": f"Running Table {table_id}", "status": "running"},
        {"id": table_id+"2", "name": f"Running Table {table_id}", "status": "running"},
    ]
