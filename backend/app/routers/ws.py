from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, List

router = APIRouter(
    prefix="/ws/conversations",
    tags=["websocket"]
)

# Store active connections per conversation
active_connections: Dict[str, List[WebSocket]] = {}

@router.websocket("/{conversation_id}")
async def conversation_websocket(websocket: WebSocket, conversation_id: str):
    await websocket.accept()

    # Add to active connections
    if conversation_id not in active_connections:
        active_connections[conversation_id] = []
    active_connections[conversation_id].append(websocket)

    try:
        while True:
            data = await websocket.receive_text()

            # Broadcast to all clients in this conversation
            for connection in active_connections[conversation_id]:
                await connection.send_text(f"[{conversation_id}] {data}")

    except WebSocketDisconnect:
        # Remove disconnected websocket
        active_connections[conversation_id].remove(websocket)
        if not active_connections[conversation_id]:
            del active_connections[conversation_id]
        print(f"WebSocket disconnected for conversation {conversation_id}")
