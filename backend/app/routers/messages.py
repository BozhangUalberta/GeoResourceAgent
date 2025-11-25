from fastapi import APIRouter, Depends, HTTPException
from datetime import datetime, timezone
from typing import List

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Message as MessageModel
from app.db import get_db
from app.schemas.messages import (
    MessageCreateRequest,
    MessageResponse,
    ListMessages
)
from app.utils.auth import verify_access_token

router = APIRouter(
    prefix="/api/v1/conversations/{conversation_id}/messages",
    tags=["messages"],
)

@router.get("/", response_model=List[MessageResponse])
async def list_messages(
    conversation_id: int,
    db: AsyncSession = Depends(get_db),
    _: None = Depends(verify_access_token)
    ):
    """
    List all messages in a conversation.
    """
    stmt = (
        select(MessageModel)
        .where(MessageModel.conversation_id == conversation_id)
        .order_by(MessageModel.timestamp.asc())
    )
    result = await db.execute(stmt)
    records = result.scalars().all()
    return [
        MessageResponse(
            id=msg.message_id,
            conversation_id=msg.conversation_id,
            content=msg.content,
            created_at=msg.timestamp,
            type=msg.type
        )
        for msg in records
    ]

@router.post("/", response_model=MessageResponse)
async def create_message(
        payload: MessageCreateRequest,
        db: AsyncSession = Depends(get_db),
        _: None = Depends(verify_access_token)):
    """
    Create a new message in a conversation.
    """
    new_message = MessageModel(
        conversation_id=payload.conversation_id,
        type="user",
        content=payload.content,
        timestamp=datetime.now(timezone.utc),
    )
    db.add(new_message)
    await db.commit()
    await db.refresh(new_message)
    
    return MessageResponse(
        id=new_message.message_id,
        conversation_id=new_message.conversation_id,
        type=new_message.type,
        content=new_message.content,
        created_at=new_message.timestamp,
    )



#not used yet
@router.get("/{message_id}", response_model=MessageResponse)
def get_message(conversation_id: int, message_id: int, db: AsyncSession = Depends(get_db)):
    """
    Retrieve a single message by ID.
    """
    message = (
        db.query(MessageModel)
        .filter(
            MessageModel.conversation_id == conversation_id,
            MessageModel.message_id == message_id
        )
        .first()
    )
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")
    return MessageResponse(
        id=message.message_id,
        conversation_id=message.conversation_id,
        sender=message.sender,
        content=message.content,
        timestamp=message.timestamp,
    )
