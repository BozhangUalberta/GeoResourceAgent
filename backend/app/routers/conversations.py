from fastapi import APIRouter, Depends, HTTPException
from typing import List
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.utils.auth import verify_access_token
from app.db import get_db
from app.schemas.conversations import (
    ConversationCreateRequest,
    ConversationResponse,
    ConversationUpdateRequest,
)
from app.models import Conversation as ConversationModel, User as UserModel
from app.utils.auth import get_current_user

router = APIRouter(
    prefix="/api/v1/conversations",
    tags=["conversations"],
)

@router.get("/", response_model=List[ConversationResponse])
async def list_conversations(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(ConversationModel))
    records = result.scalars().all()
    return [
        ConversationResponse(
            id=conv.conversation_id,
            title=conv.title,
            created_at=conv.created_at
        )
        for conv in records
    ]

@router.post("/", response_model=ConversationResponse)
async def create_conversation(payload: ConversationCreateRequest, db: AsyncSession = Depends(get_db), current_user: UserModel = Depends(get_current_user)):
    print(payload)
    new_conversation = ConversationModel(
        title=payload.title or "Untitled",
        user_id=current_user.user_id,
        created_at=datetime.now(timezone.utc),
    )
    db.add(new_conversation)
    await db.commit()
    await db.refresh(new_conversation)

    return ConversationResponse(
        id=new_conversation.conversation_id,
        title=new_conversation.title,
        created_at=new_conversation.created_at,
    )   

#not used
@router.get("/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(conversation_id: int, db: AsyncSession = Depends(get_db)):
    stmt = select(ConversationModel).where(ConversationModel.conversation_id == conversation_id)
    result = await db.execute(stmt)
    conversation = result.scalar_one_or_none()
    if not conversation:
        return {"detail": "Conversation not found"}
    return ConversationResponse(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
    )

@router.patch("/{conversation_id}", response_model=ConversationResponse)
def update_conversation(conversation_id: int, payload: ConversationUpdateRequest, db: AsyncSession = Depends(get_db)):
    conversation = db.query(ConversationModel).filter(ConversationModel.id == conversation_id).first()
    if not conversation:
        return {"detail": "Conversation not found"}

    if payload.title is not None:
        conversation.title = payload.title
    db.commit()
    db.refresh(conversation)

    return ConversationResponse(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
    )

@router.delete("/{conversation_id}")
def delete_conversation(conversation_id: int, db: AsyncSession = Depends(get_db)):
    conversation = db.query(ConversationModel).filter(ConversationModel.id == conversation_id).first()
    if not conversation:
        return {"detail": "Conversation not found"}
    db.delete(conversation)
    db.commit()
    return {"detail": f"Conversation {conversation_id} deleted."}
