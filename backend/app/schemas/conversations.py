from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class ConversationCreateRequest(BaseModel):
    """
    Request to create a new conversation.
    """
    title: Optional[str] = Field(
        None,
        description="Optional conversation title",
        example="My Research Chat"
)

class ConversationResponse(BaseModel):
    """
    Conversation metadata response.
    """
    id: int = Field(
        ...,
        description="Conversation ID",
        example=42
    )
    title: Optional[str] = Field(
        None,
        description="Conversation title",
        example="My Research Chat"
    )
    created_at: datetime = Field(
        ...,
        description="Conversation creation timestamp",
        example="2024-01-01T12:00:00Z"
    )

class ConversationUpdateRequest(BaseModel):
    """
    Request to update conversation metadata.
    """
    title: Optional[str] = Field(
        None,
        description="New conversation title",
        example="Updated Title"
    )


