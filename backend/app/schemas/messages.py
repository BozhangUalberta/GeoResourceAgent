from pydantic import BaseModel, Field

from datetime import datetime
from app.db import Base


class MessageCreateRequest(BaseModel):
    """
    Request to create a new message.
    """
    conversation_id: int = Field(
        ...,
        description="Conversation ID",
        example=5
    )
    content: str = Field(
        ...,
        description="Message text content",
        example="Hello everyone!"
    )

class MessageResponse(BaseModel):
    """
    Response representing a message.
    """
    id: int = Field(
        ...,
        description="Message ID",
        example=101
    )
    conversation_id: int = Field(
        ...,
        description="Conversation ID",
        example=5
    )
    content: str = Field(
        ...,
        description="Message content",
        example="Hello everyone!"
    )
    created_at: datetime = Field(
        ...,
        description="Timestamp when the message was created",
        example="2024-01-01T12:00:00Z"
    )
    type: str = Field(
        ...,
        description="Sender type",
        example="agent"
    )

    #type: Literal["user", "agent", "system"]

class ListMessages(BaseModel): 
    conversationId: int = Field(
        ...,
        description="Conversation ID",
        example=5
    )


