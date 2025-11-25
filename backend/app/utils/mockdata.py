from sqlalchemy.orm import Session
from app.db import SessionLocal
from app.models import User, Conversation, Message

from datetime import datetime, timedelta
from typing import List, Dict


def get_mock_conversations(count: int = 10) -> List[Dict]:
    """
    Generate a list of mock conversation dictionaries.
    """
    base_time = datetime.utcnow()
    return [
        {
            "title": f"Mock Conversation {i + 1}",
            "created_at": base_time - timedelta(days=i)
        }
        for i in range(count)
    ]


def get_mock_messages(conversation_id: int, count: int = 10) -> List[Dict]:
    """
    Generate a list of mock message dictionaries for a given conversation.
    """
    base_time = datetime.utcnow()
    return [
        {
            "conversation_id": conversation_id,
            "content": f"Sample message {i + 1} in conversation {conversation_id}",
            "created_at": base_time - timedelta(minutes=i)
        }
        for i in range(count)
    ]


def get_mock_message(conversation_id: int, message_id: int) -> Dict:
    """
    Generate a single mock message dictionary.
    """
    return {
        "conversation_id": conversation_id,
        "content": f"Single mock message with ID {message_id}",
        "created_at": datetime.utcnow()
    }

def get_test_running_table():
    return {
        "id": "test-table",
        "name": "Mock Running Table",
        "status": "running"
    }


def mock_data():
    """
    Insert a user, multiple conversations, and multiple messages into the database.
    """
    db: Session = SessionLocal()
    try:
        # Skip if User already exists
        if db.query(User).first():
            return

        # 1. Build User
        user = User(
            email="test@example.com",
            hashed_password="fakehashedpassword",
            full_name="Test User"
        )
        db.add(user)
        db.commit()
        db.refresh(user)

        # 2. Conversations
        conversations_data = get_mock_conversations(count=15)
        conversation_objs = []
        for c in conversations_data:
            convo = Conversation(
                user_id=user.user_id,
                title=c["title"],
                created_at=c["created_at"]
            )
            db.add(convo)
            conversation_objs.append(convo)
        db.commit()

        # 3. Generate Messages for each Conversation
        for convo in conversation_objs:
            messages_data = get_mock_messages(
                conversation_id=convo.conversation_id,
                count=15
            )
            for m in messages_data:
                msg = Message(
                    conversation_id=m["conversation_id"],
                    sender=user.full_name,
                    content=m["content"],
                    timestamp=m["created_at"]
                )
                db.add(msg)
        db.commit()

        # 4. Insert a single message
        single_message_data = get_mock_message(
            conversation_id=conversation_objs[0].conversation_id,
            message_id=999
        )
        msg = Message(
            conversation_id=single_message_data["conversation_id"],
            sender=user.full_name,
            content=single_message_data["content"],
            timestamp=single_message_data["created_at"]
        )
        db.add(msg)
        db.commit()

    finally:
        db.close()
