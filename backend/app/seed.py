import asyncio
from datetime import datetime, timedelta, timezone
from sqlalchemy import select
from passlib.context import CryptContext

from app.db import SessionLocal
from app.models import User, Conversation, Message


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
def fake_hash_password(pw: str) -> str:
    return pwd_context.hash(pw)


async def seed_fake_data():
    async with SessionLocal() as session:
        # Check if user existed
        res = await session.execute(select(User))
        if res.scalar():
            print("Users already exist, skip seeding.")
            return

        # Add Users
        user = User(
            email="123@123.com",
            hashed_password=fake_hash_password("123"),
            full_name="User 123 Backend",
            created_at=datetime.now(timezone.utc),
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)

        # Add Conversations
        convos = []
        for i in range(14):
            convo = Conversation(
                title=f"Demo Conversation {i+1}",
                created_at=datetime.now(timezone.utc) - timedelta(days=i+1),
                user_id=user.user_id,
            )
            session.add(convo)
            convos.append(convo)
        await session.commit()
        for c in convos:
            await session.refresh(c)

        # Add Messages to each Conversation
        for convo in convos:
            for j in range(35):
                msg = Message(
                    conversation_id=convo.conversation_id,
                    type=["user", "agent", "system"][j%3],
                    content=f"Sample message {j+1} in {convo.title}",
                    timestamp=convo.created_at + timedelta(minutes=j),
                )
                session.add(msg)
        await session.commit()
        print("Fake users, conversations, and messages seeded!")

if __name__ == "__main__":
    asyncio.run(seed_fake_data())
