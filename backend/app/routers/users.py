from fastapi import APIRouter, Depends, HTTPException, Header, status
from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession
from passlib.context import CryptContext
from app.db import get_db
from app.models import User
from app.schemas.users import UserResponse, TokenResponse, UserRegisterRequest, UserLoginRequest
from app.utils.auth import get_current_user
from app.utils.auth import create_access_token, verify_access_token, get_current_user   

router = APIRouter(
    prefix="/api/v1/users",
    tags=["users"],
)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

@router.post("/register", response_model=UserResponse)
async def register_user(payload: UserRegisterRequest, db: AsyncSession = Depends(get_db)):
    user = User(
        email=payload.email,
        hashed_password=hash_password(payload.password),
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    return UserResponse(
        id=user.user_id,
        email=user.email,
    )


@router.post("/login", response_model=TokenResponse)
async def login_user(payload: UserLoginRequest, db: AsyncSession = Depends(get_db)):
    stmt = select(User).where(User.email == payload.email)
    result = await db.execute(stmt)
    user = result.scalars().first()
    token = create_access_token({"sub": user.email})
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return TokenResponse(
        access_token=token,
        token_type="bearer"
    )



@router.post("/logout")
async def logout_user(payload: dict = Depends(verify_access_token)):
    username = payload.get("sub")
    return {
        "detail": f"User {username} logged out successfully."
    }


@router.get("/me", response_model=UserResponse)
async def fetch_me(current_user=Depends(get_current_user)):
    return UserResponse(
        id=current_user.user_id,
        email=current_user.email,
        full_name=current_user.full_name
    )