from pydantic import BaseModel, EmailStr, Field

from app.db import Base


class UserRegisterRequest(BaseModel):
    """
    Request body for user registration.
    """
    email: EmailStr = Field(
        ...,
        description="User email address",
        example="johndoe@example.com"
    )
    password: str = Field(
        ...,
        min_length=6,
        description="User password (min 6 characters)",
        example="securePassword123"
    )
    # full_name: str = Field(
    #     ...,
    #     description="Full name of the user",
    #     example="John Doe"
    # )


class UserLoginRequest(BaseModel):
    """
    Request body for user login.
    """
    email: EmailStr = Field(
        ...,
        description="User email address",
        example="johndoe@example.com"
    )
    password: str = Field(
        ...,
        description="User password",
        example="securePassword123"
    )


class UserResponse(BaseModel):
    """
    Response returned after registration or fetching user info.
    """
    id: int = Field(
        ...,
        description="Unique user identifier",
        example=1
    )
    email: EmailStr = Field(
        ...,
        description="User email address",
        example="johndoe@example.com"
    )
    full_name: str = Field(
        ...,
        description="Full name of the user",
        example="John Doe"
    )


class TokenResponse(BaseModel):
    """
    Response containing authentication token.
    """
    access_token: str = Field(
        ...,
        description="JWT access token",
        example="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
    )
    token_type: str = Field(
        default="bearer",
        description="Token type",
        example="bearer"
    )

