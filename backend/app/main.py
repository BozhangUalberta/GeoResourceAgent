import os

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

from app.routers import users, conversations, messages, ws, files, health, running_tables
from app.db import engine, Base
from app.seed import seed_fake_data

from contextlib import asynccontextmanager
from dotenv import load_dotenv
load_dotenv(".env")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup: create tables, seed, etc.
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    await seed_fake_data()

    yield

    # shutdown: drop tables, clean up
    if os.getenv("ENV", "development") == "development":
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await engine.dispose()
        try:
            os.remove(os.getenv("DATABASE_URL").replace("sqlite+aiosqlite:///", ""))
        except FileNotFoundError:
            pass


app = FastAPI(
    title="GeoResourceAgent API",
    version="1.0.0",
    lifespan=lifespan,
)


# CORS
origins = os.getenv("FRONTEND_ORIGINS", "").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all routers
app.include_router(users.router)
app.include_router(conversations.router)
app.include_router(messages.router)
app.include_router(ws.router)
app.include_router(files.router)
app.include_router(health.router)
app.include_router(running_tables.router)


