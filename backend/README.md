# Backend

## Commands

```shell
# Init or update venv
poetry install
poetry shell

# Adding new package dependency
poetry add

# Start app
poetry run uvicorn app.main:app --reload
```


## Project Structure
```bash
backend/
│
├── app/
│   ├── main.py               # FastAPI entrance，app object and routers
│   ├── db.py                 # Database engine/session/Base/init
│   ├── models/               # All SQLAlchemy ORM models (including users and services)
│   │   ├── __init__.py
│   │   ├── user.py           # User ORM model
│   │   ├── conversation.py   # Conversation ORM model
│   │   └── ...               # Other models (like Message, RunningTable etc.)
│   ├── schemas/              # Pydantic models (verify/serialization, request/response)
│   │   ├── __init__.py
│   │   ├── user.py           # UserRegisterRequest/UserRead/UserUpdate
│   │   ├── conversation.py
│   │   └── ...
│   ├── routers/              # Routers (api groups, users, conversations, etc.）
│   │   ├── __init__.py
│   │   ├── users.py
│   │   ├── conversations.py
│   │   └── ...
│   ├── seed.py               # Seed script, fake data
│   └── ...
│
├── tests/                    # Test codes
├── .env                      # ENV variables (Not included in git)
├── .env.example              # ENV variables template
├── pyproject.toml            # dependency
└── README.md
```