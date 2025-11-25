# GeoResourceAgent
University of Alberta GeoResource Cloud Research Group (https://www.georesourcecloud.com/)

Current collaborators:
Yiding Sun;
Rahhim Khan;
Usaid Ahmed;
Walid Ben Saleh;

Major past contributors:
Shuxin Qiao;
Ziming Xu;
Jingwen Zheng;
Jeffery Bian;

## Structure

### Backend
- Poetry (Dependency management & packaging)
- Fastapi (Web framework)

### Frontend
- React (UI)
- Vite (Build tool)
- Tailwind (CSS framework)

### src
- older agent, not compatible with current agent
- agents ready
- tools in the integration have all tools

## WebSocket Logic

### 1. User Authentication and Conversation Selection

- User logs in and obtains an access token.
- User selects an existing conversation or starts a new one.

### 2. User Sends a Message

- User composes a message and clicks send.
- Frontend POSTs the message to the backend (e.g., `POST /messages` with `conversation_id`, `type="user"`, `content`).
- Backend persists the user message in the database.

### 3. Agent (LLM) Handles and Streams a Reply

- Backend detects a new user message and triggers the agent (LLM).
- Agent/LLM starts generating a reply.
- Backend streams the reply token-by-token (or chunk-by-chunk) to the frontend via WebSocket, using a dedicated endpoint (e.g., `/ws/conversations/{conversation_id}`).

### 4. Frontend Handles Streaming

- Frontend listens on the WebSocket and displays incoming tokens in a "streaming" agent message bubble.
- The UI should clearly distinguish between completed and streaming/in-progress agent messages.

### 5. Finalizing and Persisting the Agent Reply

- Once generation finishes (or is interrupted), backend saves the full agent reply as a new message in the database (type="agent").
- Backend sends a special WS event to indicate streaming is done (e.g., `{ "type": "stream_end" }`).

### 6. UI Consistency and Sync

- After receiving the "stream_end" event, the frontend should fetch the updated message list via REST (e.g., `GET /conversations/conversation_id/messages`).
- This ensures the UI and database are fully consistent (especially in case of lost tokens, errors, or reloads).

### 7. Key Points & Best Practices

- All persisted messages should live in the DB—streamed output is just for user experience.
- Only fetch or update messages from the DB on key state changes (e.g., after "stream_end").
- If the user reloads, always display the conversation by fetching from REST, not by replaying old streaming data.
- For message types, use a `type` field such as "user", "agent", or "system" in the DB.
- All authorization and conversation context must be managed on both REST and WS endpoints.

### Flow Summary (Sequence)

- User sends message via REST → saved to DB.
- Backend triggers LLM → streams agent reply via WS to frontend.
- Frontend renders streaming tokens as they arrive.
- Backend signals stream end and saves agent reply in DB.
- Frontend refreshes messages from REST to ensure full sync.


## To Do List - Backend

### [ ] 1. Project & Collaboration Standards

- [ ] Use a unified API version prefix: `/api/v1/`
- [ ] All API routes follow RESTful resource naming (plural/action-specific)
- [ ] Clear route grouping by domain (users, chats, files, ws, etc.)
- [ ] All requests/responses use JSON, fields in `snake_case`
- [ ] Every API request/response is defined using Pydantic models with proper type hints and comments
- [ ] Leverage FastAPI's auto-generated Swagger docs for frontend reference
- [ ] Notify the frontend of any field changes and keep `/docs` in sync
- [ ] Structure project directories by functional modules (routers, models, services, etc.)

---

### [ ] 2. Core API Design & Implementation

#### [ ] User Endpoints

- [x] User registration: `POST /api/v1/users/register`
- [x] User login: `POST /api/v1/users/login`
- [x] Get current user info: `GET /api/v1/users/me`

#### [ ] Conversations & Messages

- [x] List conversations for a user: `GET /api/v1/conversations/`
- [x] Create a new conversation: `POST /api/v1/conversations/`
- [x] Get a conversation's metadata (title, created_at, etc.): `GET /api/v1/conversations/{conversation_id}`
- [x] Rename or update a conversation: `PATCH /api/v1/conversations/{conversation_id}`
- [x] Delete a conversation: `DELETE /api/v1/conversations/{conversation_id}`

#### [ ] Message Management

- [x] List all messages in a conversation: `GET /api/v1/conversations/{conversation_id}/messages`
- [x] Send/add a new message: `POST /api/v1/conversations/{conversation_id}/messages`
- [x] (Optional) Get a single message: `GET /api/v1/conversations/{conversation_id}/messages/{message_id}`
#### [ ] WebSocket Real-Time Messaging

- [x] Conversation real-time chat WebSocket: `/ws/conversations/{conversation_id}`

#### [ ] File Upload/Download

- [x] File upload: `POST /api/v1/files/upload`
- [x] File download: `GET /api/v1/files/download/{file_id}`

#### [ ] System Health

- [x] Health check: `GET /api/v1/health`

---

### [ ] 3. Pydantic Data Modeling

- [x] Define every API request/response with a Pydantic BaseModel
- [x] Use `XxxRequest` for input, `XxxResponse` for output models
- [x] Use proper type annotations, `Field` for descriptions/examples
- [x] Define core business objects like `User`, `ChatMessage`, `File` as separate models

---

### [ ] 4. Project Directory & Code Structure

- [x] `app/main.py` as the entry point
- [x] `app/routers/` with routes grouped by domain (users.py, chats.py, files.py, ws.py)
- [x] `app/models/` for all Pydantic data models

---

### Database Structure Suggestions

- `users` (user_id, ...)
- `conversations` (conversation_id, user_id, title, created_at, ...)
- `messages` (message_id, conversation_id, sender, content, timestamp, ...)


## To Do List - Frontend

### [ ] 1. UI Framework

- [ ] Global theme/configuration
- [x] Basic layout: sidebar, main window area, header, etc.
- [ ] Responsive/mobile adaptation

---

### [ ] 2. API Integration & Alignment

- [ ] Set up API base config (axios/fetch, baseURL, token handling)
- [ ] Design API client modules for:
    - [ ] Authentication (login/register)
    - [ ] Conversation CRUD (list, create, rename, delete)
    - [ ] Message CRUD (fetch/send)
    - [ ] WebSocket real-time messaging
    - [ ] File upload/download (if needed)
- [ ] Use backend Swagger docs (`/docs`) as main source for API params/returns

---

### [ ] 3. Core Logic

- [ ] User authentication flow (login/register/session persistence)
- [ ] Display conversation list, create/delete/rename conversation
- [x] Show chat history (messages in a conversation)
- [ ] Real-time messaging UI with WebSocket (send/receive)
- [ ] Message input, emoji/file/image support
- [ ] Handle file upload/download
- [ ] Error & edge case handling (API errors, session expired, etc.)

---

### [ ] 4. Mock Data & Offline API Testing

- [x] Implement mock API responses (using Mock Service Worker/MSW or similar tool)
- [ ] Create mock data for all main endpoints (conversations, messages, auth, etc.)
- [x] Add config/env flag to switch between mock and real backend
- [x] Develop and test UI using mock data when backend is not ready or unavailable
- [ ] Keep mock data in sync with backend API schema (update as needed)
