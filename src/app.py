from fastapi import FastAPI, WebSocket, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.websockets import WebSocketState
from fastapi_users.authentication import JWTStrategy
from fastapi.encoders import jsonable_encoder

# User Managment
from fastapi import Depends
from src.users.db import User, create_db_and_tables
from src.users.schemas import UserCreate, UserRead, UserUpdate
from src.users.users import auth_backend, current_active_user, fastapi_users, get_jwt_strategy, get_user_manager

import uvicorn

from contextlib import asynccontextmanager
from pydantic import BaseModel
from dotenv import load_dotenv
import subprocess
import asyncio
import json 
import os
import re

# from src.utils.prod_agent import async_get_answer_and_docs
from src.utils.db_utils import *
from src.utils.logger_utils import get_logger, attach_ws_handler, detach_ws_handler
from src.utils.chroma_utils import init_chroma_client

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
import aiosqlite


load_dotenv()


# ----- Chroma Settings -----
import chromadb
from chromadb.config import Settings

CHROMA_PATH = "./chroma"

# Set Chroma Persistent
settings = Settings(is_persistent=True)


# ----- Agent Setting -----
import glob
import importlib.util
import os

def import_module_from_file(file_path):
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

CHECKPOINTER_DB = "sqlite"
SAVER_PATH = "checkpoints.db"

async def get_available_agents():
    files = glob.glob(os.path.join('src','agents') + "/*.py")

    for file in files:
        mod = import_module_from_file(file)
        # Check if the module has the attribute AGENT_NAME
        if hasattr(mod, 'AGENT_NAME'):
            agent_name = mod.AGENT_NAME
            app.state.AGENTS[mod.AGENT_NAME] = {
                "description": mod.AGENT_DESCRIPTION,
                "graph": mod.AGENT_GRAPH,
            }

    return



# Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start ChromaDB
    # try:
    #     print("Preparing ChromaDB server...")
    #     chroma_process = subprocess.Popen(["chroma", "run", "--path", CHROMA_PATH])
    # except Exception as e:
    #     print(f"Failed to start chroma: {e}")
    
    # try:
    #     print("Preparing ChromaDB async client...")
    #     await init_chroma_client()
    # except Exception as e:
    #     print(f"Failed to init a chroma async client: {e}")

    # Start UserDB
    await create_db_and_tables()
    
    # Start Agents
    app.state.AGENTS = {}
    await get_available_agents()
    

    print(f"Available Agents:\n{app.state.AGENTS}")

    yield

    # Stop ChromaDB
    # if chroma_process and chroma_process.poll() is None:
    #     print("Stopping Chroma process...")
    #     chroma_process.terminate()
    #     try:
    #         chroma_process.wait(timeout=5)
    #     except subprocess.TimeoutExpired:
    #         print("Chroma did not stop in time; forcing termination.")
    #         chroma_process.kill()
    #     print("Chroma process stopped.")
    # else:
    #     print("Chroma process was not running.")



   


app = FastAPI(
    title="Agent API",
    description="A not simple Agent",
    version="0.1",
    lifespan=lifespan,
)

origins = [
    "http://localhost:3000",
    f"http://{os.getenv("HOST_IP", "127.0.0.1")}:3000",
    "ws://localhost:8000",
    f"ws://{os.getenv("HOST_IP", "127.0.0.1")}:3000",
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],  # Allows all headers
)

# Serve static files (e.g., images) from the 'static' directory
app.mount("/static", StaticFiles(directory="src/static"), name="static")



# User Managment
app.include_router(
    fastapi_users.get_auth_router(auth_backend), prefix="/auth/jwt", tags=["auth"]
)
app.include_router(
    fastapi_users.get_register_router(UserRead, UserCreate),
    prefix="/auth",
    tags=["auth"],
)
app.include_router(
    fastapi_users.get_reset_password_router(),
    prefix="/auth",
    tags=["auth"],
)
app.include_router(
    fastapi_users.get_verify_router(UserRead),
    prefix="/auth",
    tags=["auth"],
)
app.include_router(
    fastapi_users.get_users_router(UserRead, UserUpdate),
    prefix="/users",
    tags=["users"],
)


@app.get("/authenticated-route")
async def authenticated_route(user: User = Depends(current_active_user)):
    return {"message": f"Hello {user.email}!"}



# LangChain Related
class Message(BaseModel):
    message: str


# Chat API
async def websocket_jwt(websocket: WebSocket, jwt: JWTStrategy, user_manager):
    token = websocket.query_params.get("token")

    if not token:
        await websocket.close(1008, "unauthorized")
        return
    
    user = await jwt.read_token(token, user_manager)
    if not user:
        await websocket.close(1008, "jwt error")
        return

    return user

async def async_get_answer_and_docs(agent_graph, question: str, userID: str, conversationID: str):
    user_input = {
        "messages": [("user", question)], 
        "userID": f"{userID}_{conversationID}", 
        "full_history": [("user", question)], 
        # "summary": "", 
        # "running_table": "", 
        # "whole_db": "", 
        # "title_summary": ""
    }

    config = {"configurable": {"thread_id": userID + "_" + conversationID}}

    async for event in agent_graph.astream_events(user_input, config=config, version="v2"):
        event_type = event['event']

        if event_type == "on_chat_model_stream":
            yield {
                "event_type": event_type,
                "content": event['data']['chunk'].content
            }
        elif event_type == "on_chat_model_end":
            yield {
                "event_type": event_type,
                "content": "",
            }
        
    yield {
        "event_type": "done",
    }

@app.websocket('/async_chat')
async def async_chat(websocket: WebSocket, jwt: JWTStrategy = Depends(get_jwt_strategy), user_manager = Depends(get_user_manager)):
    await websocket.accept()

    # JWT
    user = await websocket_jwt(websocket, jwt, user_manager)
    if user is None:
        return
    
    userID = str(user.id)
    conversationID = websocket.query_params.get("conversationID")
    agent_name = websocket.query_params.get("agentName")

    userConvID = f"{userID}_{conversationID}"

    # Get the logger for this user
    logger = get_logger(userConvID)

    # Attach a WebSocket handler
    ws_handler = attach_ws_handler(logger, websocket)

    try:
        # DB init or recap if reconnected
        db_check(db_name=userConvID)
        table_check(db_name=userConvID, table_name="running")

        while True:
            question = await websocket.receive_text()
            print(f"Received question in the backend: {question}\nfrom user: {userConvID}")
            
            async with AsyncSqliteSaver.from_conn_string(SAVER_PATH) as saver:

                agent_graph = app.state.AGENTS[agent_name]["graph"].compile(checkpointer=saver)

                # Stream the results from async_get_answer_and_docs
                async for event in async_get_answer_and_docs(agent_graph, question, userID, conversationID):
                    if event['event_type'] == 'done':
                        await websocket.close()
                        return
                    else:
                        # Stream the event data (including image URLs) back to the client
                        await websocket.send_text(json.dumps(event))

    except Exception as e:
        print(f"async_chat Error: {e}")
        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.close()

    finally:
        # Detach the WebSocket handler to clean up
        detach_ws_handler(logger, ws_handler)
        ws_handler.close()


# Agent Availability API
def reshape_agents(agents):
    agents_ui = []

    for idx, agent in enumerate(agents.items()):
        agents_ui.append(
            {
                "id": idx,
                "label": agent[0],
                "desc": agent[1]["description"],
                "selected": False,
            }
        )
        
    return agents_ui

@app.get("/agents")
async def get_agents(user: User = Depends(current_active_user)):
    try:
        return jsonable_encoder(reshape_agents(app.state.AGENTS))
    except Exception as e:
        print(f"get_agents Error: {e}")


# History API
@app.get("/history")
async def get_history_list(user: User = Depends(current_active_user)):
    query = """
    SELECT DISTINCT SUBSTR(thread_id, LENGTH(?) + 2) AS conversationID
    FROM checkpoints
    WHERE thread_id LIKE ?
    """
    pattern = f"{user.id}_%"
    conversation_ids_ts = []

    async with aiosqlite.connect(SAVER_PATH) as db:
        async with db.execute(query, (str(user.id), pattern)) as cursor:
            rows = await cursor.fetchall()
            for row in rows:
                conversation_ids_ts.append({
                    "conversation_id": row[0],
                })
    
    async with AsyncSqliteSaver.from_conn_string(SAVER_PATH) as db:
        for idx, conversation in enumerate(conversation_ids_ts):
            checkpoint_tuple = await db.aget_tuple({ "configurable" : {"thread_id": f"{user.id}_{conversation["conversation_id"]}"} })
            conversation["ts"] = checkpoint_tuple.checkpoint["ts"]
            # print(f"{idx} : {checkpoint_tuple.checkpoint["channel_values"].keys()}")
            try:
                conversation["title_summary"] = checkpoint_tuple.checkpoint["channel_values"]["title_summary"]
            except Exception as e:
                conversation["title_summary"] = "New Chat"
   
    return conversation_ids_ts

@app.get("/get_history_conversation/{conversationID}")
async def get_history_conversation(conversationID: str, user: User = Depends(current_active_user)):
    full_history_list = []
    tool_msg_pattern = r"^branch\(.*\)$"

    async with AsyncSqliteSaver.from_conn_string(SAVER_PATH) as db:
        history_list = db.alist({ "configurable" : {"thread_id": f"{user.id}_{conversationID}"} })
        async for checkpoint in history_list:
            checkpoint_keys = checkpoint.checkpoint["channel_values"].keys()

            if "__start__" not in checkpoint_keys:
                if 'start:assistant' in checkpoint_keys:
                    full_history_list.append({"sender":"You", "message": checkpoint.checkpoint["channel_values"]["messages"][-1].content})
                elif "assistant" in checkpoint_keys and not any(re.match(tool_msg_pattern, key) for key in checkpoint_keys):
                    full_history_list.append({"sender":"Agent", "message": checkpoint.checkpoint["channel_values"]["messages"][-1].content})
    return full_history_list    



# DB view update API
@app.get("/db/{conversationID}")
async def db_view(conversationID: str, user: User = Depends(current_active_user)):
    try:
        # print("getting DB request.")
        # print(get_db_json(db_name="test.db", table_name=userID,))
        results = get_db_json(db_name=f"{user.id}_{conversationID}", table_name="running", output_type="list")
        return results

    except Exception as e:
        print(f"db_view Error: {e}")

# User upload API
@app.post("/upload/{conversationID}")
async def receive_upload(file: UploadFile, conversationID: str, user: User = Depends(current_active_user)):
    try:
        user_dir = os.path.join('src','static', f"{user.id}_{conversationID}")
        # Ensure the directory exists
        os.makedirs(user_dir, exist_ok=True)

        # Save the file to static/userid_conversationid/xxx directory
        file_path = os.path.join(user_dir, file.filename)
        
        with open(file_path, 'wb+') as destination:
            destination.write(file.file.read())

        description = f"User uploads file '{file.filename}', saving location is in this row."

        db_check(db_name=f"{user.id}_{conversationID}")
        table_check(db_name=f"{user.id}_{conversationID}", table_name="running")
        rowid = store_data_sqlite3(filename=f"{user.id}_{conversationID}", table="running", data=f"{file_path}", type="userinput", description=description)

        print(f"Uploaded file '{file.filename}' received from user: {f"{user.id}_{conversationID}"}")

        return {'message': 'File uploaded successfully!', 'file_path_db_rowid': rowid}
    
    except Exception as e:
        print(f"receive_upload Error: {e}")



if __name__ == "__main__":
    host_ip = os.getenv("HOST_IP", "127.0.0.1")  # Default to localhost if not set
    host_port = int(os.getenv("HOST_PORT", "8000"))  # Default to port 8000 if not set

    uvicorn.run(app, host=host_ip, port=host_port, reload=True)