from src.utils.build_langgraph import build_langgraph
from decouple import config
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(openai_api_key=config("OPENAI_API_KEY"), model="gpt-4o-mini", temperature=0.2)
graph = build_langgraph(llm)
config = {"configurable": {"thread_id": "1"}}

async def async_get_answer_and_docs(question: str):
    user_input = {"messages": [("user", question)]}
    
    async for event in graph.astream_events(user_input, config=config, version="v1"):
        event_type = event['event']
        if event_type == "on_chat_model_stream":
            yield {
                "event_type": event_type,
                "content": event['data']['chunk'].content
            }

        if 'image_url' in event['data']:
            yield {
                "event_type": "on_image_stream",
                "image_url": event['data']['image_url']
            }
    
    yield {
        "event_type": "done",
    }