from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, RemoveMessage, trim_messages
from openai import OpenAI
from decouple import config

def filter_messages(messages):
    # Trim till last 2 messages
    return messages[-2:]


def summarize_conversation(state, trim_length=5):
    # First, we get any existing summary
    summary = state.get("summary", "")

    # Create our summarization prompt
    if summary:
        # A summary already exists
        summary_message = (
            f"This is a summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"

    messages = []
    
    cur_length = 0

    while cur_length < trim_length:
        msg = state["messages"][cur_length]

        if isinstance(msg, HumanMessage):   
            messages.append({"role": "user", "content": msg.content})
            cur_length += 1

        if isinstance(msg, AIMessage):
            if "tool_calls" in msg.additional_kwargs:
                num_calls = len(msg.additional_kwargs["tool_calls"])

                for tool_call_idx in range(num_calls):
                    tool_calls = [{
                        "id": msg.additional_kwargs["tool_calls"][tool_call_idx]["id"],
                        "function": {
                            "name": msg.additional_kwargs["tool_calls"][tool_call_idx]["function"]["name"],
                            "arguments": msg.additional_kwargs["tool_calls"][tool_call_idx]["function"]["arguments"],
                        },
                        "type": "function"
                    }]

                    tool_msg = state["messages"][cur_length+tool_call_idx+1]

                    messages.append({"role": "assistant", "content": msg.content, "tool_calls": tool_calls})
                    messages.append({"role": "tool", "tool_call_id": tool_msg.tool_call_id, "content": tool_msg.content})

                cur_length += num_calls + 1
            else:
                messages.append({"role": "assistant", "content": msg.content})
                cur_length += 1

    messages.append({"role": "user", "content": summary_message})

    # Init a summary model
    client = OpenAI(api_key=config("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.3,
    )

    response = response.choices[0].message.content

    # Delete all but the 'trim_length' most recent messages
    # delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-trim_length+additional_index_buffer]]
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:cur_length]]
    return {"summary": response, "messages": delete_messages}


def buffer_msg(state, detect_length=10, trim_length=5):
    """
    Detect if messages over 'detect_length' conversations.
    If over, trim and summarize it into 'trim_length'.
    """
    if len(state["messages"]) >= detect_length:
        # print("\n\n\nCalled!!!!!\n\n\n")
        return summarize_conversation(state, trim_length)

    return {"summary": state["summary"], "messages": []}



def summarize_title(state, summary_len=2):
    """
    Summary title as state into checkpoint.
    Only summarize if title_summary is empty.
    """
    # print(f"\n\ninside summarize_title()\n{state}\n")
    msg = state["messages"]
    title_summary = state["title_summary"]
    
    if title_summary == "":
        return "New Chat"
    elif title_summary == "New Chat" and len(msg) >= summary_len:
        messages = []
        cur_length = 0

        while cur_length < summary_len:
            msg = state["messages"][cur_length]

            if isinstance(msg, HumanMessage):   
                messages.append({"role": "user", "content": msg.content})
                cur_length += 1

            if isinstance(msg, AIMessage):
                if "tool_calls" in msg.additional_kwargs:
                    num_calls = len(msg.additional_kwargs["tool_calls"])

                    for tool_call_idx in range(num_calls):
                        tool_calls = [{
                            "id": msg.additional_kwargs["tool_calls"][tool_call_idx]["id"],
                            "function": {
                                "name": msg.additional_kwargs["tool_calls"][tool_call_idx]["function"]["name"],
                                "arguments": msg.additional_kwargs["tool_calls"][tool_call_idx]["function"]["arguments"],
                            },
                            "type": "function"
                        }]

                        tool_msg = state["messages"][cur_length+tool_call_idx+1]

                        messages.append({"role": "assistant", "content": msg.content, "tool_calls": tool_calls})
                        messages.append({"role": "tool", "tool_call_id": tool_msg.tool_call_id, "content": tool_msg.content})

                    cur_length += num_calls + 1
                else:
                    messages.append({"role": "assistant", "content": msg.content})
                    cur_length += 1

        summary_prompt = (
            f"""
            Your task is to generate a concise and descriptive title for the conversation by reviewing the initial few round chats with the user.
            Avoid summarize conversation as greetings or interactions unless you found user explicitly chat with you for casual issues.
            Example Title: 'Processing Featured Datasheet'.
            Return output with marker <title summary> with '\n' attached, strictly obey in following format but summarized title should use user's language (e.g. English, French and etc):
            <title summary>\nProcessing Featured Datasheet
            """
        )

        messages.append({"role": "user", "content": summary_prompt})
        
        # invoke a summary model
        client = OpenAI(api_key=config("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3
        )

        response = response.choices[0].message.content
        split_parts = response.split("<title summary>\n")

        if len(split_parts) > 1:
            title_summary = split_parts[1].strip()
        else:
            client = OpenAI(api_key=config("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=msg,
                temperature=0.5
            )

            response = response.choices[0].message.content
            split_parts = response.split("<title summary>\n")

            if len(split_parts) > 1:
                title_summary = split_parts[1].strip()
            else:
                title_summary = "Failed to generate"

        return title_summary
    else:
        return title_summary