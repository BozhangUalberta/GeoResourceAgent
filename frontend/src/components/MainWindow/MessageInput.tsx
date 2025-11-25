import { useRef, useState } from "react";

import { sendMessage } from "../../api/conversations";

import { useUser } from "../../context/UserContext";
import { useCurrentConversation } from "../../context/CurrentConversationContext";

function MessageInput() {
  const { token, isAuthenticated } = useUser();
  const { conversationId, messages, setMessages } = useCurrentConversation();

  const [value, setValue] = useState("");
  const inputRef = useRef(null);

  function handleInput(e) {
    setValue(e.target.innerText);
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  async function handleSend() {
    const trimmed = value.trim();
    if (!trimmed) return;

    const tempId = `temp-${Date.now()}`;
    const tempMessage = {
      message_id: tempId,
      conversation_id: conversationId,
      content: trimmed,
      type: "user",
      timestamp: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, tempMessage]);
    setValue("");
    if (inputRef.current) inputRef.current.innerText = "";

    try {
      const msgFromServer = await sendMessage(token, conversationId, trimmed);
      setMessages((prev) =>
        prev.map((m) =>
          m.message_id === tempId ? msgFromServer : m
        )
      );
    } catch (err) {
      setMessages((prev) =>
        prev.map((m) =>
          m.message_id === tempId ? { ...m, error: true } : m
        )
      );
    }
    
    
  }

  return (
    <div className="flex h-full w-full flex-col max-w-[48rem]">
      <div className="w-full flex flex-col p-3 justify-center border rounded-2xl border-gray-300 bg-white">
        <div className="flex item-center pb-3 w-full mx-auto">
          <div
            ref={inputRef}
            contentEditable
            spellCheck={true}
            tabIndex={0}
            className={`
              flex-1 p-2 min-h-[5em] max-h-[15rem]
              focus:outline-none focus:ring-2 focus:ring-blue-200
              rounded-lg
              transparent-scrollbar
              overflow-y-auto
              transition-all
              text-base
            `}
            placeholder="Ask anything"
            onInput={handleInput}
            onKeyDown={handleKeyDown}
            style={{
              whiteSpace: "pre-wrap",
              wordBreak: "break-word",
            }}
            aria-label="Type your message"
          />
        </div>

        <div className="flex">
          <button
            className="ml-auto mr-0 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition disabled:bg-gray-300 disabled:cursor-not-allowed"
            onClick={handleSend}
            disabled={!value.trim()}
          >
            Send
          </button>
        </div>
      </div>
    
    <div className="left-0 bottom-0 w-full z-20 text-center text-xs text-gray-500 py-2 bg-gradient-to-t from-white via-white/80 to-transparent pointer-events-none select-none px-3">
      GeoResourceAgent says hello to you. Check important info.
      <span className="underline ml-2 cursor-pointer pointer-events-auto select-auto">Placeholder</span>
    </div>
  
  </div>
    
  );
}

export default MessageInput;
