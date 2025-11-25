import { useEffect, useState, useRef } from "react";
import { useUser } from "../../context/UserContext";
import { useCurrentConversation } from "../../context/CurrentConversationContext";

import ChatMessage from "./ChatMessage";

import { fetchMessagesList } from "../../api/conversations";


function MessageList() {
  const { token, isAuthenticated } = useUser();
  const { conversationId, messages } = useCurrentConversation();

  const [list, setList] = useState([]);
  const bottomRef = useRef(null);
  

  useEffect(() => {
    if (!isAuthenticated) {
      setList([]);
      return;
    }
    fetchMessagesList(token, conversationId)
      .then(setList)
      .catch(console.error);
  }, [token, isAuthenticated, conversationId, messages]);
  
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className="h-full w-full bg-transparent max-w-[48rem] mt-20 mb-60">
      <div className="overflow-hidden">
        <ul>
          {list.map(item => (
            <ChatMessage key={item.id} item={item} />
          ))}
        </ul>
      </div>
      <div ref={bottomRef} />
    </div>
    
  );
}


export default MessageList;