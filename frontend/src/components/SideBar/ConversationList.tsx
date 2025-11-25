import { useEffect, useState } from "react";
import { useUser } from "../../context/UserContext";
import { fetchConversationsList, fetchMessagesList } from "../../api/conversations";
import { useCurrentConversation } from "../../context/CurrentConversationContext";

function ConversationList() {
  const { token, isAuthenticated } = useUser();

  const [list, setList] = useState([]);
  const { conversationId, setConversationId, setConversationName, setMessages } = useCurrentConversation();

  function handleClick(item) {
    setConversationId(item.id);
    setConversationName(item.title);
    fetchMessagesList(token, item.id)
      .then(setMessages)
      .catch(console.error);
  } 

  useEffect(() => {
    if (!isAuthenticated) {
      setList([]);
      return;
    }
    fetchConversationsList(token)
      .then(setList)
      .catch(console.error);
  }, [token, isAuthenticated, conversationId, ]);

  return (
    <div className="p-4">
      <ul>
        {list.map(item => (
          <li
            key={item.id}
            className={`
              ${item.id === conversationId ? "bg-blue-200 font-bold" : "bg-white"}
              p-2 mb-1 rounded shadow hover:bg-blue-100 cursor-pointer
            `}
            onClick={() => handleClick(item)}
          >
            {item.title}
          </li>
        ))}
      </ul>
    </div>
  )
}

export default ConversationList;