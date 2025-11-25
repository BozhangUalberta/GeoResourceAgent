import { useState, useEffect } from "react";
import { useUser } from "../../context/UserContext";
import { useCurrentConversation } from "../../context/CurrentConversationContext";

import { fetchRunningTablesList } from "../../api/conversations";

function RunningTable() {
  const { token, isAuthenticated } = useUser();
  const { conversationId } = useCurrentConversation();

  const [list, setList] = useState([]);
  
  useEffect(() => {
    if (!isAuthenticated) {
      setList([]);
      return;
    }
    fetchRunningTablesList(token, conversationId)
      .then(setList)
      .catch(console.error);
  }, [token, isAuthenticated, conversationId]);

  return (
    <div className="p-4">
      <ul>
        {list.map(item => (
          <li
            key={item.id}
            className="p-2 mb-1 bg-white rounded shadow hover:bg-blue-100 cursor-pointer"
          >
            {item.name}
          </li>
        ))}
      </ul>
    </div>
  )
}

export default RunningTable;