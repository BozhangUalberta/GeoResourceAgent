export const API_URL = import.meta.env.VITE_API_URL;

export async function fetchConversationsList(token) {
  const res = await fetch(`${API_URL}/api/v1/conversations/`, {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  });
  if (!res.ok) throw new Error("Network error");
  return res.json();
}

export async function fetchRunningTablesList(token, conversationId) {
  const res = await fetch(`${API_URL}/api/v1/conversations/${conversationId}/running-tables/test`, {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  });
  if (!res.ok) throw new Error("Network error");
  return res.json();
}

export async function fetchMessagesList(token, conversationId) {
  const res = await fetch(`${API_URL}/api/v1/conversations/${conversationId}/messages/`, {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  });
  if (!res.ok) throw new Error("Network error");
  return res.json();
}

export async function sendMessage(token, conversationId, content) {
  const res = await fetch(`${API_URL}/api/v1/conversations/${conversationId}/messages/`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify({
      "conversation_id": conversationId,
      "content": content,
    }),
  });
  if (!res.ok) {
    let errMsg = "Failed to send message";
    try {
      const data = await res.json();
      errMsg = data.detail || errMsg;
    } catch {}
    throw new Error(errMsg);
  }
  return res.json(); // { MessageResponce }
}

export async function createConversation(token, title) {
  const res = await fetch(`${API_URL}/api/v1/conversations/`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify({
      "title": title,
    }),
  });
  if (!res.ok) {
    let errMsg = "Failed to create conversation";
    try {
      const data = await res.json();
      errMsg = data.detail || errMsg;
    } catch {}
    throw new Error(errMsg);
  }
  return res.json(); // { ConversationResponse }
}