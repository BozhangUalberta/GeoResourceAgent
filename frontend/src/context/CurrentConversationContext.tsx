import { createContext, useContext, useState } from "react";

type Message = {
  id: number;
  name: string;
  type: string;
  content: string;
};

type CurrentConversationContextType = {
  conversationName: string | null;
  SetConversationName: (name: string) => void;
  conversationId: number | null;
  setConversationId: (id: number) => void;
  messages: Message[];
  setMessages: (msgs: Message[]) => void;
};

const CurrentConversationContext = createContext<CurrentConversationContextType | undefined>(undefined);

export function useCurrentConversation() {
  const ctx = useContext(CurrentConversationContext);
  if (!ctx) throw new Error("useCurrentConversation must be used within Provider");
  return ctx;
}

export function CurrentConversationProvider({ children }) {
  const [conversationId, setConversationId] = useState<number | null>(null);
  const [conversationName, setConversationName] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);

  return (
    <CurrentConversationContext.Provider value={{
      conversationId,
      setConversationId,
      conversationName,
      setConversationName,
      messages,
      setMessages,
    }}>
      {children}
    </CurrentConversationContext.Provider>
  );
}
