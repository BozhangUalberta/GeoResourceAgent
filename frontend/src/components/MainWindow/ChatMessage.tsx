import React from "react";

/**
 * Props:
 * - item: { id, name, type: "user" | "agent" | "system", content }
 */
function ChatMessage({ item }) {
  if (item.type === "system") {
    return (
      <article
        className="flex justify-center my-2"
        aria-label="System message"
      >
        <div className="text-xs text-gray-400 bg-gray-50 px-3 py-1 rounded">
          {item.content}
        </div>
      </article>
    );
  }

  // type: user/agent
  const isUser = item.type === "user";

  return (
    <article
      className={
        isUser
          ? "flex justify-end my-2"
          : "flex justify-start my-2"
      }
      aria-label={isUser ? "User message" : "Agent message"}
    >
      <div
        className={[
          "px-4 py-2 rounded-2xl max-w-[70%] shadow",
          isUser
            ? "bg-blue-500 text-white rounded-br-md"
            : "bg-gray-100 text-gray-900 rounded-bl-md"
        ].join(" ")}
        title={item.name}
      >
        {/* <div className="text-xs text-gray-500 mb-1">{item.name}</div> */}
        {item.content}
      </div>
    </article>
  );
}

export default ChatMessage;
