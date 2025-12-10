import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import { Textarea } from "./ui/textarea";
import { Button } from "./ui/button";
import { Send } from "lucide-react";

const BASE_API_URL = ""; // Empty because Vite proxy handles routing

export default function ChatSupport() {
  const [messages, setMessages] = useState([
    { 
      role: "bot", 
      text: "👋 Hi! I'm your MediVision AI assistant!\n\n📖 User Manual:\n• How to use the platform\n• Features & benefits\n• Step-by-step guides\n\n📞 Support:\n• Contact information\n• Report issues\n• Get help\n\n💡 For medical AI assistant, please log in and visit the Chat page!\n\nHow can I help?" 
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userText = input.trim();
    setMessages((prev) => [...prev, { role: "user", text: userText }]);
    setInput("");
    setLoading(true);

    try {
      // Always use public-chat endpoint (no authentication)
      const res = await axios.post(
        `${BASE_API_URL}/public-chat/`, 
        { message: userText },
        {
          headers: {
            'Content-Type': 'application/json'
          }
        }
      );

      const botReply = res?.data?.response || "Sorry, I didn't get that. Can you try asking differently?";
      setMessages((prev) => [...prev, { role: "bot", text: botReply }]);
    } catch (err) {
      console.error("Chat error:", err);
      const errorMsg = "I'm having trouble connecting right now. 😔\n\nYou can:\n• Try asking again\n• Email us: contact@medivision.ai\n• Call: +91 (080) 1234-5678";
      setMessages((prev) => [
        ...prev,
        { role: "bot", text: errorMsg },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed bottom-20 right-4 z-50 w-[95%] sm:w-96 h-[400px] rounded-lg border bg-background shadow-xl flex flex-col">
      {/* Header */}
      <div className="p-4 border-b bg-primary/5">
        <h4 className="text-lg font-semibold">📖 Help & Support</h4>
        <p className="text-xs text-muted-foreground mt-1">User Manual & Contact</p>
      </div>

      {/* Messages (Scrollable) */}
      <div className="flex-1 overflow-y-auto px-3 py-2 space-y-2">
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`w-fit max-w-[80%] rounded-lg px-4 py-2 text-sm break-words whitespace-pre-line ${msg.role === "user"
                ? "ml-auto bg-primary text-primary-foreground"
                : "mr-auto bg-muted text-muted-foreground"
              }`}
          >
            {msg.text}
          </div>
        ))}
        {loading && (
          <div className="mr-auto rounded-lg bg-muted px-4 py-2 text-sm text-muted-foreground">
            Typing...
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <form onSubmit={handleSubmit} className="flex items-end gap-2 border-t p-3">
        <Textarea
          placeholder="Ask how to use, report issues, get help..."
          className="flex-grow resize-none h-10"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            // Submit on Enter (but allow Shift+Enter for new line)
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              handleSubmit(e);
            }
          }}
          rows={1}
          disabled={loading}
        />
        <Button type="submit" size="icon" disabled={loading || !input.trim()}>
          <Send className="h-4 w-4" />
          <span className="sr-only">Send</span>
        </Button>
      </form>
    </div>
  );
}
