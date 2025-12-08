import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import { FiSend } from "react-icons/fi";
import { SiChatbot } from "react-icons/si";
import "./App.css";

const API_URL = process.env.REACT_APP_API_URL || "http://localhost:5000";

function App() {
  const [messages, setMessages] = useState([
    { from: "bot", text: "Hey! I'm your AI chatbot. Ask me anything 😊" },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);

  useEffect(() => {
    if (bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, loading]);

  const handleSend = async () => {
    const trimmed = input.trim();
    if (!trimmed || loading) return;

    const userMessage = { from: "user", text: trimmed };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    try {
      const res = await axios.post(`${API_URL}/api/chat`, {
        message: trimmed,
      });

      const botMessage = {
        from: "bot",
        text: res.data.reply,
        meta: {
          intent: res.data.intent,
          confidence: res.data.confidence,
        },
      };

      setMessages((prev) => [...prev, botMessage]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          from: "bot",
          text: "Oops, I couldn't reach the server. Please try again in a moment.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="app-root">
      <div className="chat-shell">
        <header className="chat-header">
          <div className="chat-brand">
            <div className="chat-logo">
              <SiChatbot size={26} />
            </div>
            <div>
              <div className="chat-title">Rizzler chatbot</div>
              <div className="chat-subtitle">Your Rizzler ML-powered assistant</div>
            </div>
          </div>
          <div className="status-pill">
            <span className="status-dot" />
            Online
          </div>
        </header>

        <main className="chat-main">
          <div className="chat-messages">
            {messages.map((msg, idx) => (
              <div
                key={idx}
                className={
                  msg.from === "user" ? "msg-row msg-row-user" : "msg-row msg-row-bot"
                }
              >
                <div
                  className={
                    msg.from === "user"
                      ? "msg-bubble msg-bubble-user"
                      : "msg-bubble msg-bubble-bot"
                  }
                >
                  <div className="msg-text">{msg.text}</div>
                  {msg.meta && (
                    <div className="msg-meta">
                      intent: <b>{msg.meta.intent}</b> • conf:{" "}
                      {msg.meta.confidence.toFixed(2)}
                    </div>
                  )}
                </div>
              </div>
            ))}
            {loading && (
              <div className="msg-row msg-row-bot">
                <div className="msg-bubble msg-bubble-bot typing-bubble">
                  <span className="dot" />
                  <span className="dot" />
                  <span className="dot" />
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>
        </main>

        <footer className="chat-footer">
          <textarea
            className="chat-input"
            placeholder="Type your message and press Enter..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={1}
          />
          <button
            className="send-button"
            onClick={handleSend}
            disabled={!input.trim() || loading}
          >
            <FiSend size={18} />
          </button>
        </footer>
      </div>
    </div>
  );
}

export default App;