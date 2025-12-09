// App.js
import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import { FiSend } from "react-icons/fi";
import { SiChatbot } from "react-icons/si";
import "./App.css";

// If you prefer to use CRA proxy, set API_URL to "" or "/api"
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

  // Optionally set axios defaults:
  axios.defaults.baseURL = API_URL;
  // axios.defaults.withCredentials = true; // only if you need cookies/auth

  const handleSend = async () => {
    const trimmed = input.trim();
    if (!trimmed || loading) return;

    const userMessage = { from: "user", text: trimmed };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    try {
      const res = await axios.post("/api/chat", { message: trimmed });
      // If you didn't use a proxy, this will send to http://localhost:5000/api/chat
      const data = res && res.data ? res.data : null;

      if (!data) {
        throw new Error("No response data");
      }

      const botMessage = {
        from: "bot",
        text: data.reply || "Hmm, I didn't get a reply text.",
        meta: {
          intent: data.intent || "unknown",
          confidence: typeof data.confidence === "number" ? data.confidence : 0,
        },
      };

      setMessages((prev) => [...prev, botMessage]);
    } catch (err) {
      console.error("Chat request failed:", err);

      // Prefer server-provided message if present
      let errorText = "Oops, I couldn't reach the server. Please try again in a moment.";
      if (err.response && err.response.data && err.response.data.reply) {
        errorText = err.response.data.reply;
      } else if (err.message) {
        // show the error message for debugging in dev
        errorText = `Network error: ${err.message}`;
      }

      setMessages((prev) => [
        ...prev,
        {
          from: "bot",
          text: errorText,
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
                      {Number(msg.meta.confidence).toFixed(2)}
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
