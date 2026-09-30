import { useState, useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import "./App.css";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

function App() {
  // Upload & Document State
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  const [uploadedDoc, setUploadedDoc] = useState(null);

  // Chat & Conversation State
  const [conversationId, setConversationId] = useState(null);
  const [chatQuestion, setChatQuestion] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const [chatError, setChatError] = useState(null);
  const [messages, setMessages] = useState([]);

  // Backend Health State
  const [backendHealthy, setBackendHealthy] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  // Toast notification state
  const [toastVisible, setToastVisible] = useState(false);
  const toastTimerRef = useRef(null);

  // Auto-scroll ref
  const messagesEndRef = useRef(null);
  const chatInputRef = useRef(null);

  // Check Backend Health on Mount
  useEffect(() => {
    const checkBackend = async () => {
      try {
        const res = await fetch(`${API_URL}/`);
        if (res.ok) {
          setBackendHealthy(true);
        } else {
          setBackendHealthy(false);
        }
      } catch (e) {
        setBackendHealthy(false);
      }
    };
    checkBackend();
  }, []);

  // Auto-scroll to bottom of chat
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, chatLoading]);

  // Auto-resize chat textarea based on content (capped at 160px)
  useEffect(() => {
    const textarea = chatInputRef.current;
    if (!textarea) return;
    textarea.style.height = "auto";
    const newHeight = Math.min(textarea.scrollHeight, 160);
    textarea.style.height = `${Math.max(newHeight, 48)}px`;
  }, [chatQuestion]);

  // Drag & Drop State
  const [isDragging, setIsDragging] = useState(false);

  // Common File Validation & Selection Handler
  const processSelectedFile = (selected) => {
    if (!selected) return;
    if (!selected.name.toLowerCase().endsWith(".pdf")) {
      setUploadError("Please select a valid .pdf file.");
      setFile(null);
      return;
    }
    setFile(selected);
    setUploadError(null);
  };

  // Handle File Input Selection
  const handleFileChange = (e) => {
    const selected = e.target.files && e.target.files[0];
    processSelectedFile(selected);
  };

  // Drag Event Handlers
  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (!uploading && !isDragging) {
      setIsDragging(true);
    }
  };

  const handleDragEnter = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (!uploading) {
      setIsDragging(true);
    }
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.currentTarget.contains(e.relatedTarget)) {
      return;
    }
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    if (uploading) return;

    if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const droppedFile = e.dataTransfer.files[0];
      processSelectedFile(droppedFile);
    }
  };

  // Handle PDF Upload
  const handleUpload = async (e) => {
    e.preventDefault();
    if (!file) {
      setUploadError("Please choose a PDF file to upload.");
      return;
    }

    setUploading(true);
    setUploadError(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_URL}/api/documents/upload`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        const errorMsg = data.detail || "Failed to upload and process PDF.";
        throw new Error(typeof errorMsg === "string" ? errorMsg : JSON.stringify(errorMsg));
      }

      setUploadedDoc(data);
      setFile(null);
      // Reset file input element if needed
      const fileInput = document.getElementById("pdf-file-input");
      if (fileInput) fileInput.value = "";
    } catch (err) {
      setUploadError(err.message || "An error occurred during upload. Please ensure the backend is running.");
    } finally {
      setUploading(false);
    }
  };

  // Handle Chat Submission
  const handleChat = async (e) => {
    e.preventDefault();
    const q = chatQuestion.trim();

    if (!q) {
      setChatError("Please enter a question.");
      return;
    }

    setChatLoading(true);
    setChatError(null);

    // Optimistic UI: Add user message immediately
    const userMsg = { role: "user", content: q, timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) };
    const updatedMessages = [...messages, userMsg];
    setMessages(updatedMessages);
    setChatQuestion("");

    try {
      const response = await fetch(`${API_URL}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          conversation_id: conversationId,
          question: q,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        const errorMsg = data.detail || "Failed to generate answer.";
        throw new Error(typeof errorMsg === "string" ? errorMsg : JSON.stringify(errorMsg));
      }

      // Update active conversation ID for memory continuity
      setConversationId(data.conversation_id);

      // Append assistant answer with sources
      const assistantMsg = {
        role: "assistant",
        content: data.answer,
        sources: data.sources || [],
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };

      setMessages([...updatedMessages, assistantMsg]);
    } catch (err) {
      setChatError(err.message || "Failed to connect to the chatbot server. Please check your backend.");
    } finally {
      setChatLoading(false);
    }
  };

  // Keyboard shortcut handler for textarea (Enter to submit, Shift+Enter for newline)
  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!chatLoading && chatQuestion.trim()) {
        handleChat(e);
      }
    }
  };

  // Handle New Conversation (Reset Session)
  const handleNewConversation = async () => {
    const hadMessages = messages.length > 0;
    if (conversationId) {
      try {
        await fetch(`${API_URL}/api/chat/${conversationId}`, { method: "DELETE" });
      } catch (err) {
        // Silent reset even if network fails
      }
    }
    setConversationId(null);
    setMessages([]);
    setChatError(null);
    setChatQuestion("");
    if (chatInputRef.current) {
      chatInputRef.current.focus();
    }
    // Show toast only when there was an actual conversation to clear
    if (hadMessages) {
      if (toastTimerRef.current) clearTimeout(toastTimerRef.current);
      setToastVisible(true);
      toastTimerRef.current = setTimeout(() => setToastVisible(false), 3000);
    }
  };

  // Sample prompt helper
  const handleSamplePrompt = (promptText) => {
    setChatQuestion(promptText);
    if (chatInputRef.current) {
      chatInputRef.current.focus();
    }
  };

  return (
    <div className="layout-root">
      {/* ── Clear-Chat Toast ── */}
      {toastVisible && (
        <div
          className="clear-toast"
          role="status"
          aria-live="polite"
          aria-atomic="true"
        >
          <span className="clear-toast-icon" aria-hidden="true">✓</span>
          Conversation cleared
        </div>
      )}

      {/* ── Mobile Sidebar Toggle Overlay ── */}
      {sidebarOpen && (
        <div className="sidebar-backdrop" onClick={() => setSidebarOpen(false)} />
      )}

      {/* ── Sidebar ─────────────────────────────────────────────────── */}
      <aside className={`app-sidebar ${sidebarOpen ? "open" : ""}`}>
        <div className="sidebar-brand">
          <div className="brand-icon">📄</div>
          <div className="brand-info">
            <h2>PDF Chatbot</h2>
            <span className="brand-subtitle">RAG + Multi-Turn AI</span>
          </div>
        </div>

        {/* Backend Status Indicator */}
        <div className="backend-status-pill">
          <span className={`status-dot ${backendHealthy ? "online" : "offline"}`} />
          <span className="status-label">
            {backendHealthy === null
              ? "Checking Server..."
              : backendHealthy
              ? "Backend Online"
              : "Backend Disconnected"}
          </span>
        </div>

        {/* PDF Upload Section */}
        <div className="sidebar-section upload-section">
          <h3 className="sidebar-heading">Upload Document</h3>
          <form onSubmit={handleUpload} className="sidebar-upload-form">
            <div className="custom-file-dropzone">
              <input
                type="file"
                id="pdf-file-input"
                accept=".pdf,application/pdf"
                onChange={handleFileChange}
                className="hidden-file-input"
              />
              <label
                htmlFor="pdf-file-input"
                className={`file-drop-label ${isDragging ? "drag-active" : ""}`}
                onDragOver={handleDragOver}
                onDragEnter={handleDragEnter}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
              >
                <span className="file-icon">{isDragging ? "📥" : "📁"}</span>
                <span className="file-text">
                  {isDragging
                    ? "Drop PDF file here..."
                    : file
                    ? file.name
                    : "Choose or drag PDF..."}
                </span>
                <span className="file-hint">Max 50MB (.pdf)</span>
              </label>
            </div>

            <button
              type="submit"
              disabled={!file || uploading}
              className="btn btn-primary btn-block"
            >
              {uploading ? (
                <>
                  <span className="spinner" /> Uploading & Processing...
                </>
              ) : (
                "Upload & Index PDF"
              )}
            </button>
          </form>

          {uploadError && (
            <div className="alert alert-error">
              <strong>Error:</strong> {uploadError}
            </div>
          )}

          {uploading && (
            <div className="alert alert-info">
              Parsing PDF → Chunking → Generating Embeddings → Storing in Chroma Cloud...
            </div>
          )}
        </div>

        {/* Uploaded Document Info Card */}
        {uploadedDoc && (
          <div className="sidebar-section doc-info-card">
            <div className="doc-info-header">
              <span className="doc-icon">📑</span>
              <div className="doc-title-box">
                <h4 className="doc-filename" title={uploadedDoc.filename}>
                  {uploadedDoc.filename}
                </h4>
                <span className="badge badge-success">Indexed & Ready</span>
              </div>
            </div>

            <div className="doc-stats-grid">
              <div className="stat-box">
                <span className="stat-number">{uploadedDoc.total_pages}</span>
                <span className="stat-name">Pages</span>
              </div>
              <div className="stat-box">
                <span className="stat-number">{uploadedDoc.total_chunks}</span>
                <span className="stat-name">Chunks</span>
              </div>
              <div className="stat-box">
                <span className="stat-number">{uploadedDoc.vectors_stored}</span>
                <span className="stat-name">Vectors</span>
              </div>
              <div className="stat-box">
                <span className="stat-number">{uploadedDoc.embedding_dimension}d</span>
                <span className="stat-name">Dimension</span>
              </div>
            </div>

            <div className="doc-meta-footer">
              <small>Vector Store: <strong>{uploadedDoc.vector_store_status}</strong></small>
              <small>Model: <strong>{uploadedDoc.embedding_model}</strong></small>
            </div>
          </div>
        )}

        {/* Conversation Action */}
        <div className="sidebar-footer">
          <button
            type="button"
            onClick={handleNewConversation}
            className="btn btn-secondary btn-block"
            title="Start a fresh conversation memory session"
          >
            ➕ New Conversation
          </button>
          {conversationId && (
            <div className="session-id-tag">
              <small>Session: {conversationId.slice(0, 8)}...</small>
            </div>
          )}
        </div>
      </aside>

      {/* ── Main Chat Area ───────────────────────────────────────────── */}
      <main className="chat-main">
        {/* Backend Offline Banner */}
        {backendHealthy === false && (
          <div className="backend-offline-banner" role="alert" aria-live="assertive">
            <span className="offline-banner-icon" aria-hidden="true">⚠️</span>
            <span>
              <strong>Backend unavailable.</strong> The server at{" "}
              <code>{API_URL}</code> is not responding. Please start the backend and refresh.
            </span>
          </div>
        )}
        {/* Top Chat Header */}
        <header className="chat-header">
          <div className="header-left">
            <button
              className="mobile-menu-btn"
              onClick={() => setSidebarOpen(!sidebarOpen)}
              aria-label="Toggle Sidebar"
            >
              ☰
            </button>
            <div className="chat-title-group">
              <h1>DocuChat AI</h1>
              <span className="chat-status-sub">
                {uploadedDoc
                  ? `Active Document: ${uploadedDoc.filename}`
                  : "No document uploaded yet"}
              </span>
            </div>
          </div>

          <div className="header-right">
            {conversationId && (
              <span className="active-session-badge">
                Session Active
              </span>
            )}
            <button
              type="button"
              onClick={handleNewConversation}
              className="btn-text-action"
              title="Reset conversation memory"
            >
              Clear Chat
            </button>
          </div>
        </header>

        {/* Chat Messages Container */}
        <div className="chat-messages-scroll">
          {messages.length === 0 ? (
            !uploadedDoc ? (
              <div className="chat-empty-state pre-upload-state">
                <div className="empty-icon-wrap" aria-hidden="true">📄</div>
                <h2>Upload a PDF to get started</h2>
                <p>
                  Upload a PDF document using the left sidebar to start asking questions.
                  Your answers will be strictly grounded in the document you upload with exact page citations.
                </p>

                <div className="pre-upload-guide-box">
                  <div className="guide-item">
                    <span className="guide-icon" aria-hidden="true">⚡</span>
                    <span className="guide-text">
                      <strong>Smart Ingestion:</strong> Fast text parsing and semantic chunking
                    </span>
                  </div>
                  <div className="guide-item">
                    <span className="guide-icon" aria-hidden="true">🔍</span>
                    <span className="guide-text">
                      <strong>Chroma Cloud Vectors:</strong> High-precision semantic similarity retrieval
                    </span>
                  </div>
                  <div className="guide-item">
                    <span className="guide-icon" aria-hidden="true">💬</span>
                    <span className="guide-text">
                      <strong>Grounded Answers:</strong> Qwen AI responses with direct page citations
                    </span>
                  </div>
                </div>
              </div>
            ) : (
              <div className="chat-empty-state post-upload-state">
                <div className="empty-icon-wrap" aria-hidden="true">✨</div>
                <h2>Your document is ready</h2>
                <p>
                  <strong>{uploadedDoc.filename}</strong> ({uploadedDoc.total_pages} {uploadedDoc.total_pages === 1 ? "page" : "pages"}) is indexed. Ask any question below or choose a prompt:
                </p>

                <div className="sample-prompts-container">
                  <span className="prompts-title">Try asking:</span>
                  <div className="prompt-chips">
                    <button
                      type="button"
                      className="chip"
                      onClick={() => handleSamplePrompt("What is the main summary of this document?")}
                    >
                      💡 What is the main summary of this document?
                    </button>
                    <button
                      type="button"
                      className="chip"
                      onClick={() => handleSamplePrompt("What are the key findings or conclusions?")}
                    >
                      🔍 What are the key findings or conclusions?
                    </button>
                    <button
                      type="button"
                      className="chip"
                      onClick={() => handleSamplePrompt("Explain the core concept in simple terms.")}
                    >
                      📖 Explain the core concept in simple terms.
                    </button>
                    <button
                      type="button"
                      className="chip"
                      onClick={() => handleSamplePrompt("What are the most important takeaways?")}
                    >
                      📌 What are the most important takeaways?
                    </button>
                  </div>
                </div>
              </div>
            )
          ) : (
            <div className="messages-list">
              {messages.map((msg, index) => (
                <div
                  key={index}
                  className={`message-row ${msg.role === "user" ? "user-row" : "assistant-row"}`}
                >
                  <div className="message-avatar">
                    {msg.role === "user" ? "👤" : "✨"}
                  </div>

                  <div className="message-bubble-content">
                    <div className="message-sender-bar">
                      <span className="sender-name">
                        {msg.role === "user" ? "You" : "DocuChat AI"}
                      </span>
                      {msg.timestamp && (
                        <span className="message-time">{msg.timestamp}</span>
                      )}
                    </div>

                    <div className="message-text">
                      {msg.role === "user" ? (
                        msg.content.split("\n").map((para, pIdx) => (
                          <p key={pIdx}>{para}</p>
                        ))
                      ) : (
                        <div className="markdown-content">
                          <ReactMarkdown>{msg.content}</ReactMarkdown>
                        </div>
                      )}
                    </div>

                    {/* Sources & Citations */}
                    {msg.role === "assistant" && msg.sources && msg.sources.length > 0 && (
                      <div className="message-sources-box" aria-label="Sources and citations">
                        <div className="sources-header">
                          <span className="sources-icon" aria-hidden="true">📚</span>
                          <span>Sources & Citations:</span>
                        </div>
                        <div className="source-cards-list">
                          {msg.sources.map((src, sIdx) => (
                            <div key={sIdx} className="source-card">
                              <div className="source-card-title">
                                <span className="source-filename" title={src.source || "Document"}>
                                  <span className="source-file-icon" aria-hidden="true">📄</span>
                                  <span className="source-file-name-text">{src.source || "Document"}</span>
                                </span>
                                <span className="source-page-tag">
                                  Page {src.page_number != null && src.page_number >= 0 ? src.page_number : 1}
                                </span>
                              </div>
                              {src.text && typeof src.text === "string" && src.text.trim() && (
                                <div className="source-snippet" title={src.text.trim()}>
                                  "{src.text.trim()}"
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ))}

              {/* Loading Indicator */}
              {chatLoading && (
                <div className="message-row assistant-row">
                  <div className="message-avatar">✨</div>
                  <div className="message-bubble-content loading-bubble">
                    <div className="message-sender-bar">
                      <span className="sender-name">DocuChat AI</span>
                    </div>
                    <div className="typing-indicator">
                      <span className="dot" />
                      <span className="dot" />
                      <span className="dot" />
                      <span className="typing-label">AI is thinking & retrieving context...</span>
                    </div>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Bottom Input Area */}
        <div className="chat-input-area">
          {chatError && (
            <div className="chat-error-banner">
              <span>⚠️ {chatError}</span>
              <button onClick={() => setChatError(null)} className="error-close-btn">×</button>
            </div>
          )}

          <form onSubmit={handleChat} className="chat-input-form">
            <textarea
              ref={chatInputRef}
              id="chat-input-field"
              rows={1}
              value={chatQuestion}
              onChange={(e) => setChatQuestion(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={
                uploadedDoc
                  ? "Ask anything about the PDF (e.g. What is supervised learning?)..."
                  : "Upload a PDF or ask a question..."
              }
              className="chat-text-input"
              disabled={chatLoading}
              aria-label="Ask a question"
            />
            <button
              type="submit"
              disabled={chatLoading || !chatQuestion.trim()}
              className="btn btn-send"
              aria-label="Send message"
            >
              {chatLoading ? (
                <span className="spinner-small" />
              ) : (
                <span>Send ➔</span>
              )}
            </button>
          </form>
          <div className="chat-input-footer">
            <small>Grounded RAG answers powered by Chroma Cloud & Hugging Face LLM with Conversation Memory.</small>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
