import { useState, useEffect, useRef } from "react";
import "./App.css";

import Sidebar from "./components/Sidebar";
import Header from "./components/Header";
import MessageItem from "./components/MessageItem";
import Composer from "./components/Composer";
import SourceDrawer from "./components/SourceDrawer";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

function App() {
  // Upload & Document State
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  const [uploadedDoc, setUploadedDoc] = useState(null);
  const [isDragging, setIsDragging] = useState(false);

  // Chat & Conversation State
  const [conversationId, setConversationId] = useState(null);
  const [chatQuestion, setChatQuestion] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const [chatError, setChatError] = useState(null);
  const [messages, setMessages] = useState([]);

  // Source Drawer & Preview State
  const [activeSource, setActiveSource] = useState(null);
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);

  // Backend Health & UI Navigation State
  const [backendHealthy, setBackendHealthy] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  // Toast notification state
  const [toastVisible, setToastVisible] = useState(false);
  const toastTimerRef = useRef(null);

  // Auto-scroll ref & textarea ref
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

  // File Validation & Selection Handler
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
    if (e) e.preventDefault();
    const q = chatQuestion.trim();

    if (!q) {
      setChatError("Please enter a question.");
      return;
    }

    setChatLoading(true);
    setChatError(null);

    // Optimistic UI: Add user message immediately
    const userMsg = {
      role: "user",
      content: q,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };
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
    setActiveSource(null);
    setIsDrawerOpen(false);
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
        <div className="clear-toast" role="status">
          <span className="clear-toast-icon">✓</span>
          Conversation cleared
        </div>
      )}

      {/* ── Left Sidebar ───────────────────────────────────────────── */}
      <Sidebar
        sidebarOpen={sidebarOpen}
        setSidebarOpen={setSidebarOpen}
        backendHealthy={backendHealthy}
        file={file}
        uploading={uploading}
        uploadError={uploadError}
        uploadedDoc={uploadedDoc}
        isDragging={isDragging}
        handleFileChange={handleFileChange}
        handleDragOver={handleDragOver}
        handleDragEnter={handleDragEnter}
        handleDragLeave={handleDragLeave}
        handleDrop={handleDrop}
        handleUpload={handleUpload}
        handleNewConversation={handleNewConversation}
        conversationId={conversationId}
      />

      {/* ── Main Chat Area ───────────────────────────────────────────── */}
      <main className="chat-main-wrapper">
        {/* Backend Offline Banner */}
        {backendHealthy === false && (
          <div className="offline-banner" role="alert">
            <span className="material-symbols-outlined text-[20px]">warning</span>
            <span>
              <strong>Backend unavailable.</strong> The server at <code>{API_URL}</code> is not responding. Please start the backend.
            </span>
          </div>
        )}

        {/* Top Header */}
        <Header
          sidebarOpen={sidebarOpen}
          setSidebarOpen={setSidebarOpen}
          uploadedDoc={uploadedDoc}
          conversationId={conversationId}
          handleNewConversation={handleNewConversation}
          onToggleDrawer={() => setIsDrawerOpen(!isDrawerOpen)}
        />

        {/* Chat Messages Container */}
        <div className="chat-scroll-canvas">
          <div className="chat-center-container">
            {/* Topic Session Header */}
            <div className="topic-session-header">
              <div className="topic-header-left">
                <span className="doc-context-label">Document Context</span>
                <span className="active-doc-badge">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 inline-block"></span>
                  {uploadedDoc ? `${uploadedDoc.filename} Active` : "No Document Active"}
                </span>
              </div>
              <button
                className="view-drawer-btn"
                type="button"
                onClick={() => setIsDrawerOpen(!isDrawerOpen)}
              >
                <span className="material-symbols-outlined text-[16px]">auto_stories</span>
                <span>View Source Drawer</span>
              </button>
            </div>

            {/* Chat Content */}
            {messages.length === 0 ? (
              <div className="chat-empty-canvas">
                <div className="empty-icon-circle">
                  <span className="material-symbols-outlined text-[28px]">smart_toy</span>
                </div>
                <h2>DocuChat AI Workspace</h2>
                <p>
                  {uploadedDoc
                    ? `Ready to query ${uploadedDoc.filename} (${uploadedDoc.total_pages} pages indexed). Ask any question or pick a sample prompt.`
                    : "Upload a PDF document to start asking questions. Your answers will be strictly grounded with exact page citations."}
                </p>

                {uploadedDoc ? (
                  <div className="sample-prompts-wrap">
                    <span className="text-xs text-gray-400 font-semibold uppercase tracking-wider">Suggested Queries</span>
                    <div className="sample-chips-grid">
                      <button
                        type="button"
                        className="sample-chip"
                        onClick={() => handleSamplePrompt("What is the main summary of this document?")}
                      >
                        💡 What is the main summary of this document?
                      </button>
                      <button
                        type="button"
                        className="sample-chip"
                        onClick={() => handleSamplePrompt("What are the key findings or conclusions?")}
                      >
                        🔍 What are the key findings or conclusions?
                      </button>
                      <button
                        type="button"
                        className="sample-chip"
                        onClick={() => handleSamplePrompt("Explain the core concept in simple terms.")}
                      >
                        📖 Explain the core concept in simple terms.
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="feature-cards-grid">
                    <div className="feature-card">
                      <div className="feature-card-icon">
                        <span className="material-symbols-outlined">bolt</span>
                      </div>
                      <div className="feature-card-title">Fast Ingestion</div>
                      <div className="feature-card-desc">Page text parsing & metadata-aware chunking</div>
                    </div>
                    <div className="feature-card">
                      <div className="feature-card-icon">
                        <span className="material-symbols-outlined">database</span>
                      </div>
                      <div className="feature-card-title">Chroma Cloud</div>
                      <div className="feature-card-desc">Hosted MiniLM 384-d semantic vector retrieval</div>
                    </div>
                    <div className="feature-card">
                      <div className="feature-card-icon">
                        <span className="material-symbols-outlined">verified</span>
                      </div>
                      <div className="feature-card-title">Grounded Answers</div>
                      <div className="feature-card-desc">LLM response synthesis with verified citations</div>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="dialogue-thread">
                {messages.map((msg, index) => (
                  <MessageItem
                    key={index}
                    msg={msg}
                    onSelectSource={(source) => {
                      setActiveSource(source);
                      setIsDrawerOpen(true);
                    }}
                  />
                ))}

                {/* Loading Indicator */}
                {chatLoading && (
                  <div className="ai-msg-row">
                    <div className="ai-avatar-circle">
                      <span className="material-symbols-outlined text-[18px] animate-spin">sync</span>
                    </div>
                    <div className="ai-msg-body">
                      <div className="ai-sender-header">
                        <span className="ai-sender-name">DocuChat AI</span>
                        <span className="text-xs text-gray-400">Searching vectors & generating...</span>
                      </div>
                    </div>
                  </div>
                )}

                <div ref={messagesEndRef} />
              </div>
            )}
          </div>
        </div>

        {/* Bottom Message Composer */}
        <Composer
          chatQuestion={chatQuestion}
          setChatQuestion={setChatQuestion}
          chatLoading={chatLoading}
          chatError={chatError}
          setChatError={setChatError}
          handleChat={handleChat}
          handleKeyDown={handleKeyDown}
          chatInputRef={chatInputRef}
          uploadedDoc={uploadedDoc}
        />
      </main>

      {/* ── Source Details Drawer ───────────────────────────────────── */}
      <SourceDrawer
        source={activeSource}
        isOpen={isDrawerOpen}
        onClose={() => {
          setActiveSource(null);
          setIsDrawerOpen(false);
        }}
      />
    </div>
  );
}

export default App;
