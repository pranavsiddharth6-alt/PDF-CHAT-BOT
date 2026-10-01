import React from "react";

export default function Sidebar({
  sidebarOpen,
  setSidebarOpen,
  backendHealthy,
  file,
  uploading,
  uploadError,
  uploadedDoc,
  isDragging,
  handleFileChange,
  handleDragOver,
  handleDragEnter,
  handleDragLeave,
  handleDrop,
  handleUpload,
  handleNewConversation,
  conversationId,
}) {
  return (
    <>
      {/* Mobile Backdrop Overlay */}
      {sidebarOpen && (
        <div
          className="sidebar-backdrop"
          onClick={() => setSidebarOpen(false)}
          aria-hidden="true"
        />
      )}

      <aside className={`app-sidebar ${sidebarOpen ? "open" : ""}`}>
        <div className="sidebar-main-content">
          {/* Brand Header */}
          <div className="sidebar-brand">
            <div className="brand-icon-box">
              <span className="material-symbols-outlined text-[19px]">smart_toy</span>
            </div>
            <span className="brand-title">DocuChat AI</span>
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

          {/* New Chat Action */}
          <div className="sidebar-action-wrap">
            <button
              type="button"
              onClick={() => {
                handleNewConversation();
                if (window.innerWidth <= 768) setSidebarOpen(false);
              }}
              className="btn-new-chat"
              title="Start a new conversation session"
            >
              <span className="material-symbols-outlined text-[18px]">add</span>
              <span>New chat</span>
            </button>
          </div>

          {/* Scrollable Nav Area */}
          <div className="sidebar-scroll-area">
            {/* Document Upload Section */}
            <div className="sidebar-section-header">
              <span className="sidebar-section-title">Your Documents</span>
              <button
                className="user-settings-btn"
                title="Upload document"
                type="button"
                onClick={() => document.getElementById("pdf-file-input")?.click()}
              >
                <span className="material-symbols-outlined text-[18px]">upload_file</span>
              </button>
            </div>

            <form onSubmit={handleUpload} className="sidebar-upload-box">
              <div
                className={`sidebar-dropzone ${isDragging ? "drag-active" : ""}`}
                onDragOver={handleDragOver}
                onDragEnter={handleDragEnter}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => document.getElementById("pdf-file-input")?.click()}
              >
                <input
                  type="file"
                  id="pdf-file-input"
                  accept=".pdf,application/pdf"
                  onChange={handleFileChange}
                  className="hidden-file-input"
                />
                <span className="material-symbols-outlined dropzone-icon text-[24px]">
                  {isDragging ? "download" : file ? "picture_as_pdf" : "upload_file"}
                </span>
                <span className="dropzone-text">
                  {isDragging
                    ? "Drop PDF here..."
                    : file
                    ? file.name
                    : "Choose or drag PDF..."}
                </span>
                <span className="dropzone-hint">PDF up to 50MB</span>
              </div>

              {file && (
                <button
                  type="submit"
                  disabled={uploading}
                  className="btn-upload-submit"
                >
                  {uploading ? (
                    <>
                      <span className="material-symbols-outlined text-[16px] animate-spin">sync</span>
                      Indexing PDF...
                    </>
                  ) : (
                    <>
                      <span className="material-symbols-outlined text-[16px]">cloud_upload</span>
                      Upload & Index
                    </>
                  )}
                </button>
              )}

              {uploadError && (
                <div className="text-red-400 text-xs mt-2 p-2 bg-red-950/30 rounded border border-red-900/50">
                  {uploadError}
                </div>
              )}
            </form>

            {/* Uploaded Documents Nav — only shown when a real document exists */}
            {uploadedDoc && (
              <nav className="nav-list">
                <div className="nav-item active">
                  <span className="material-symbols-outlined nav-item-icon text-[18px]">description</span>
                  <span className="nav-item-text">{uploadedDoc.filename}</span>
                </div>
              </nav>
            )}

            {/* Active Document Details Card */}
            {uploadedDoc && (
              <div className="doc-stats-card">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Document Stats</span>
                  <span className="text-[10px] bg-emerald-950 text-emerald-400 px-1.5 py-0.5 rounded border border-emerald-800 font-mono">Ready</span>
                </div>
                <div className="doc-stats-grid">
                  <div className="stat-box">
                    <div className="stat-value">{uploadedDoc.total_pages}</div>
                    <div className="stat-label">Pages</div>
                  </div>
                  <div className="stat-box">
                    <div className="stat-value">{uploadedDoc.total_chunks}</div>
                    <div className="stat-label">Chunks</div>
                  </div>
                  <div className="stat-box">
                    <div className="stat-value">{uploadedDoc.vectors_stored}</div>
                    <div className="stat-label">Vectors</div>
                  </div>
                  <div className="stat-box">
                    <div className="stat-value">{uploadedDoc.embedding_dimension || 384}d</div>
                    <div className="stat-label">Dimension</div>
                  </div>
                </div>
              </div>
            )}

            {/* Active Session — only shown when a real session exists */}
            {conversationId && (
              <>
                <div className="sidebar-section-header" style={{ marginTop: "1rem" }}>
                  <span className="sidebar-section-title">Active Session</span>
                </div>
                <nav className="nav-list">
                  <div className="nav-item active">
                    <span className="material-symbols-outlined nav-item-icon text-[18px]">chat_bubble</span>
                    <span className="nav-item-text">
                      Session {conversationId.slice(0, 8)}...
                    </span>
                  </div>
                </nav>
              </>
            )}
          </div>
        </div>

        {/* Sidebar Footer — Session ID and version only */}
        <div className="sidebar-user-footer">
          <div className="user-info">
            <div className="user-avatar">
              <span className="material-symbols-outlined text-[18px]">smart_toy</span>
            </div>
            <div className="user-details">
              <span className="user-name">DocuChat AI</span>
              <span className="user-plan">v1.0 · RAG Assistant</span>
            </div>
          </div>
        </div>
      </aside>
    </>
  );
}
