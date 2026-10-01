import React from "react";

export default function Header({
  sidebarOpen,
  setSidebarOpen,
  uploadedDoc,
  conversationId,
  handleNewConversation,
  onToggleDrawer,
}) {
  return (
    <header className="app-header">
      <div className="header-left">
        <button
          className="mobile-menu-toggle"
          onClick={() => setSidebarOpen(!sidebarOpen)}
          aria-label="Toggle Sidebar"
          type="button"
        >
          <span className="material-symbols-outlined">menu</span>
        </button>

        {uploadedDoc ? (
          <div className="doc-picker-pill">
            <span className="material-symbols-outlined text-[18px] text-[#a3a3a3]">menu_book</span>
            <span className="font-medium text-[#ececec]">
              {uploadedDoc.filename}
            </span>
            <span className="doc-page-count">
              · {uploadedDoc.total_pages} {uploadedDoc.total_pages === 1 ? "page" : "pages"}
            </span>
          </div>
        ) : (
          <div className="doc-picker-pill">
            <span className="material-symbols-outlined text-[18px] text-[#a3a3a3]">upload_file</span>
            <span className="text-[#a3a3a3]">No document uploaded</span>
          </div>
        )}
      </div>

      <div className="header-right">
        {/* Source Drawer toggle — only functional when sources exist */}
        <button
          className="header-btn"
          type="button"
          onClick={onToggleDrawer}
          title="View source citations panel"
        >
          <span className="material-symbols-outlined text-[18px]">dock_to_left</span>
          <span className="hidden sm:inline">Sources</span>
        </button>

        {conversationId && (
          <button
            type="button"
            onClick={handleNewConversation}
            className="header-btn"
            title="Clear active conversation memory"
          >
            <span className="material-symbols-outlined text-[18px]">cleaning_services</span>
            <span className="hidden sm:inline">Clear Chat</span>
          </button>
        )}
      </div>
    </header>
  );
}
