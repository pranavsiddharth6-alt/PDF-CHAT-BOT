import React from "react";

export default function Composer({
  chatQuestion,
  setChatQuestion,
  chatLoading,
  chatError,
  setChatError,
  handleChat,
  handleKeyDown,
  chatInputRef,
  uploadedDoc,
}) {
  const isDocUploaded = Boolean(uploadedDoc && uploadedDoc.filename);

  return (
    <div className="composer-fixed-dock">
      <div className="composer-center-wrap">
        {chatError && (
          <div className="w-full mb-2 p-3 bg-red-950/80 border border-red-800 text-red-200 text-sm rounded-xl flex items-center justify-between pointer-events-auto">
            <span>⚠️ {chatError}</span>
            <button
              onClick={() => setChatError(null)}
              className="text-red-400 hover:text-white font-bold ml-2"
              type="button"
            >
              ×
            </button>
          </div>
        )}

        <div className="composer-card">
          {/* Active Document Pill inside Composer — rendered only when a document is uploaded */}
          {isDocUploaded && (
            <div className="active-doc-chip">
              <span className="material-symbols-outlined text-[16px] text-[#ececec]">description</span>
              <span className="doc-chip-name">{uploadedDoc.filename}</span>
            </div>
          )}

          {/* Text Input */}
          <textarea
            ref={chatInputRef}
            id="chat-input-field"
            rows={1}
            value={chatQuestion}
            onChange={(e) => setChatQuestion(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              isDocUploaded
                ? `Ask anything about ${uploadedDoc.filename}...`
                : "Upload a PDF to start chatting..."
            }
            className="composer-input-field"
            disabled={chatLoading || !isDocUploaded}
            aria-label="Ask a question"
          />

          {/* Send Button */}
          <button
            type="button"
            onClick={handleChat}
            disabled={chatLoading || !chatQuestion.trim() || !isDocUploaded}
            className="send-btn-circle"
            aria-label="Send message"
          >
            {chatLoading ? (
              <span className="material-symbols-outlined text-[20px] animate-spin">sync</span>
            ) : (
              <span className="material-symbols-outlined text-[20px] font-semibold">arrow_upward</span>
            )}
          </button>
        </div>

        <p className="composer-helper-text">
          {isDocUploaded
            ? "DocuChat AI references your uploaded document. Citations are linked to exact PDF pages."
            : "Upload a PDF document from the sidebar to begin querying."}
        </p>
      </div>
    </div>
  );
}
