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
          {/* Active Document Pill inside Composer */}
          <div className="active-doc-chip">
            <span className="material-symbols-outlined text-[16px] text-[#ececec]">description</span>
            <span className="doc-chip-name">
              {uploadedDoc ? uploadedDoc.filename : "LLM_and_RAG.pdf"}
            </span>
          </div>

          {/* Text Input */}
          <textarea
            ref={chatInputRef}
            id="chat-input-field"
            rows={1}
            value={chatQuestion}
            onChange={(e) => setChatQuestion(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              uploadedDoc
                ? `Ask anything about ${uploadedDoc.filename}...`
                : "Ask anything about your document..."
            }
            className="composer-input-field"
            disabled={chatLoading}
            aria-label="Ask a question"
          />

          {/* Send Button */}
          <button
            type="button"
            onClick={handleChat}
            disabled={chatLoading || !chatQuestion.trim()}
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
          DocuChat AI references your uploaded document. Citations are linked to exact PDF pages.
        </p>
      </div>
    </div>
  );
}
