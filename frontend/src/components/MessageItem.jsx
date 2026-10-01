import React, { useState } from "react";
import ReactMarkdown from "react-markdown";

export default function MessageItem({ msg, onSelectSource }) {
  const isUser = msg.role === "user";
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    if (msg.content) {
      navigator.clipboard.writeText(msg.content).catch(() => {});
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  if (isUser) {
    return (
      <div className="user-msg-row">
        <div className="user-msg-column">
          <div className="user-bubble">
            <p>{msg.content}</p>
          </div>
          {msg.timestamp && <span className="msg-timestamp">{msg.timestamp}</span>}
        </div>
        <div className="user-avatar-circle">
          <span className="material-symbols-outlined text-[17px]">person</span>
        </div>
      </div>
    );
  }

  const hasSources = msg.sources && msg.sources.length > 0;

  return (
    <div className="ai-msg-row">
      <div className="ai-avatar-circle">
        <span className="material-symbols-outlined text-[18px]">smart_toy</span>
      </div>

      <div className="ai-msg-body">
        <div className="ai-sender-header">
          <span className="ai-sender-name">DocuChat AI</span>
          {/* Show source verification tag only when actual sources were returned */}
          {hasSources && (
            <span className="verified-tag">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 inline-block"></span>
              {msg.sources.length} {msg.sources.length === 1 ? "source" : "sources"} cited
            </span>
          )}
        </div>

        <div className="prose-dark">
          <ReactMarkdown>{msg.content}</ReactMarkdown>
        </div>

        {/* Sources & Citations Section — only rendered from real API data */}
        {hasSources && (
          <div className="sources-section-box">
            <div className="sources-section-header">
              <div className="sources-title-group">
                <span className="material-symbols-outlined text-[17px] text-[#a3a3a3]">format_quote</span>
                <span>Sources from Document</span>
              </div>
              <span className="sources-count-label">
                {msg.sources.length} {msg.sources.length === 1 ? "citation" : "citations"}
              </span>
            </div>

            <div className="sources-grid">
              {msg.sources.map((src, sIdx) => {
                const pageNum = src.page_number != null && src.page_number >= 0 ? src.page_number : "—";
                return (
                  <div
                    key={sIdx}
                    className="source-card-btn group"
                    onClick={() => onSelectSource && onSelectSource(src)}
                    role="button"
                    tabIndex={0}
                  >
                    <div className="source-card-top">
                      <span className="source-page-badge">
                        <span className="material-symbols-outlined text-[15px] text-[#a3a3a3]">description</span>
                        Page {pageNum}
                      </span>
                      <span className="inspect-text">
                        View
                        <span className="material-symbols-outlined text-[14px]">arrow_forward</span>
                      </span>
                    </div>
                    {src.text && typeof src.text === "string" && src.text.trim() && (
                      <p className="source-card-snippet">
                        &ldquo;{src.text.trim()}&rdquo;
                      </p>
                    )}
                  </div>
                );
              })}
            </div>

            {/* Action Bar — only Copy is functional */}
            <div className="ai-action-bar">
              <div className="action-buttons-group">
                <button
                  className="icon-action-btn"
                  onClick={handleCopy}
                  title="Copy answer to clipboard"
                  type="button"
                >
                  <span className="material-symbols-outlined text-[18px]">
                    {copied ? "check" : "content_copy"}
                  </span>
                </button>
              </div>

              {copied && (
                <span className="text-xs text-[#ececec]">Copied to clipboard</span>
              )}
            </div>
          </div>
        )}

        {/* Copy button when there are no sources */}
        {!hasSources && (
          <div className="ai-action-bar">
            <div className="action-buttons-group">
              <button
                className="icon-action-btn"
                onClick={handleCopy}
                title="Copy answer to clipboard"
                type="button"
              >
                <span className="material-symbols-outlined text-[18px]">
                  {copied ? "check" : "content_copy"}
                </span>
              </button>
            </div>
            {copied && (
              <span className="text-xs text-[#ececec]">Copied to clipboard</span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
