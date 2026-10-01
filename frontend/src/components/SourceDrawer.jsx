import React, { useEffect, useState } from "react";

export default function SourceDrawer({ source, onClose, isOpen, onToggleDrawer }) {
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === "Escape") {
        onClose();
      }
    };
    if (source || isOpen) {
      window.addEventListener("keydown", handleKeyDown);
    }
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [source, isOpen, onClose]);

  const show = Boolean(source || isOpen);
  if (!show) return null;

  const pageNum = source && source.page_number != null && source.page_number >= 0 ? source.page_number : 1;
  const docName = source && source.source ? source.source : "Document Context";
  const excerptText = source && source.text ? source.text.trim() : null;

  const handleCopyExcerpt = () => {
    if (excerptText) {
      navigator.clipboard.writeText(excerptText).catch(() => {});
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <>
      <div
        className={`drawer-backdrop-overlay ${show ? "open" : ""}`}
        onClick={onClose}
        aria-hidden="true"
      />

      <aside className={`source-drawer-aside ${show ? "open" : ""}`} role="dialog" aria-label="Source Citation Preview">
        {/* Drawer Header */}
        <div className="drawer-header-bar">
          <div className="drawer-header-left">
            <span className="material-symbols-outlined text-[20px] text-[#ececec]">auto_stories</span>
            <div className="flex flex-col min-w-0">
              <span className="drawer-doc-title">{docName}</span>
              <span className="drawer-page-sub">Page {pageNum} of Document</span>
            </div>
          </div>
          <button onClick={onClose} className="drawer-close-btn" aria-label="Close drawer" type="button">
            <span className="material-symbols-outlined text-[20px]">close</span>
          </button>
        </div>

        {/* Toolbar Bar */}
        <div className="px-5 py-2.5 bg-[#171717] border-b border-[#262626] flex items-center justify-between">
          <div className="flex items-center gap-2">
            <button className="p-1 rounded hover:bg-[#262626] text-[#737373] hover:text-[#ececec] transition-colors" type="button">
              <span className="material-symbols-outlined text-[18px]">chevron_left</span>
            </button>
            <span className="font-mono text-xs text-[#ececec] font-medium">Page {pageNum}</span>
            <button className="p-1 rounded hover:bg-[#262626] text-[#737373] hover:text-[#ececec] transition-colors" type="button">
              <span className="material-symbols-outlined text-[18px]">chevron_right</span>
            </button>
          </div>
          <span className="px-2.5 py-0.5 rounded-full bg-[#262626] border border-[#383838] text-[#ececec] text-xs font-medium">
            Matched section
          </span>
        </div>

        {/* Viewport Content */}
        <div className="drawer-body-viewport">
          <div className="paper-sheet-card">
            <div className="sheet-top-meta">
              <span>RETRIEVAL-AUGMENTED CONTEXT</span>
              <span className="font-semibold text-[#ececec]">PAGE {pageNum}</span>
            </div>

            <h3 className="text-base font-semibold text-[#ececec]">
              Document Passage Excerpt
            </h3>

            <div className="highlighted-excerpt-box">
              <div className="excerpt-label-header">
                <span className="material-symbols-outlined text-[16px]">verified</span>
                <span>Citation In Context</span>
              </div>
              <p className="excerpt-quote-text">
                {excerptText ? `“${excerptText}”` : "No exact snippet text provided for this chunk reference."}
              </p>
            </div>

            <div className="p-3 bg-[#212121] rounded-xl border border-[#2f2f2f] flex items-center justify-between">
              <div className="flex items-center gap-2 text-xs text-[#737373]">
                <span className="material-symbols-outlined text-[18px]">search_insights</span>
                <span className="text-[#ececec]">Citation match</span>
              </div>
              <span className="text-xs text-[#ececec] font-mono font-medium">Retrieved Passage</span>
            </div>
          </div>
        </div>

        {/* Drawer Footer Actions */}
        <div className="drawer-footer-bar">
          <button
            className="header-btn"
            onClick={handleCopyExcerpt}
            type="button"
          >
            <span className="material-symbols-outlined text-[18px]">
              {copied ? "check" : "copy_all"}
            </span>
            <span>{copied ? "Copied!" : "Copy Excerpt"}</span>
          </button>

          <button onClick={onClose} className="drawer-done-btn" type="button">
            Close Preview
          </button>
        </div>
      </aside>
    </>
  );
}
