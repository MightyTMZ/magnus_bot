"use client";

import { useEffect, useState, useCallback, useRef } from "react";

const GAME_URL_PATTERN = /^https:\/\/www\.chess\.com\/game\/\d+/;
const BOT_URL_PATTERN = /^https:\/\/www\.chess\.com\/play\/computer\/.+/;

export default function Home() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [viewerUrl, setViewerUrl] = useState<string | null>(null);
  const [currentUrl, setCurrentUrl] = useState<string>("");
  const [gameUrl, setGameUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showBoardView, setShowBoardView] = useState(false);
  const [showAbandonConfirm, setShowAbandonConfirm] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Create or restore Steel session on mount
  useEffect(() => {
    async function init() {
      try {
        // Try to restore an existing session from localStorage
        const stored = localStorage.getItem("steelSession");
        if (stored) {
          const { sessionId: storedId } = JSON.parse(stored);
          const checkRes = await fetch(`/api/steel?sessionId=${storedId}`);
          if (checkRes.ok) {
            const data = await checkRes.json();
            if (data.status === "live") {
              setSessionId(data.sessionId);
              setViewerUrl(data.viewerUrl);
              setLoading(false);
              return;
            }
          }
          // Session is dead, clear it
          localStorage.removeItem("steelSession");
        }

        // No valid session — create a new one
        const res = await fetch("/api/steel", { method: "POST" });
        if (!res.ok) throw new Error("Failed to create session");
        const data = await res.json();
        setSessionId(data.sessionId);
        setViewerUrl(data.viewerUrl);

        // Persist to localStorage
        localStorage.setItem(
          "steelSession",
          JSON.stringify({ sessionId: data.sessionId, viewerUrl: data.viewerUrl })
        );

        // Navigate to chess.com
        await fetch("/api/steel/navigate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            sessionId: data.sessionId,
            url: "https://www.chess.com/",
          }),
        });
      } catch (err) {
        setError(err instanceof Error ? err.message : "Unknown error");
      } finally {
        setLoading(false);
      }
    }
    init();
  }, []);

  // Poll for URL changes via CDP
  useEffect(() => {
    if (!sessionId) return;

    pollRef.current = setInterval(async () => {
      try {
        const res = await fetch(`/api/steel/url?sessionId=${sessionId}`);
        if (res.ok) {
          const data = await res.json();
          setCurrentUrl(data.url ?? "");
          const url = data.url ?? "";
          if (GAME_URL_PATTERN.test(url) || BOT_URL_PATTERN.test(url)) {
            setGameUrl(data.url);
          } else {
            setGameUrl(null);
          }
        }
      } catch {
        // Silently ignore polling errors
      }
    }, 3000);

    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [sessionId]);

  // Note: session is intentionally NOT released on unmount so it persists across refreshes.
  // Sessions auto-expire after the Steel timeout (15 min).

  const isInGame = gameUrl !== null;

  const handleAbandon = useCallback(() => {
    setShowAbandonConfirm(true);
  }, []);

  const confirmAbandon = useCallback(() => {
    setShowAbandonConfirm(false);
    setShowBoardView(false);
    if (gameUrl) {
      window.open(gameUrl, "_blank");
    }
  }, [gameUrl]);

  const handleGoToGame = useCallback(() => {
    if (gameUrl) {
      window.open(gameUrl, "_blank");
    }
  }, [gameUrl]);

  if (error) {
    return (
      <div className="flex flex-1 items-center justify-center">
        <p className="text-red-500">Error: {error}</p>
      </div>
    );
  }

  return (
    <div className="flex flex-1 h-screen overflow-hidden">
      {/* Main browser frame */}
      <div className="flex-1 flex flex-col relative">
        {/* Top bar */}
        <div className="flex items-center justify-between px-4 py-2 border-b border-zinc-200 dark:border-zinc-800">
          <span className="text-sm text-zinc-500 dark:text-zinc-400 truncate max-w-[60%]">
            {currentUrl || "Loading..."}
          </span>
          <button
            onClick={() => setShowBoardView(true)}
            className="px-3 py-1.5 text-sm rounded-md border border-zinc-200 dark:border-zinc-700 hover:bg-zinc-100 dark:hover:bg-zinc-800 transition-colors"
          >
            Board view
          </button>
        </div>

        {/* Browser iframe */}
        <div className="flex-1 relative">
          {loading ? (
            <div className="flex items-center justify-center h-full">
              <p className="text-zinc-500 dark:text-zinc-400">
                Starting browser session...
              </p>
            </div>
          ) : viewerUrl ? (
            <iframe
              src={viewerUrl}
              className="w-full h-full border-0"
              allow="clipboard-read; clipboard-write"
            />
          ) : null}
        </div>
      </div>

      {/* Right sidebar */}
      <div className="w-56 flex flex-col gap-3 p-4 border-l border-zinc-200 dark:border-zinc-800">
        <button
          onClick={handleAbandon}
          disabled={!isInGame}
          className="px-4 py-2.5 rounded-md text-sm font-medium text-white bg-red-600 hover:bg-red-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          Abandon over the board
        </button>
        <button
          onClick={handleGoToGame}
          disabled={!isInGame}
          className="px-4 py-2.5 rounded-md text-sm font-medium text-white bg-green-600 hover:bg-green-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          Go to the live game
        </button>
      </div>

      {/* Board view modal */}
      {showBoardView && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="bg-white dark:bg-zinc-900 rounded-lg shadow-xl w-[600px] h-[500px] flex flex-col">
            <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-200 dark:border-zinc-800">
              <span className="text-sm font-medium">Board View</span>
              <button
                onClick={() => setShowBoardView(false)}
                className="text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-200 transition-colors"
              >
                &#x2715;
              </button>
            </div>
            <div className="flex-1 flex items-center justify-center text-zinc-400 dark:text-zinc-500">
              Camera feed coming soon
            </div>
          </div>
        </div>
      )}

      {/* Abandon confirmation modal */}
      {showAbandonConfirm && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="bg-white dark:bg-zinc-900 rounded-lg shadow-xl p-6 max-w-sm">
            <p className="text-sm mb-4">
              Are you sure you want to abandon the over-the-board game?
            </p>
            <div className="flex gap-3 justify-end">
              <button
                onClick={() => setShowAbandonConfirm(false)}
                className="px-4 py-2 text-sm rounded-md border border-zinc-200 dark:border-zinc-700 hover:bg-zinc-100 dark:hover:bg-zinc-800 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={confirmAbandon}
                className="px-4 py-2 text-sm rounded-md text-white bg-red-600 hover:bg-red-700 transition-colors"
              >
                Yes, abandon
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
