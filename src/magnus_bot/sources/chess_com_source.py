"""Move source that polls the chess.com monitor for the opponent's moves.

The Next.js app (``chess_com/magnus_bot``) runs a Steel browser session connected to a chess.com
game.  Its ``/api/steel/monitor`` endpoint screenshots the board, diffs against the previous
frame, and asks Claude Vision to identify any new move.  This source polls that endpoint and
returns each new move as it appears.

It also exposes :meth:`play_move` to push a UCI move *to* chess.com (click the squares in the
browser) via ``/api/steel/play-move``.
"""

from __future__ import annotations

import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import chess

from magnus_bot.sources.move_source import MoveSource, validate_move


def _post_json(url: str, body: dict, timeout: float = 15.0) -> dict:
    """POST JSON to *url* and return the parsed response.  Stdlib only."""
    data = json.dumps(body).encode()
    req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except HTTPError as e:
        # The Next routes put the real reason in the JSON body; surface it instead of a bare 500.
        e.msg = f"{e.msg}: {e.read().decode(errors='replace')[:500]}"
        raise


class ChessComSource(MoveSource):
    """Polls the chess.com Next.js monitor API for new moves.

    Parameters
    ----------
    session_id:
        The Steel browser session ID connected to the chess.com game.
    base_url:
        Root URL of the Next.js app (default ``http://localhost:3000``).
    poll_interval:
        Seconds between monitor polls (default 1.0).
    timeout:
        Give up after this many seconds with no move detected (default 300).
    """

    def __init__(
        self,
        session_id: str,
        base_url: str = "http://localhost:3000",
        poll_interval: float = 1.0,
        timeout: float = 300.0,
    ) -> None:
        self.session_id = session_id
        self.base_url = base_url.rstrip("/")
        self.poll_interval = poll_interval
        self.timeout = timeout
        self._last_consumed: str = ""  # guard against the same poll returning a move twice
        self._pending: str = ""  # a move seen while priming (a fast reply), served next

    def next_move(self, board: chess.Board) -> chess.Move:
        """Block until the chess.com monitor reports a new move, then return it."""
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            uci, self._pending = self._pending or self._poll_once(), ""
            if uci and uci != self._last_consumed:
                move = chess.Move.from_uci(uci)
                if move in board.legal_moves:
                    self._last_consumed = uci
                    return validate_move(board, move)
                # Vision hallucinated an illegal move; ignore and keep polling.
            time.sleep(self.poll_interval)
        raise TimeoutError(
            f"no move from chess.com within {self.timeout:.0f} s "
            f"(polled {self.base_url}/api/steel/monitor)"
        )

    def play_move(self, move: chess.Move) -> dict:
        """Click ``move`` on chess.com via the Steel browser.

        Returns the JSON response from the play-move endpoint.  Raises on HTTP errors.
        """
        url = f"{self.base_url}/api/steel/play-move"
        return _post_json(url, {"sessionId": self.session_id, "move": move.uci()}, timeout=30)

    def prime(self) -> None:
        """Advance the monitor's stored screenshot so the next diff starts fresh.

        Call this after :meth:`play_move`: the click changed the board, and without priming,
        the next :meth:`next_move` poll would see our own click as a diff and might hallucinate
        a move from it.  Two polls with a gap let the board settle and the animation finish.
        A move either poll reports (the opponent replied fast) is kept for :meth:`next_move`,
        which still checks it is legal.
        """
        self._pending = self._poll_once()
        time.sleep(1.5)  # let chess.com animate and the opponent start thinking
        self._pending = self._poll_once() or self._pending

    # ------------------------------------------------------------------
    def _poll_once(self) -> str:
        """One call to the monitor endpoint.  Returns a UCI string or ``""``."""
        url = f"{self.base_url}/api/steel/monitor"
        try:
            data = _post_json(url, {"sessionId": self.session_id, "consumer": "arms"})
            return data.get("move", "")
        except (HTTPError, URLError, ValueError, OSError):
            return ""
