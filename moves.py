import chess

def expected_occupancy(chess_board):
    occupancy = {}

    for sq in chess.SQUARES:
        piece = chess_board.piece_at(sq)

        if piece is None:
            continue
        else:
            name = chess.square_name(sq)
            if piece.color == chess.WHITE:
                color = "W"
            else:
                color = "B"
            occupancy[name] = color
    return occupancy

def mismatch(a,b):
    squares = set(a) | set(b)
    count = 0
    for sq in squares:
        if a.get(sq) != b.get(sq):
            count += 1
    return count

def looks_like_move(expected, observed):
    """A piece left a square and a piece arrived. A vanished piece alone is a miss."""
    left = False
    arrived = False
    for sq in set(expected) | set(observed):
        before = expected.get(sq)
        after = observed.get(sq)
        if before == after:
            continue
        if before is not None:
            left = True
        if after is not None:
            arrived = True
    return left and arrived

def find_move(chess_board, observed):
    """Return (move, illegal).

    move is the legal move that explains the board, or None.
    illegal is True when the board changed like a move and no legal move fits.
    That covers moving while in check, moving the wrong piece, and any other illegal placement.
    """
    expected = expected_occupancy(chess_board)
    no_move_score = mismatch(expected, observed)
    best_move = None
    best_score = no_move_score
    for move in chess_board.legal_moves:
        test = chess_board.copy()
        test.push(move)
        score = mismatch(expected_occupancy(test), observed)
        if score < best_score:
            best_score = score
            best_move = move
    if best_move is not None:
        return best_move, False
    return None, looks_like_move(expected, observed)

     # e2e4
