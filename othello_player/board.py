from random import randint, choice
from typing import List, Tuple


class OthelloBoard:
    def __init__(self, height: int = 8, width: int = 8):
        self.height = height
        self.width = width
        self.total_tokens = self.height * self.width

        self.board = []
        for _ in range(self.height):
            self.board.append([0 for _ in range(self.width)])

        self.tokens = {}
        self.player = 1

    def __str__(self) -> str:
        result = '{}{}.{}.'.format(self.player, self.height, self.width)
        for r in self.board:
            for c in r:
                result += str(c)
            # result += '\n'
        return result

    @staticmethod
    def from_string(bstring: str):
        bits = bstring.split('.')

        player = int(bits[0][0])
        height = int(bits[0][1:])
        width = int(bits[1])

        result = OthelloBoard(height=height, width=width)
        result.player = player

        pos = 0
        for i in range(height):
            result.board[i] = [int(bits[2][pos + j]) for j in range(width)]
            pos += width

        result.recompute_tokens()

        return result

    @property
    def gameover(self) -> bool:
        return self.total_tokens == 0

    @property
    def winner(self) -> int:
        return 1 if self.tokens['white'] > self.tokens['black'] else 2

    @staticmethod
    def decode_column(col: str) -> int:
        res = 0
        col = col.upper().strip()
        for i in range(len(col)):
            ch = col[-(i + 1)]
            res += (ord(ch) - ord('A') + 1) * (1 if i == 0 else (i * 26))
        return res

    def reset_board(self):
        for h in range(self.height):
            for w in range(self.width):
                self.board[h][w] = 0

        mid_h = (self.height >> 1) - 1
        mid_w = (self.width >> 1) - 1

        self.board[mid_h][mid_w] = 1
        self.board[mid_h][mid_w + 1] = 2
        self.board[mid_h + 1][mid_w] = 2
        self.board[mid_h + 1][mid_w + 1] = 1

        self.total_tokens = self.height * self.width - 4

        self.tokens = {'white': 2, 'black': 2}

        self.player = randint(1, 2)

    def recompute_tokens(self):
        self.total_tokens = self.width * self.height
        self.tokens = {'white': 0, 'black': 0}
        for row in self.board:
            for col in row:
                if col != 0:
                    self.tokens['white' if col == 1 else 'black'] += 1
                    self.total_tokens -= 1

    def is_legal(self, r: int, c: int) -> bool:
        if 0 <= r < self.height and 0 <= c < self.width:
            if self.board[r][c] == 0:
                directions = [(0, 1), (1, 0), (1, 1), (0, -1), (-1, 0), (-1, -1), (-1, 1), (1, -1)]
                dcounts = []
                for d in directions:
                    rd = r + d[0]
                    cd = c + d[1]
                    count = 0
                    while 0 <= rd < self.height and 0 <= cd < self.width:
                        if self.board[rd][cd] != 0 and self.board[rd][cd] != self.player:
                            count += 1
                            rd += d[0]
                            cd += d[1]
                            if (rd == self.height and d[0] == 1) or (rd < 0 and d[0] == -1) or \
                                    (cd == self.width and d[1] == 1) or (cd < 0 and d[1] == -1):
                                count = 0
                                break
                        else:
                            if self.board[rd][cd] == 0:
                                count = 0
                            break
                    dcounts.append(count)
                return sum(dcounts) > 0
            else:
                return False
        else:
            return False

    def place(self, r: int, c: int) -> bool:
        if self.is_legal(r, c):
            self.board[r][c] = self.player
            self.tokens['white' if self.player == 1 else 'black'] += 1
            directions = [(0, 1), (1, 0), (1, 1), (0, -1), (-1, 0), (-1, -1), (-1, 1), (1, -1)]
            for d in directions:
                rd = r + d[0]
                cd = c + d[1]
                count = 0

                while 0 <= rd < self.height and 0 <= cd < self.width:
                    if self.board[rd][cd] != 0 and self.board[rd][cd] != self.player:
                        count += 1
                        rd += d[0]
                        cd += d[1]
                        if (rd == self.height and d[0] == 1) or (rd < 0 and d[0] == -1) or \
                                (cd == self.width and d[1] == 1) or (cd < 0 and d[1] == -1):
                            count = 0
                            break
                    else:
                        if self.board[rd][cd] == 0:
                            count = 0
                        break

                if count > 0:
                    rd = r + d[0]
                    cd = c + d[1]
                    while self.board[rd][cd] != 0 and self.board[rd][cd] != self.player:
                        self.board[rd][cd] = self.player
                        self.tokens['white' if self.player == 1 else 'black'] += 1
                        self.tokens['black' if self.player == 1 else 'white'] -= 1
                        rd += d[0]
                        cd += d[1]

            self.total_tokens -= 1
            self.player = 1 if self.player == 2 else 2
            return True
        return False


def find_next_boards(b: OthelloBoard) -> List[Tuple[int, int]]:
    # Start in the middle every time, and find all of the spaces that are filled, then find the valid positions
    mid_h = (b.height >> 1) - 1
    mid_w = (b.width >> 1) - 1

    edge_positions = []

    visited = []
    for _ in range(b.height):
        visited.append([False for _ in range(b.width)])

    q = [(mid_h, mid_w)]

    while not len(q) == 0:
        coord = q[0]
        q.pop(0)
        r, c = coord

        visited[r][c] = True

        edges = [(0, 1), (1, 0), (1, 1), (0, -1), (-1, 0), (-1, -1), (1, -1), (-1, 1)]
        for e in edges:
            rd, cd = e
            rt = r + rd
            ct = c + cd
            if 0 <= rt < b.height and 0 <= ct < b.width:
                if b.board[rt][ct] != 0:
                    if not visited[rt][ct]:
                        q.append((rt, ct))
                else:
                    if not visited[rt][ct]:
                        visited[rt][ct] = True
                        edge_positions.append((rt, ct))

    # Iterate over all of the edge positions and determine if
    return [ep for ep in edge_positions if b.is_legal(ep[0], ep[1])]


def generate_random_board() -> OthelloBoard:
    b = OthelloBoard()
    d = randint(0, 64)
    for i in range(d):
        poss = find_next_boards(b)
        pos = choice(poss)
        b.place(pos[0], pos[1])
    return b
