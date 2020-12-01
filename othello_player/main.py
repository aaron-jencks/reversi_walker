import pygame as pg
from typing import Tuple

from board import OthelloBoard

pg.init()


class BoardDisplayWrapper:
    def __init__(self, board_width: int = 320, board_height: int = 320):
        self.bwidth = board_width
        self.bheight = board_height

        self.board = OthelloBoard()
        self.board.reset_board()
        self.screen = pg.display.set_mode([500, 500])
        self.board_surf = pg.Surface((self.bwidth, self.bheight))
        self.text_surf = pg.Surface((500 - self.bwidth, 500))

        self.screen.fill((255, 255, 255))
        self.screen.blit(self.board_surf, (10, 10))

        self.draw()

    def text(self, pos: Tuple[int, int], msg: str):
        font = pg.font.SysFont('roboto', 16)
        lines = msg.splitlines(False)
        for i, l in enumerate(lines):
            img = font.render(l, True, (0, 0, 0))
            self.text_surf.blit(img, (pos[0], pos[1] + (i * 16)))

    def draw(self):
        self.board_surf.fill((50, 168, 82))
        self.text_surf.fill((255, 255, 255))

        pg.draw.rect(self.board_surf, (0, 0, 0), (0, 0, self.bwidth, self.bheight), 1)
        cw = self.bwidth // self.board.width
        ch = self.bheight // self.board.height
        for c in range(self.board.width):
            for r in range(self.board.height):
                x = c * cw - c
                y = r * ch - r

                pg.draw.rect(self.board_surf, (0, 0, 0), (x, y, cw, ch), 1)

                if self.board.board[r][c] != 0:
                    p = self.board.board[r][c]
                    pg.draw.circle(self.board_surf, (255, 255, 255) if p == 1 else (0, 0, 0),
                                   (x + (cw >> 1), y + (ch >> 1)), (cw >> 1) - 2)
                elif self.board.is_legal(r, c):
                    pg.draw.rect(self.board_surf, (55, 247, 250), (x + 1, y + 1, cw - 2, ch - 2), 1)

        self.screen.blit(self.board_surf, (10, 10))

        pg.draw.rect(self.text_surf, (0, 0, 0), (0, 0, 486 - self.bwidth, 488), 1)
        self.text((2, 2), "Tokens:\nWhite: {}\nBlack: {}\nIt is {}'s turn".format(self.board.tokens['white'],
                                                                                  self.board.tokens['black'],
                                                                                  "White" if self.board.player == 1
                                                                                  else "Black"))
        self.screen.blit(self.text_surf, (10 + self.bwidth + 2, 10))

    def parse_mouse_click(self, pos: Tuple[int, int]):
        # print("Received a mouse click at {}".format(pos))
        cw = self.bwidth // self.board.width
        ch = self.bheight // self.board.height
        c = (pos[0] - 10) // cw
        r = (pos[1] - 10) // ch
        # print("Attempting to place at ({}, {})".format(r, c))
        self.board.place(r, c)


if __name__ == '__main__':
    running = True
    b = BoardDisplayWrapper()
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
                break
            elif event.type == pg.MOUSEBUTTONUP:
                b.parse_mouse_click(pg.mouse.get_pos())
                b.draw()

        pg.display.flip()

    pg.quit()
