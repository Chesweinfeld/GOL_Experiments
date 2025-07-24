#!/usr/bin/env python3
"""
Simple Game-of-Life viewer.

▪ Fixed board 200 × 200.
▪ If --pattern is given, any smaller pattern is centered automatically.
▪ Otherwise fills with random soup (live-cell probability = --prob).
"""

import argparse, numpy as np
import pygame, sys

ROWS = COLS = 200         # board size
CELL  = 3                 # pixel size of one cell
FPS   = 30

##############################################################################
# Life kernel
##############################################################################

def step(board: np.ndarray) -> None:
    nbrs = sum(np.roll(np.roll(board, i, 0), j, 1)
               for i in (-1, 0, 1) for j in (-1, 0, 1) if i or j)
    board[:] = (nbrs == 3) | (board & (nbrs == 2))

##############################################################################
# Main loop
##############################################################################

def run(board: np.ndarray, fps: int) -> None:
    pygame.init()
    screen = pygame.display.set_mode((COLS*CELL, ROWS*CELL))
    clock  = pygame.time.Clock()

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()

        # draw
        screen.fill((0, 0, 0))
        ys, xs = np.where(board)
        for y, x in zip(ys, xs):
            pygame.draw.rect(screen, (255, 255, 255),
                             (x*CELL, y*CELL, CELL, CELL))
        pygame.display.flip()

        step(board)
        clock.tick(fps)

##############################################################################
# CLI
##############################################################################

def cli():
    ap = argparse.ArgumentParser(description="Game of Life viewer")
    ap.add_argument("--pattern", type=str, help=".npy file to load")
    ap.add_argument("--prob",    type=float, default=0.25,
                    help="Live-cell probability if no pattern (default 0.25)")
    ap.add_argument("--fps",     type=int,   default=FPS,
                    help="Frames per second")
    return ap.parse_args()

def main():
    args = cli()
    board = np.zeros((ROWS, COLS), dtype=bool)

    if args.pattern:
        patt = np.load(args.pattern, allow_pickle=False)
        pr, pc = patt.shape
        if pr > ROWS or pc > COLS:
            raise ValueError(f"Pattern {patt.shape} larger than board {(ROWS, COLS)}")
        r0 = ROWS//2 - pr//2
        c0 = COLS//2 - pc//2
        board[r0:r0+pr, c0:c0+pc] = patt.astype(bool)
    else:
        board[:] = np.random.rand(ROWS, COLS) < args.prob

    run(board, args.fps)

if __name__ == "__main__":
    main()