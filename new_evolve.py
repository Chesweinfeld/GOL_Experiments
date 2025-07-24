#!/usr/bin/env python3
"""
Evolution search for Life patterns.

By default this script saves only the final champion as `winner.npy`.
Optionally, you can tile the top N final-round patterns into a single board
and save that board to a file you specify with --tile_output.
"""

import argparse, numpy as np, multiprocessing as mp
from tqdm import tqdm

##############################################################################
# Life helpers
##############################################################################

def step(b: np.ndarray) -> None:
    nbrs = sum(np.roll(np.roll(b, i, 0), j, 1)
               for i in (-1,0,1) for j in (-1,0,1) if i or j)
    b[:] = (nbrs == 3) | (b & (nbrs == 2))

def mutate(patt: np.ndarray, flips: int, rng) -> np.ndarray:
    q = patt.copy()
    idx = rng.choice(q.size, size=flips, replace=False)
    q.flat[idx] ^= True
    return q

##############################################################################
# Fitness
##############################################################################

def fitness(seed: np.ndarray, board_size: int, steps: int,
            copy_tol: int, sustain_extra: int) -> float:

    B = np.zeros((board_size, board_size), dtype=bool)
    r0 = board_size//2 - seed.shape[0]//2
    c0 = board_size//2 - seed.shape[1]//2
    B[r0:r0+seed.shape[0], c0:c0+seed.shape[1]] = seed

    for _ in range(steps):
        step(B)

    sr, sc = seed.shape
    repro = 0.0
    for y in range(board_size-sr+1):
        for x in range(board_size-sc+1):
            d = np.count_nonzero(B[y:y+sr, x:x+sc] ^ seed)
            if d <= copy_tol:
                repro += (copy_tol + 1 - d)

    alive = 0
    for _ in range(sustain_extra):
        if not B.any(): break
        alive += 1
        step(B)
    survival = 2.0 * alive / sustain_extra
    growth   = 0.05 * (B.sum()/20)

    return repro + survival + growth


def tile_patterns(patterns, margin: int = 4) -> np.ndarray:
    """
    Return a large board containing all `patterns` tiled in a square grid
    with `margin` dead-cell spacing. Assumes all patterns share the same shape.
    """
    if not patterns:
        raise ValueError("No patterns to tile.")
    pr, pc = patterns[0].shape
    n = len(patterns)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    height = rows * pr + (rows - 1) * margin
    width  = cols * pc + (cols - 1) * margin
    board = np.zeros((height, width), dtype=bool)
    k = 0
    for r in range(rows):
        for c in range(cols):
            if k >= n:
                break
            top  = r * (pr + margin)
            left = c * (pc + margin)
            board[top:top + pr, left:left + pc] = patterns[k]
            k += 1
    return board

##############################################################################
# Evolution loop
##############################################################################

def evolve(a):
    rng = np.random.default_rng()
    R, K = a.round_size, a.keepers
    assert K <= R//2, "--keepers must be ≤ round_size/2"

    # initial random population
    pop=[]
    for _ in range(R):
        p=np.zeros((30,30),bool)
        idx=rng.choice(900,20,False)
        p.flat[idx]=True
        pop.append(p)

    for round_no in range(1, a.rounds+1):
        tqdm.write(f"\nROUND {round_no}: evaluating {R} patterns")

        args=[(p,a.boardsize,a.steps,a.copy_tol,a.sustain_extra) for p in pop]
        with mp.Pool(processes=a.workers) as pool:
            scores=list(tqdm(pool.starmap(fitness,args),
                             total=R,ncols=80,desc="scoring"))

        top_idx=np.argsort(scores)[-K:]
        elites=[pop[i] for i in top_idx]

        mutants=[mutate(e,a.mut_flips,rng) for e in elites]
        new_pop=elites+mutants
        while len(new_pop)<R:
            q=np.zeros((30,30),bool)
            q.flat[rng.choice(900,20,False)]=True
            new_pop.append(q)
        pop=new_pop[:R]

    # final champion
    finals=[fitness(p,a.boardsize,a.steps,a.copy_tol,a.sustain_extra) for p in pop]
    champ=pop[int(np.argmax(finals))]
    np.save("winner.npy",champ)
    print(f"\n🏆 Champion saved to winner.npy  (score {max(finals):.2f})")

    # Optional: save a tiled board of top patterns
    if a.tile_output and a.tile_n > 0:
        sorted_idx = np.argsort(finals)[::-1][:a.tile_n]
        patterns_to_tile = [pop[i] for i in sorted_idx]
        tiled = tile_patterns(patterns_to_tile, margin=a.tile_margin)
        np.save(a.tile_output, tiled)
        print(f"Tiled top {a.tile_n} patterns → {a.tile_output}")


##############################################################################
# CLI
##############################################################################

def get_args():
    ap=argparse.ArgumentParser(description="Evolutionary Life search")
    ap.add_argument("--round_size", type=int, default=20)
    ap.add_argument("--keepers",    type=int, default=10)
    ap.add_argument("--mut_flips",  type=int, default=3)
    ap.add_argument("--rounds",     type=int, default=5)
    ap.add_argument("--steps",      type=int, default=500)
    ap.add_argument("--boardsize",  type=int, default=200)
    ap.add_argument("--copy_tol",   type=int, default=10)
    ap.add_argument("--sustain_extra", type=int, default=100)
    ap.add_argument("--workers",    type=int, default=mp.cpu_count())
    ap.add_argument("--tile_output", type=str, default="",
                    help="Path to save a tiled board of top patterns (optional).")
    ap.add_argument("--tile_n", type=int, default=0,
                    help="How many top patterns to tile (0 = skip).")
    ap.add_argument("--tile_margin", type=int, default=4,
                    help="Dead-cell spacing between tiled patterns.")
    return ap.parse_args()

if __name__ == "__main__":
    mp.freeze_support()
    evolve(get_args())