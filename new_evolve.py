#!/usr/bin/env python3
"""
Evolution search for Life patterns.

Each round:
• Evaluate `round_size` patterns.
• Keep top `keepers` elites.
• Mutate each elite (flip `mut_flips` cells) and refill to `round_size`.
• Optionally save & tile the top N patterns of the final round.

Run example:
    python3 new_evolve.py --round_size 32 --keepers 16 --mut_flips 4 \
                          --rounds 6 --steps 400 --show_n 20 --tile_margin 6
Replay:
    python3 myGOL.py --pattern winner.npy
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

##############################################################################
# Tiling utility
##############################################################################

def tile_patterns(patterns, margin=4):
    if not patterns: raise ValueError("no patterns")
    pr, pc = patterns[0].shape
    n = len(patterns)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n/cols))
    H = rows*pr + (rows-1)*margin
    W = cols*pc + (cols-1)*margin
    board = np.zeros((H, W), dtype=bool)
    k=0
    for r in range(rows):
        for c in range(cols):
            if k>=n: break
            top  = r*(pr+margin)
            left = c*(pc+margin)
            board[top:top+pr, left:left+pc] = patterns[k]
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
        for r,p in enumerate(elites,1):
            np.save(f"round{round_no:02d}_elite{r:02d}.npy",p)

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

    if a.show_n>0:
        idx=np.argsort(finals)[::-1][:a.show_n]
        board=tile_patterns([pop[i] for i in idx], margin=a.tile_margin)
        np.save("show.npy",board)
        print(f"Top {a.show_n} tiled → show.npy")

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
    ap.add_argument("--show_n",     type=int, default=0,
                    help="Tile top N patterns into show.npy (0=skip)")
    ap.add_argument("--tile_margin", type=int, default=4)
    return ap.parse_args()

if __name__ == "__main__":
    mp.freeze_support()
    evolve(get_args())