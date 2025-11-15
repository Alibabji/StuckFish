from .utils import chess_manager, GameContext
from chess import Move
import random
import time, os, subprocess

os.system('./scripts/setup.sh')
while not os.path.isdir("libtorch"): time.sleep(0.1)
engine = subprocess.Popen(
    ['./suckfish', '--nnue-path', 'nnue.ot'],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
    text=True,
    bufsize=1,
)
print('============== Done setup ==============')

@chess_manager.entrypoint
def test_func(ctx: GameContext):
    fen = ctx.board.fen()
    engine.stdin.write(f'go {ctx.timeLeft} {fen}\n')
    engine.stdin.flush()
    uci = engine.stdout.readline().strip()
    print(uci)
    move = Move.from_uci(uci)
    return move


@chess_manager.reset
def reset_func(ctx: GameContext):
    engine.stdin.write(f'newgame\n')
    engine.stdin.flush()
    while engine.stdout.readline().strip() != 'newgame ready':
        time.sleep(0.01)
    pass
