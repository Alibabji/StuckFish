from .utils import chess_manager, GameContext
from chess import Move
import random
import time, os, subprocess

setup = subprocess.Popen(
    ['sh', '-c', './scripts/setup.sh'],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
    text=True,
    bufsize=1,
)
for line in setup.stdout:
    if not line:
        break
    print(line, end='', flush=True)
retcode = setup.wait()
if retcode not in [0, 2]:
    raise RuntimeError(f"setup failed with exit code {retcode}")

LIBTORCH = os.path.join(os.getcwd(), "libtorch")
os.environ["LIBTORCH"] = LIBTORCH
# PATH
old_path = os.environ.get("PATH", "")
os.environ["PATH"] = f"{LIBTORCH}/bin" + (f":{old_path}" if old_path else "")
# LD_LIBRARY_PATH
old_ld = os.environ.get("LD_LIBRARY_PATH", "")
os.environ["LD_LIBRARY_PATH"] = f"{LIBTORCH}/lib" + (f":{old_ld}" if old_ld else "")

engine = subprocess.Popen(
    [os.path.join(os.getcwd(), 'suckfish'), '--nnue-path', 'nnue.ot'],
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
