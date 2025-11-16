from .utils import chess_manager, GameContext
from chess import Move
import random
import time, os, subprocess

import pathlib
import shutil
import requests
import zipfile

def run_with_out(cmd, **kwargs):
    p = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
        **kwargs,
    )
    print('===Run subprocess', cmd, flush=True)
    for line in p.stdout:
        if not line:
            break
        print(line, end='', flush=True)
    print('===Done subprocess', cmd, flush=True)
    return p.wait()

def run_with_err(cmd, **kwargs):
    p = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        **kwargs,
    )
    print('===Run subprocess', cmd, flush=True)
    for line in p.stderr:
        if not line:
            break
        print(line, end='', flush=True)
    print('===Done subprocess', cmd, flush=True)
    return p.wait()

def download(url, path):
    r = requests.get(url)
    print(path, " downloading...", flush = True)
    with open(path, 'wb') as f:
        f.write(r.content)
    if os.path.isfile(path):
        print(path, "downloaded.")
    else:
        print("FAILED DOWNLOAD", path)
        exit(1)

download('https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.9.0%2Bcpu.zip', 'libtorch.zip')
with zipfile.ZipFile('libtorch.zip', 'r') as zf:
    zf.extractall('.')

env = os.environ.copy()
LIBTORCH = pathlib.Path(os.getcwd()) / 'libtorch'
print(LIBTORCH)
env["LIBTORCH"] = LIBTORCH
old_path = os.environ.get("PATH", "")
env["PATH"] = f"{LIBTORCH}/bin" + (f":{old_path}" if old_path else "")
old_ld = os.environ.get("LD_LIBRARY_PATH", "")
env["LD_LIBRARY_PATH"] = f"{LIBTORCH}/lib" + (f":{old_ld}" if old_ld else "")

print("Libtorch set.")

download('https://sh.rustup.rs', 'rust.sh')
run_with_err(
    ['sh', 'rust.sh', '-qy'],
    env=env
)

home = pathlib.Path(os.environ['HOME'])
env["PATH"] = f'{home / ".cargo/bin"}' + f':{env["PATH"]}'

run_with_err(
    ['cargo', 'build', '--release', '--bin', 'suckfish'],
    env=env,
    cwd='engine',
)

engine = subprocess.Popen(
    ['engine/target/release/suckfish', '--nnue-path', os.path.join(os.getcwd(), 'nnue.ot')],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
    text=True,
    bufsize=1,
    env=env,
)
# for line in engine.stderr:
#     if not line:
#         break
#     print(line, end='', flush=True)
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
