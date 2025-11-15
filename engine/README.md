# SuckFish

Rust based handmade chess engine with an optional NNUE (efficiently updatable neural network) evaluation.

## Installation

### Start

```bash
# If rust is not yet installed:
curl https://sh.rustup.rs -sSf | sh
# Don't forget to activate source cargo env

# If cuda is not yet installed and it's debian:
wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update
apt-get -y install cuda-toolkit-12-8

# Libtorch
wget "https://download.pytorch.org/libtorch/cu128/libtorch-shared-with-deps-2.9.0%2Bcu128.zip"
mv libtorch* libtorch.zip
unzip libtorch.zip
rm libtorch.zip

# Now clone the repository
git clone https://github.com/Undecember/suckfish
cd suckfish

# Setup python venv
python3 -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### Env var setup
Append to `.bashrc`
```
export PATH="/usr/local/cuda-12.8/bin${PATH:+:${PATH}}"
export LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export LIBTORCH="/home/libtorch"
export PATH="$LIBTORCH/bin${PATH:+:${PATH}}"
export LD_LIBRARY_PATH="$LIBTORCH/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
```

### Build binaries

```bash
cargo build --release
cp ./target/release/train_nnue ./
cp ./target/release/suckfish ./
cargo clean
```

### Download training data

Fetch drive link from https://drive.google.com/file/d/1Z6dAszKEsGk9WBjq3G8wnMyNOcMWOCRu/view?usp=drive_link

### Official Stockfish dataset

If you prefer to start from the upstream Stockfish NNUE dumps, follow
[`docs/stockfish_dataset.md`](docs/stockfish_dataset.md) for the download and
sampling steps. It shows how to fetch one of the monthly `.binpack` archives
from Hugging Face and convert it into a ~10M row CSV that `train_nnue` can read.

## Usage

### Training steps

```bash
mkdir nnue
# Epoch 1
./train_nnue --csv dataset.csv --output nnue/epoch-1.ot --learning-rate 0.1
# Epoch 2
./train_nnue --csv dataset.csv --init-weights nnue/epoch-1.ot --output nnue/epoch-2.ot --learning-rate 0.01
# Epoch 3~
./train_nnue --csv dataset.csv --init-weights nnue/epoch-2.ot --output nnue/epoch-3.ot
```

### Main Engine

```bash
./suckfish -h
```
