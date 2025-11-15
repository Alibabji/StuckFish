echo "Checking rust"
if ! type rustup > /dev/null; then
    echo "Rust is not installed. Now installing..."
    curl https://sh.rustup.rs -sSf --output rust.sh
    chmod +x rust.sh
    ./rust.sh -qy
    rm rust.sh
    echo "Rust is now installed."
else
    echo "Rust is already installed."
fi
. "$HOME/.cargo/env"

echo "Checking CUDA"
if ! type nvcc > /dev/null; then
    echo "CUDA is not installed. Now installing..."
    wget "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb"
    dpkg -i cuda-keyring_1.1-1_all.deb
    apt update
    apt -y install cuda-toolkit-12-8
    apt upgrade -y
    apt autoremove -y
    rm cuda-keyring_1.1-1_all.deb
    echo "CUDA is now installed."
else
    echo "CUDA is already installed."
fi
export PATH="/usr/local/cuda-12.8/bin${PATH:+:${PATH}}"
export LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

echo "Checking libtorch"
if [ -d "./libtorch" ]; then
    echo "libtorch is already installed."
else
    echo "libtorch is not installed. Now installing..."
    wget -q "https://download.pytorch.org/libtorch/cu128/libtorch-shared-with-deps-2.9.0%2Bcu128.zip"
    mv libtorch* libtorch.zip
    rm -rf ./libtorch
    unzip libtorch.zip
    rm libtorch.zip
    echo "libtorch is now installed."
fi

export LIBTORCH="$(pwd)/libtorch"
export PATH="$LIBTORCH/bin${PATH:+:${PATH}}"
export LD_LIBRARY_PATH="$LIBTORCH/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

cd engine
cargo build --release --bin suckfish
cp target/release/suckfish ../
cd ..
