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

echo "Checking libtorch"
if [ -d "./libtorch" ]; then
    echo "libtorch is already installed."
else
    echo "libtorch is not installed. Now installing..."
    wget -q "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.9.0%2Bcpu.zip"
    mv libtorch*.zip libtorch.zip
    rm -rf ./libtorch
    unzip libtorch.zip
    rm libtorch.zip
    echo "libtorch is now installed."
fi

export LIBTORCH="$(pwd)/libtorch"
export PATH="$LIBTORCH/bin${PATH:+:${PATH}}"
export LD_LIBRARY_PATH="$LIBTORCH/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

cd engine
. "$HOME/.cargo/env"
cargo build --release --bin suckfish
cp target/release/suckfish ../
