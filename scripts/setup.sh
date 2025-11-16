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

echo test1
cd engine
echo test2
# Use cargo directly from rustup's default install location to avoid sourcing.
CARGO_BIN="$HOME/.cargo/bin/cargo"
echo $CARGO_BIN
if [ ! -x "$CARGO_BIN" ]; then
    CARGO_BIN="$(command -v cargo || true)"
fi
if [ ! -x "$CARGO_BIN" ]; then
    echo "cargo binary not found; ensure rustup installed correctly."
    exit 1
fi
"$CARGO_BIN" build --release --bin suckfish
cp target/release/suckfish ../
