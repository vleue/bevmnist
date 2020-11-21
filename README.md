# bevmnist

Running MNIST inference in [bevy](https://github.com/bevyengine/bevy) with [tract](https://github.com/sonos/tract). And in wasm!

![demo](https://raw.githubusercontent.com/vleue/bevmnist/main/demo.gif)


## Optional Setup
To install the required target, WASM utils, and HTTP server:
```
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli https
```

## Build in wasm

```
cp -r assets wasm/
cargo build --release --target wasm32-unknown-unknown --no-default-features
wasm-bindgen --no-typescript --out-name bevmnist --out-dir wasm --target web ${CARGO_TARGET_DIR:-target}/wasm32-unknown-unknown/release/bevmnist.wasm
```
and then serve with your favorite http server the `wasm` folder (eg. `cd wasm; http`)
