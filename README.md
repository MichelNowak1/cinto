
<p align="center">
    <img src="figures/cinto_logo.jpg" alt="cinto_logo">
</p>

Cinto is a particle transport playground designed to test new ideas and algorithms.

## Requirements

### Rust

<p align="center">
<img src="figures/rust_logo.png" alt="rust_logo" width="300"/>
</p>

cinto is written in [Rust](https://www.rust-lang.org/). The recommended way to install Rust is
through [Rustup](https://rustup.rs/):
```
curl https://sh.rustup.rs -sSf | sh
```

This will install `rustup` as well as the Rust build tool and package manager `cargo`. Once the installation is
complete add Cargo's bin directory (default $HOME/.cargo/bin) to your PATH environment variable:
```
export PATH="$HOME/.cargo/bin:$PATH"
```

cinto uses dependencies which are only available on the Rust `nightly` toolchain. To set it as a default for
this project use `rustup override` in the project directory:
```
rustup override set nightly
```
or define `nightly` as default for your entire system:
```
rustup default nightly
```

### PyNE
<p align="center">
<img src="figures/pyne_logo.png" alt="pyne_logo" width="300"/>
</p>
Cinto cross sections are handled by [PyNE](https://pyne.io/). Since PyNE source code is required to use cinto,
Pyne has been added as a submodule of the current project. To download and build it:

```
cd pyne/
sudo apt-get install libhdf5-dev gcc gfortran
python3 setup.py install --user
python3 scripts/nuc_data_make
```

Cross sections are in ACE format, the folder that contains the evaluations should be put in the environment variable `CINTO_DATA`. 
For exemple:

```
export CINTO_DATA=/home/rustme/data/endf71x/
```

### ROOT
<p align="center">
<img src="figures/root_logo.png" alt="root_logo" width="300"/>
</p>
cinto geometries are handled by the [ROOT data analysis framework](https://root.cern.ch). To install it please
follow the instructions provided [here]( https://root.cern.ch/downloading-root)

## Compilation
```
cargo build
```

By default, the target is compiled in debug mode. To compile in release run:

```
cargo build --release
```

## Run
```
cd examples/bypass
cargo run
```

or 

```
cargo run --release
```

To see backtrace upon runtime errors set the environment variable `RUST_BACKTRACE` to `1` or simply use:
```
RUST_BACKTRACE=1 cargo run
```

For Apple, look at ~/.cargo/confg
add this:

```
[target.x86_64-apple-darwin]
rustflags = [
  "-C", "link-arg=-undefined",
  "-C", "link-arg=dynamic_lookup",
]

[target.aarch64-apple-darwin]
rustflags = [
  "-C", "link-arg=-undefined",
  "-C", "link-arg=dynamic_lookup",
]
```

In ~/.bashrc
``` bash
export OUT_DIR=~/cinto/target
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${OUT_DIR}
```
