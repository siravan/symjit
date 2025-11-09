# How to Compile Symjit from Source?

A quick guide on how to compile *Symjit* from source. As of version 2.7, you need to build from source if you want to use *Symjit* on a RISC-V machine.

## Prerequisites

1. You nead working git, Python (>3.10) and Rust toolchains. The easiest way to install Rust is to use [rustup](https://rustup.rs/) Rust toolchain installer.

2. In addition, you need to install setuptools and setuptools-rust (see [setuptools](https://setuptools.pypa.io/)).

## Download Symjit

Clone *Symjit* by running

```bash
git clone https://github.com/siravan/symjit
```

## Install Symjit

1. Go to symjit directory (created after cloning).

2. Optionally (but preferably) make a new Python environment. For example,

```bash
python -m venv .symjit
source .symjit/bin/activate
```

3. Compile and install Symjit by running `python -m pip install -e .` from Symjit root directory.

4. Test the installation. For example, try `python examples/test.py`.
