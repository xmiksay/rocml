export CARGO_BUILD_JOBS := 4

.PHONY: build test test-unit test-integration lint fmt clean

build:
	cargo build --workspace

test:
	cargo test --workspace

test-unit:
	cargo test --workspace --lib

test-integration:
	cargo test --workspace --test '*'

lint:
	cargo clippy --workspace --all-targets -- -D warnings
	cargo fmt --all -- --check

fmt:
	cargo fmt --all

clean:
	cargo clean
