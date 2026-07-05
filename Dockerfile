# syntax=docker/dockerfile:1

# ---- Build stage ----
FROM rust:1.89-slim AS builder
WORKDIR /app

# Cache dependencies first.
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY crates ./crates
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/app/target \
    cargo build --release -p reranklab-node && \
    cp target/release/reranklab /usr/local/bin/reranklab

# ---- Runtime stage ----
FROM debian:bookworm-slim AS runtime
RUN groupadd --gid 10001 rank && \
    useradd --uid 10001 --gid rank --no-create-home --shell /usr/sbin/nologin rank
COPY --from=builder /usr/local/bin/reranklab /usr/local/bin/reranklab
USER 10001:10001
EXPOSE 8080
ENV RERANKLAB_BIND=0.0.0.0:8080 RUST_LOG=info,reranklab=debug
ENTRYPOINT ["/usr/local/bin/reranklab"]
CMD ["serve"]
