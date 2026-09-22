#!/usr/bin/env bash
# Compatibility entrypoint for the two serving benchmark drivers.
set -u

DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

case "${1:-}" in
  fair|max)
    exec "$DIR/bench-serving-capacity.sh" "$@"
    ;;
  native|packed)
    exec "$DIR/bench-serving-image.sh" "$@"
    ;;
  --help|-h)
    cat <<'EOF'
usage: bench-serving.sh <fair|max> <ctx> [C_fair]
       bench-serving.sh <native|packed> <in> <out> <concurrency>

fair/max use the local Docker-container capacity driver.
native/packed use the standalone production-image driver used by Modal.
EOF
    ;;
  *)
    echo "usage: $0 <fair|max> <ctx> [C_fair]" >&2
    echo "       $0 <native|packed> <in> <out> <concurrency>" >&2
    exit 2
    ;;
esac
