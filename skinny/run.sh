#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -z "${PYTHON:-}" ]]; then
    if [[ -x "$ROOT_DIR/.venv313/bin/python" ]]; then
        PYTHON="$ROOT_DIR/.venv313/bin/python"
    else
        PYTHON="python3"
    fi
fi
CXX="${CXX:-g++}"
CASE="${1:-all}"

MAX_CORRELATION="${MAX_CORRELATION:-1000}"
MAX_TRAILS="${MAX_TRAILS:-30}"
SEARCH_TIME="${SEARCH_TIME:-3600}"
GUROBI_THREADS="${GUROBI_THREADS:-16}"
CPP_THREADS="${CPP_THREADS:-36}"

case "$CASE" in
    all|d18|d19|d20|d21|d22|d23)
        ;;
    *)
        echo "Usage: bash run.sh [all|d18|d19|d20|d21|d22|d23]" >&2
        exit 2
        ;;
esac

require_positive_integer() {
    local name="$1"
    local value="$2"
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "$name must be a positive integer; got: $value" >&2
        exit 2
    fi
}

require_positive_integer MAX_CORRELATION "$MAX_CORRELATION"
require_positive_integer MAX_TRAILS "$MAX_TRAILS"
require_positive_integer SEARCH_TIME "$SEARCH_TIME"
require_positive_integer CPP_THREADS "$CPP_THREADS"
require_positive_integer GUROBI_THREADS "$GUROBI_THREADS"

if ! command -v "$PYTHON" >/dev/null 2>&1 && [[ ! -x "$PYTHON" ]]; then
    echo "Python interpreter not found: $PYTHON" >&2
    exit 1
fi

if ! command -v "$CXX" >/dev/null 2>&1; then
    echo "C++ compiler not found: $CXX" >&2
    exit 1
fi

GUROBI_VERSION="$("$PYTHON" -c 'import gurobipy as gp; print(gp.__version__)')"
if [[ "$GUROBI_VERSION" != "13.0.1" ]]; then
    echo "This code requires gurobipy 13.0.1; found: $GUROBI_VERSION" >&2
    exit 1
fi

"$PYTHON" -c 'import numpy, pybind11, tqdm'

EXT_SUFFIX="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
read -r -a PYBIND_INCLUDES <<< "$("$PYTHON" -m pybind11 --includes)"

echo "Compiling C++ helper with $CPP_THREADS physical-core threads..."
"$CXX" -O3 -shared -std=c++17 -fPIC -pthread \
    -DNUM_THREADS="$CPP_THREADS" \
    "${PYBIND_INCLUDES[@]}" \
    ./helper/search_trails_parallel.cpp \
    -o "./helper/skinny_cpp${EXT_SUFFIX}"

"$PYTHON" -c 'import helper.skinny_cpp'

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RESULT_DIR="${RESULT_DIR:-$ROOT_DIR/results/${CASE}_${RUN_STAMP}}"
mkdir -p "$RESULT_DIR"

echo "Case: $CASE"
echo "Python: $PYTHON"
echo "gurobipy: $GUROBI_VERSION"
echo "C++ threads: $CPP_THREADS"
echo "Gurobi threads: $GUROBI_THREADS"
echo "Maximum trails: $MAX_TRAILS"
echo "Time limit per Gurobi solve: $SEARCH_TIME seconds"
echo "Maximum correlation exponent: $MAX_CORRELATION"
echo "Results: $RESULT_DIR"

run_case() {
    local input_file="$1"
    local sbox_size="$2"
    local tk="$3"
    local rounds="$4"
    local key_size="$5"
    local output_name="$6"
    local output_file="$RESULT_DIR/$output_name.txt"

    echo "Running $output_name..."
    "$PYTHON" -u main.py "$input_file" \
        "$sbox_size" "$tk" "$rounds" "$key_size" \
        "$MAX_CORRELATION" "$MAX_TRAILS" "$SEARCH_TIME" "$GUROBI_THREADS" \
        2>&1 | tee "$output_file"
}

if [[ "$CASE" == "all" || "$CASE" == "d18" ]]; then
    run_case ./data/trail_TK2_R5.txt 4 2 5 2 trail_TK2_R5
fi
if [[ "$CASE" == "all" || "$CASE" == "d19" ]]; then
    run_case ./data/trail_SK_R7.txt 4 1 7 1 trail_SK_R7
fi
if [[ "$CASE" == "all" || "$CASE" == "d20" ]]; then
    run_case ./data/trail_TK2_R13.txt 4 2 13 2 trail_TK2_R13
fi
if [[ "$CASE" == "all" || "$CASE" == "d21" ]]; then
    run_case ./data/trail_TK3_R15.txt 4 3 15 3 trail_TK3_R15
fi

if [[ "$CASE" == "all" || "$CASE" == "d22" || "$CASE" == "d23" ]]; then
    if [[ ! -f ./data/quasidifferential_sbox8.npy ]]; then
        echo "Warning: the first 128-bit case will create an approximately 32 GiB qdtm file." >&2
    fi
fi
if [[ "$CASE" == "all" || "$CASE" == "d22" ]]; then
    run_case ./data/trail128_TK1_R14.txt 8 1 14 1 trail128_TK1_R14
fi
if [[ "$CASE" == "all" || "$CASE" == "d23" ]]; then
    run_case ./data/trail128_TK2_R16.txt 8 2 16 2 trail128_TK2_R16
fi

echo "Completed successfully. Results are in: $RESULT_DIR"
