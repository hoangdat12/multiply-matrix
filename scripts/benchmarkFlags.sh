#!/bin/bash

# =========================
# CONFIG
# =========================

SRC="../multiplyMatrixAMX2.c"
OUTDIR="../build"
LOGDIR="../logs"

RUNS=10

BASE_FLAGS="-mavx2 -march=native"

FLAGS=(
    "-O0"
    "-O1"
    "-O2"
    "-O3"
)

FUNC_LABELS=(
    "1."
    "2."
    "3."
    "4."
    "5."
)

FUNC_DISPLAY=(
    "ijk"
    "ikj"
    "ikj Block"
    "ikj Block AVX2"
    "ikj AVX2"
)

# =========================
# SETUP
# =========================

mkdir -p $OUTDIR
mkdir -p $LOGDIR

SUMMARY_LOG="$LOGDIR/logsbenchmark.log"

echo "Benchmark started at $(date)" > $SUMMARY_LOG
echo "=========================================" >> $SUMMARY_LOG
echo "" >> $SUMMARY_LOG

declare -A FINAL_AVG

# =========================
# BUILD + BENCHMARK
# =========================

for FLAG in "${FLAGS[@]}"; do
    NAME=$(echo $FLAG | tr ' ' '_' | tr '-' '_')
    BIN="$OUTDIR/bm_${NAME}"

    echo "================================="
    echo "BUILD + RUN: $FLAG $BASE_FLAGS"
    echo "================================="
    
    # gcc multiplyMatrixAMX2.c -O3 -mavx2 -march=native -o multiplyMatrixAMX2
    gcc $SRC $FLAG $BASE_FLAGS -o $BIN

    if [ $? -ne 0 ]; then
        echo "Build failed: $FLAG"
        for label in "${FUNC_LABELS[@]}"; do
            FINAL_AVG["${FLAG}___${label}"]="N/A"
        done
        continue
    fi

    # Không hiện output của hàm thực thi ra command line
    $BIN > /dev/null 2>&1

    declare -A TOTAL
    for label in "${FUNC_LABELS[@]}"; do
        TOTAL["$label"]="0"
    done

    for ((i=1; i<=RUNS; i++)); do
        OUTPUT=$($BIN)
        for label in "${FUNC_LABELS[@]}"; do
            TIME=$(echo "$OUTPUT" | grep "^[[:space:]]*${label}" | grep -oP '[\d.]+(?=\s*ms)')
            if [ -z "$TIME" ]; then TIME="0"; fi
            TOTAL["$label"]=$(awk "BEGIN {printf \"%.6f\", ${TOTAL[$label]} + $TIME}")
        done
        echo -n "."
    done
    echo " done"

    for label in "${FUNC_LABELS[@]}"; do
        AVG=$(awk "BEGIN {printf \"%.6f\", ${TOTAL[$label]} / $RUNS}")
        FINAL_AVG["${FLAG}___${label}"]=$AVG
    done

    echo ""
done

# =========================
# SUMMARY TABLE
# =========================

print_table() {
    printf "%-6s" "Flag"
    for display in "${FUNC_DISPLAY[@]}"; do
        printf " | %-16s" "$display"
    done
    printf "\n"

    printf "%s" "------"
    for display in "${FUNC_DISPLAY[@]}"; do
        printf "%s" "-+------------------"
    done
    printf "\n"

    for FLAG in "${FLAGS[@]}"; do
        printf "%-6s" "$FLAG"
        for label in "${FUNC_LABELS[@]}"; do
            KEY="${FLAG}___${label}"
            VAL="${FINAL_AVG[$KEY]}"
            if [ -z "$VAL" ]; then VAL="N/A"; fi
            printf " | %-16s" "${VAL} ms"
        done
        printf "\n"
    done
}

echo ""
echo "========================================="
echo "   SUMMARY TABLE (avg over $RUNS runs, ms)"
echo "========================================="
print_table
echo "========================================="

{
    echo ""
    echo "========================================="
    echo "   SUMMARY TABLE (avg over $RUNS runs, ms)"
    echo "========================================="
    print_table
    echo "========================================="
    echo ""
    echo "Benchmark finished at $(date)"
} >> $SUMMARY_LOG

echo ""
echo "DONE — Logs: $LOGDIR"