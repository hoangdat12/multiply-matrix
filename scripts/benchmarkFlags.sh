#!/bin/bash

# =========================
# CONFIG
# =========================

SRC="../multiplyMatrix.c"
OUTDIR="../build"
LOGDIR="../logs"

RUNS=10

BASE_FLAGS="-mavx2 -march=native -lpthread"

FLAGS=(
    "-O0"
    "-O1"
    "-O2"
    "-O3"
)

# Prefix khớp với printf trong file C
FUNC_LABELS=(
    "1. Basic ijk"
    "2. Basic ikj"
    "3. Block ikj"
    "4. AVX2 No-Block ikj"
    "5. AVX2 Block ikj"
    "6. ThreadPool AVX2+Block"
    "7. ThreadPool AVX2 NoBl"
)

# Tên hiển thị trong bảng tổng kết
FUNC_DISPLAY=(
    "ijk"
    "ikj"
    "Block ikj"
    "AVX2 NoBl"
    "AVX2 Block"
    "Pool+Block"
    "Pool NoBl"
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

    gcc $SRC $FLAG $BASE_FLAGS -o $BIN

    if [ $? -ne 0 ]; then
        echo "Build failed: $FLAG"
        for label in "${FUNC_LABELS[@]}"; do
            FINAL_AVG["${FLAG}___${label}"]="N/A"
        done
        continue
    fi

    # Warm-up run (không tính)
    $BIN > /dev/null 2>&1

    declare -A TOTAL
    for label in "${FUNC_LABELS[@]}"; do
        TOTAL["$label"]="0"
    done

    for ((i=1; i<=RUNS; i++)); do
        OUTPUT=$($BIN)
        for label in "${FUNC_LABELS[@]}"; do
            # Output format: "1. Basic ijk               X.XXXX ms  ..."
            # grep dòng bắt đầu bằng label (có thể có khoảng trắng đầu dòng)
            # lấy số float đầu tiên trước chữ "ms"
            TIME=$(echo "$OUTPUT" | grep "^\s*$(echo "$label" | sed 's/[.+]/\\&/g')" \
                   | grep -oP '[\d]+\.[\d]+(?=\s*ms)' | head -1)
            if [ -z "$TIME" ]; then TIME="0"; fi
            TOTAL["$label"]=$(awk "BEGIN {printf \"%.6f\", ${TOTAL[$label]} + $TIME}")
        done
        echo -n "."
    done
    echo " done"

    for label in "${FUNC_LABELS[@]}"; do
        AVG=$(awk "BEGIN {printf \"%.4f\", ${TOTAL[$label]} / $RUNS}")
        FINAL_AVG["${FLAG}___${label}"]=$AVG
    done

    echo ""
done

# =========================
# SUMMARY TABLE
# =========================

print_table() {
    # Header
    printf "%-6s" "Flag"
    for display in "${FUNC_DISPLAY[@]}"; do
        printf " | %-12s" "$display"
    done
    printf "\n"

    # Separator
    printf "%s" "------"
    for display in "${FUNC_DISPLAY[@]}"; do
        printf "%s" "-+--------------"
    done
    printf "\n"

    # Rows
    for FLAG in "${FLAGS[@]}"; do
        printf "%-6s" "$FLAG"
        for label in "${FUNC_LABELS[@]}"; do
            KEY="${FLAG}___${label}"
            VAL="${FINAL_AVG[$KEY]}"
            if [ -z "$VAL" ]; then VAL="N/A"; fi
            printf " | %-12s" "${VAL} ms"
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