#!/bin/bash

# =========================
# CONFIG
# =========================

SRC="../multiplyMatrix.c"
OUTDIR="../build"
LOGDIR="../logs"

RUNS=10

BASE_FLAGS="-O3 -mavx2 -march=native -lpthread"

# Cố định
K_DIM=4
N_DIM=32
KC_VALUE=4
NC_VALUE=32
NUM_THREAD=8

# Sweep Mc
MC_LIST=(64 128 192)

# =========================
# SETUP
# =========================

mkdir -p "$OUTDIR"
mkdir -p "$LOGDIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SWEEP_LOG="$LOGDIR/sweep_mc_${TIMESTAMP}.log"

echo "Sweep Mc started at $(date)" | tee "$SWEEP_LOG"
echo "Fixed: Kc=$KC_VALUE  Nc=$NC_VALUE  threads=$NUM_THREAD" | tee -a "$SWEEP_LOG"
echo "Sweep: Mc = ${MC_LIST[*]}" | tee -a "$SWEEP_LOG"
echo "==========================================" | tee -a "$SWEEP_LOG"
echo "" | tee -a "$SWEEP_LOG"

# Label khớp với printf trong multiplyMatrix.c
FUNC_LABELS=(
    "1. Basic ijk"
    "2. Basic ikj"
    "3. Block ikj"
    "4. AVX2 No-Block ikj"
    "5. AVX2 Block ikj"
    "6. ThreadPool AVX2+Block"
    "7. ThreadPool AVX2 NoBl"
)

FUNC_DISPLAY=(
    "ijk"
    "ikj"
    "Block"
    "AVX2 NoBl"
    "AVX2 Blk"
    "Pool+Blk"
    "Pool NoBl"
)

declare -A FINAL_AVG   # key: "Mc___label"

# =========================
# BUILD + BENCHMARK
# =========================

for MC in "${MC_LIST[@]}"; do
    BIN="$OUTDIR/sweep_mc${MC}"

    echo "================================="
    echo "BUILD: Mc=$MC  Kc=$KC_VALUE  Nc=$NC_VALUE  threads=$NUM_THREAD"
    echo "================================="

    gcc $SRC $BASE_FLAGS \
        -DK_DIM=$K_DIM \
        -DN_DIM=$N_DIM \
        -DMC_VALUE=$MC \
        -DKC_VALUE=$KC_VALUE \
        -DNC_VALUE=$NC_VALUE \
        -DNUM_THREAD=$NUM_THREAD \
        -o "$BIN"

    if [ $? -ne 0 ]; then
        echo "[ERROR] Build failed cho Mc=$MC"
        for label in "${FUNC_LABELS[@]}"; do
            FINAL_AVG["${MC}___${label}"]="N/A"
        done
        continue
    fi

    # Warm-up
    "$BIN" > /dev/null 2>&1

    declare -A TOTAL
    for label in "${FUNC_LABELS[@]}"; do
        TOTAL["$label"]="0"
    done

    for ((i=1; i<=RUNS; i++)); do
        OUTPUT=$("$BIN")
        for label in "${FUNC_LABELS[@]}"; do
            TIME=$(echo "$OUTPUT" \
                   | grep "^\s*$(echo "$label" | sed 's/[.+]/\\&/g')" \
                   | grep -oP '[\d]+\.[\d]+(?=\s*ms)' \
                   | head -1)
            if [ -z "$TIME" ]; then TIME="0"; fi
            TOTAL["$label"]=$(awk "BEGIN {printf \"%.6f\", ${TOTAL[$label]} + $TIME}")
        done
        echo -n "."
    done
    echo " done (Mc=$MC)"

    for label in "${FUNC_LABELS[@]}"; do
        AVG=$(awk "BEGIN {printf \"%.4f\", ${TOTAL[$label]} / $RUNS}")
        FINAL_AVG["${MC}___${label}"]=$AVG
    done

    echo ""
done

# =========================
# SUMMARY TABLE
# =========================

print_table() {
    # Header
    printf "%-6s" "Mc"
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

    # Rows – mỗi row là 1 giá trị Mc
    for MC in "${MC_LIST[@]}"; do
        printf "%-6s" "$MC"
        for label in "${FUNC_LABELS[@]}"; do
            KEY="${MC}___${label}"
            VAL="${FINAL_AVG[$KEY]}"
            [ -z "$VAL" ] && VAL="N/A"
            printf " | %-12s" "${VAL} ms"
        done
        printf "\n"
    done
}

echo ""
echo "=========================================="
echo "   SWEEP Mc — avg over $RUNS runs (ms)"
echo "   Kc=$KC_VALUE  Nc=$NC_VALUE  threads=$NUM_THREAD"
echo "=========================================="
print_table
echo "=========================================="

{
    echo ""
    echo "=========================================="
    echo "   SWEEP Mc — avg over $RUNS runs (ms)"
    echo "   Kc=$KC_VALUE  Nc=$NC_VALUE  threads=$NUM_THREAD"
    echo "=========================================="
    print_table
    echo "=========================================="
    echo ""
    echo "Sweep finished at $(date)"
} >> "$SWEEP_LOG"

echo ""
echo "DONE — Log: $SWEEP_LOG"