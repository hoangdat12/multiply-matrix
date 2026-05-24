CC      := gcc
BUILD   := build
TARGET  := $(BUILD)/multiplyMatrix
SRC     := multiplyMatrix.c

# Flags
CFLAGS  := -O3 -mavx2 -march=native -Wall -Wextra
LDFLAGS := -lpthread

.PHONY: all run clean

all: $(TARGET)

$(BUILD):
	mkdir -p $(BUILD)

$(TARGET): $(SRC) | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(BUILD)