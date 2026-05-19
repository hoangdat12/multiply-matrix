CC = gcc
CFLAGS = -Wall -g 
TARGET = fftImplement.exe

SRC = fftImplement.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) -lm

clean:
	rm -f $(TARGET)