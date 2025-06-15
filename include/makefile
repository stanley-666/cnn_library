
# Host build
CC_x86 = gcc
CFLAGS_x86 = -Wall -O2
LDFLAGS_x86 = -lm
# RISC-V build
CC_riscv = riscv64-unknown-elf-gcc
CFLAGS_riscv = -std=gnu99 -O2 -fno-common -fno-builtin-printf -Wall -specs=htif_nano.specs
LDFLAGS_riscv = -static -specs=htif_nano.specs -lm -u _printf_float

# dir
SRCDIR = src
OBJDIR = obj
TARGET = main

SRCS = $(wildcard $(SRCDIR)/*.c)
OBJS = $(SRCS:.c=.o)

# 預設編譯 x86
all: $(TARGET)

# x86 build
$(TARGET): $(OBJS)
	$(CC_x86) -o $@ $^ $(LDFLAGS_x86)
	@rm -f $(OBJS)
	@echo "Build complete (x86), object files cleaned."

%.o: %.c
	$(CC_x86) -c $< -o $@ $(CFLAGS_x86)

# riscv build
$(TARGET).riscv: $(SRCS)
	$(CC_riscv) $(CFLAGS_riscv) -c $(SRCDIR)/main.c -o main.o
	$(CC_riscv) $(CFLAGS_riscv) -c $(SRCDIR)/cnn.c -o cnn.o
	$(CC_riscv) $(CFLAGS_riscv) -c $(SRCDIR)/cnn_utils.c -o cnn_utils.o
	$(CC_riscv) $(CFLAGS_riscv) -c $(SRCDIR)/cnn_layer.c -o cnn_layer.o
	$(CC_riscv) $(CFLAGS_riscv) -c $(SRCDIR)/cnn_train.c -o cnn_train.o
	$(CC_riscv) main.o cnn.o cnn_utils.o cnn_layer.o cnn_train.o -o $@ $(LDFLAGS_riscv)
	@rm -f main.o cnn.o cnn_utils.o cnn_layer.o cnn_train.o
	@echo "Build complete (RISC-V), object files cleaned."

clean:
	rm -f $(OBJS) $(TARGET) *.o $(TARGET).riscv
.PHONY: all clean
