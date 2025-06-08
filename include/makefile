CC = gcc
CFLAGS = -Wall -O2 -lm
SRCDIR = src
OBJDIR = obj
TARGET = main

SRCS = $(wildcard $(SRCDIR)/*.c)
OBJS = $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) -o $@ $^ $(CFLAGS)
	@rm -f $(OBJS)
	@echo "Build complete, object files cleaned."

%.o: %.c
	$(CC) -c $< -o $@ $(CFLAGS)

clean:
	rm -f $(OBJS) $(TARGET)
.PHONY: all clean
