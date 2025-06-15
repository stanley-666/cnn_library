CC = gcc
CFLAGS = -Wall -O2 -lm
TARGET = main
SRCS = main.c cnn.c cnn_utils.c cnn_layer.c cnn_train.c
OBJS = $(SRCS:.c=.o)

all: $(TARGET) clean

$(TARGET): $(OBJS)
	$(CC) -o $(TARGET) $(OBJS) $(CFLAGS)

%.o: %.c
	$(CC) -c $< -o $@ $(CFLAGS)

clean:
	rm -f $(OBJS)
