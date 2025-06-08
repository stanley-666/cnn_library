#ifndef CNN_H
#define CNN_H
typedef enum { RELU, SIGMOID, TANH, LEAKY_RELU, NONE } ActivationType;
typedef enum { CONV, POOL, FC } LayerType;
typedef enum { MAX_POOL, AVG_POOL } PoolType;
#define MAX_INPUT_CHANNEL  32 // 你模型最大 in channel
#define MAX_K             5   // 最大 kernel size
typedef struct {
    int filterSize, stride, padding, numFilters;
    const void *weights;
    float *bias;
    int sizeofWeights; // 用於計算梯度
    int sizeofBias; // 用於計算梯度
} ConvParams;

typedef struct {
    int poolSize, stride;
    PoolType type;
} PoolParams;

typedef struct {
    int inputSize, outputSize;
    float *weights, *bias;
    int sizeofWeights; // 用於計算梯度
    int sizeofBias; // 用於計算梯度
} FCParams;

typedef struct Layer {
    LayerType type;
    ActivationType activation;
    union {
        ConvParams conv;
        PoolParams pool;
        FCParams fc;
    } params;
    int inputWidth, inputHeight, inputChannels;
    int outputWidth, outputHeight, outputChannels;
    float *input, *output, *deltas;
    float *weightGrads, *biasGrads;
    struct Layer *prev, *next;
} Layer;

typedef struct {
    Layer *firstLayer, *lastLayer;
    int numLayers;
} CNN;

CNN *createCNN();
void freeCNN(CNN *net);
void printCNN(CNN *net);
float *forward(CNN *net, float *input);
void backward(CNN *net, float *target);
void updateWeights(CNN *net, float learningRate);

#endif
