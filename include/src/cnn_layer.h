// cnn_layer.h
#ifndef CNN_LAYER_H
#define CNN_LAYER_H

void addConvLayer(CNN *net, int filterSize, int numFilters, int stride, int padding,
                  ActivationType activation, int inputWidth, int inputHeight, int inputChannels);

void addPoolLayer(CNN *net, int poolSize, int stride, PoolType poolType,
                  int inputWidth, int inputHeight, int inputChannels);

void addFCLayer(CNN *net, int outputSize, ActivationType activation, int inputSize);

void forwardConv(Layer *layer, float *input);
void forwardPool(Layer *layer, float *input);
void forwardFC(Layer *layer, float *input);

#endif
