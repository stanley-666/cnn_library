#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include "cnn.h"
#include "cnn_utils.h"   // safeMalloc
#include "cnn_layer.h"   // forwardConv, etc.
#include "cnn_train.h"   // backwardFC, updateWeightsFC, etc.

extern size_t g_total_allocated;
// 創建新的CNN and initialize it
CNN *createCNN()
{
    CNN *net = (CNN *)safeMalloc("CNN net", sizeof(CNN));
    net->firstLayer = NULL;
    net->lastLayer = NULL;
    net->numLayers = 0;
    return net;
}

void freeCNN(CNN *net)
{
    Layer *currentLayer = net->firstLayer;

    while (currentLayer != NULL)
    {
        Layer *nextLayer = currentLayer->next;

        // 釋放層特定資源
        if (currentLayer->type == CONV)
        {
            //free(currentLayer->params.conv.weights);
            free(currentLayer->params.conv.bias);
        }
        else if (currentLayer->type == FC)
        {
            //free(currentLayer->params.fc.weights);
            free(currentLayer->params.fc.bias);
        }

        free(currentLayer->output);
        free(currentLayer);

        currentLayer = nextLayer;
    }

    free(net);
}

void printCNN(CNN *net)
{
    printf("\n=== CNN Architecture: ===\n"); // cnn架構
    Layer *currentLayer = net->firstLayer;
    int layerNum = 1;

    while (currentLayer != NULL)
    {
        printf("Layer %d: ", layerNum++);

        switch (currentLayer->type)
        {
        case CONV:
            printf("Convolution Layer\n");
            printf("  filterSize: %d\n", currentLayer->params.conv.filterSize);
            printf("  numFilters: %d\n", currentLayer->params.conv.numFilters);
            printf("  stride: %d\n", currentLayer->params.conv.stride);
            printf("  padding: %d\n", currentLayer->params.conv.padding);
            break;
        case POOL:
            printf("%s Pooling Layer\n",
                   currentLayer->params.pool.type == MAX_POOL ? "Max" : "Average");
            printf("  poolSize: %d\n", currentLayer->params.pool.poolSize);
            printf("  stride: %d\n", currentLayer->params.pool.stride);
            break;
        case FC:
            printf("Fully Connected Layer\n");
            printf("  inputSize: %d\n", currentLayer->params.fc.inputSize);
            printf("  outputSize: %d\n", currentLayer->params.fc.outputSize);
            break;
        }

        printf("  Activation function: ");
        switch (currentLayer->activation)
        {
        case RELU:
            printf("ReLU\n");
            break;
        case SIGMOID:
            printf("Sigmoid\n");
            break;
        case TANH:
            printf("Tanh\n");
            break;
        case LEAKY_RELU:
            printf("Leaky ReLU\n");
            break;
        case NONE:
            printf("None\n");
            break;
        }

        printf("  Input size: %d x %d x %d\n",
               currentLayer->inputWidth, currentLayer->inputHeight, currentLayer->inputChannels);
        printf("  Output size: %d x %d x %d\n",
               currentLayer->outputWidth, currentLayer->outputHeight, currentLayer->outputChannels);
        printf("\n");

        currentLayer = currentLayer->next;
    }
    printf("=========================\n");
}

float *forward(CNN *net, float *input)
{
    Layer *currentLayer = net->firstLayer;
    float *currentInput = input; // save the input to the first layer
    float *flattenedInput = NULL;

    while (currentLayer != NULL)
    {
        fprintf(stderr, "[Memory Message] Total dynamically allocate memory size : %lu bytes\n", g_total_allocated);
        currentLayer->input = currentInput;
        // 根據層類型進行處理
        switch (currentLayer->type)
        {
        case CONV:
            forwardConv(currentLayer, currentInput);
            break;
        case POOL:
            forwardPool(currentLayer, currentInput);
            break;
        case FC:
            // 如果前一層不是FC，把輸入變1-dim
            if (currentLayer != net->firstLayer && 
                (currentLayer->inputWidth * currentLayer->inputHeight * currentLayer->inputChannels != 
                 currentLayer->params.fc.inputSize)) {
                
                if (flattenedInput) {
                    free(flattenedInput);
                }
                flattenedInput = flattenOutput(currentLayer->next && currentLayer->next->type == FC ? 
                                           currentLayer : net->lastLayer);
                forwardFC(currentLayer, flattenedInput);
            } else {
                forwardFC(currentLayer, currentInput);
            }
            break;
        }

        // 更新到下一層
        currentInput = currentLayer->output;
        currentLayer = currentLayer->next;
    }

    // 釋放臨時要flatten buffer
    if (flattenedInput) {
        free(flattenedInput);
    }

    // 返回最後一層的輸出
    return net->lastLayer->output;
}

void backward(CNN* net, float* target) {
    Layer* layer = net->lastLayer;
    while (layer) {
        if (layer->type == FC) backwardFC(layer, target);
        else if (layer->type == POOL) { backwardPool(layer); }
        else if (layer->type == CONV) { backwardConv(layer); }
        layer = layer->prev;
    }
}

void updateWeights(CNN* net, float learningRate) {
    Layer* layer = net->firstLayer;
    while (layer) {
        if (layer->type == FC) {
            updateWeightsFC(layer, learningRate);
        }
        else if (layer->type == POOL) {
            ;// 池化層不需要更新權重
        }
        else if (layer->type == CONV) {
            updateWeightsConv(layer, learningRate);
        }
        layer = layer->next;
    }
}