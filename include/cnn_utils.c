#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cnn_utils.h"

void* safeMalloc(size_t size) {
    void* ptr = malloc(size);
    if (ptr == NULL) {
        fprintf(stderr, "Memory allocation failed! Requested size: %zu bytes\n", size);
        exit(EXIT_FAILURE);
    }
    return ptr;
}

void* safeCalloc(size_t count, size_t size) {
    void* ptr = calloc(count, size); // Allocates memory and initializes it to 0
    if (ptr == NULL) {
        fprintf(stderr, "Memory allocation failed! Requested size: %zu bytes\n", count * size);
        exit(EXIT_FAILURE);
    }
    return ptr;
}

float activate(float x, ActivationType type) {
    switch (type) {
        case RELU: return x > 0 ? x : 0;
        case SIGMOID: return 1.0f / (1.0f + expf(-x));
        case TANH: return tanhf(x);
        case LEAKY_RELU: return x > 0 ? x : 0.01f * x;
        case NONE: default: return x;
    }
}


float activateDerivative(float x, ActivationType type) {
    switch (type) {
    case RELU: return x > 0 ? 1.0f : 0.0f;
    case SIGMOID: return x * (1.0f - x);
    case TANH: return 1.0f - x * x;
    case LEAKY_RELU: return x > 0 ? 1.0f : 0.01f;
    case NONE: default: return 1.0f;
    }
}

float computeMSE(float *output, float *target, int size) {
    float loss = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = output[i] - target[i];
        loss += diff * diff;
    }
    return loss / size;
}

int validateConvParams(int inputWidth, int inputHeight, int filterSize, int stride, int padding) {
    int outWidth = ((inputWidth + 2 * padding - filterSize) / stride) + 1;
    int outHeight = ((inputHeight + 2 * padding - filterSize) / stride) + 1;
    
    if (outWidth <= 0 || outHeight <= 0) {
        fprintf(stderr, "Error: Invalid convolution parameters resulting in output size %dx%d\n", 
                outWidth, outHeight);
        return 0;
    }
    return 1;
}


int validatePoolParams(int inputWidth, int inputHeight, int poolSize, int stride) {
    int outWidth = (inputWidth - poolSize) / stride + 1;
    int outHeight = (inputHeight - poolSize) / stride + 1;
    
    if (outWidth <= 0 || outHeight <= 0) {
        fprintf(stderr, "Error: Invalid pooling parameters resulting in output size %dx%d\n", 
                outWidth, outHeight);
        return 0;
    }
    return 1;
}

float* flattenOutput(Layer* layer) {
    int totalSize = layer->outputWidth * layer->outputHeight * layer->outputChannels;
    float* flattenedOutput = (float*)safeCalloc(totalSize, sizeof(float));
    
    memcpy(flattenedOutput, layer->output, totalSize * sizeof(float));
    return flattenedOutput;
}