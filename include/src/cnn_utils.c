#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cnn_utils.h"

size_t g_total_allocated = 0; 

void* safeMalloc(char *name, size_t size) {
    void* ptr = malloc(size);
    if (ptr == NULL) {
        fprintf(stderr, "[Memory Message] Memory allocation for %s failed! Requested size: %lu bytes\n", name, size);
        exit(EXIT_FAILURE);
    }
    printf("[Memory Message] Allocating for %s, %lu bytes\n", name, size);
    g_total_allocated += size ;

    return ptr;
}

void* safeCalloc(char *name, size_t count, size_t size) {
    size_t request_size = count * size ;
    void* ptr = calloc(count, size);
    if (ptr == NULL) {
        fprintf(stderr, "[Memory Message] Memory allocation failed! Requested size: %lu bytes\n", count * size);
        exit(EXIT_FAILURE);
    }
    printf("[Memory Message] Allocating for %s, %lu elements, each size %lu bytes, total size: %lu bytes\n", name, count, size, request_size);
    g_total_allocated += request_size ;

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
    float* flattenedOutput = (float*)safeCalloc("flattenedOutput", totalSize, sizeof(float));
    
    memcpy(flattenedOutput, layer->output, totalSize * sizeof(float));
    return flattenedOutput;
}