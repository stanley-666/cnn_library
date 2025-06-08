// cnn_utils.h
#ifndef CNN_UTILS_H
#define CNN_UTILS_H
#include <math.h>
#include "cnn.h"
#include <stdlib.h>

void *safeMalloc(size_t size);
void *safeCalloc(size_t count, size_t size);

float activate(float x, ActivationType type);
float activateDerivative(float x, ActivationType type);
float computeMSE(float *output, float *target, int size);
int validateConvParams(int inputWidth, int inputHeight, int filterSize, int stride, int padding);
int validatePoolParams(int inputWidth, int inputHeight, int poolSize, int stride);
float* flattenOutput(Layer* layer);
#endif
