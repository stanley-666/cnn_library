#include <stdio.h>
#include <stdlib.h>
#include "cnn.h"
#include "cnn_train.h"
#include "cnn_utils.h"

void backwardFC(Layer* layer, float* target) {
    int inSize = layer->params.fc.inputSize;
    int outSize = layer->params.fc.outputSize;

    if (!layer->deltas) 
        layer->deltas = (float*)safeCalloc(outSize, sizeof(float));
    if (!layer->params.fc.weights || !layer->params.fc.bias) {
        printf("[ERROR] FC layer missing weights/bias!\n");
        exit(EXIT_FAILURE);
    }
    if (!layer->weightGrads) 
        layer->weightGrads = (float*)safeCalloc(inSize * outSize, sizeof(float));
    if (!layer->biasGrads)
        layer->biasGrads = (float*)safeCalloc(outSize, sizeof(float));

    if (layer->next == NULL) {
        // 最後一層 (output layer)
        for (int o = 0; o < outSize; o++) {
            float outputVal = layer->output[o];
            float error = outputVal - target[o];
            layer->deltas[o] = error * activateDerivative(outputVal, layer->activation);
        }
    } else {
        // 中間層 (hidden layer)
        for (int i = 0; i < inSize; i++) {
            float sum = 0.0f;
            for (int o = 0; o < outSize; o++) {
                sum += layer->next->deltas[o] * layer->params.fc.weights[o * inSize + i];
            }
            float outputVal = layer->output[i];
            layer->deltas[i] = sum * activateDerivative(outputVal, layer->activation);
        }
    }

    // 計算 weight gradients 與 bias gradients
    for (int o = 0; o < outSize; o++) {
        for (int i = 0; i < inSize; i++) {
            layer->weightGrads[o * inSize + i] = layer->deltas[o] * layer->input[i];
        }
        layer->biasGrads[o] = layer->deltas[o];
    }
}

void backwardPool(Layer* layer) {
    int inW = layer->inputWidth;
    int inH = layer->inputHeight;
    int inC = layer->inputChannels;
    int outW = layer->outputWidth;
    int outH = layer->outputHeight;
    int poolSize = layer->params.pool.poolSize;
    int stride = layer->params.pool.stride;
    PoolType poolType = layer->params.pool.type;

    if (!layer->deltas) layer->deltas = (float*)calloc(inW * inH * inC, sizeof(float));
    // update deltas
    for (int c = 0; c < inC; c++) {
        for (int oh = 0; oh < outH; oh++) {
            for (int ow = 0; ow < outW; ow++) {
                int outIdx = (c * outH + oh) * outW + ow;
                float grad = (layer->next) ? layer->next->deltas[outIdx] : layer->deltas[outIdx]; // *

                if (poolType == MAX_POOL) {
                    float maxVal = -INFINITY;
                    int maxH = -1, maxW = -1;
                    for (int ph = 0; ph < poolSize; ph++) {
                        for (int pw = 0; pw < poolSize; pw++) {
                            int ih = oh * stride + ph;
                            int iw = ow * stride + pw;
                            if (ih < inH && iw < inW) {
                                int inIdx = (c * inH + ih) * inW + iw;
                                if (layer->input[inIdx] > maxVal) {
                                    maxVal = layer->input[inIdx];
                                    maxH = ih;
                                    maxW = iw;
                                }
                            }
                        }
                    }
                    if (maxH >= 0 && maxW >= 0) {
                        int maxIdx = (c * inH + maxH) * inW + maxW;
                        layer->deltas[maxIdx] += grad;
                    }
                } else if (poolType == AVG_POOL) {
                    float gradAvg = grad / (poolSize * poolSize);
                    for (int ph = 0; ph < poolSize; ph++) {
                        for (int pw = 0; pw < poolSize; pw++) {
                            int ih = oh * stride + ph;
                            int iw = ow * stride + pw;
                            if (ih < inH && iw < inW) {
                                int inIdx = (c * inH + ih) * inW + iw;
                                layer->deltas[inIdx] += gradAvg;
                            }
                        }
                    }
                }
            }
        }
    }
}

void backwardConv(Layer* layer) {
    int inW = layer->inputWidth;
    int inH = layer->inputHeight;
    int inC = layer->inputChannels;
    int outW = layer->outputWidth;
    int outH = layer->outputHeight;
    int outC = layer->outputChannels;
    int filterSize = layer->params.conv.filterSize;
    int stride = layer->params.conv.stride;
    int padding = layer->params.conv.padding;

    if (!layer->deltas)
        layer->deltas = (float*)safeCalloc(inW * inH * inC, sizeof(float));
    if (!layer->weightGrads)
        layer->weightGrads = (float*)safeCalloc(filterSize * filterSize * inC * outC, sizeof(float));
    if (!layer->biasGrads)
        layer->biasGrads = (float*)safeCalloc(outC, sizeof(float));

    // backward delta
    for (int oc = 0; oc < outC; oc++) {
        for (int oh = 0; oh < outH; oh++) {
            for (int ow = 0; ow < outW; ow++) {
                int outIdx = (oc * outH + oh) * outW + ow;
                float gradOut = (layer->next) ? layer->next->deltas[outIdx] : layer->deltas[outIdx];
                float activatedVal = layer->output[outIdx];
                float grad = gradOut * activateDerivative(activatedVal, layer->activation);

                for (int ic = 0; ic < inC; ic++) {
                    for (int fh = 0; fh < filterSize; fh++) {
                        for (int fw = 0; fw < filterSize; fw++) {
                            int ih = oh * stride + fh - padding;
                            int iw = ow * stride + fw - padding;
                            if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                                int inIdx = (ic * inH + ih) * inW + iw;
                                int filterIdx = ((oc * inC + ic) * filterSize + fh) * filterSize + fw;

                                //layer->deltas[inIdx] += grad * layer->params.conv.weights[filterIdx];
                                layer->weightGrads[filterIdx] += grad * layer->input[inIdx];
                            }
                        }
                    }
                }
                layer->biasGrads[oc] += grad;
            }
        }
    }
}

void updateWeightsFC(Layer* layer, float learningRate) {
    int inSize = layer->params.fc.inputSize;
    int outSize = layer->params.fc.outputSize;

    for (int o = 0; o < outSize; o++) {
        for (int i = 0; i < inSize; i++) {
            int idx = o * inSize + i;
            layer->params.fc.weights[idx] -= learningRate * layer->weightGrads[idx];
        }
        layer->params.fc.bias[o] -= learningRate * layer->biasGrads[o];
    }
}

void updateWeightsConv(Layer* layer, float learningRate) {
    int filterSize = layer->params.conv.filterSize;
    int inC = layer->inputChannels;
    int outC = layer->outputChannels;
    int weightCount = filterSize * filterSize * inC * outC;

    for (int i = 0; i < weightCount; i++) {
        ;//layer->params.conv.weights[i] -= learningRate * layer->weightGrads[i];
    }
    for (int oc = 0; oc < outC; oc++) {
        layer->params.conv.bias[oc] -= learningRate * layer->biasGrads[oc];
    }
}
