```c
#include <stdio.h>
#include <stdlib.h>
#include "cnn.h"
#include "cnn_layer.h"
#include "cnn_utils.h"

void addConvLayer(CNN *net, int filterSize, int numFilters, int stride, int padding, ActivationType activation, int inputWidth, int inputHeight, int inputChannels)
{
  
    if (!validateConvParams(inputWidth, inputHeight, filterSize, stride, padding)) {
        exit(EXIT_FAILURE);
    }

    Layer *layer = (Layer *)safeMalloc(sizeof(Layer));
    layer->type = CONV;
    layer->activation = activation;

    // 設置卷積層參數
    layer->params.conv.filterSize = filterSize; // kernel是幾乘幾的
    layer->params.conv.numFilters = numFilters; // 幾個feature map
    layer->params.conv.stride = stride;
    layer->params.conv.padding = padding;

    // 設置輸入尺寸
    layer->inputWidth = inputWidth;
    layer->inputHeight = inputHeight;
    layer->inputChannels = inputChannels;

    // 計算輸出尺寸
    layer->outputWidth = ((inputWidth + (2 * padding) - filterSize) / stride) + 1;
    layer->outputHeight = ((inputHeight + (2 * padding) - filterSize) / stride) + 1;
    layer->outputChannels = numFilters;

    // 分配權重和偏置空間
    int weightSize = filterSize * filterSize * inputChannels * numFilters;
    layer->params.conv.weights = (float *)safeCalloc(weightSize, sizeof(float));
    layer->params.conv.bias = (float *)safeCalloc(numFilters, sizeof(float));

    // 使用He初始化weights
    float scale = sqrtf(2.0f / (inputChannels * filterSize * filterSize));
    for (int i = 0; i < weightSize; i++)
    {
        layer->params.conv.weights[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale;
    }

    for (int i = 0; i < numFilters; i++)
    {
        layer->params.conv.bias[i] = 0.0f;
    }

    // 分配輸出空間
    int outputSize = layer->outputWidth * layer->outputHeight * layer->outputChannels;
    layer->output = (float *)safeCalloc(outputSize, sizeof(float));

    // 將層添加到網絡
    layer->prev = net->lastLayer;
    layer->next = NULL;
    if (net->lastLayer == NULL)
    {
        net->firstLayer = layer;
    }
    else
    {
        net->lastLayer->next = layer;
    }

    net->lastLayer = layer;
    net->numLayers++;
    
    printf("Added Conv Layer: Input: %dx%dx%d, Output: %dx%dx%d, FilterSize: %d, NumFilters: %d\n",
           inputWidth, inputHeight, inputChannels,
           layer->outputWidth, layer->outputHeight, layer->outputChannels,
           filterSize, numFilters);
}

// 為CNN添加池化層
void addPoolLayer(CNN *net, int poolSize, int stride, PoolType poolType, int inputWidth, int inputHeight, int inputChannels)
{
    if (!validatePoolParams(inputWidth, inputHeight, poolSize, stride)) {
        exit(EXIT_FAILURE);
    }

    Layer *layer = (Layer *)safeMalloc(sizeof(Layer));
    layer->type = POOL;
    layer->activation = NONE; // 池化層通常不需要激活函數

    // 設置池化層參數
    layer->params.pool.poolSize = poolSize;
    layer->params.pool.stride = stride;
    layer->params.pool.type = poolType;

    // 設置輸入尺寸
    layer->inputWidth = inputWidth;
    layer->inputHeight = inputHeight;
    layer->inputChannels = inputChannels;

    // 計算輸出尺寸
    layer->outputWidth = (inputWidth - poolSize) / stride + 1;
    layer->outputHeight = (inputHeight - poolSize) / stride + 1;
    layer->outputChannels = inputChannels;

    // 分配輸出空間
    int outputSize = layer->outputWidth * layer->outputHeight * layer->outputChannels;
    layer->output = (float *)safeCalloc(outputSize, sizeof(float));

    // 將層添加到網絡
    layer->prev = net->lastLayer;
    layer->next = NULL;
    if (net->lastLayer == NULL)
    {
        net->firstLayer = layer;
    }
    else
    {
        net->lastLayer->next = layer;
    }

    net->lastLayer = layer;
    net->numLayers++;
    
    printf("Added Pool Layer: Input: %dx%dx%d, Output: %dx%dx%d, PoolSize: %d, Type: %s\n",
           inputWidth, inputHeight, inputChannels,
           layer->outputWidth, layer->outputHeight, layer->outputChannels,
           poolSize, poolType == MAX_POOL ? "Max" : "Avg");
}

// 為CNN添加全連接層
void addFCLayer(CNN *net, int outputSize, ActivationType activation, int inputSize)
{
    if (inputSize <= 0 || outputSize <= 0) {
        fprintf(stderr, "Error: Invalid FC layer dimensions: input=%d, output=%d\n", 
                inputSize, outputSize);
        exit(EXIT_FAILURE);
    }

    Layer *layer = (Layer *)safeMalloc(sizeof(Layer));
    layer->type = FC;
    layer->activation = activation;

    // 設置全連接層參數
    layer->params.fc.inputSize = inputSize;
    layer->params.fc.outputSize = outputSize;

    // FC輸入輸出為一維
    layer->inputWidth = inputSize;
    layer->inputHeight = 1;
    layer->inputChannels = 1;

    layer->outputWidth = outputSize;
    layer->outputHeight = 1;
    layer->outputChannels = 1;

    // 分配權重和偏置空間
    layer->params.fc.weights = (float *)safeCalloc(inputSize * outputSize, sizeof(float));
    layer->params.fc.bias = (float *)safeCalloc(outputSize, sizeof(float));

    // Xavier初始化
    float scale = sqrtf(6.0f / (inputSize + outputSize));
    for (int i = 0; i < inputSize * outputSize; i++)
    {
        layer->params.fc.weights[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale;
    }

    for (int i = 0; i < outputSize; i++)
    {
        layer->params.fc.bias[i] = 0.0f;
    }

    // 分配輸出空間
    layer->output = (float *)safeCalloc(outputSize, sizeof(float));

    // 將層添加到網絡
    layer->prev = net->lastLayer;
    layer->next = NULL;
    if (net->lastLayer == NULL)
    {
        net->firstLayer = layer;
    }
    else
    {
        net->lastLayer->next = layer;
    }

    net->lastLayer = layer;
    net->numLayers++;
    
    printf("Added FC Layer: Input: %d, Output: %d\n", inputSize, outputSize);
}

// 前向傳播卷積層
void forwardConv(Layer *layer, float *input)
{
    //printf("Forwarding Conv Layer...\n");
    int outW = layer->outputWidth;
    int outH = layer->outputHeight;
    int outC = layer->outputChannels;
    int inW = layer->inputWidth;
    int inH = layer->inputHeight;
    int inC = layer->inputChannels;
    int filterSize = layer->params.conv.filterSize; // kernel size
    int stride = layer->params.conv.stride;
    int padding = layer->params.conv.padding;
    /*
    //printf("=== Convolution Layer Parameters ===\n");
    //printf("Input  (W x H x C): %d x %d x %d\n", inW, inH, inC);
    //printf("Output (W x H x C): %d x %d x %d\n", outW, outH, outC);
    //printf("Filter Size: %d\n", filterSize);
    //printf("Stride: %d\n", stride);
    //printf("Padding: %d\n", padding);
    //printf("====================================\n");
    */
    if (inW <= 0 || inH <= 0 || inC <= 0 || padding < 0) {
        printf("Invalid input dimensions or padding in convolution\n");
        exit(EXIT_FAILURE);
    }

    // padding = 1 矩陣外維繞一圈補0
    int paddedW = inW + 2 * padding;
    int paddedH = inH + 2 * padding;

    float *paddedInput = (float *)calloc(paddedW * paddedH * inC, sizeof(float));
    if (!paddedInput) {
        printf("Memory allocation failed for padded input in convolution\n");
        exit(EXIT_FAILURE);
    }
/*
    Input:
    +---+---+---+---+
    | 1 | 2 | 3 | 4 |
    +---+---+---+---+
    | 5 | 6 | 7 | 8 |  (inW = 4, inH = 4)
    +---+---+---+---+
    | 9 | 10| 11| 12|
    +---+---+---+---+
    | 13| 14| 15| 16|
    +---+---+---+---+

    Padding (padding = 1):
    +---+---+---+---+---+---+
    | 0 | 0 | 0 | 0 | 0 | 0 |
    +---+---+---+---+---+---+
    | 0 | 1 | 2 | 3 | 4 | 0 |
    +---+---+---+---+---+---+
    | 0 | 5 | 6 | 7 | 8 | 0 |  (paddedW = 6, paddedH = 6)
    +---+---+---+---+---+---+
    | 0 | 9 | 10| 11| 12| 0 |
    +---+---+---+---+---+---+
    | 0 | 13| 14| 15| 16| 0 |
    +---+---+---+---+---+---+
    | 0 | 0 | 0 | 0 | 0 | 0 |
    +---+---+---+---+---+---+
*/
    for (int c = 0; c < inC; c++) {
        for (int h = 0; h < inH; h++) {
            for (int w = 0; w < inW; w++) {
                int inputIndex = c * inH * inW + h * inW + w;
                int paddedIndex = c * paddedH * paddedW + (h + padding) * paddedW + (w + padding);
                paddedInput[paddedIndex] = input[inputIndex];
            }
        }
    }

    /*
    Padded Input:             Kernel (filterSize = 3):

    +---+---+---+---+---+---+    +---+---+---+
    | 0 | 0 | 0 | 0 | 0 | 0 |    | w1| w2| w3|
    +---+---+---+---+---+---+    +---+---+---+
    | 0 | 1 | 2 | 3 | 4 | 0 |    | w4| w5| w6|
    +---+---+---+---+---+---+    +---+---+---+
    | 0 | 5 | 6 | 7 | 8 | 0 |    | w7| w8| w9|
    +---+---+---+---+---+---+    +---+---+---+
    | 0 | 9 | 10| 11| 12| 0 |
    +---+---+---+---+---+---+
    | 0 | 13| 14| 15| 16| 0 |
    +---+---+---+---+---+---+
    | 0 | 0 | 0 | 0 | 0 | 0 |
    +---+---+---+---+---+---+
*/
// 被kernel掃到的對應的值相乘之後的總和為卷積的其中之一結果(conv.weight)
// output size設很小，代表kernel只會掃一部分
    for (int oc = 0; oc < outC; oc++) {
        for (int oh = 0; oh < outH; oh++) {
            for (int ow = 0; ow < outW; ow++) {
                float sum = 0.0f;

                for (int ic = 0; ic < inC; ic++) {
                    int baseIdx = (oc * inC + ic) * filterSize * filterSize;

                    for (int fh = 0; fh < filterSize; fh++) {
                        for (int fw = 0; fw < filterSize; fw++) {
                            int ih = oh * stride + fh;
                            int iw = ow * stride + fw;

                            if (ih >= 0 && ih < paddedH && iw >= 0 && iw < paddedW) {
                                int paddedIdx = ic * paddedH * paddedW + ih * paddedW + iw;
                                int filterIdx = baseIdx + fh * filterSize + fw;

                                sum += paddedInput[paddedIdx] * layer->params.conv.weights[filterIdx];
                            }
                        }
                    }
                }

                // 加上 bias
                sum += layer->params.conv.bias[oc];

                // 經過activation function存到output
                layer->output[(oc * outH + oh) * outW + ow] = activate(sum, layer->activation);
            }
        }
    }

    //printf("卷積完成\n");
    free(paddedInput);
}

// 前向傳播池化層
void forwardPool(Layer *layer, float *input)
{
    int outW = layer->outputWidth;
    int outH = layer->outputHeight;
    int outC = layer->outputChannels; // 與輸入通道數相同
    int inW = layer->inputWidth;
    int inH = layer->inputHeight;
    int poolSize = layer->params.pool.poolSize; // ex. 2x2
    int stride = layer->params.pool.stride;
    PoolType poolType = layer->params.pool.type;

    for (int c = 0; c < outC; c++)
    { // next channel
        for (int oh = 0; oh < outH; oh++)
        { // next height 上到下
            for (int ow = 0; ow < outW; ow++)
            { // next width 左到右
                float value = (poolType == MAX_POOL) ? -INFINITY : 0.0f;
                int count = 0;

                // 池化窗口
                for (int ph = 0; ph < poolSize; ph++)
                {
                    for (int pw = 0; pw < poolSize; pw++)
                    {
                        int ih = oh * stride + ph;
                        int iw = ow * stride + pw;

                        // 確保在輸入範圍內
                        if (ih < inH && iw < inW)
                        {
                            float inVal = input[(c * inH + ih) * inW + iw];

                            if (poolType == MAX_POOL)
                            {
                                value = fmaxf(value, inVal);
                            }
                            else
                            { // 平均池化
                                value += inVal;
                                count++;
                            }
                        }
                    }
                }

                // 對於平均池化，計算平均值
                if (poolType == AVG_POOL) {
                    if (count > 0) {
                        value /= count;
                    } else {
                        value = 0.0f; // 防止除以零
                    }
                } else if (poolType == MAX_POOL && value == -INFINITY) {
                    value = 0.0f; // 沒有找到最大值
                }

                layer->output[(c * outH + oh) * outW + ow] = value;
            }
        }
    }
}

// 前向傳播全連接層
void forwardFC(Layer *layer, float *input)
{
    int inSize = layer->params.fc.inputSize;
    int outSize = layer->params.fc.outputSize;

    for (int o = 0; o < outSize; o++)
    { // iterate over every output neurons
        float sum = 0.0f;

        
        for (int i = 0; i < inSize; i++)
        { // more inputs to one output neuron
            sum += input[i] * layer->params.fc.weights[o * inSize + i];
        }

        // 加上bias
        sum += layer->params.fc.bias[o];

        // 加上activation function後再算下一個output
        layer->output[o] = activate(sum, layer->activation);
    }
}


```