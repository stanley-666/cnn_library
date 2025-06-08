#include <stdio.h>
#include <stdlib.h>
#include "cnn.h"
#include "cnn_layer.h"
#include "cnn_utils.h"

void addConvLayer(CNN *net, int filterSize, int numFilters, int stride, int padding,
                  ActivationType activation, int inputWidth, int inputHeight, int inputChannels)
{
    if (!validateConvParams(inputWidth, inputHeight, filterSize, stride, padding)) {
        exit(EXIT_FAILURE);
    }

    Layer *layer = (Layer *)safeMalloc(sizeof(Layer));
    layer->type = CONV;
    layer->activation = activation;

    // 設置卷積層參數
    layer->params.conv.filterSize = filterSize;
    layer->params.conv.numFilters = numFilters;
    layer->params.conv.stride     = stride;
    layer->params.conv.padding    = padding;

    // 設置輸入／輸出尺寸
    layer->inputWidth   = inputWidth;
    layer->inputHeight  = inputHeight;
    layer->inputChannels= inputChannels;
    layer->outputWidth  = ((inputWidth  + 2*padding - filterSize) / stride) + 1;
    layer->outputHeight = ((inputHeight + 2*padding - filterSize) / stride) + 1;
    layer->outputChannels = numFilters;

    // 分配權重和偏置
    int wcount = filterSize * filterSize * inputChannels * numFilters;
    int bcount = numFilters;
    layer->params.conv.sizeofWeights = wcount;
    layer->params.conv.sizeofBias    = bcount;
    //layer->params.conv.weights = (float *)safeCalloc(wcount, sizeof(float));
    layer->params.conv.bias = NULL;
    layer->output = NULL;


    // **在這裡一次把你想要的全部印出來：**
    printf("=== Added Conv Layer ===\n");
    printf(" Activation: %s\n",
        layer->activation==RELU      ? "RELU" :
        layer->activation==SIGMOID   ? "SIGMOID" :
        layer->activation==TANH      ? "TANH" :
        layer->activation==LEAKY_RELU? "LEAKY_RELU" : "NONE");
    printf(" Input  : %d x %d x %d\n",
           layer->inputWidth, layer->inputHeight, layer->inputChannels);
    printf(" Output : %d x %d x %d\n",
           layer->outputWidth, layer->outputHeight, layer->outputChannels);
    printf(" Kernel : %d x %d\n", filterSize, filterSize);
    printf(" Stride : %d, Padding: %d\n", stride, padding);
    printf(" NumFilters: %d\n", numFilters);
    printf(" Weights count: %d, Bias count: %d\n",
           layer->params.conv.sizeofWeights,
           layer->params.conv.sizeofBias);
    printf("=========================\n");

    // 把 layer 掛到 net 上
    layer->prev = net->lastLayer;
    layer->next = NULL;
    if (net->lastLayer) net->lastLayer->next = layer;
    else                net->firstLayer = layer;
    net->lastLayer = layer;
    net->numLayers++;
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
    static int layerCount = 0;
    // [ outputChannels, inputChannels, filterHeight, filterWidth ]
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
    if (inW <= 0 || inH <= 0 || inC <= 0 || padding < 0) {
        printf("Invalid input dimensions or padding in convolution\n");
        exit(EXIT_FAILURE);
    }

    // padding = 1 矩陣外維繞一圈補0
    
    int paddedW = inW + 2 * padding;
    int paddedH = inH + 2 * padding;

    float *paddedInput = (float *)safeCalloc(paddedW * paddedH * inC, sizeof(float));
    if (!paddedInput) {
        printf("Memory allocation failed for padded input in convolution\n");
        exit(EXIT_FAILURE);
    }

    int inputIndex = 0, paddedIndex = 0;
    for (int c = 0; c < inC; c++) {
        for (int h = 0; h < inH; h++) {
            for (int w = 0; w < inW; w++) {
                inputIndex = c * inH * inW + h * inW + w;
                //printf("inputIndex: %d, c: %d, h: %d, w: %d, paddedW: %d, paddedH: %d\n", inputIndex, c, h, w, paddedW, paddedH);
                paddedIndex = c * paddedH * paddedW + (h + padding) * paddedW + (w + padding);
                //printf("paddedIndex: %d, c: %d, h: %d, w: %d\n", paddedIndex, c, h + padding, w + padding);
                paddedInput[paddedIndex] = input[inputIndex];
            }
        }
        //printf("intput index %d, padded index %d\n", inputIndex, paddedIndex);
    }

// 被kernel掃到的對應的值相乘之後的總和為卷積的其中之一結果(conv.weight)
    const float (*W)[inC][filterSize][filterSize] = layer->params.conv.weights;
    printf("Conv weights shape: [%d][%d][%d][%d]\n", 
           outC, inC, filterSize, filterSize);
    layer->output = (float *)safeCalloc(outC * outH * outW, sizeof(float));
    printf( "%d x %d x %d output allocated at %p\n", 
            outC, outH, outW, layer->output);
    for (int oc = 0; oc < outC; oc++) {
        for (int oh = 0; oh < outH; oh++) {
            for (int ow = 0; ow < outW; ow++) {
                float sum = 0.0f;
                for (int ic = 0; ic < inC; ic++) {
                    for (int fh = 0; fh < filterSize; fh++) {
                        for (int fw = 0; fw < filterSize; fw++) {
                            int ih = oh * stride + fh;
                            int iw = ow * stride + fw;
                            if (ih >= 0 && ih < paddedH && iw >= 0 && iw < paddedW) {
                                float w = W[oc][ic][fh][fw];   // 直接四維存取！
                                //printf("oc: %d, ic: %d, fh: %d, fw: %d, w: %f\n", oc, ic, fh, fw, w);
                                float x = paddedInput[ic * paddedH * paddedW + ih * paddedW + iw];
                                sum += x * w;

                                //printf("paddedInput[%d][%d][%d] * W[%d][%d][%d][%d] = %f * %f = %f\n", 
                                //ic, ih, iw, oc, ic, fh, fw, paddedInput[ic * paddedH * paddedW + ih * paddedW + iw], w, x * w);
                            }
                        }
                    }
                }
                //printf("oc: %d, oh: %d, ow: %d, sum: %f\n", oc, oh, ow, sum);
                // Activation、存 output
                sum += layer->params.conv.bias[oc];
                if (isnan(sum) || isinf(sum)) {
                    printf("Invalid sum value at [%d][%d][%d]: %f\n", oc, oh, ow, sum);
                    exit(EXIT_FAILURE);
                }
                
                float temp = activate(sum, layer->activation);

                if (layerCount == 0) {
                    layer->output[(oc * outH + oh) * outW + ow] = temp*0.0014;
                }           
                else if(layerCount == 1) {
                    layer->output[(oc * outH + oh) * outW + ow] = temp*0.002;                }
                else if(layerCount == 2) {
                    layer->output[(oc * outH + oh) * outW + ow] = temp*0.02;
                }
                else if(layerCount == 3) {
                    layer->output[(oc * outH + oh) * outW + ow] = temp*0.01;
                }
                else {
                    layer->output[(oc * outH + oh) * outW + ow] = temp;
                }

            }
        }
    }
    layerCount++;
    free(paddedInput);
    if ( layerCount == 10) {
        layerCount = 0; // reset for next conv layer
    }
}

/*
void forwardConv_old(Layer *layer, float *input)
{   
    // [ outputChannels, inputChannels, filterHeight, filterWidth ]
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
    if (inW <= 0 || inH <= 0 || inC <= 0 || padding < 0) {
        printf("Invalid input dimensions or padding in convolution\n");
        exit(EXIT_FAILURE);
    }

    // padding = 1 矩陣外維繞一圈補0
    
    int paddedW = inW + 2 * padding;
    int paddedH = inH + 2 * padding;

    float *paddedInput = (float *)safeCalloc(paddedW * paddedH * inC, sizeof(float));
    if (!paddedInput) {
        printf("Memory allocation failed for padded input in convolution\n");
        exit(EXIT_FAILURE);
    }

    int inputIndex = 0, paddedIndex = 0;
    for (int c = 0; c < inC; c++) {
        for (int h = 0; h < inH; h++) {
            for (int w = 0; w < inW; w++) {
                inputIndex = c * inH * inW + h * inW + w;
                //printf("inputIndex: %d, c: %d, h: %d, w: %d, paddedW: %d, paddedH: %d\n", inputIndex, c, h, w, paddedW, paddedH);
                paddedIndex = c * paddedH * paddedW + (h + padding) * paddedW + (w + padding);
                //printf("paddedIndex: %d, c: %d, h: %d, w: %d\n", paddedIndex, c, h + padding, w + padding);
                paddedInput[paddedIndex] = input[inputIndex];
            }
        }
        //printf("intput index %d, padded index %d\n", inputIndex, paddedIndex);
    }

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
                                float temp = paddedInput[paddedIdx] * layer->params.conv.weights[filterIdx];
                                //printf("temp = paddedInput[%d] * layer->params.conv.weights[%d] = %f * %f = %f\n", 
                                //paddedIdx, filterIdx, paddedInput[paddedIdx], layer->params.conv.weights[filterIdx],temp);
                                if (isnan(temp) || isinf(temp)) {
                                    printf("Invalid output value at [%d][%d][%d]: %f\n", oc, oh, ow, temp);
                                    exit(EXIT_FAILURE);
                                }
                                sum += temp;
                                
                            }
                        }
                    }
                }

                // 加上 bias
                //sum += layer->params.conv.bias[oc];
                //printf("sum %f\n ", sum);
                // 經過activation function存到output
                float temp = activate(sum, layer->activation);
                //printf("Conv output[%d][%d][%d] = %f\n", oc, oh, ow, temp);
                //int idx = (oc * outH + oh) * outW + ow;
                //printf("Conv output %d [%d][%d][%d] = %f\n",idx,  oc, oh, ow, temp);
                if (isnan(temp) || isinf(temp)) {
                    printf("Invalid output value at [%d][%d][%d]: %f\n", oc, oh, ow, temp);
                    exit(EXIT_FAILURE);
                }
                //printf("Conv output[%d][%d][%d] = %f\n", oc, oh, ow, temp);
                layer->output[(oc * outH + oh) * outW + ow] = temp;

            }
        }
    }

    //printf("卷積完成\n");
    free(paddedInput);
}
*/
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