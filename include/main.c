#include <stdio.h>
#include <stdlib.h>
#include "cnn.h"
#include "cnn_utils.h"
#include "cnn_layer.h"
#include "cnn_train.h"
#include "Weight1.h"
#include "Weight2.h"
#include <string.h> // for memset
// #include "Weight2.h"
#include "image_data_array.h"
#include <unistd.h> // pause
#include <float.h> // for INFINITY
#include <math.h> // for fmaxf
void create_model2(float* image_data);

void test_single_maxpool() {
    // 建立一個 CNN
    CNN* net = createCNN();

    // 設定 input 大小
    int inputWidth = 4;
    int inputHeight = 4;
    int inputChannels = 1; // 單通道

    // 加一層 Max Pooling
    int poolSize = 2;
    int stride = 2;
    PoolType poolType = MAX_POOL;

    addPoolLayer(net, poolSize, stride, poolType, inputWidth, inputHeight, inputChannels);

    // 更新 input size (通常 create_cnn_sample 裡面會更新)
    inputWidth = (inputWidth - poolSize) / stride + 1;
    inputHeight = (inputHeight - poolSize) / stride + 1;

    // 準備輸入：用剛剛的例子
    float inputData[16] = {
         1,  3,  2,  4,
         5,  6,  7,  8,
         9, 10, 11, 12,
        13, 14, 15, 16
    };

    // forward
    float* output = forward(net, inputData);

    // 輸出結果
    printf("Pooling output:\n");
    int outputSize = net->lastLayer->outputWidth *
                     net->lastLayer->outputHeight *
                     net->lastLayer->outputChannels;

    for (int i = 0; i < outputSize; i++) {
        printf("%d: %.1f\n", i, output[i]);
    }

    // 收尾
    freeCNN(net);
}

void test_single_avgpool() {
    // 建立一個 CNN
    CNN* net = createCNN();

    // 設定 input 大小
    int inputWidth = 4;
    int inputHeight = 4;
    int inputChannels = 1; // 單通道

    // 加一層 Max Pooling
    int poolSize = 2;
    int stride = 2;
    PoolType poolType = AVG_POOL;

    addPoolLayer(net, poolSize, stride, poolType, inputWidth, inputHeight, inputChannels);

    // 更新 input size (通常 create_cnn_sample 裡面會更新)
    inputWidth = (inputWidth - poolSize) / stride + 1;
    inputHeight = (inputHeight - poolSize) / stride + 1;

    // 準備輸入：用剛剛的例子
    float inputData[16] = {
         1,  3,  2,  4,
         5,  6,  7,  8,
         9, 10, 11, 12,
        13, 14, 15, 16
    };

    // forward
    float* output = forward(net, inputData);

    // 輸出結果
    printf("Pooling output:\n");
    int outputSize = net->lastLayer->outputWidth *
                     net->lastLayer->outputHeight *
                     net->lastLayer->outputChannels;

    for (int i = 0; i < outputSize; i++) {
        printf("%d: %.1f\n", i, output[i]);
    }

    // 收尾
    freeCNN(net);
}

void test_single_conv() {
    CNN* net = createCNN();

    // 輸入為 4x4 單通道
    int inputWidth = 4, inputHeight = 4, inputChannels = 1;

    // 加入一層卷積層：filterSize=2, stride=2, padding=0
    addConvLayer(net, 2, 1, 2, 0, NONE, inputWidth, inputHeight, inputChannels);

    // 取得卷積層
    Layer* convLayer = net->lastLayer;

    // 將所有 filter weights 設為 1，bias 設為 0
    for (int i = 0; i < 2 * 2 * 1 * 1; i++) {
        convLayer->params.conv.weights[i] = 1.0f;
    }
    convLayer->params.conv.bias[0] = 0.0f;

    // 測試輸入資料
    float inputData[16] = {
         1,  3,  2,  4,
         5,  6,  7,  8,
         9, 10, 11, 12,
        13, 14, 15, 16
    };

    float* output = forward(net, inputData);

    printf("Conv output:\n");
    for (int i = 0; i < convLayer->outputWidth * convLayer->outputHeight; i++) {
        printf("%d: %.1f\n", i, output[i]);
    }

    freeCNN(net);
}

void test_single_fc() {
    CNN* net = createCNN();

    int inputSize = 16;
    int outputSize = 2;

    addFCLayer(net, outputSize, RELU, inputSize);

    Layer* fcLayer = net->lastLayer;

    // 設定 weight = 1, bias = 0
    for (int i = 0; i < inputSize * outputSize; i++) {
        fcLayer->params.fc.weights[i] = 1.0f;
    }
    for (int i = 0; i < outputSize; i++) {
        fcLayer->params.fc.bias[i] = 0.0f;
    }

    // 輸入資料
    float inputData[16] = {
         1,  3,  2,  4,
         5,  6,  7,  8,
         9, 10, 11, 12,
        13, 14, 15, 16
    };

    float* output = forward(net, inputData);

    printf("FC output:\n");
    for (int i = 0; i < outputSize; i++) {
        printf("%d: %.1f\n", i, output[i]);
    }

    freeCNN(net);
}

void training_sample1() {
    /*
    5x5 -> 3x3 -> flatten -> 9x1 -> 1
    */
    CNN* net = createCNN();
    
    // 定義一個簡單的5x5黑白圖像 (數字1的簡單表示)
    int inputWidth = 5, inputHeight = 5, inputChannels = 1;
    float inputImage[25] = {
        0, 0, 1, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 1, 0, 0
    };
    
    // 添加一個卷積層
    addConvLayer(net, 3, 1, 1, 0, RELU, inputWidth, inputHeight, inputChannels);
    
    // 添加一個全連接層 (將卷積輸出展平後連接)
    int convOutputSize = net->firstLayer->outputWidth * net->firstLayer->outputHeight * net->firstLayer->outputChannels;
    addFCLayer(net, 1, SIGMOID, convOutputSize);  // 輸出一個值，使用Sigmoid激活
    
    printCNN(net); // 打印網絡結構
    // 目標輸出 (我們希望網絡能識別這是數字1)
    float target = 1.0f;
    
    //printf("CLOCKS_PER_SEC: %ld\n", CLOCKS_PER_SEC); // 1Mhz
    printf("Starting training...\n");
    //time_t start, end ;
    //start = clock();
    //printf("Start time: %s", ctime(&start));
    int epochs = 1000;
    float learning_rate = 0.01f;
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        // 前向傳播
        float* output = forward(net, inputImage); // lastlayer->output
        
        // 計算損失 (使用均方誤差)
        float loss = (output[0] - target) * (output[0] - target);
        //computeMSE(output, float *target, int size)
        // 反向傳播
        backward(net, &target);
        updateWeights(net, learning_rate);
        
        if (epoch % 100 == 0) {
            printf("Epoch %d, Output: %.4f, Target: %.1f, Loss: %.6f\n", 
                  epoch, output[0], target, loss);
        }
    }
    //end = clock();
    //double elapsed = ((double)(end - start) * 1000) / CLOCKS_PER_SEC;;
    // 測試訓練結果
    //printf("\nTraining take %.2f ms completed. Final output: %.4f (Target: %.1f)\n", elapsed, net->lastLayer->output[0], target);
    // 打印卷積層的權重和偏置
    Layer* convLayer = net->firstLayer;
    printf("\nConvolution layer weights (3x3):\n");
    for (int fh = 0; fh < 3; fh++) {
        for (int fw = 0; fw < 3; fw++) {
            printf("%.4f ", convLayer->params.conv.weights[fh * 3 + fw]);
        }
        printf("\n");
    }
    printf("Bias: %.4f\n", convLayer->params.conv.bias[0]);
    
    freeCNN(net);
}

// 將權重從 src[ (inputChannels*filterHeight*filterWidth) ][ numFilters ]
// 轉成連續一維陣列，排列順序為:
// filterHeight -> filterWidth -> inputChannels -> numFilters

void convertConvWeights(
    float *dst,
    const float src[][32],
    int filterHeight,
    int filterWidth,
    int inputChannels,
    int outputChannels
) {
    int kernelArea = filterHeight * filterWidth;
    for (int oc = 0; oc < outputChannels; ++oc) {
        for (int ic = 0; ic < inputChannels; ++ic) {
            for (int fh = 0; fh < filterHeight; ++fh) {
                for (int fw = 0; fw < filterWidth; ++fw) {
                    int k = ic * kernelArea + fh * filterWidth + fw;
                    int dstIdx = ((oc * inputChannels + ic) * filterHeight + fh) * filterWidth + fw;
                    dst[dstIdx] = src[k][oc];
                }
            }
        }
    }
}



void printWeights(float *weights, int filterHeight, int filterWidth, int inputChannels, int numFilters) {
    printf("Weights:\n");
    for (int c = 0; c < numFilters; c++) {
        printf("Filter %d:\n", c);
        for (int h = 0; h < filterHeight; h++) {
            for (int w = 0; w < filterWidth; w++) {
                for (int ic = 0; ic < inputChannels; ic++) {
                    printf("%.4f ", weights[(h * filterWidth + w) * inputChannels * numFilters + ic * numFilters + c]);
                }
                printf("\n");
            }
        }
    }
}

void printoutput(float *output, int ele , int outputWidth, int outputHeight, int outputChannels) {
    printf("Output:\n");
    for (int i = 0; i < outputChannels*outputHeight*outputWidth; i++) {
        printf("[%d] Element %d: %f\n", ele, i, output[i]);
    }
}

void save_density_pgm(const float *outputPtr, int W, int H, const char *filename) {
    // 先在所有像素裡找出 min/max，用來做視覺化的線性縮放
    float vmin =  FLT_MAX, vmax = -FLT_MAX;
    for (int i = 0; i < W*H; i++) {
        if (outputPtr[i] < vmin) vmin = outputPtr[i];
        if (outputPtr[i] > vmax) vmax = outputPtr[i];
    }
    // 如果全部相等，就直接輸出全黑或全白
    if (vmax - vmin < 1e-6f) {
        vmin = 0; vmax = 1;
    }

    // 開檔，輸出 PGM Header
    FILE *f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "Cannot open file %s for writing\n", filename);
        return;
    }
    // P5 = binary grayscale, max val 255
    fprintf(f, "P5\n%d %d\n255\n", W, H);

    // 每個像素做 (v - vmin)/(vmax-vmin) → [0,1] → [0,255]
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            float v = outputPtr[y*W + x];
            // 線性 Normalize
            float norm = (v - vmin) / (vmax - vmin);
            if (norm < 0) norm = 0;
            if (norm > 1) norm = 1;
            unsigned char gray = (unsigned char)(norm * 255.0f);
            fputc(gray, f);
        }
    }
    fclose(f);
    printf("Saved density map to %s  (min=%f, max=%f)\n", filename, vmin, vmax);
}

void create_model1() {

    // for density map
    CNN* model1 = createCNN();
    float *image = (float*)image_data; // 假設 image_data_array 是一個 float 陣列
    int inputWidth = 512, inputHeight = 512, inputChannels = 3; // (C,H,W) 1D flatten channels first image
    addConvLayer(model1, 3, 32, 2, 1, RELU, inputWidth, inputHeight, inputChannels);
    Layer *layer = model1->lastLayer;
    convertConvWeights(layer->params.conv.weights, cnn_conv_1_w, 3, 3, inputChannels, 32);
    layer->params.conv.bias = (float*)cnn_conv_1_b;
    //printoutput(layer->output, 1, layer->outputWidth, layer->outputHeight, layer->outputChannels);

    addConvLayer(model1, 3, 32, 2, 1, RELU, layer->outputWidth, layer->outputHeight, layer->outputChannels);
    layer = model1->lastLayer;
    convertConvWeights(layer->params.conv.weights,cnn_conv_2_w, 3, 3, layer->inputChannels, 32);
    layer->params.conv.bias = (float*)cnn_conv_2_b;
    //printoutput(layer->output, 2, layer->outputWidth, layer->outputHeight, layer->outputChannels);

    addConvLayer(model1, 3, 32, 1, 1, RELU, layer->outputWidth, layer->outputHeight, layer->outputChannels);
    layer = model1->lastLayer;
    convertConvWeights(layer->params.conv.weights,cnn_conv_3_w, 3, 3, layer->inputChannels, 32);
    layer->params.conv.bias = (float*)cnn_conv_3_b;
    //printoutput(layer->output, 3, layer->outputWidth, layer->outputHeight, layer->outputChannels);

    addConvLayer(model1, 3, 32, 1, 1, RELU, layer->outputWidth, layer->outputHeight, layer->outputChannels);
    layer = model1->lastLayer;
    convertConvWeights(layer->params.conv.weights,cnn_conv_4_w, 3, 3, layer->inputChannels, 32);
    layer->params.conv.bias = (float*)cnn_conv_4_b;
    //printoutput(layer->output, 4, layer->outputWidth, layer->outputHeight, layer->outputChannels);

    addConvLayer(model1, 3, 32, 1, 1, RELU, layer->outputWidth, layer->outputHeight, layer->outputChannels);
    layer = model1->lastLayer;
    convertConvWeights(layer->params.conv.weights,cnn_conv_5_w, 3, 3, layer->inputChannels, 32);
    layer->params.conv.bias = (float*)cnn_conv_5_b;
    //printoutput(layer->output, 5, layer->outputWidth, layer->outputHeight, layer->outputChannels);

    addConvLayer(model1, 3, 32, 1, 1, RELU, layer->outputWidth, layer->outputHeight, layer->outputChannels);
    layer = model1->lastLayer;
    convertConvWeights(layer->params.conv.weights,cnn_conv_6_w, 3, 3, layer->inputChannels, 32);
    layer->params.conv.bias = (float*)cnn_conv_6_b;
    //printoutput(layer->output, 6, layer->outputWidth, layer->outputHeight, layer->outputChannels);

    addConvLayer(model1, 3, 32, 1, 1, RELU, layer->outputWidth, layer->outputHeight, layer->outputChannels);
    layer = model1->lastLayer;
    convertConvWeights(layer->params.conv.weights, cnn_conv_7_w, 3, 3, layer->inputChannels, 32);
    layer->params.conv.bias = (float*)cnn_conv_7_b;
    //printoutput(layer->output, 7, layer->outputWidth, layer->outputHeight, layer->outputChannels);

    addConvLayer(model1, 3, 32, 1, 1, RELU, layer->outputWidth, layer->outputHeight, layer->outputChannels);
    layer = model1->lastLayer;
    convertConvWeights(layer->params.conv.weights, cnn_conv_8_w, 3, 3, layer->inputChannels, 32);
    layer->params.conv.bias = (float*)cnn_conv_8_b;
    //printoutput(layer->output, 8, layer->outputWidth, layer->outputHeight, layer->outputChannels);

    addConvLayer(model1, 3, 32, 1, 1, RELU, layer->outputWidth, layer->outputHeight, layer->outputChannels);
    layer = model1->lastLayer;
    convertConvWeights(layer->params.conv.weights, cnn_conv_9_w, 3, 3, layer->inputChannels, 32);
    layer->params.conv.bias = (float*)cnn_conv_9_b;
    //printoutput(layer->output, 9, layer->outputWidth, layer->outputHeight, layer->outputChannels);

    addConvLayer(model1, 1, 1, 1, 0, NONE, layer->outputWidth, layer->outputHeight, layer->outputChannels);
    layer = model1->lastLayer;
    memcpy(layer->params.conv.weights,cnn_conv_10_w,32 * sizeof(float));
    layer->params.conv.bias = (float*)cnn_conv_10_b;

    forward(model1, (float*)image);
    layer = model1->lastLayer;
    printoutput(layer->output, 10, layer->outputWidth, layer->outputHeight, layer->outputChannels);
    save_density_pgm(layer->output, layer->outputWidth, layer->outputHeight, "density_map.pgm");
    create_model2(layer->output);

}

void create_model2(float* image_data) {
    /*
    128x128x1 圖片

    到(kernel_size, kernel, strdie)
    3-32-2 -> 4-32-4 -> 4-8-4 -> 4-1-1
    */
    // for density map
    CNN* density_map = createCNN();
    int inputWidth = 128, inputHeight = 128, inputChannels = 1;
    float *image = (float*)image_data;
    addConvLayer(density_map, 3, 32, 2, 1, RELU, inputWidth, inputHeight, inputChannels);
    Layer *layer = density_map->lastLayer;
    convertConvWeights(layer->params.conv.weights, model2_cnn_conv_1_w, 3, 3, layer->inputChannels, 32);
    layer->params.conv.bias = (float*)model2_cnn_conv_1_b;
    printf("cnn_conv_1_w %f, %f\n", *layer->params.conv.weights, *layer->params.conv.bias ) ;
    addConvLayer(density_map, 4, 32, 4, 1, RELU, layer->outputWidth, layer->outputHeight, layer->outputChannels);
    layer = density_map->lastLayer;
    convertConvWeights(layer->params.conv.weights, model2_cnn_conv_2_w, 4, 4, layer->inputChannels, 32);
    layer->params.conv.bias = (float*)model2_cnn_conv_2_b;
    addConvLayer(density_map, 4, 8, 4, 1, RELU, layer->outputWidth, layer->outputHeight, layer->outputChannels);
    layer = density_map->lastLayer;
    convertConvWeights(layer->params.conv.weights, model2_cnn_conv_3_w, 4, 4, layer->inputChannels, 8);
    layer->params.conv.bias = (float*)model2_cnn_conv_3_b;
    addConvLayer(density_map, 4, 1, 1, 0, NONE, layer->outputWidth, layer->outputHeight, layer->outputChannels);
    layer = density_map->lastLayer;
    convertConvWeights(layer->params.conv.weights, model2_cnn_conv_4_w, 4, 4, layer->inputChannels, 1);
    layer->params.conv.bias = (float*)model2_cnn_conv_4_b;
    forward(density_map, (float*)image);
    printoutput(layer->output, 1, layer->outputWidth, layer->outputHeight, layer->outputChannels);
}


void test_model1() {

    // for density map
    CNN* model1 = createCNN();
    float *image = (float*)image_data; // 假設 image_data_array 是一個 float 陣列
    int inputWidth = 512, inputHeight = 512, inputChannels = 3;
    addConvLayer(model1, 3, 32, 2, 1, RELU, inputWidth, inputHeight, inputChannels);
    Layer *layer = model1->lastLayer;
    convertConvWeights(layer->params.conv.weights, cnn_conv_1_w, 3, 3, inputChannels, 32);
    layer->params.conv.bias = (float*)cnn_conv_1_b;
    printWeights(layer->params.conv.weights, 3, 3, inputChannels, 32);
    //printoutput(layer->output, 1, layer->outputWidth, layer->outputHeight, layer->outputChannels);
    forward(model1, image);
    layer = model1->lastLayer;
}

int main() {
    // input: [C][H][W] flattened
    // paddedInput: [C][H+2P][W+2P] flattened
    // weights: [OC][IC][FH][FW] flattened
    // output: [OC][OH][OW] flattened

    create_model1();
    //test_model1();
    return 0;
}