#include <stdio.h>
#include <stdlib.h>
#include "cnn.h"
#include "cnn_utils.h"
#include "cnn_layer.h"
#include "cnn_train.h"

float expf(float x) {
    float result = 1.0f;
    float term = 1.0f;
    for (int i = 1; i < 10; i++) { // Taylor series expansion
        term *= x / i;
        result += term;
    }
    return result;
}

// Approximate hyperbolic tangent function
float tanhf(float x) {
    if (x > 10) return 1.0f;  // Approximation for large x
    if (x < -10) return -1.0f; // Approximation for very small x

    float e2x = expf(2 * x);
    return (e2x - 1) / (e2x + 1);
}


// Approximate square root function
float sqrtf(float x) {
    if (x < 0) return -1.0f; // Handle negative input
    if (x == 0) return 0.0f;  // Square root of 0 is 0

    float guess = x / 2.0f;
    for (int i = 0; i < 10; i++) {
        float next_guess = (guess + x / guess) / 2.0f;
        if (fabs(next_guess - guess) < 0.000001f) break;
        guess = next_guess;
    }
    return guess;
}


// Maximum value between two floats
float fmaxf(float a, float b) {
    if (a != a) return b; // Handle NaN
    if (b != b) return a; // Handle NaN
    return (a > b) ? a : b;
}
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

int main() {
    training_sample1();
    //create_cnn_sample();
    test_single_maxpool();
    test_single_avgpool();
    test_single_conv();
    test_single_fc();
    return 0;
}