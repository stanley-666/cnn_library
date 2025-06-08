#include <stdio.h>
#include <stdlib.h>
#include "cnn.h"
#include "cnn_utils.h"
#include "cnn_layer.h"
#include "cnn_train.h"
#include "converted_Weight1.h"
#include <string.h> // for memset
#include "image_data_array.h"
#include <unistd.h> // pause
#include <float.h> // for INFINITY
#include <math.h> // for fmaxf
void create_model2(float* image_data);

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
    layer->params.conv.weights = cnn_conv_1_w;
    layer->params.conv.bias = (float*)cnn_conv_1_b;
    //printoutput(layer->output, 1, layer->outputWidth, layer->outputHeight, layer->outputChannels);

    addConvLayer(model1, 3, 32, 2, 1, RELU, layer->outputWidth, layer->outputHeight, layer->outputChannels);
    layer = model1->lastLayer;
    layer->params.conv.weights = cnn_conv_2_w;
    layer->params.conv.bias = (float*)cnn_conv_2_b ;
    //printoutput(layer->output, 2, layer->outputWidth, layer->outputHeight, layer->outputChannels);

    addConvLayer(model1, 3, 32, 1, 1, RELU, layer->outputWidth, layer->outputHeight, layer->outputChannels);
    layer = model1->lastLayer;
    layer->params.conv.weights = cnn_conv_3_w;
    layer->params.conv.bias = (float*) cnn_conv_3_b;
    //printoutput(layer->output, 3, layer->outputWidth, layer->outputHeight, layer->outputChannels);

    addConvLayer(model1, 3, 32, 1, 1, RELU, layer->outputWidth, layer->outputHeight, layer->outputChannels);
    layer = model1->lastLayer;
    layer->params.conv.weights = cnn_conv_4_w;
    layer->params.conv.bias = (float*) cnn_conv_4_b;
    //printoutput(layer->output, 4, layer->outputWidth, layer->outputHeight, layer->outputChannels);

    addConvLayer(model1, 3, 32, 1, 1, RELU, layer->outputWidth, layer->outputHeight, layer->outputChannels);
    layer = model1->lastLayer;
    layer->params.conv.weights = cnn_conv_5_w;
    layer->params.conv.bias = (float*) cnn_conv_5_b;
    //printoutput(layer->output, 5, layer->outputWidth, layer->outputHeight, layer->outputChannels);

    addConvLayer(model1, 3, 32, 1, 1, RELU, layer->outputWidth, layer->outputHeight, layer->outputChannels);
    layer = model1->lastLayer;
    layer->params.conv.weights = cnn_conv_6_w;
    layer->params.conv.bias = (float*) cnn_conv_6_b;
    //printoutput(layer->output, 6, layer->outputWidth, layer->outputHeight, layer->outputChannels);

    addConvLayer(model1, 3, 32, 1, 1, RELU, layer->outputWidth, layer->outputHeight, layer->outputChannels);
    layer = model1->lastLayer;
    layer->params.conv.weights = cnn_conv_7_w;
    layer->params.conv.bias = (float*) cnn_conv_7_b;
    //printoutput(layer->output, 7, layer->outputWidth, layer->outputHeight, layer->outputChannels);

    addConvLayer(model1, 3, 32, 1, 1, RELU, layer->outputWidth, layer->outputHeight, layer->outputChannels);
    layer = model1->lastLayer;
    layer->params.conv.weights = cnn_conv_8_w;
    layer->params.conv.bias = (float*) cnn_conv_8_b;
    //printoutput(layer->output, 8, layer->outputWidth, layer->outputHeight, layer->outputChannels);

    addConvLayer(model1, 3, 32, 1, 1, RELU, layer->outputWidth, layer->outputHeight, layer->outputChannels);
    layer = model1->lastLayer;
    layer->params.conv.weights = cnn_conv_9_w;
    layer->params.conv.bias = (float*) cnn_conv_9_b;
    //printoutput(layer->output, 9, layer->outputWidth, layer->outputHeight, layer->outputChannels);

    addConvLayer(model1, 1, 1, 1, 0, NONE, layer->outputWidth, layer->outputHeight, layer->outputChannels);
    layer = model1->lastLayer;
    layer->params.conv.weights = cnn_conv_10_w;
    layer->params.conv.bias = (float*) cnn_conv_10_b;

    printf("\nModel1 created with %d layers.\n", model1->numLayers);
    forward(model1, (float*)image);
    layer = model1->lastLayer;
    //printoutput(layer->output, 10, layer->outputWidth, layer->outputHeight, layer->outputChannels);
    save_density_pgm(layer->output, layer->outputWidth, layer->outputHeight, "heapmap/density_map.pgm");
    //create_model2(layer->output);
}

void create_model2(float* image_data) {
    /*
    128x128x1 圖片

    到(kernel_size, kernel, strdie)
    3-32-2 -> 4-32-4 -> 4-8-4 -> 4-1-1
    */
    // for density map
    printf("Creating model2 for inference...\n");
    CNN* density_map = createCNN();
    int inputWidth = 128, inputHeight = 128, inputChannels = 1;
    float *image = (float*)image_data;
    addConvLayer(density_map, 3, 32, 2, 1, RELU, inputWidth, inputHeight, inputChannels);
    Layer *layer = density_map->lastLayer;
    //convertConvWeights(layer->params.conv.weights, model2_cnn_conv_1_w, 3, 3, layer->inputChannels, layer->outputChannels);
    //print oc, inputChannels, filterHeight, filterWidth
    printf("[%d][%d][%d][%d]\n", layer->outputChannels, layer->inputChannels, 3, 3);
    //layer->params.conv.bias = (float*)model2_cnn_conv_1_b;
    addConvLayer(density_map, 4, 32, 4, 1, RELU, layer->outputWidth, layer->outputHeight, layer->outputChannels);
    layer = density_map->lastLayer;
    //convertConvWeights(layer->params.conv.weights, model2_cnn_conv_2_w, 4, 4, layer->inputChannels, layer->outputChannels);
    printf("[%d][%d][%d][%d]\n", layer->outputChannels, layer->inputChannels, 3, 3);
    //layer->params.conv.bias = (float*)model2_cnn_conv_2_b;
    addConvLayer(density_map, 4, 8, 4, 1, RELU, layer->outputWidth, layer->outputHeight, layer->outputChannels);
    layer = density_map->lastLayer;
    //convertConvWeights(layer->params.conv.weights, model2_cnn_conv_3_w, 4, 4, layer->inputChannels, layer->outputChannels);
    //layer->params.conv.bias = (float*)model2_cnn_conv_3_b;
    addConvLayer(density_map, 4, 1, 1, 0, NONE, layer->outputWidth, layer->outputHeight, layer->outputChannels);
    layer = density_map->lastLayer;
    //convertConvWeights(layer->params.conv.weights, model2_cnn_conv_4_w, 4, 4, layer->inputChannels, layer->outputChannels);
    //layer->params.conv.bias = (float*)model2_cnn_conv_4_b;
    forward(density_map, (float*)image);
    //printoutput(layer->output, 1, layer->outputWidth, layer->outputHeight, layer->outputChannels);
}


int main() {
    // input: [C][H][W] flattened
    // paddedInput: [C][H+2P][W+2P] flattened
    // weights: [OC][IC][FH][FW] flattened
    // output: [OC][OH][OW] flattened
    create_model1() ;
    return 0;
}