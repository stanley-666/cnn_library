// cnn_train.h
#ifndef CNN_TRAIN_H
#define CNN_TRAIN_H
#pragma once
#include "cnn.h"     // for CNN*, Layer*
#include "cnn_utils.h"

void backwardConv(Layer *layer);
void backwardPool(Layer *layer);
void backwardFC(Layer *layer, float *target);

void updateWeightsConv(Layer *layer, float learningRate);
void updateWeightsFC(Layer *layer, float learningRate);

#endif
