#include <stdio.h>
#include <stdlib.h>
#include <jpeglib.h>

void read_jpg_to_float_array(const char *filename, float **image_array, int *width, int *height, int *channels) {
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    FILE *file = fopen(filename, "rb");

    if (!file) {
        printf("無法開啟檔案 %s\n", filename);
        return;
    }

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, file);
    jpeg_read_header(&cinfo, TRUE);
    
    jpeg_start_decompress(&cinfo);
    *width = cinfo.output_width;
    *height = cinfo.output_height;
    *channels = cinfo.output_components;

    printf("圖片大小: %dx%d, 色彩通道: %d\n", *width, *height, *channels);

    // 建立陣列來儲存圖片數據
    int image_size = (*width) * (*height) * (*channels);
    *image_array = (float *)malloc(image_size * sizeof(float));
    unsigned char *row_pointer = (unsigned char *)malloc((*width) * (*channels));

    for (int i = 0; i < *height; i++) {
        jpeg_read_scanlines(&cinfo, &row_pointer, 1);
        for (int j = 0; j < (*width) * (*channels); j++) {
            (*image_array)[i * (*width) * (*channels) + j] = row_pointer[j] / 255.0f;  // 正規化到 [0,1]
        }
    }

    free(row_pointer);
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(file);
}

void write_array_to_txt(const char *filename, float *image_array, int width, int height, int channels) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        printf("無法開啟檔案 %s 進行寫入\n", filename);
        return;
    }

    fprintf(file, "圖片大小: %dx%d, 色彩通道: %d\n", width, height, channels);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width * channels; j++) {
            fprintf(file, "%.6f ", image_array[i * width * channels + j]);  // 以浮點數格式寫入
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

int main() {
    float *image_array;
    int width, height, channels;

    read_jpg_to_float_array("41.jpg", &image_array, &width, &height, &channels);
    write_array_to_txt("output.txt", image_array, width, height, channels);

    free(image_array);
    return 0;
}
