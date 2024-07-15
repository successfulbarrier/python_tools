/*
在K210上使用C语言推理onnx模型
安装Kendryte SDK:
下载并安装Kendryte SDK。你可以从Kendryte的官方GitHub页面获取：Kendryte SDK
*/


/*
转换ONNX模型:
使用Kendryte提供的工具将ONNX模型转换为K210可以使用的Kmodel格式。你可以使用nncase工具进行转换：nncase
python

   # 安装nncase
   pip install nncase

   # 转换ONNX模型为Kmodel
   nncase compile --input-model path/to/your/model.onnx --output-model path/to/your/model.kmodel --target k210
*/

// 通过SD卡读取模型文件并推理
#include <stdio.h>
#include <stdlib.h>
#include "kpu.h"
#include "plic.h"
#include "sysctl.h"
#include "uarths.h"
#include "utils.h"
#include "ff.h" // FatFs文件系统库

#define KMODEL_SIZE (380 * 1024) // 根据你的模型大小调整

uint8_t model_data[KMODEL_SIZE];

int main() {
    // 初始化系统
    sysctl_pll_set_freq(SYSCTL_PLL0, 800000000UL);
    sysctl_pll_set_freq(SYSCTL_PLL1, 400000000UL);
    sysctl_pll_set_freq(SYSCTL_PLL2, 45158400UL);
    uarths_init();

    // 初始化SD卡
    FATFS sdcard_fs;
    FIL file;
    FRESULT res;
    res = f_mount(&sdcard_fs, "0:", 1);
    if (res != FR_OK) {
        printf("挂载SD卡失败\n");
        return -1;
    }

    // 打开Kmodel文件
    res = f_open(&file, "0:/model.kmodel", FA_READ);
    if (res != FR_OK) {
        printf("打开模型文件失败\n");
        return -1;
    }

    // 读取Kmodel文件
    UINT br;
    res = f_read(&file, model_data, KMODEL_SIZE, &br);
    if (res != FR_OK || br != KMODEL_SIZE) {
        printf("读取模型文件失败\n");
        f_close(&file);
        return -1;
    }
    f_close(&file);

    // 初始化KPU
    kpu_model_context_t task;
    if (kpu_load_kmodel(&task, model_data) != 0) {
        printf("加载Kmodel失败\n");
        return -1;
    }

    // 准备输入数据
    uint8_t *input_data = malloc(224 * 224 * 3); // 根据你的模型输入大小调整
    // 填充输入数据
    for (int i = 0; i < 224 * 224 * 3; i++) {
        input_data[i] = 0; // 示例数据
    }

    // 运行推理
    size_t output_size;
    uint8_t *output_data;
    if (kpu_run_kmodel(&task, input_data, DMAC_CHANNEL5, &output_data, &output_size) != 0) {
        printf("推理失败\n");
        free(input_data);
        return -1;
    }

    // 处理输出结果
    printf("推理结果: %d\n", output_data[0]);

    // 释放资源
    free(input_data);

    return 0;
}