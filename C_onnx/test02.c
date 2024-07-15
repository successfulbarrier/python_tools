/*
在RK3588上推理onnx模型
安装RKNN Toolkit:
下载并安装RKNN Toolkit。你可以从Rockchip的官方GitHub页面获取：RKNN Toolkit
*/

/*
转换ONNX模型:
python
   from rknn.api import RKNN

   # 创建RKNN对象
   rknn = RKNN()

   # 加载ONNX模型
   rknn.load_onnx(model='path/to/your/model.onnx')

   # 配置模型
   rknn.config(mean_values=[[0, 0, 0]], std_values=[[1, 1, 1]])

   # 编译模型
   rknn.build(do_quantization=False)

   # 导出RKNN模型
   rknn.export_rknn('path/to/your/model.rknn')

   # 释放RKNN对象
   rknn.release()
*/

// 使用C语言推理模型
#include <stdio.h>
#include <stdlib.h>
#include "rknn_api.h"

int main() {
    // 初始化RKNN
    rknn_context ctx;
    int ret = rknn_init(&ctx, "path/to/your/model.rknn", 0, 0, NULL);
    if (ret < 0) {
        fprintf(stderr, "初始化RKNN失败: %d\n", ret);
        return -1;
    }

    // 准备输入数据
    int input_size = 224 * 224 * 3; // 假设输入大小为224x224x3
    unsigned char* input_data = (unsigned char*)malloc(input_size);
    // 填充输入数据
    for (int i = 0; i < input_size; i++) {
        input_data[i] = 0; // 示例数据
    }

    // 创建输入张量
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = input_size;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].buf = input_data;

    // 设置输入
    ret = rknn_inputs_set(ctx, 1, inputs);
    if (ret < 0) {
        fprintf(stderr, "设置输入失败: %d\n", ret);
        free(input_data);
        rknn_destroy(ctx);
        return -1;
    }

    // 运行推理
    ret = rknn_run(ctx, NULL);
    if (ret < 0) {
        fprintf(stderr, "推理失败: %d\n", ret);
        free(input_data);
        rknn_destroy(ctx);
        return -1;
    }

    // 获取输出
    rknn_output outputs[1];
    memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 1;
    ret = rknn_outputs_get(ctx, 1, outputs, NULL);
    if (ret < 0) {
        fprintf(stderr, "获取输出失败: %d\n", ret);
        free(input_data);
        rknn_destroy(ctx);
        return -1;
    }

    // 处理输出结果
    float* output_data = (float*)outputs[0].buf;
    printf("推理结果: %f\n", output_data[0]);

    // 释放资源
    rknn_outputs_release(ctx, 1, outputs);
    free(input_data);
    rknn_destroy(ctx);

    return 0;
}