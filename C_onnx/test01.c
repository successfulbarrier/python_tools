/*
在PC上用C语言推理onnx模型
下载并安装ONNX Runtime库，这是一个高效的推理引擎，支持多种平台和硬件加速。
*/

#include <stdio.h>
#include <stdlib.h>
#include <onnxruntime/core/session/onnxruntime_c_api.h>

int main() {
    // 初始化ONNX Runtime
    OrtEnv* env;
    OrtStatus* status = OrtCreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env);
    if (status != NULL) {
        fprintf(stderr, "创建ONNX Runtime环境失败: %s\n", OrtGetErrorMessage(status));
        OrtReleaseStatus(status);
        return -1;
    }

    // 创建会话选项
    OrtSessionOptions* session_options;
    OrtCreateSessionOptions(&session_options);

    // 加载ONNX模型
    const char* model_path = "path/to/your/model.onnx";
    OrtSession* session;
    status = OrtCreateSession(env, model_path, session_options, &session);
    if (status != NULL) {
        fprintf(stderr, "加载模型失败: %s\n", OrtGetErrorMessage(status));
        OrtReleaseStatus(status);
        OrtReleaseSessionOptions(session_options);
        OrtReleaseEnv(env);
        return -1;
    }

    // 准备输入数据（假设模型有一个输入，形状为[1, 3, 224, 224]）
    size_t input_tensor_size = 1 * 3 * 224 * 224;
    float* input_tensor_values = (float*)malloc(input_tensor_size * sizeof(float));
    // 填充输入数据
    for (size_t i = 0; i < input_tensor_size; i++) {
        input_tensor_values[i] = 1.0f; // 示例数据
    }

    // 获取输入节点信息
    size_t num_input_nodes;
    OrtAllocator* allocator;
    OrtCreateDefaultAllocator(&allocator);
    OrtSessionGetInputCount(session, &num_input_nodes);
    char* input_name;
    OrtSessionGetInputName(session, 0, allocator, &input_name);

    // 创建输入张量
    OrtMemoryInfo* memory_info;
    OrtCreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
    const int64_t input_shape[] = {1, 3, 224, 224};
    OrtValue* input_tensor = NULL;
    OrtCreateTensorWithDataAsOrtValue(memory_info, input_tensor_values, input_tensor_size * sizeof(float), input_shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor);
    OrtReleaseMemoryInfo(memory_info);

    // 准备输入和输出
    const OrtValue* input_tensors[] = {input_tensor};
    const char* input_names[] = {input_name};
    OrtValue* output_tensor = NULL;
    const char* output_names[] = {"output"};

    // 运行推理
    status = OrtRun(session, NULL, input_names, input_tensors, 1, output_names, 1, &output_tensor);
    if (status != NULL) {
        fprintf(stderr, "推理失败: %s\n", OrtGetErrorMessage(status));
        OrtReleaseStatus(status);
        OrtReleaseValue(input_tensor);
        OrtReleaseSession(session);
        OrtReleaseSessionOptions(session_options);
        OrtReleaseEnv(env);
        return -1;
    }

    // 处理输出结果
    float* output_data;
    OrtGetTensorMutableData(output_tensor, (void**)&output_data);
    printf("推理结果: %f\n", output_data[0]);

    // 释放资源
    OrtReleaseValue(output_tensor);
    OrtReleaseValue(input_tensor);
    OrtReleaseSession(session);
    OrtReleaseSessionOptions(session_options);
    OrtReleaseEnv(env);
    free(input_tensor_values);
    OrtReleaseAllocator(allocator);

    return 0;
}