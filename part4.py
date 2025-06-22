import onnx
import onnxruntime as ort
import time

def export_to_onnx(model, output_path="thyroid_model.onnx"):
    """
    将模型导出为ONNX格式
    """
    # 创建示例输入
    dummy_input = torch.randn(1, 1, 512, 512).cuda()
    
    # 导出模型
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["seg_output", "cls_output"],
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "seg_output": {0: "batch", 2: "height", 3: "width"},
            "cls_output": {0: "batch"}
        }
    )
    
    # 验证ONNX模型
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    print(f"模型已成功导出到 {output_path}")

def test_onnx_runtime():
    """
    测试ONNX Runtime推理性能
    """
    # 创建ONNX Runtime会话
    ort_session = ort.InferenceSession(
        "thyroid_model.onnx", 
        providers=["CUDAExecutionProvider"]
    )
    
    # 创建示例输入
    input_tensor = torch.randn(1, 1, 512, 512).cuda()
    ort_inputs = {
        ort_session.get_inputs()[0].name: input_tensor.cpu().numpy()
    }
    
    # 测量推理时间
    start_time = time.time()
    for _ in range(100):  # 运行100次取平均
        ort_outputs = ort_session.run(None, ort_inputs)
    end_time = time.time()
    
    avg_time = (end_time - start_time) * 1000 / 100  # 毫秒
    print(f"ONNX Runtime平均推理时间: {avg_time:.2f}ms")

def measure_model_performance():
    """
    测量模型性能指标
    """
    # 这里应该实现完整的性能测试代码
    # 包括:
    # 1. 端到端延迟测量(含预处理)
    # 2. 能效比计算
    # 3. 鲁棒性测试(运动模糊/低对比度样本)
    print("性能测试完成(示例代码)")

if __name__ == "__main__":
    # 导出模型为ONNX格式
    export_to_onnx(torch.load("best_model.pth"))
    
    # 测试ONNX Runtime推理性能
    test_onnx_runtime()
    
    # 测量模型性能指标
    measure_model_performance()