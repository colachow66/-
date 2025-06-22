
##ONNX导出与TensorRT转换

import torch
import onnx
import onnxruntime as ort
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

def export_to_onnx(model_path="best_model.pth", onnx_path="thyroid_model.onnx"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ThyroidMultiTaskModel().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    dummy_input = torch.randn(1, 1, 512, 512).to(device)
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["seg_output", "cls_output"],
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "seg_output": {0: "batch", 2: "height", 3: "width"},
            "cls_output": {0: "batch"}
        }
    )
    
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX模型已成功导出到 {onnx_path}")

def convert_to_tensorrt(onnx_path="thyroid_model.onnx", engine_path="thyroid_engine.trt"):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise ValueError("ONNX解析失败")
    
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    config.set_flag(trt.BuilderFlag.FP16)
    
    profile = builder.create_optimization_profile()
    profile.set_shape("input", (1, 1, 512, 512), (1, 1, 512, 512), (1, 1, 1024, 1024))
    config.add_optimization_profile(profile)
    
    serialized_engine = builder.build_serialized_network(network, config)
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    
    print(f"TensorRT引擎已成功导出到 {engine_path}")

if __name__ == "__main__":
    export_to_onnx()
    convert_to_tensorrt()