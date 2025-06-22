import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import time

class TensorRTInfer:
    def __init__(self, engine_path="thyroid_engine.trt"):
        self.logger = trt.Logger(trt.Logger.INFO)
        self.engine = self._load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        
        # 分配内存
        self.h_input = cuda.pagelocked_empty(
            trt.volume(self.engine.get_binding_shape(0)),
            dtype=np.float32
        )
        self.h_output_seg = cuda.pagelocked_empty(
            trt.volume(self.engine.get_binding_shape(1)),
            dtype=np.float32
        )
        self.h_output_cls = cuda.pagelocked_empty(
            trt.volume(self.engine.get_binding_shape(2)),
            dtype=np.float32
        )
        
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output_seg = cuda.mem_alloc(self.h_output_seg.nbytes)
        self.d_output_cls = cuda.mem_alloc(self.h_output_cls.nbytes)
    
    def _load_engine(self, engine_path):
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    
    def infer(self, image):
        # 预处理
        image = cv2.resize(image, (512, 512))
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=(0, 1))
        
        # 拷贝到主机内存
        np.copyto(self.h_input, image.ravel())
        
        # 传输到设备
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        
        # 执行推理
        self.context.execute_async_v2(
            bindings=[int(self.d_input), int(self.d_output_seg), int(self.d_output_cls)],
            stream_handle=self.stream.handle
        )
        
        # 传输结果回主机
        cuda.memcpy_dtoh_async(self.h_output_seg, self.d_output_seg, self.stream)
        cuda.memcpy_dtoh_async(self.h_output_cls, self.d_output_cls, self.stream)
        
        # 同步流
        self.stream.synchronize()
        
        # 后处理
        seg_mask = self.h_output_seg.reshape(1, 512, 512)
        cls_logits = self.h_output_cls.reshape(4)
        
        return seg_mask, cls_logits
    
    def __del__(self):
        self.d_input.free()
        self.d_output_seg.free()
        self.d_output_cls.free()

if __name__ == "__main__":
    infer = TensorRTInfer()
    
    # 测试图像
    test_image = cv2.imread("test.png", cv2.IMREAD_GRAYSCALE)
    
    # 预热
    for _ in range(10):
        infer.infer(test_image)
    
    # 性能测试
    start_time = time.time()
    for _ in range(100):
        seg_mask, cls_logits = infer.infer(test_image)
    end_time = time.time()
    
    avg_time = (end_time - start_time) * 1000 / 100
    print(f"TensorRT平均推理时间: {avg_time:.2f}ms")
    
    # 可视化结果
    seg_mask = (seg_mask > 0.5).astype(np.uint8)
    plt.imshow(seg_mask[0], cmap='gray')
    plt.title("Segmentation Result")
    plt.show()