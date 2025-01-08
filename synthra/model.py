import onnxruntime as ort


def load_model(model_path="models/linux/weights-high.onnx"):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    
    ort_session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    return ort_session

