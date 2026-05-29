import abc
import os

import numpy as np
import onnxruntime


class EngineBase(abc.ABC):
    def __init__(self, model_path):
        if not os.path.isfile(model_path):
            raise Exception("The model path [%s] can't be found!" % model_path)
        assert model_path.endswith('.onnx'), 'Model path must be a .onnx file.'
        self._framework_type = None

    @property
    def framework_type(self):
        if self._framework_type is None:
            raise Exception("Framework type can't be None")
        return self._framework_type

    @framework_type.setter
    def framework_type(self, value):
        if not isinstance(value, str):
            raise Exception("Framework type need be str")
        self._framework_type = value

    @abc.abstractmethod
    def get_engine_input_shape(self):
        return NotImplemented

    @abc.abstractmethod
    def get_engine_output_shape(self):
        return NotImplemented

    @abc.abstractmethod
    def engine_inference(self):
        return NotImplemented


def _create_session_options(num_threads=None):
    opts = onnxruntime.SessionOptions()
    opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.enable_mem_pattern = True
    opts.enable_mem_reuse = True
    opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    if num_threads is not None and num_threads > 0:
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = 1
    else:
        opts.intra_op_num_threads = 0
        opts.inter_op_num_threads = 0
    return opts


class OnnxEngine(EngineBase):
    def __init__(self, onnx_file_path, num_threads=None):
        EngineBase.__init__(self, onnx_file_path)

        sess_opts = _create_session_options(num_threads)

        providers = onnxruntime.get_available_providers()
        if 'CUDAExecutionProvider' in providers:
            self.session = onnxruntime.InferenceSession(
                onnx_file_path, sess_opts, providers=['CUDAExecutionProvider']
            )
            print("[INFO] ONNX Runtime using GPU (CUDA)")
        else:
            if num_threads is None:
                half_cores = max(1, os.cpu_count() // 2)
                sess_opts.intra_op_num_threads = half_cores
                sess_opts.inter_op_num_threads = 1
            self.session = onnxruntime.InferenceSession(
                onnx_file_path, sess_opts, providers=['CPUExecutionProvider']
            )
            print(f"[INFO] ONNX Runtime using CPU (intra_op_threads={sess_opts.intra_op_num_threads})")

        self.providers = self.session.get_providers()
        self.engine_dtype = (
            np.float16 if 'float16' in self.session.get_inputs()[0].type else np.float32
        )
        self.framework_type = "onnx"
        self.__load_engine_interface()

    def __load_engine_interface(self):
        self.__input_shape = [input.shape for input in self.session.get_inputs()]
        self.__input_names = [input.name for input in self.session.get_inputs()]
        self.__output_shape = [output.shape for output in self.session.get_outputs()]
        self.__output_names = [output.name for output in self.session.get_outputs()]

    def get_engine_input_shape(self):
        return self.__input_shape[0]

    def get_engine_output_shape(self):
        return self.__output_shape, self.__output_names

    def engine_inference(self, input_tensor):
        output = self.session.run(
            self.__output_names, {self.__input_names[0]: input_tensor}
        )
        return output