"""TensorRT inference session — prefers cupy, falls back to pycuda (no network needed)."""

from __future__ import annotations

import numpy as np
import tensorrt as trt

_USE_CUPY = False
try:
    import cupy as cp
    import cupyx

    _USE_CUPY = True
except ImportError:
    import pycuda.driver as cuda

    # Prefer the device primary context (shared with DeepStream/nvinfer) instead of
    # pycuda.autoinit, which creates a separate context and can SIGSEGV the pipeline.
    cuda.init()
    _pycuda_ctx = cuda.Device(0).retain_primary_context()
    _pycuda_ctx.push()


class HostDeviceMem:
    """Pinned host + device buffer pair."""

    def __init__(self, size, dtype):
        self.size = size
        self.dtype = np.dtype(dtype)
        if _USE_CUPY:
            self.host = cupyx.zeros_pinned(size, self.dtype)
            self.device = cp.zeros(size, self.dtype)
            self._stream = None
        else:
            self.host = cuda.pagelocked_empty(size, self.dtype)
            self.device = cuda.mem_alloc(int(self.host.nbytes))

    @property
    def nbytes(self):
        return int(self.host.nbytes)

    @property
    def hostptr(self):
        if _USE_CUPY:
            return self.host.ctypes.data
        return self.host.base.get_pointer() if hasattr(self.host, "base") else self.host.__array_interface__["data"][0]

    @property
    def devptr(self):
        if _USE_CUPY:
            return self.device.data.ptr
        return int(self.device)

    def copy_htod_async(self, stream):
        if _USE_CUPY:
            self.device.data.copy_from_host_async(self.hostptr, self.nbytes, stream)
        else:
            cuda.memcpy_htod_async(self.device, self.host, stream)

    def copy_dtoh_async(self, stream):
        if _USE_CUPY:
            self.device.data.copy_to_host_async(self.hostptr, self.nbytes, stream)
        else:
            cuda.memcpy_dtoh_async(self.host, self.device, stream)


class TensorRTInferenceSession:
    def __init__(self, engine_path, use_gpu=True, gpu_id=0):
        if not use_gpu:
            raise ValueError("TensorRT requires GPU")
        if _USE_CUPY:
            cp.cuda.Device(gpu_id).use()
            self.stream = cp.cuda.Stream(non_blocking=True)
            self._stream_ptr = lambda: self.stream.ptr
        else:
            # pycuda.autoinit already bound a context
            self.stream = cuda.Stream()
            self._stream_ptr = lambda: self.stream.handle

        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError("Failed to load TRT engine")

        self.context = self.engine.create_execution_context()
        self.is_trt10 = hasattr(self.engine, "num_io_tensors")
        self.input_names = []
        self.output_names = []
        self.input_buffers = {}
        self.output_buffers = {}
        self.bindings = []

        if self.is_trt10:
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                shape = self.engine.get_tensor_shape(name)
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    self.input_names.append(name)
                else:
                    self.output_names.append(name)
                setattr(self, f"{name}_shape", shape)
        else:
            for i in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(i)
                shape = self.engine.get_binding_shape(i)
                if self.engine.binding_is_input(i):
                    self.input_names.append(name)
                else:
                    self.output_names.append(name)
                setattr(self, f"{name}_shape", shape)

        backend = "cupy" if _USE_CUPY else "pycuda"
        print(f"TensorRTInferenceSession ready ({backend})")

    def _allocate_buffers(self, input_shapes):
        self.input_buffers.clear()
        self.output_buffers.clear()
        bindings = []

        if self.is_trt10:
            for name in self.input_names:
                shape = input_shapes[name]
                size = trt.volume(shape)
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                buf = HostDeviceMem(size, dtype)
                self.input_buffers[name] = buf
                bindings.append(buf.devptr)
            for name in self.output_names:
                shape = self.engine.get_tensor_shape(name)
                if -1 in shape:
                    batch_size = list(input_shapes.values())[0][0]
                    shape = tuple(s if s != -1 else batch_size for s in shape)
                size = trt.volume(shape)
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                buf = HostDeviceMem(size, dtype)
                self.output_buffers[name] = buf
                bindings.append(buf.devptr)
            return bindings

        for name in self.input_names + self.output_names:
            shape = getattr(self, f"{name}_shape", None)
            if name in input_shapes:
                shape = input_shapes[name]
            elif shape is None:
                raise RuntimeError(f"Unknown shape for {name}")
            size = trt.volume(shape)
            dtype = trt.nptype(self.engine.get_binding_dtype(self.engine[name]))
            buf = HostDeviceMem(size, dtype)
            if name in self.input_names:
                self.input_buffers[name] = buf
            else:
                self.output_buffers[name] = buf
            bindings.append(buf.devptr)
        return bindings

    def run(self, input_dict):
        input_shapes = {name: arr.shape for name, arr in input_dict.items()}
        self.bindings = self._allocate_buffers(input_shapes)

        for name, arr in input_dict.items():
            host_arr = np.ascontiguousarray(arr, dtype=np.float32)
            buf = self.input_buffers[name]
            if trt.volume(host_arr.shape) != buf.size:
                self.bindings = self._allocate_buffers(input_shapes)
                buf = self.input_buffers[name]
            np.copyto(buf.host[: host_arr.size], host_arr.ravel())
            buf.copy_htod_async(self.stream)

        if self.is_trt10:
            for name, shape in input_shapes.items():
                self.context.set_input_shape(name, shape)
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                self.context.set_tensor_address(name, self.bindings[i])
            self.context.execute_async_v3(stream_handle=self._stream_ptr())
        else:
            for i, name in enumerate(self.input_names):
                self.context.set_binding_shape(i, input_shapes[name])
            self.context.execute_async_v2(
                bindings=self.bindings, stream_handle=self._stream_ptr()
            )

        for name, buf in self.output_buffers.items():
            buf.copy_dtoh_async(self.stream)

        if _USE_CUPY:
            self.stream.synchronize()
        else:
            self.stream.synchronize()

        outputs = []
        for name in self.output_names:
            buf = self.output_buffers[name]
            if self.is_trt10:
                out_shape = tuple(self.context.get_tensor_shape(name))
            else:
                out_shape = tuple(self.context.get_binding_shape(self.engine[name]))
            outputs.append(np.array(buf.host[: trt.volume(out_shape)]).reshape(out_shape))
        return outputs
