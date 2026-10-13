# DeepStream LPR Service — architecture, TensorRT, and session challenges

FastAPI control plane around a DeepStream detect pipeline plus GPU OCR, uploading plate crops to per-camera MinIO buckets.

For `pipeline.py` internals (GStreamer graph, probe flow, start/stop diagrams) see **[PIPELINE.md](./PIPELINE.md)**.

This document records **what runs where**, **how TensorRT is used**, and the **challenges hit while bringing the service up** (verify session).

---

## Quick start

```bash
# Base image must already be local (no docker pull from NGC):
#   nvcr.io/nvidia/deepstream:7.1-samples-multiarch

./deepstream/ds_lpr_service/build.sh          # cache-only build → gpu-decoding:ds-lpr-service
NUM_CAMS=2 ./scripts/rtsp_emulator/start_rtsp_emu.sh
./deepstream/ds_lpr_service/run.sh            # host network + GPU

curl -X POST http://127.0.0.1:8003/start
curl 'http://127.0.0.1:8003/results?limit=20'
curl -X POST http://127.0.0.1:8003/stop
```

- API / local docs: http://127.0.0.1:8003/docs  
- Config: `cameras.yaml` (RTSP URLs, detector/OCR engine paths, MinIO)

---

## End-to-end data path

```
RTSP (nvurisrcbin × N)
        ↓
nvstreammux  (nvbuf-memory-type=3 = CUDA UNIFIED)
        ↓
nvinfer PGIE  →  YOLO plate detector TensorRT engine  [GPU / VRAM]
        ↓
nvvideoconvert → RGBA (still NVMM)
        ↓
pad probe (pyds)
   • map surface → CPU numpy crop
   • send crop to OCR child process
        ↓
OCR subprocess  →  RobustScanner TensorRT engine  [GPU / VRAM]
        ↓
MinIO  (plate JPG + JSON)  +  in-memory /results ring buffer
        ↓
tiler → nvdsosd → fakesink   (preview path; not required for hits)
```

**Inference is on the GPU for both models.**  
CPU does crop marshalling, JPEG encode, HTTP, and MinIO I/O — not the heavy nets.

---

## Two TensorRT engines (different integration styles)

| Role | Engine (typical path) | How TRT is invoked |
|------|------------------------|--------------------|
| **Detector** | `/workspace/models/yolo12_plate_ds_fp16.engine` | DeepStream **`nvinfer`** loads the engine via `config_infer_primary_plate.txt` + DeepStream-Yolo bbox parser `.so` |
| **OCR** | `/workspace/license-plate/models/rb_scaner_fp16.engine` | Standalone **TensorRT Runtime** in Python (`TensorRTInferenceSession` in `license-plate/src/trt_loader.py`), driven by `recognizer_gpu.py` |

### Detector path (DeepStream / nvinfer)

- Engine is deserialized inside the `nvinfer` plugin (C++), not by our Python TRT wrapper.
- Config points at:
  - `model-engine-file=...yolo12_plate_ds_fp16.engine`
  - `custom-lib-path=.../libnvdsinfer_custom_impl_Yolo.so`
  - `parse-bbox-func-name=NvDsInferParseYolo`
- Batching is owned by `nvstreammux` + `nvinfer` (`batch-size` in the infer txt vs mux `batch-size=N` cams).
- Precision: FP16 engine (`network-mode=2` in the infer config).
- Outputs become `NvDsObjectMeta` (class id, bbox, confidence). Plate class id is **30** in our export.

### OCR path (standalone TensorRT + pycuda)

RobustScanner expects **three** named inputs (TRT 10 I/O tensors), roughly:

| Tensor name | Role | Typical dtype |
|-------------|------|----------------|
| `x` | Normalized image batch `[N,C,H,W]` | **float32** |
| `data_0` | Valid width ratios | **float32** |
| `data_1` | Word / position indices (length 40) | **int32** (engine binding) |

Python builds those arrays in `LicensePlateRecognition.preprocess_images()`, then:

```text
TensorRTInferenceSession.run({
  "x": norm_img_batch,      # float32
  "data_0": valid_ratios,   # float32
  "data_1": word_positions, # int64 in numpy → must match engine dtype on copy
})
```

Execution uses:

1. `trt.Runtime.deserialize_cuda_engine(engine_bytes)`
2. `create_execution_context()`
3. Host pinned buffers + device buffers via **pycuda** (or cupy if present)
4. `execute_async_v3` / `execute_async_v2` on a CUDA stream
5. Device→host copy of logits → `SARLabelDecode` postprocess

**Important:** OCR does **not** go through `nvinfer`. It is a second TRT consumer on the same GPU.

---

## Why OCR is a subprocess (CUDA / TensorRT challenge)

### The problem

DeepStream’s `nvinfer` already owns a CUDA context for the detector engine.  
If the same OS process also does:

```python
import pycuda.autoinit   # or retain_primary_context + push
# load rb_scaner_fp16.engine and run inference
```

then under **uvicorn + GStreamer pad probes on streaming threads**, the two TRT users fight over the device context. Observed failures:

- Process **SIGSEGV (exit 139)** shortly after OCR engine load  
- Or frames never reaching the pad probe (`frames={}`) if OCR init runs too early  
- Or pad-probe OCR failing with `explicit_context_dependent … no currently active context`

This is a classic **multi-context / wrong-thread CUDA** issue, not “OCR is too slow.”

### The fix used here

`ocr_proc.py` starts a **`multiprocessing` spawn child**:

- Parent (DeepStream + FastAPI): **never imports pycuda/tensorrt for OCR**  
- Child: loads `rb_scaner_fp16.engine`, owns its own CUDA context, serves crops over a Queue  

Crops are `uint8` BGR numpy arrays (CPU). The child uploads them to GPU for TRT.  
So: **still VRAM inference**, isolated address space / CUDA context.

Tradeoff: crop crosses process boundary (pickle + H2D in the child). Acceptable for plate ROIs; avoids crashing the DS pipeline.

---

## TensorRT dtype bug (session blocker)

### Symptom

Hundreds of `ocr_attempts`, zero `ocr_hits`, logs like:

```text
OCR failed: Cannot cast array data from dtype('float32') to dtype('int32') according to the rule 'same_kind'
```

### Root cause

In `trt_loader.TensorRTInferenceSession.run()`, every input was forced to float32:

```python
host_arr = np.ascontiguousarray(arr, dtype=np.float32)  # WRONG for all tensors
np.copyto(buf.host[...], host_arr.ravel())              # buf for data_1 is int32
```

`HostDeviceMem` is allocated with **`trt.nptype(engine.get_tensor_dtype(name))`**, so `data_1` host memory is **int32**. Copying float32 into that buffer raises the cast error. Inference never produced usable text → nothing uploaded to MinIO.

### Fix

```python
host_arr = np.ascontiguousarray(arr, dtype=buf.dtype)  # float32 for x/data_0, int32 for data_1
```

After this, OCR returned plates and MinIO filled `plates-cam0` / `plates-cam1`.

**Lesson for multi-input TRT engines:** never assume “everything is FP32.” Always honor binding dtypes (and shapes) from the engine.

---

## Other challenges (non-TRT but blocking)

### 1. DeepStream base image missing GStreamer GI typelibs

```text
ValueError: Namespace Gst not available
```

`python3-gi` was present; `gir1.2-gstreamer-1.0` / `python3-gst-1.0` were not.  
**Fix:** offline Ubuntu `.deb`s in `vendor/debs/`, installed in the Dockerfile (no NGC apt).

### 2. GIO libproxy abort on RTSP

```text
terminate called after throwing an instance of 'std::runtime_error'
what(): Unable to read configuration
```

Known DeepStream + `libgiolibproxy` / `libproxy` issue when resolving RTSP URLs.  
**Fix:** `entrypoint.sh` removes the GIO proxy module and sets `GIO_USE_PROXY=no`.

### 3. Stale GStreamer plugin registry

Build-time `Gst.init` without `libcuda` can blacklist DeepStream plugins.  
**Fix:** entrypoint deletes `/root/.cache/gstreamer-1.0` before uvicorn starts; runtime has NVIDIA libs.

### 4. Unified memory for surface map

Pad probe uses `pyds.get_nvds_buf_surface`. On dGPU this needs:

- `nvstreammux` / convert with **`nvbuf-memory-type=3`** (CUDA UNIFIED)  
- RGBA before the probe  

Wrong memory type → segfault or map failure (the “surface-map” class of bugs).

### 5. Cache-only Docker policy

No `docker pull` from NGC during build. Base must already exist (`docker image inspect`).  
Python deps from local `wheelhouse/`; libpython + Gst GI from local debs.  
Tag: `gpu-decoding:ds-lpr-service`.

### 6. Blank Swagger `/docs`

Default FastAPI docs load Swagger UI from `cdn.jsdelivr.net`. Offline / filtered browsers show an empty page even though HTTP 200.  
**Fix:** self-contained HTML at `/docs` that only fetches `/openapi.json`.

---

## Config surface (`cameras.yaml`)

```yaml
detector:
  engine: /workspace/models/yolo12_plate_ds_fp16.engine
  infer_config: /workspace/deepstream/plate_pipeline/config_infer_primary_plate.txt
  plate_class_id: 30
ocr:
  engine: /workspace/license-plate/models/rb_scaner_fp16.engine
  dict: /workspace/license-plate/dictionary_plate/dict90.txt
  conf_thresh: 0.25
cameras:
  - id: cam0
    url: rtsp://127.0.0.1:8554/poll0   # etc.
    bucket: plates-cam0                 # unique MinIO bucket per camera
minio:
  endpoint: 127.0.0.1:9000
  access_key: minioadmin
  secret_key: minioadmin
  secure: false
```

API contract (same shape as the older OpenCV plate server):

| Method | Path | Notes |
|--------|------|--------|
| GET | `/health` | `running`, cameras, OCR counters, frames |
| GET | `/get-config` | YAML + running flag |
| PUT | `/config` | Rejected while running |
| POST | `/start` | Build pipeline, ensure buckets, start OCR worker |
| POST | `/stop` | EOS + join + stop OCR worker |
| GET | `/results?limit=` | Recent hits with `s3://bucket/key` |

Hot-add of RTSP pads without restart is **out of scope** (mux sources fixed at build).

---

## Package layout

| File | Role |
|------|------|
| `app.py` | FastAPI control plane + local `/docs` |
| `pipeline.py` | DeepStream launch string, pad probe, results ring |
| `ocr_proc.py` | Spawn OCR worker (isolated TRT/pycuda) |
| `minio_sink.py` | Ensure buckets; put JPG + JSON |
| `cameras.yaml` | Runtime config |
| `Dockerfile` / `build.sh` | Local-base, wheelhouse, debs |
| `run.sh` / `entrypoint.sh` | GPU + host network; GIO/GST hygiene |

---

## Challenges summary (to this point)

1. **Two TRT consumers on one GPU** — detector inside `nvinfer`, OCR via Python Runtime → must not share a broken CUDA context in one process under GStreamer threads.  
2. **Multi-input OCR engine dtypes** — FP16/FP32 image tensors vs **int32** index tensor; host copies must match bindings.  
3. **DeepStream Python image gaps** — Gst GI typelibs, GIO proxy, plugin registry vs CUDA.  
4. **Operational constraints** — no NGC pull; bake wheels/debs; host-network for RTSP + MinIO.  
5. **DX** — CDN-based Swagger unusable offline → local docs.

What is **not** claimed: a pure “everything inside one nvinfer SGIE” LPR graph. OCR is deliberately a second TensorRT engine outside DeepStream elements, still GPU-accelerated.

---

## Verify gate (done in session)

- [x] Base present locally; image build with `--pull=false`  
- [x] RTSP emu + existing MinIO  
- [x] `POST /start` → detections + OCR attempts/hits  
- [x] `GET /results` returns `cam_id`, `ocr`, `conf`, `object_path`  
- [x] MinIO buckets contain matching `.jpg` + `.json`  
- [x] `POST /stop` returns idle; config can be changed after stop  
