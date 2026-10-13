# `pipeline.py` — DeepStream LPR runner

Long-lived DeepStream detect → OCR pipeline used by the FastAPI control plane (`app.py`).  
Builds mux → nvinfer → RGBA → pad probe → tiler → OSD → fakesink from `cameras.yaml`, runs until `stop()`, and pushes plate hits into a ring buffer (optional MinIO upload).

---

## GStreamer graph

```mermaid
flowchart LR
  subgraph sources ["Sources (N cameras)"]
    U0["nvurisrcbin cam0"]
    U1["nvurisrcbin cam1"]
    Un["nvurisrcbin camN-1"]
  end

  U0 --> Q0[queue] --> M0["m.sink_0"]
  U1 --> Q1[queue] --> M1["m.sink_1"]
  Un --> Qn[queue] --> Mn["m.sink_N-1"]

  M0 --> MUX["nvstreammux<br/>batch-size=N<br/>nvbuf-memory-type=3"]
  M1 --> MUX
  Mn --> MUX

  MUX --> PGIE["nvinfer PGIE<br/>YOLO plate TRT"]
  PGIE --> CONV["nvvideoconvert<br/>name=conv<br/>RGBA + UNIFIED"]
  CONV --> PROBE{{"pad probe<br/>_pgie_probe"}}
  PROBE --> TILER[nvmultistreamtiler]
  TILER --> CONV2[nvvideoconvert]
  CONV2 --> OSD["nvdsosd<br/>boxes + OCR text"]
  OSD --> SINK["preview.sink:<br/>fakesink | file (.ts) | egl"]
```

Built by `build_launch()`. The probe is attached on **`conv` src pad** (after RGBA), so crops see a CPU-mappable surface.

**Watch overlays:** set `preview.sink: file` and open `output/ds_lpr_preview.ts` in VLC/ffplay (green plate boxes + OCR labels).

---

## Processes and threads

```mermaid
flowchart TB
  subgraph api ["API process"]
    APP[FastAPI app.py]
    RUN["pipeline_runner<br/>DsLprPipeline"]
    APP -->|start / stop / health / results| RUN

    subgraph gst_thread ["Thread: ds-lpr-pipeline"]
      GST["Gst.parse_launch + PLAYING"]
      LOOP[GLib.MainLoop]
      BUS[_bus_call EOS / ERROR]
      PROBE2["_pgie_probe on each buffer"]
      GST --> LOOP
      LOOP --> BUS
      LOOP --> PROBE2
    end
    RUN --> gst_thread
  end

  subgraph ocr_child ["OCR child process (spawn)"]
    OCR["OcrProcessClient worker<br/>RobustScanner TRT + pycuda"]
  end

  PROBE2 -->|"crop ndarray via Queue"| OCR
  OCR -->|"text, conf"| PROBE2
  PROBE2 -->|PlateHit| RING["ring buffer _results"]
  PROBE2 -->|optional| MINIO[HitUploader / MinIO]
```

OCR never loads pycuda in the DeepStream process (avoids CUDA-context SIGSEGV).

---

## Pad probe (plate → OCR → hit)

```mermaid
flowchart TD
  BUF[Gst buffer on conv src] --> META[gst_buffer_get_nvds_batch_meta]
  META --> FRAME["For each NvDsFrameMeta"]
  FRAME --> SID["source_id → source_map → CameraRow"]
  SID --> OBJ["For each NvDsObjectMeta"]
  OBJ --> CHK{"class_id == plate_class<br/>and OCR interval OK?"}
  CHK -->|no| NEXT[next object]
  CHK -->|yes| MAP["_surface_to_bgr<br/>get_nvds_buf_surface → copy → unmap"]
  MAP --> CROP["_ocr_one: bbox crop"]
  CROP --> TRT["ocr.process_images crop"]
  TRT --> FILT{"conf ≥ thresh<br/>and text not junk?"}
  FILT -->|no| NEXT
  FILT -->|yes| HIT["PlateHit → ring buffer<br/>+ optional MinIO upload"]
  HIT --> OSDTXT["set display_text for nvdsosd"]
```

Camera identity always comes from `frame_meta.source_id` (mux sink index), not from frame arrival order.

---

## Camera → `source_id` map

```text
cameras.yaml order          nvstreammux              probe
─────────────────          ─────────────            ─────
cameras[0]  ──►  nvurisrcbin → m.sink_0  ──►  source_id = 0  →  cam_id / bucket
cameras[1]  ──►  nvurisrcbin → m.sink_1  ──►  source_id = 1  →  cam_id / bucket
cameras[i]  ──►  nvurisrcbin → m.sink_i  ──►  source_id = i  →  cam_id / bucket
```

Parsed by `cameras_from_config()` into `CameraRow(source_id, cam_id, url, bucket)`.

---

## Start / stop lifecycle

```mermaid
sequenceDiagram
  participant API as FastAPI
  participant P as DsLprPipeline
  participant OCR as OCR child
  participant T as Gst thread

  API->>P: start(cfg, uploader)
  P->>P: cameras_from_config + engine checks
  P->>OCR: ocr_client.start()
  P->>T: Thread(_thread_main, launch)
  T->>T: parse_launch, add_probe, PLAYING
  T-->>P: _started_event
  P-->>API: {status: started, cameras: N}

  loop while PLAYING
    T->>T: _pgie_probe → OCR → enqueue hit
  end

  API->>P: stop()
  P->>T: EOS / force quit
  P->>OCR: ocr.stop()
  P-->>API: {status: stopped}
```

---

## Main types and API

| Symbol | Role |
|--------|------|
| `CameraRow` | One camera: `source_id`, `cam_id`, `url`, `bucket` |
| `PlateHit` | Accepted OCR result for `/results` |
| `_Runtime` | Probe state: OCR client, uploader, counters, `source_map` |
| `HitUploader` | Protocol for MinIO (or other) upload |
| `DsLprPipeline` | start/stop/health/get_results + Gst thread |
| `pipeline_runner` | Process-wide singleton used by `app.py` |
| `build_launch()` | gst-launch string (UNIFIED mem + RGBA) |
| `_surface_to_bgr()` | Map NVMM RGBA → BGR numpy (needs `nvbuf-memory-type=3`) |
| `_pgie_probe()` | Per-buffer detect meta → crop → OCR → hit |

### Public methods

```text
start(cfg, uploader=None)  →  {status, cameras}
stop(timeout=30)           →  {status}
health()                   →  running, frames, plate_dets, ocr_*
get_results(limit=50)      →  newest PlateHit dicts
```

---

## Why UNIFIED + RGBA

| Setting | Value | Reason |
|---------|--------|--------|
| `nvbuf-memory-type` | `3` (`CUDA_UNIFIED`) | `pyds.get_nvds_buf_surface` safe on dGPU |
| Format before probe | `RGBA` | Required layout for surface map |
| OCR process | spawn child | Isolates TRT/pycuda from DeepStream CUDA |

Wrong memory type or format → surface-map segfault. In-process OCR → CUDA-context segfault (fixed via `ocr_proc.py`).

---

## Related files

| File | Relation to `pipeline.py` |
|------|---------------------------|
| `cameras.yaml` | Config consumed by `start()` |
| `ocr_proc.py` | OCR child client (`OcrProcessClient`) |
| `minio_sink.py` | Implements `HitUploader` |
| `app.py` | Calls `pipeline_runner` |
| `plate_pipeline/config_infer_primary_plate.txt` | `nvinfer` engine / parser config |
| `README.md` | Service-wide architecture and ops |
