# Vehicle-CV-ADAS: ONNX CPU Setup & Documentation

This project has been modified to run on **ONNX Runtime (CPU)**, removing the strict dependency on NVIDIA TensorRT and pycuda. This allows the ADAS simulation to run on a wider range of hardware.

## 🛠️ Work Performed

### 1. Environment Optimization
- **ONNX Migration**: Reconfigured the core engine to prioritize ONNX models.
- **CPU Fallback**: Implemented a silent execution provider validation in `coreEngine.py` to skip broken GPU drivers and use the CPU without noisy error logs.
- **GUI Fix**: Reinstalled `opencv-python` to ensure the simulation window displays correctly on Windows.

### 2. Model Conversion
- **YOLOv8n**: Exported to ONNX for object detection.
- **CULane ResNet18**: Converted from PyTorch to ONNX using custom CPU-compatible scripts.
- Location: `ObjectDetector/models/` and `TrafficLaneDetector/models/`.

### 3. Core Code Fixes (NumPy 1.24+ Compatibility)
Fixed critical bugs caused by deprecated NumPy features in the latest environments:
- Resolved `AttributeError: module 'numpy' has no attribute 'float'` in ByteTrack.
- Fixed `ValueError` ambiguous array comparisons in detection utilities.
- Cleaned up comparison logic (e.g., `if arr != []` → `if arr.size > 0`).

## 📋 How to Set Up & Run

### Prerequisites
- Python 3.10+
- Recommended: `pip install -r requirements.txt` (adjusted for ONNX)

### Installation
```powershell
pip install onnxruntime opencv-python ultralytics numpy scipy manual numba
```

### Running the Demo
- **Primary Demo**:
  ```powershell
  python demo.py
  ```
  Processes `./demo/demo_video.mp4`.
- **Secondary Demo**:
  ```powershell
  python demo2.py
  ```
  Processes `./demo/Simple_Driving_Simulation_Video.mp4`.

- Real-time progress is logged to the terminal.
- **Graceful Exit**: Press **'q'** or **'Ctrl+C'** to stop the simulation safely. Resources are automatically released.

### 🎥 Working Demo Recording
A pre-processed demo recording of the system working with the provided video has been saved to the root directory for quick review:
- **File**: `working_video.mp4`

## 📄 Repository Structure (Modified)
- `coreEngine.py`: Updated with silent provider fallback.
- `demo.py`: Configured for ONNX, local video paths, and graceful crash/interrupt handling.
- `ObjectDetector/utils.py`: NumPy compatibility fixes.
- `ObjectTracker/byteTrack/`: ByteTrack logic updated for modern NumPy.
- `TrafficLaneDetector/convertPytorchToONNX.py`: Enabled CPU-only model conversion.

---
*Documentation updated with final system improvements.*
