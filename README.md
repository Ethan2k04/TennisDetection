# Real-Time Tennis Ball Impact Detection and Scoring System

![tennis_logo](https://github.com/user-attachments/assets/1f677a89-d2d2-4111-8825-0d71ab92f9a4)

A real-time computer vision system that detects tennis balls hitting a scoring target, locates the impact point, and computes the score. It runs on an embedded edge device, the **Orange Pi 5 Max (Rockchip RK3588)**, with a single camera, and was developed for a commercial tennis training product.

## Highlights

- **Embedded real-time deployment**: runs entirely on an ARM single-board computer. Phase 1 reached **60 FPS** with on-device NPU inference.
- **Two detection approaches, compared on real deployment data**:
  - **Phase 1**: YOLO11 detector converted to RKNN for the RK3588 NPU.
  - **Phase 2**: a lightweight classical vision pipeline (background subtraction and connected-component analysis). It replaced the detector because it handles fast, motion-blurred balls more robustly in this setting.
- **End-to-end pipeline**: target localization, ball detection, impact detection, impact-point estimation, and scoring.

## System Pipeline (current version)

1. **Target localization**: detect the target board using shape-based detection.
2. **Background modeling**: locate the black 10-point zone and sample its background color statistics (hue, saturation, value).
3. **Ball candidate detection**: within the 10-point zone, remove the background and run connected-component analysis to extract ball candidates.
4. **Ball appearance modeling**: estimate the ball's color statistics from the candidates. These statistics are used to detect the ball over the whole target.
5. **Full-target ball detection**: detect the ball over the whole target board.
6. **Impact detection** (`hit_manager.py`): apply background subtraction to the target ROI, resize the result, and count non-zero pixels. Each frame is classified as *no hit*, *uncertain*, *hit*, or *outlier*.
7. **Impact point and scoring**: estimate the impact point from the ball trajectory and map it to the corresponding scoring zone.

## Development History

### Phase 1: Deep-learning detector on the edge NPU (v1.13 – v1.30)

- Trained a **YOLO11** tennis ball detector and deployed it on the RK3588 NPU via **RKNN**.
- Replaced the activation functions with **ReLU** during training for better NPU compatibility and speed.
- **Decoupled camera capture and detection into separate processes**, raising real-time throughput to **60 FPS**.
- Fetched and detected frames concurrently with multiple workers, using a **reorder buffer** so results are emitted in frame order. Multi-process detection was later found to reduce bounding-box tracking latency compared with multi-threading.
- Simplified target detection from color filtering to **shape detection**, which removed sensitivity to lighting changes.
- Designed an impact criterion based on the **change in ball bounding-box size**: an impact is registered when the box size reaches a global minimum. A check that the minimum ranks among the three smallest values filters out noise.

### Phase 2: Lightweight classical vision pipeline (v1.2 – v2.3)

Analysis of the deployed system showed that a deep detector was not the best fit for this task:

1. A tennis ball has very few distinctive visual features, so a deep network offers limited benefit.
2. The detector was not trained on images from the actual deployment scene.
3. Deep-network inference is less efficient on the edge device.
4. Fast-moving balls are heavily motion-blurred and easily blend into a white background, so the detector misses them.

The system was therefore re-designed around background subtraction and color/connectivity analysis (see *System Pipeline*). Further refinements:

- **v2.0**: impact detection by background subtraction on the target ROI.
- **Frame-rate tuning**: above 100 FPS, many ball-like noise blobs are detected, so the frame rate is capped at about 40 FPS to balance responsiveness and stability.
- **v2.3**: after an impact is detected, ball detection is suspended for 1.5 seconds. The target lights up on impact, which would otherwise corrupt the background-subtraction statistics.

## Project Structure

```
├── main.py                 # Entry point and main loop
├── target_manager.py       # Target board and scoring-zone localization
├── tennis_ball_manager.py  # Ball detection and color modeling
├── hit_manager.py          # Impact detection via background subtraction
├── network.py              # Network communication
├── tools.py                # Utility functions
├── constants.py            # Constants
├── meta/
│   ├── config.json         # Configuration
│   └── settings.json       # Runtime settings
├── requirements.txt
└── tennis_logo.png
```

## Requirements

```
numpy
requests==2.32.3
opencv-python==4.10.0.84
```

```bash
pip install -r requirements.txt
python main.py
```

## Usage

- Press **`h`** to relocate the target. This re-detects the target board and the 10-point zone, then updates the background and ball color models.

## Scene Requirements

- Use green tennis balls.
- Avoid white backgrounds behind the target, which reduce contrast with blurred balls.

## Known Limitations

- Impact detection accuracy can be further improved.
- Detection performance depends on lighting and background contrast.
