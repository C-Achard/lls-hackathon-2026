# Lemanic Life Sciences Hackathon 2026 - Integrating foundation model detectors with DeepLabCut

This project aims to integrate foundation model detectors with DeepLabCut, a popular tool for markerless pose estimation in animals. 

Current top-down pose estimation pipelines require retraining a detector for each new dataset, increasing resource requirements and potentially requiring additional annotation work. 
By leveraging foundation model detectors, we can potentially bypass the need for retraining, allowing for more efficient and scalable pose estimation across diverse datasets.

## Scope of the project

We implemented a proof-of-concept integration and compared several detectors:

- DART (SAM3-based)
- GroundingDINO
- RF-DETR
- YOLO26-E

<!-- ./images/DART.svg -->
![DART](https://raw.githubusercontent.com/C-Achard/lls-hackathon-2026/main/images/DART.SVG)
<p align="center">
  <em><span style="color: #888888;">Figure 1: Example output from the DART (SAM3-based) detector on trimice (left) and fish (right).</span></em>
</p>

## Results

### Detection performance

We compared the zero-shot performance of the foundation model detectors to a purpose-trained DeepLabCut detector, showing that mAP-50 and mAR-50 are comparable for certain models, namely GroundingDINO and DART, even lightly outperforming DeepLabCut on trimice, without any fine-tuning. 

![Detection Performance](https://raw.githubusercontent.com/C-Achard/lls-hackathon-2026/main/images/DET_BENCH.SVG)
<p align="center">
  <em><span style="color: #888888;">Figure 2: Comparison of foundation model detectors performance (zero-shot), compared to a purpose-trained DeepLabCut detector, on two datasets.</span></em>
</p>

### Pose estimation performance

We then compared, without retraining the pose estimation model, the performance of the full end-to-end pose pipeline, showing that previously top-performing detectors achieve comparable performance to the original DeepLabCut model.
We also use ground truth detections estimated from keypoints to show the upper bound of performance for the pose estimation model.

**Metrics**:

- RMSE: Root Mean Square Error (lower is better)
- mAP: Mean Average Precision (higher is better)
- mAR: Mean Average Recall (higher is better)

Performance is noticeably better on the fish dataset for foundation model detectors, reaching comparable or better performance than DeepLabCut, while on the trimice dataset, performance is worse.
Investigation the failure modes of detectors on the second dataset would be an interesting next step, as well as exploring the impact of retraining the pose estimation model with detections from foundation models, to see if the gap in performance can resolved, and whether foundation model performance inference FPS matches that of DeepLabCut, allowing to determine which approach works best in which scenarios.

#### Fish dataset

![Pose estimation Performance](https://raw.githubusercontent.com/C-Achard/lls-hackathon-2026/main/images/pose_bench1.SVG)
<p align="center">
  <em><span style="color: #888888;">Figure 3: Comparison of pose estimation performance using foundation model detectors versus a purpose-trained DeepLabCut detector and ground truth (GT) detections on the fish dataset.</span></em>
</p>

#### Trimice dataset

![Pose estimation Performance](https://raw.githubusercontent.com/C-Achard/lls-hackathon-2026/main/images/pose_bench2.SVG)
<p align="center">
  <em><span style="color: #888888;">Figure 4: Comparison of pose estimation performance using foundation model detectors versus a purpose-trained DeepLabCut detector and ground truth (GT) detections on the trimice dataset.</span></em>
</p>

<!-- ---
title: DeepLabCut Model Zoo
emoji: 🐕🐁🐴😻🐘🐆🐿🐂🦘🦒
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 6.11.0
python_version: "3.10"
app_file: app.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference -->