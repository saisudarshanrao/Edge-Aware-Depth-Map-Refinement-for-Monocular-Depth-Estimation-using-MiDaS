# Edge-Aware Depth Map Refinement for Monocular Depth Estimation using MiDaS

**Author:** Sai Gandham

## Overview

This project improves the visual quality of monocular depth estimation outputs produced by MiDaS. The baseline MiDaS model generates strong depth maps, but the results may still contain blurred boundaries, noisy flat regions, and weak foreground/background transitions. To address this, the project adds a lightweight edge-aware refinement stage using guided filtering and image-guided post-processing.

The idea is simple: first generate a depth map using MiDaS, then refine it using the original RGB image so that object boundaries become sharper and the final depth output looks cleaner and more structurally consistent.

## How This Project Was Derived

This project was developed as a coursework project in deep learning and computer vision. It was derived from the idea of improving a pretrained monocular depth estimation model without retraining it. Instead of modifying the neural network architecture, the project focuses on a practical post-processing enhancement that can be applied to MiDaS outputs.

The final pipeline combines:

* MiDaS as the baseline depth estimator
* edge-aware guided filtering as the refinement step
* side-by-side comparison images for analysis

This makes the project easy to run, easy to explain, and strong enough for a course submission.

## Features

* Uses MiDaS for monocular depth estimation
* Refines depth maps using edge-aware filtering
* Preserves sharp object boundaries
* Reduces noise in smooth regions
* Generates comparison panels for qualitative analysis
* Works on both indoor and outdoor images

## Project Structure

```text
MiDaS/
├── input_images/        # Input RGB images
├── output/              # Raw MiDaS depth maps
├── refined_output/      # Refined depth maps and comparisons
├── weights/             # Pretrained model weights
├── run.py               # MiDaS inference script
├── refine_depth.py      # Edge-aware refinement script
└── README.md            # Project documentation
```

## Requirements

* Python 3.9 or higher
* OpenCV
* OpenCV Contrib
* NumPy
* PyTorch
* timm
* imutils

## Installation

Install the required dependencies using:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install torch torchvision timm imutils numpy opencv-python opencv-contrib-python
```

If your system has issues with the default PyTorch installation, use the CPU wheel index:

```bash
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install timm imutils numpy opencv-python opencv-contrib-python
```

## Download Model Weights

MiDaS requires pretrained weights before inference.

Example:

```bash
mkdir -p weights
cd weights
curl -L -o dpt_large_384.pt https://github.com/isl-org/MiDaS/releases/download/v3/dpt_large_384.pt
cd ..
```

If the download command does not work, download the weight file manually and place it inside the `weights/` folder.

## How to Run the Baseline MiDaS Model

Put your input images inside the `input_images/` folder.

Then run:

```bash
python3 run.py --model_type dpt_large_384 --input_path input_images --output_path output --grayscale
```

This will generate raw depth maps in the `output/` folder.

## How to Run the Refinement Step

After the baseline depth maps are generated, run:

```bash
python3 refine_depth.py --input_dir input_images --depth_dir output --output_dir refined_output
```

This will create:

* `refined_output/raw/` for raw depth maps
* `refined_output/refined/` for refined depth maps
* `refined_output/compare/` for side-by-side comparison panels

## Expected Output

For each image, the project produces:

* the original RGB image
* the raw MiDaS depth map
* the refined depth map
* a comparison image showing all three side by side

## Method Summary

1. Input RGB image is passed to MiDaS.
2. MiDaS predicts a baseline depth map.
3. The depth map is normalized.
4. The original image is used as guidance for edge-aware refinement.
5. The refined output is saved and compared with the raw output.

## Why This Improvement Helps

The guided refinement step improves the visual quality of depth maps by:

* sharpening object boundaries
* keeping structural edges more consistent
* smoothing noisy flat areas
* improving the overall visual appearance of the depth map

## Limitations

* This project does not retrain the model.
* The method improves visual quality more than numerical accuracy.
* Very fine texture details may still be smoothed slightly.

## Use Cases

This project can be useful for:

* computer vision coursework
* depth map visualization
* AR/VR preprocessing
* robotics perception demos
* scene understanding experiments

## Author

**Sai sudarshan**

**Sai chandra raju**

**Bhagya shree**

## License

This project is created for academic and educational purposes.

## Acknowledgements

* MiDaS for the baseline depth estimation model
* OpenCV for image processing utilities
* Guided filtering ideas from edge-preserving image smoothing methods
