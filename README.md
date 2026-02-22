# Auto-Netlist: Sketch-to-SPICE AI

## Project Overview
Auto-Netlist is a computer vision and graph theory application designed to bridge the gap between hand-drawn analog schematics and Electronic Design Automation (EDA) software. By leveraging a fine-tuned object detection model and morphological image processing, this pipeline automatically translates raw sketches of electronic circuits into machine-readable LTspice netlists (.net).

This project demonstrates a practical intersection of deep learning, computer vision, and core circuit theory.
try live: https://huggingface.co/spaces/DSAkash/Sketch-To-Spice
Example
![Uploading image.png…]()


## Key Features
* Automated Component Detection: Utilizes a custom-trained YOLOv8 model to identify standard circuit components (Op-Amps, BJTs, FETs, Resistors, Capacitors, etc.) from hand-drawn images.
* Topological Wire Extraction: Applies OpenCV-based binarization, component masking, and morphological skeletonization to isolate connecting wires.
* Nodal Analysis & Mapping: Extracts the topological graph using connected-component labeling to identify distinct equipotential electrical nodes.
* SPICE Netlist Generation: An intersection algorithm maps component bounding boxes to extracted electrical nodes, dynamically generating syntactically correct SPICE directives.
* Interactive Web Interface: Packaged and deployed using Gradio for real-time inference and netlist generation.

## Technical Pipeline
1. Object Detection: A YOLOv8 Nano model, trained on a custom dataset of hand-drawn circuits, outputs spatial coordinates (bounding boxes) and class labels for each component.
2. Image Preprocessing & Masking: The original image is converted to a binary format. The bounding box coordinates from the detection phase are used to mask (erase) the components, leaving only the hand-drawn wires.
3. Skeletonization: The isolated wires are thinned to 1-pixel-wide paths using the Zhang-Suen skeletonization algorithm, ensuring consistent mathematical representation of the traces.
4. Graph Extraction: Connected component labeling assigns a unique integer ID to every continuous wire path, treating each as a distinct electrical node.
5. Intersection Algorithm: The script expands the perimeter of each component's bounding box to detect intersections with the labeled node paths, effectively determining which pins connect to which nodes.
6. Netlist Formatting: The extracted relationships are formatted into standard SPICE syntax (e.g., `R1 N001 N002 resistor_val`).

## Model Performance
* Architecture: YOLOv8n (Nano)
* Validation Metrics: Achieved an overall mAP50 of 94.1%.
* High-Confidence Classes: Operational Amplifiers (97.3%), BJT Transistors (98.3%).
* Inference Speed: ~40ms per image on a standard GPU.

## Installation and Usage

### Prerequisites
Ensure you have Python 3.8+ installed.

### Setup Instructions
1. Clone this repository to your local machine.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

   Run the Gradio web application:

Bash
python app.py
Open the provided localhost URL in your web browser. Upload a sketch of a circuit to generate the corresponding SPICE netlist.

Repository Structure
app.py: The main application script containing the AI inference, OpenCV tracing logic, and Gradio interface.

best.pt: The custom-trained YOLOv8 weights file.

requirements.txt: List of Python dependencies.

/test_images: Sample hand-drawn circuits for testing the pipeline.

Future Enhancements
Optical Character Recognition (OCR): Integrate OCR to read handwritten component values (e.g., "10k", "100uF") and dynamically inject them into the netlist.

Component Polarity: Implement directional logic for asymmetric components (diodes, polarized capacitors, transistors) to map source/drain and emitter/collector pins accurately.
