import cv2
import numpy as np
import gradio as gr
from ultralytics import YOLO
from skimage.morphology import skeletonize

#model
model = YOLO('best.pt')


def process_circuit(img):

    # basic color conversions
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    results = model.predict(img_bgr)

    plotted = results[0].plot()
    res_img_rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
    _, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
    wires_only = thresh.copy()

    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    names = results[0].names

    # remove component regions from wire mask
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)

        # small padding so wires touching box edges are removed properly
        cv2.rectangle(
            wires_only,
            (max(0, x1 - 5), max(0, y1 - 5)),
            (x2 + 5, y2 + 5),
            0,
            -1
        )

    
    contours, _ = cv2.findContours(
        wires_only,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    clean_wires = np.zeros_like(wires_only)

    for cnt in contours:
        if cv2.contourArea(cnt) > 30:  # ignore tiny regions
            cv2.drawContours(clean_wires, [cnt], -1, 255, thickness=cv2.FILLED)

    # skeletonize so wires become single pixel thick
    skeleton = skeletonize(clean_wires > 0)
    skeleton_img = (skeleton * 255).astype(np.uint8)

    # connected components = electrical nodes
    num_labels, labels_map = cv2.connectedComponents(skeleton_img)

    colored_nodes = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    np.random.seed(42)  # fixed seed so colors stay same every run

    for label in range(1, num_labels):
        colored_nodes[labels_map == label] = np.random.randint(
            50, 255, size=3
        ).tolist()

    # slightly dilate so nodes are visible
    kernel = np.ones((5, 5), np.uint8)
    colored_nodes = cv2.dilate(colored_nodes, kernel, iterations=1)

    nodes_img_rgb = cv2.cvtColor(colored_nodes, cv2.COLOR_BGR2RGB)

    # now generating SPICE netlist
    netlist = [
        "* Auto-Generated LTspice Netlist *",
        ".backanno",
        ".end"
    ]

    comp_counts = {}

    for box, cls_id in zip(boxes, classes):

        x1, y1, x2, y2 = map(int, box)
        comp_name = names[int(cls_id)]

        pad = 8
        bx1, by1 = max(0, x1 - pad), max(0, y1 - pad)
        bx2 = min(labels_map.shape[1], x2 + pad)
        by2 = min(labels_map.shape[0], y2 + pad)

        region = labels_map[by1:by2, bx1:bx2]
        touching_nodes = np.unique(region)
        touching_nodes = touching_nodes[touching_nodes > 0]

        # counting components of same type
        comp_counts[comp_name] = comp_counts.get(comp_name, 0) + 1

        prefix = comp_name[0].upper()

        # special cases for SPICE naming
        if comp_name == "operational_amplifier":
            prefix = "X"
        elif "transistor" in comp_name:
            prefix = "Q"

        comp_id = f"{prefix}{comp_counts[comp_name]}"

        if len(touching_nodes) == 0:
            nodes_str = "NC"
        else:
            nodes_str = " ".join(
                [f"N{n:03d}" for n in touching_nodes]
            )

        netlist.insert(
            1,
            f"{comp_id} {nodes_str} {comp_name}_val"
        )

    spice_code = "\n".join(netlist)

    return res_img_rgb, nodes_img_rgb, spice_code


# gradio interface 
demo = gr.Interface(
    fn=process_circuit,
    inputs=gr.Image(type="numpy", label="Upload Circuit Sketch"),
    outputs=[
        gr.Image(label="Detected Components"),
        gr.Image(label="Extracted Nodes"),
        gr.Code(label="Generated LTspice Netlist", language="python")
    ],
    title="Sketch to SPICE Converter",
    description="Upload a hand-drawn schematic. The model detects components and generates an LTspice netlist."
)

demo.launch()
