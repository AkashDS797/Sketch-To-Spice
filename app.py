import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from skimage.morphology import skeletonize

st.set_page_config(layout="wide", page_title="Sketch-to-SPICE")

st.title("⚡ Auto-Netlist: Sketch-to-SPICE AI")
st.markdown("Upload a hand-drawn schematic. The AI will locate components, extract topological nodes, and generate an LTspice netlist.")

@st.cache_resource
def load_model():
    # Load your custom 94% accurate Brain!
    return YOLO('/kaggle/working/runs/detect/circuit_ai/run1/weights/best.pt')

model = load_model()

uploaded_file = st.file_uploader("Upload Circuit Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Read Image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. AI Component Detection")
        # Run YOLO
        results = model.predict(img)
        res_img = results[0].plot() # Image with bounding boxes
        st.image(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB), use_container_width=True)
        
    with col2:
        st.subheader("2. Node Extraction & Netlist")
        
        # 2. Binarize & Masking
        _, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
        wires_only = thresh.copy()
        
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        names = results[0].names
        
        # Erase components to isolate wires
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(wires_only, (max(0, x1-5), max(0, y1-5)), (x2+5, y2+5), 0, -1)
            
        # 3. Skeletonize
        contours, _ = cv2.findContours(wires_only, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clean_wires = np.zeros_like(wires_only)
        for cnt in contours:
            if cv2.contourArea(cnt) > 30:
                cv2.drawContours(clean_wires, [cnt], -1, 255, thickness=cv2.FILLED)
                
        skeleton = skeletonize(clean_wires > 0)
        skeleton_img = (skeleton * 255).astype(np.uint8)
        
        # 4. Connected Components (Node Mapping)
        num_labels, labels_map = cv2.connectedComponents(skeleton_img)
        
        # Color code the nodes for visualization
        colored_nodes = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        np.random.seed(42)
        for label in range(1, num_labels):
            colored_nodes[labels_map == label] = np.random.randint(50, 255, size=3).tolist()
            
        kernel = np.ones((5,5), np.uint8)
        colored_nodes = cv2.dilate(colored_nodes, kernel, iterations=1)
        st.image(cv2.cvtColor(colored_nodes, cv2.COLOR_BGR2RGB), caption=f"Extracted {num_labels-1} Distinct Electrical Nodes", use_container_width=True)

        # 5. THE INTERSECTION ALGORITHM & NETLIST GENERATION
        st.subheader("3. Generated SPICE Netlist")
        
        netlist = ["* Auto-Generated LTspice Netlist *", ".backanno", ".end"]
        comp_counts = {}
        
        for box, cls_id in zip(boxes, classes):
            x1, y1, x2, y2 = map(int, box)
            comp_name = names[int(cls_id)]
            
            # Check a slightly larger perimeter around the box to see what wires it touches
            pad = 8
            bx1, by1 = max(0, x1-pad), max(0, y1-pad)
            bx2, by2 = min(labels_map.shape[1], x2+pad), min(labels_map.shape[0], y2+pad)
            
            # Find all unique nodes inside this perimeter
            touching_region = labels_map[by1:by2, bx1:bx2]
            touching_nodes = np.unique(touching_region)
            touching_nodes = touching_nodes[touching_nodes > 0] # Remove background (0)
            
            # Format SPICE Code
            comp_counts[comp_name] = comp_counts.get(comp_name, 0) + 1
            
            # Basic formatting (R for resistor, C for cap, etc.)
            prefix = comp_name[0].upper()
            if comp_name == "operational_amplifier": prefix = "X"
            elif "transistor" in comp_name: prefix = "Q"
            
            comp_id = f"{prefix}{comp_counts[comp_name]}"
            
            nodes_str = " ".join([f"N{n:03d}" for n in touching_nodes])
            if len(touching_nodes) == 0: nodes_str = "NC" # Not Connected
                
            netlist.insert(1, f"{comp_id} {nodes_str} {comp_name}_val")
            
        spice_code = "\n".join(netlist)
        st.code(spice_code, language='spice')
