import gradio as gr
import cv2
import pandas as pd
from ultralytics import YOLOWorld
from rembg import remove

# --- MODEL LOADING ---
print("Loading AI Models... Please wait...")
model = YOLOWorld('yolov8s-world.pt')

# --- UI INPUTS MODULE ---
def gradio_inputs_for_MD_DLC():
    gr_image_input = gr.Image(type="numpy", label="Input Image")
    gr_target_input = gr.Textbox(value="fish", label="What are we tracking?")

    with gr.Accordion("⚙️ Advanced AI Settings", open=False):
        gr_mega_model_input = gr.Dropdown(
            choices=["md_v5a", "sam3.pt", "dart"], 
            value="md_v5a", 
            label="Select Detector model"
        )
        gr_dlc_model_input = gr.Dropdown(
            choices=["superanimal_quadruped_dlcrnet"], 
            value="superanimal_quadruped_dlcrnet", 
            label="Select DeepLabCut model"
        )
        gr_dlc_only_checkbox = gr.Checkbox(value=False, label="Run DLClive only?")
        
        gr_slider_conf_bboxes = gr.Slider(minimum=0.0, maximum=1.0, value=0.10, step=0.05, label="Detection Confidence")
        gr_slider_conf_keypoints = gr.Slider(minimum=0.0, maximum=1.0, value=0.4, step=0.05, label="Keypoint Confidence")
        
        gr_str_labels_checkbox = gr.Checkbox(value=True, label="Show ID Labels?")
        gr_keypt_color = gr.ColorPicker(value="#c586c0", label="Label Color")
        gr_labels_font_style = gr.Dropdown(choices=["amiko", "animals", "nature", "painter", "zen"], value="amiko", label="Font Style")
        gr_slider_font_size = gr.Slider(minimum=5, maximum=30, value=8, step=1, label="Font Size") 
        gr_slider_marker_size = gr.Slider(minimum=1, maximum=10, value=1, step=1, label="Line Thickness") 

    return [
        gr_image_input, gr_target_input, gr_mega_model_input, gr_dlc_model_input, 
        gr_dlc_only_checkbox, gr_str_labels_checkbox, gr_slider_conf_bboxes, 
        gr_slider_conf_keypoints, gr_labels_font_style, gr_slider_font_size, 
        gr_keypt_color, gr_slider_marker_size
    ]

# --- UI OUTPUTS MODULE ---
def gradio_outputs_for_MD_DLC():
    gr_image_output = gr.Image(type="numpy", label="Analysis Output")
    
    # 👇 THE DYNAMIC EXPANSION FIX 👇
    # row_count=(1, "dynamic") tells the UI to expand the table as much as needed
    gr_dataframe_output = gr.Dataframe(
        label="Coordinate Mapping (Per Detected Individual)", 
        interactive=False,
        wrap=True,
        row_count=(1, "dynamic") 
    )
    
    gr_file_download = gr.File(label="Download Full Dataset (CSV)")
    return [gr_image_output, gr_dataframe_output, gr_file_download]

# --- DESCRIPTION MODULE ---
def gradio_description_and_examples():
    title = """
        <h1 style='text-align: center; color: #c586c0; font-family: "Consolas", monospace; font-size: 2.5em;'>
            Zero-Shot Biological Tracker
        </h1>
    """
    description = """
        <div style='text-align: center; font-size: 1.1em; color: gray; font-family: "Consolas", monospace;'>
            <p>Integrated DART & YOLO Pipeline. High-capacity coordinate extraction.</p>
            <p>Test the SuperAnimal models from the <a href='http://www.mackenziemathislab.org/dlc-modelzoo' style='color: #c586c0; text-decoration: none;'>DeepLabCut ModelZoo Project</a>.</p>
        </div>
    """
    return [title, description]

# --- TRACKING ENGINE ---
def run_tracker(img, target_words, mega_model, dlc_model, dlc_only, 
                show_labels, conf_threshold, conf_keypoints, 
                font_style, font_size, keypt_color, marker_size):
    
    if img is None:
        return None, None, None

    model.set_classes([word.strip() for word in target_words.split(",")])

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if hsv[:, :, 1].mean() < 20: 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        processed_img = cv2.cvtColor(cv2.createCLAHE(clipLimit=3.0).apply(gray), cv2.COLOR_GRAY2BGR)
    else: 
        processed_img = remove(img)[:, :, :3].copy()

    # 👇 AI LIMIT FIX: Set max_det to 100 so the model doesn't ignore animals
    results = model.predict(processed_img, conf=conf_threshold, verbose=False, max_det=100)
    
    boxes = results[0].boxes.xyxy.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()
    labels = [model.names[int(c)] for c in results[0].boxes.cls.cpu().numpy()]

    display_img = img.copy()
    data_list = []

    try:
        color_hex = keypt_color.lstrip('#')
        cv_color = tuple(int(color_hex[i:i+2], 16) for i in (4, 2, 0))
    except:
        cv_color = (192, 134, 197) 
        
    cv_font_scale = max(0.3, font_size / 30.0)
    cv_thickness = max(1, int(marker_size)) 

    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        x1, y1, x2, y2 = map(int, box)
        animal_id = i + 1 
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        
        data_list.append({
            "ID": animal_id, "Target": label, "Confidence": round(float(score), 2),
            "Center_X": cx, "Center_Y": cy, "X1": x1, "Y1": y1, "X2": x2, "Y2": y2
        })
        
        cv2.rectangle(display_img, (x1, y1), (x2, y2), cv_color, cv_thickness)
        if show_labels:
            cv2.putText(display_img, f"#{animal_id}", (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, cv_font_scale, cv_color, 1)

    df = pd.DataFrame(data_list)
    csv_path = "biological_coords.csv"
    df.to_csv(csv_path, index=False)

    return display_img, df, csv_path

# --- LAUNCHER ---
with gr.Blocks() as demo:
    t, d = gradio_description_and_examples()
    gr.HTML(t); gr.HTML(d)
    
    with gr.Row():
        with gr.Column(scale=1):
            input_list = gradio_inputs_for_MD_DLC()
            btn = gr.Button("🚀 Run Analysis", variant="primary")
        with gr.Column(scale=2):
            output_list = gradio_outputs_for_MD_DLC()
            
    btn.click(fn=run_tracker, inputs=input_list, outputs=output_list)

if __name__ == "__main__":
    demo.launch(share=True, theme=gr.themes.Default(neutral_hue="slate"))