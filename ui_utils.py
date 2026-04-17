
# ui_utils.py
import gradio as gr


def gradio_inputs_for_MD_DLC(md_model_choices, dlc_model_choices):
    """
    Build UI inputs in the exact order expected by predict_pipeline():

    predict_pipeline(
        img_input,
        mega_model_input,
        dlc_model_input_str,
        flag_dlc_only,
        flag_show_str_labels,
        bbox_likelihood_th,
        kpts_likelihood_th,
        font_style,
        font_size,
        keypt_color,
        marker_size,
        bbox_color
    )
    """

    gr.Markdown(
        "Upload an image, choose detector / DeepLabCut models, and run pose estimation."
    )

    # IMPORTANT: PIL is required because app.py uses img_input.resize(...)
    gr_image_input = gr.Image(type="pil", label="Input Image")

    with gr.Accordion("⚙️ Advanced Settings", open=False):
        gr_mega_model_input = gr.Dropdown(
            choices=md_model_choices,
            value=md_model_choices[0] if md_model_choices else None,
            label="Select Detector Model",
        )

        gr_dlc_model_input = gr.Dropdown(
            choices=dlc_model_choices,
            value=dlc_model_choices[0] if dlc_model_choices else None,
            label="Select DeepLabCut Model",
        )

        with gr.Row():
            gr_dlc_only_checkbox = gr.Checkbox(
                value=False,
                label="Run DLC only? (skip detector)"
            )
            gr_str_labels_checkbox = gr.Checkbox(
                value=True,
                label="Show keypoint labels?"
            )

        gr_slider_conf_bboxes = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.10,
            step=0.05,
            label="Bounding Box Confidence Threshold",
        )

        gr_slider_conf_keypoints = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.40,
            step=0.05,
            label="Keypoint Confidence Threshold",
        )

        gr_labels_font_style = gr.Dropdown(
            choices=["amiko", "animals", "nature", "painter", "zen"],
            value="amiko",
            label="Font Style",
        )

        with gr.Row():
            gr_slider_font_size = gr.Slider(
                minimum=5,
                maximum=30,
                value=8,
                step=1,
                label="Font Size",
            )
            gr_slider_marker_size = gr.Slider(
                minimum=1,
                maximum=10,
                value=1,
                step=1,
                label="Marker Size / Line Thickness",
            )

        with gr.Row():
            gr_keypt_color = gr.ColorPicker(
                value="#c586c0",
                label="Keypoint / Label Color",
            )
            gr_bbox_color = gr.ColorPicker(
                value="#c586c0",
                label="Bounding Box Color",
            )

        gr.Markdown(
            """
            **Notes**
            - If **Run DLC only** is enabled, the detector model and bounding box threshold are effectively ignored.
            - `bbox_color` is included for API compatibility with `predict_pipeline()` even if your current drawing helper does not yet use it internally.
            """
        )

    # IMPORTANT: return order must match predict_pipeline() exactly
    return [
        gr_image_input,          # img_input
        gr_mega_model_input,     # mega_model_input
        gr_dlc_model_input,      # dlc_model_input_str
        gr_dlc_only_checkbox,    # flag_dlc_only
        gr_str_labels_checkbox,  # flag_show_str_labels
        gr_slider_conf_bboxes,   # bbox_likelihood_th
        gr_slider_conf_keypoints,# kpts_likelihood_th
        gr_labels_font_style,    # font_style
        gr_slider_font_size,     # font_size
        gr_keypt_color,          # keypt_color
        gr_slider_marker_size,   # marker_size
        gr_bbox_color,           # bbox_color
    ]


def gradio_outputs_for_MD_DLC():
    """
    Outputs must match predict_pipeline() return values:
      - image
      - download file
    """
    gr_image_output = gr.Image(type="pil", label="Analysis Output")
    gr_file_download = gr.File(label="Download Results")
    return [gr_image_output, gr_file_download]


def gradio_description_and_examples():
    """
    Return:
      [title_html, description_html, examples]
    """
    title = """
    <h1 style='text-align: center; color: #c586c0; font-family: "Consolas", monospace; font-size: 2.5em;'>
        DART + DeepLabCut Pose Estimation Demo
    </h1>
    """

    description = """
    <div style='text-align: center; font-size: 1.05em; color: gray; font-family: "Consolas", monospace;'>
        <p>Run DART + DeepLabCut SuperAnimal inference</p>
        <p>
            The inference pipeline in <code>app.py</code> remains unchanged — only the UI wiring is updated.
        </p>
    </div>
    """

    # Keep empty unless you have local example files to reference.
    # You can later add examples like:
    # ["path/to/example.jpg", "md_v5a", "superanimal_quadruped_dlcrnet", False, True, 0.10, 0.40, "amiko", 8, "#c586c0", 1, "#c586c0"]
    examples = []

    return [title, description, examples]
