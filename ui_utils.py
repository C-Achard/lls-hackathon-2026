import gradio as gr


def gradio_inputs_for_MD_DLC(md_models_list, dlc_models_list):
    # Input image
    gr_image_input = gr.Image(type="pil", label="Input Image")

    # Models
    gr_mega_model_input = gr.Dropdown(
        choices=md_models_list,
        value="md_v5a",
        type="value",
        label="Select Detector model",
    )

    gr_dlc_model_input = gr.Dropdown(
        choices=dlc_models_list,
        value="superanimal_quadruped_dlcrnet",
        type="value",
        label="Select DeepLabCut model",
    )

    # Other inputs
    gr_dlc_only_checkbox = gr.Checkbox(
        value=False,
        label="Run DLClive only, directly on input image?",
    )

    gr_str_labels_checkbox = gr.Checkbox(
        value=True,
        label="Show bodypart labels?",
    )

    # Gradio Slider signature is (minimum, maximum, value, step, ...)
    gr_slider_conf_bboxes = gr.Slider(
        minimum=0,
        maximum=1,
        value=0.2,
        step=0.05,
        label="Set confidence threshold for animal detections",
    )

    gr_slider_conf_keypoints = gr.Slider(
        minimum=0,
        maximum=1,
        value=0.4,
        step=0.05,
        label="Set confidence threshold for keypoints",
    )

    # Data viz
    gr_keypt_color = gr.ColorPicker(
        value="#862db7",
        label="Choose color for keypoint label",
    )

    gr_labels_font_style = gr.Dropdown(
        choices=["amiko", "animals", "nature", "painter", "zen"],
        value="amiko",
        type="value",
        label="Select keypoint label font",
    )

    gr_slider_font_size = gr.Slider(
        minimum=5,
        maximum=30,
        value=8,
        step=1,
        label="Set font size",
    )

    gr_slider_marker_size = gr.Slider(
        minimum=1,
        maximum=20,
        value=9,
        step=1,
        label="Set marker size",
    )

    return [
        gr_image_input,
        gr_mega_model_input,
        gr_dlc_model_input,
        gr_dlc_only_checkbox,
        gr_str_labels_checkbox,
        gr_slider_conf_bboxes,
        gr_slider_conf_keypoints,
        gr_labels_font_style,
        gr_slider_font_size,
        gr_keypt_color,
        gr_slider_marker_size,
    ]


def gradio_outputs_for_MD_DLC():
    gr_image_output = gr.Image(type="pil", label="Output Image")
    gr_file_download = gr.File(label="Download JSON file")
    return [gr_image_output, gr_file_download]


def gradio_description_and_examples():
    title = "DeepLabCut Model Zoo SuperAnimals"
    description = (
        "Test the SuperAnimal models from the "
        "<a href='http://www.mackenziemathislab.org/dlc-modelzoo'>"
        "DeepLabCut ModelZoo Project</a>, and read more on arXiv: "
        "https://arxiv.org/abs/2203.07436! Simply upload an image and see how it does. "
        "Want to run on videos on the cloud or locally? See the "
        "<a href='http://www.mackenziemathislab.org/dlc-modelzoo'>DeepLabCut ModelZoo</a>."
    )

    examples = [[
        "examples/dog.jpeg",
        "md_v5a",
        "superanimal_quadruped_dlcrnet",
        False,
        True,
        0.5,
        0.0,
        "amiko",
        9,
        "#ff0000",
        3,
    ]]

    return [title, description, examples]