# Adapted from https://huggingface.co/spaces/hlydecker/MegaDetector_v5 
# Adapted from https://huggingface.co/spaces/sofmi/MegaDetector_DLClive/blob/main/app.py
# Adapted from https://huggingface.co/spaces/Neslihan/megadetector_dlcmodels/blob/main/app.py 
# Adapted from  https://huggingface.co/spaces/DeepLabCut/MegaDetector_DeepLabCut

# app.py: Main application file for the DeepLabCut + DART Pose Estimation Demo.
import os
import yaml
import numpy as np
from matplotlib import cm
import gradio as gr
import deeplabcut
import dlclibrary
import dlclive
# import transformers

from PIL import Image, ImageColor, ImageFont, ImageDraw 
import requests

from viz_utils import save_results_as_json, draw_keypoints_on_image, draw_bbox_w_text, save_results_only_dlc
from detection_utils import predict_md, crop_animal_detections
from dlc_utils import predict_dlc
from ui_utils import gradio_inputs_for_MD_DLC, gradio_outputs_for_MD_DLC, gradio_description_and_examples

from deeplabcut.utils import auxiliaryfunctions
from dlclibrary.dlcmodelzoo.modelzoo_download import (
    download_huggingface_model,
    MODELOPTIONS,
)
from dlclive import DLCLive, Processor


# TESTING (passes) download the SuperAnimal models:
#model = 'superanimal_topviewmouse'
#train_dir = 'DLC_models/sa-tvm'
#download_huggingface_model(model, train_dir)

# grab demo data cooco cat:
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# megadetector and dlc model look up
MD_models_dict = {'md_v5a': "MD_models/md_v5a.0.0.pt", # 
                  'md_v5b': "MD_models/md_v5b.0.0.pt"}

# DLC models target  dirs
DLC_models_dict = {'superanimal_topviewmouse_dlcrnet': "DLC_models/sa-tvm",
                   'superanimal_quadruped_dlcrnet': "DLC_models/sa-q"}
                    


#####################################################
def predict_pipeline(img_input,
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
                     ):

    if not flag_dlc_only:
        ############################################################                                               
        # ### Run DART
        md_results = predict_md(img_input, 
                                MD_models_dict[mega_model_input] #mega_model_input,
                                ) #Image.fromarray(results.imgs[0])

        ################################################################
        # Obtain animal crops for bboxes with confidence above th
        list_crops = crop_animal_detections(img_input,
                                            md_results,
                                            bbox_likelihood_th)

        ############################################################

    ## Get DLC model and label map  
    
    # If model is found: do not download (previous execution is likely within same day)
    # TODO: can we ask the user whether to reload dlc model if a directory is found?
    if os.path.isdir(DLC_models_dict[dlc_model_input_str]) and \
        len(os.listdir(DLC_models_dict[dlc_model_input_str])) > 0:
        path_to_DLCmodel = DLC_models_dict[dlc_model_input_str]
    else:
        path_to_DLCmodel = DLC_models_dict[dlc_model_input_str]
        download_huggingface_model(dlc_model_input_str, path_to_DLCmodel)

    # extract map label ids to strings
    pose_cfg_path = os.path.join(DLC_models_dict[dlc_model_input_str],
                                 'pose_cfg.yaml')
    with open(pose_cfg_path, "r") as stream:
        pose_cfg_dict = yaml.safe_load(stream) 
    map_label_id_to_str = dict([(k,v) for k,v in zip([el[0] for el in pose_cfg_dict['all_joints']],  # pose_cfg_dict['all_joints'] is a list of one-element lists,
                                                     pose_cfg_dict['all_joints_names'])])


##############################################################
    # Run DLC and visualize results
    dlc_proc = Processor() #TODO: update deeplabcut.video_inference_superanimal() once merged

    # if required: ignore MD crops and run DLC on full image [mostly for testing]
    if flag_dlc_only:
        # compute kpts on input img
        list_kpts_per_crop = predict_dlc([np.asarray(img_input)],
                                         kpts_likelihood_th,
                                         path_to_DLCmodel,
                                         dlc_proc)
        # draw kpts on input img #fix!
        draw_keypoints_on_image(img_input,
                                list_kpts_per_crop[0], # a numpy array with shape [num_keypoints, 2].
                                map_label_id_to_str,
                                flag_show_str_labels,
                                use_normalized_coordinates=False,
                                font_style=font_style,
                                font_size=font_size,
                                keypt_color=keypt_color,
                                marker_size=marker_size)

        donw_file = save_results_only_dlc(list_kpts_per_crop[0], map_label_id_to_str,dlc_model_input_str)

        return img_input, donw_file

    else:
        # Compute kpts for each crop
        list_kpts_per_crop = predict_dlc(list_crops,
                                         kpts_likelihood_th,
                                         path_to_DLCmodel,
                                         dlc_proc)
        
        # resize input image to match megadetector output
        img_background = img_input.resize((md_results.ims[0].shape[1],
                                           md_results.ims[0].shape[0]))
        
        # draw keypoints on each crop and paste to background img
        for ic, (np_crop, kpts_crop) in enumerate(zip(list_crops,
                                                      list_kpts_per_crop)):

            img_crop = Image.fromarray(np_crop)

            # Draw keypts on crop
            draw_keypoints_on_image(img_crop,
                                    kpts_crop, # a numpy array with shape [num_keypoints, 2].
                                    map_label_id_to_str,
                                    flag_show_str_labels,
                                    use_normalized_coordinates=False,  # if True, then I should use md_results.xyxyn for list_kpts_crop
                                    font_style=font_style,
                                    font_size=font_size,
                                    keypt_color=keypt_color,
                                    marker_size=marker_size)

            # Paste crop in original image
            img_background.paste(img_crop, 
                                 box = tuple([int(t) for t in md_results.xyxy[0][ic,:2]]))

            # Plot bbox
            bb_per_animal = md_results.xyxy[0].tolist()[ic]
            pred = md_results.xyxy[0].tolist()[ic][4]
            if bbox_likelihood_th < pred:
                draw_bbox_w_text(img_background, 
                                    bb_per_animal,
                                    font_size=font_size)  # TODO: add selectable color for bbox?

            
        # Save detection results as json
        download_file  = save_results_as_json(md_results,list_kpts_per_crop,map_label_id_to_str, bbox_likelihood_th,dlc_model_input_str,mega_model_input)         

        return img_background, download_file



#########################################################
# Define user interface and launch

#########################################################
# Define user interface and launch

with gr.Blocks(theme=gr.themes.Default(neutral_hue="slate")) as demo:
    gr_title, gr_description, examples = gradio_description_and_examples()

    gr.HTML(gr_title)
    gr.HTML(gr_description)

    with gr.Row():
        with gr.Column(scale=1, min_width=360):
            inputs = gradio_inputs_for_MD_DLC(
                list(MD_models_dict.keys()),
                list(DLC_models_dict.keys())
            )
            run_btn = gr.Button("Run Analysis", variant="primary")

        with gr.Column(scale=2):
            outputs = gradio_outputs_for_MD_DLC()

            if examples:
                gr.Examples(
                    examples=examples,
                    inputs=inputs,
                    outputs=outputs,
                    fn=predict_pipeline,
                    cache_examples=False,
                )

    run_btn.click(
        fn=predict_pipeline,
        inputs=inputs,
        outputs=outputs,
    )

demo.queue()
demo.launch()
