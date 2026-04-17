import deeplabcut
from tkinter import W
import gradio as gr
import numpy as np
from dlclive import DLCLive, Processor


##########################################
def predict_dlc(list_np_crops,
                kpts_likelihood_th,
                dlc_model_folder,
                dlc_proc):
    
    # run dlc thru list of crops
    dlc_live = DLCLive(dlc_model_folder, processor=dlc_proc)
    dlc_live.init_inference(list_np_crops[0])

    list_kpts_per_crop = []
    all_kypts  = []
    np_aux = np.empty((1,3)) # can I avoid hardcoding here?
    for crop in list_np_crops:
        # scale crop here?
        keypts_xyp = dlc_live.get_pose(crop) # third column is llk!
        # set kpts below threhsold to nan
        
        #pdb.set_trace()
        keypts_xyp[keypts_xyp[:,-1] < kpts_likelihood_th,:] = np_aux.fill(np.nan)
        # add kpts of this crop to list 
        list_kpts_per_crop.append(keypts_xyp)
        all_kypts.append(keypts_xyp)
    
    return list_kpts_per_crop