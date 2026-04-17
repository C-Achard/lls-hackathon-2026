import json 
import numpy as np
import pdb

dict_pred = {0: 'animal', 1: 'person', 2: 'vehicle'}


def save_results(md_results, dlc_outputs,map_label_id_to_str,thr,output_file = 'dowload_predictions.json'):

    """

    write json

    """
    info = {}
    ## info megaDetector
    info['file']= md_results.files[0]
    number_bb = len(md_results.xyxy[0].tolist())
    info['number_of_bb'] = number_bb
    number_bb_thr = len(dlc_outputs)
    labels = [n for n in map_label_id_to_str.values()]
    #pdb.set_trace()
    new_index = []
    for i in range(number_bb):
        corner_x1,corner_y1,corner_x2,corner_y2,confidence, _ =  md_results.xyxy[0].tolist()[i]

        if confidence > thr:
            new_index.append(i)


    for i in range(number_bb_thr):
        aux={}
        corner_x1,corner_y1,corner_x2,corner_y2,confidence, _ =  md_results.xyxy[0].tolist()[new_index[i]]
        aux['corner_1'] = (corner_x1,corner_y1)
        aux['corner_2'] = (corner_x2,corner_y2)
        aux['predict MD'] = md_results.names[0]
        aux['confidence MD'] = confidence
        
        ## info dlc
        kypts = []
        for s in dlc_outputs[i]:
            aux1 = []
            for j in s:
                aux1.append(float(j))

            kypts.append(aux1)
        aux['dlc_pred']  = dict(zip(labels,kypts))
        info['bb_' + str(new_index[i]) ]=aux


    with open(output_file, 'w') as f:
        json.dump(info, f, indent=1)
        print('Output file saved at {}'.format(output_file))

    return output_file

