import json 
import numpy as np

from matplotlib import cm
import matplotlib
from PIL import Image, ImageColor, ImageFont, ImageDraw 
import numpy as np
import pdb
from datetime import date
today = date.today()
FONTS = {'amiko': "fonts/Amiko-Regular.ttf",
        'nature': "fonts/LoveNature.otf", 
        'painter':"fonts/PainterDecorator.otf",
        'animals': "fonts/UncialAnimals.ttf", 
        'zen': "fonts/ZEN.TTF"}

#########################################
#  Draw keypoints on image
def draw_keypoints_on_image(image,
                            keypoints,
                            map_label_id_to_str,
                            flag_show_str_labels,
                            use_normalized_coordinates=True,                           
                            font_style='amiko',
                            font_size=8,
                            keypt_color="#ff0000",
                            marker_size=2,
                            ):
    """Draws keypoints on an image.
    Modified from:
        https://www.programcreek.com/python/?code=fjchange%2Fobject_centric_VAD%2Fobject_centric_VAD-master%2Fobject_detection%2Futils%2Fvisualization_utils.py
    Args:
    image: a PIL.Image object.
    keypoints: a numpy array with shape [num_keypoints, 2].
    map_label_id_to_str: dict with keys=label number and values= label string
    flag_show_str_labels: boolean to select whether or not to show string labels
    color: color to draw the keypoints with. Default is red.
    radius: keypoint radius. Default value is 2.
    use_normalized_coordinates: if True (default), treat keypoint values as
        relative to the image.  Otherwise treat them as absolute.

    
    """
    # get a drawing context
    draw = ImageDraw.Draw(image,"RGBA")  

    im_width, im_height = image.size
    keypoints_x = [k[0] for k in keypoints]
    keypoints_y = [k[1] for k in keypoints]
    alpha = [k[2] for k in keypoints]
    norm = matplotlib.colors.Normalize(vmin=0, vmax=255)

    # debugging keypoints
    print (keypoints)
                                
    names_for_color = [i for i in map_label_id_to_str.keys()]
    colores = np.linspace(0, 255, num=len(names_for_color),dtype= int)

    # adjust keypoints coords if required
    if use_normalized_coordinates:
        keypoints_x = tuple([im_width * x for x in keypoints_x])
        keypoints_y = tuple([im_height * y for y in keypoints_y])
    
    #cmap = matplotlib.cm.get_cmap('hsv')
    cmap2 = matplotlib.cm.get_cmap('Greys')
    # draw ellipses around keypoints
    for i, (keypoint_x, keypoint_y) in enumerate(zip(keypoints_x, keypoints_y)):
        round_fill = list(cm.viridis(norm(colores[i]),bytes=True))#[round(num*255) for num in list(cmap(i))[:3]] #check!
        # handling potential nans in the keypoints
        if np.isnan(keypoint_x).any():
            continue
        
        if np.isnan(alpha[i]) == False : 
            round_fill[3] = round(alpha[i] *255)
        #print(round_fill)
        #round_outline = [round(num*255) for num in list(cmap2(alpha[i]))[:3]]
        draw.ellipse([(keypoint_x - marker_size, keypoint_y - marker_size),
                      (keypoint_x + marker_size, keypoint_y + marker_size)],
                       fill=tuple(round_fill), outline= 'black', width=1) #fill and outline: [0,255]

        # add string labels around keypoints
        if flag_show_str_labels:
            font = ImageFont.truetype(FONTS[font_style], 
                                     font_size) 
            draw.text((keypoint_x + marker_size, keypoint_y + marker_size),#(0.5*im_width, 0.5*im_height), #-------
                      map_label_id_to_str[i],
                      ImageColor.getcolor(keypt_color, "RGB"), # rgb #
                      font=font)

#########################################
#  Draw bboxes on image
def draw_bbox_w_text(img,
                     results,
                     font_style='amiko',
                     font_size=8): #TODO: select color too?
    #pdb.set_trace()
    bbxyxy = results
    w, h = bbxyxy[2], bbxyxy[3]
    shape = [(bbxyxy[0], bbxyxy[1]), (w , h)]
    imgR = ImageDraw.Draw(img)  
    imgR.rectangle(shape,  outline ="red",width=5) ##bb for animal

    confidence = bbxyxy[4]
    string_bb = 'animal ' + str(round(confidence, 2))
    font = ImageFont.truetype(FONTS[font_style], font_size) 

    text_size = font.getbbox(string_bb) # (h,w)
    position = (bbxyxy[0],bbxyxy[1] - text_size[1] -2 )
    left, top, right, bottom = imgR.textbbox(position, string_bb, font=font)
    imgR.rectangle((left, top-5, right+5, bottom+5), fill="red")
    imgR.text((bbxyxy[0] + 3 ,bbxyxy[1] - text_size[1] -2 ), string_bb, font=font, fill="black")

    return imgR

###########################################
def save_results_as_json(md_results, dlc_outputs, map_dlc_label_id_to_str, thr,model,mega_model_input, path_to_output_file = 'download_predictions.json'):

    """
    Output detections as json file

    """
    # initialise dict to save to json
    info = {}
    info['date'] = str(today)
    info['MD_model'] = str(mega_model_input)
    # info from megaDetector
    info['file']= md_results.files[0]
    number_bb = len(md_results.xyxy[0].tolist())
    info['number_of_bb'] = number_bb
    # info from DLC
    number_bb_thr = len(dlc_outputs)
    labels = [n for n in map_dlc_label_id_to_str.values()]
    
    # create list of bboxes above th
    new_index = []
    for i in range(number_bb):
        corner_x1,corner_y1,corner_x2,corner_y2,confidence, _ =  md_results.xyxy[0].tolist()[i]

        if confidence > thr:
            new_index.append(i)

    # define aux dict for every bounding box above threshold
    for i in range(number_bb_thr):
        aux={}
        # MD output
        corner_x1,corner_y1,corner_x2,corner_y2,confidence, _ =  md_results.xyxy[0].tolist()[new_index[i]]
        aux['corner_1'] = (corner_x1,corner_y1)
        aux['corner_2'] = (corner_x2,corner_y2)
        aux['predict MD'] = md_results.names[0]
        aux['confidence MD'] = confidence
        
        # DLC output
        info['dlc_model'] = model
        kypts = []
        for s in dlc_outputs[i]:
            aux1 = []
            for j in s:
                aux1.append(float(j))

            kypts.append(aux1)
        aux['dlc_pred']  = dict(zip(labels,kypts))
        info['bb_' + str(new_index[i]) ]=aux

    # save dict as json
    with open(path_to_output_file, 'w') as f:
        json.dump(info, f, indent=1)
        print('Output file saved at {}'.format(path_to_output_file))

    return path_to_output_file


def save_results_only_dlc(dlc_outputs,map_label_id_to_str,model,output_file = 'dowload_predictions_dlc.json'):

    """
    write json dlc output
    """
    info = {}
    info['date'] = str(today)
    labels = [n for n in map_label_id_to_str.values()]
    info['dlc_model'] = model
    kypts = []
    for s in dlc_outputs:
        aux1 = []
        for j in s:
            aux1.append(float(j))

        kypts.append(aux1)
    info['dlc_pred']  = dict(zip(labels,kypts))

    with open(output_file, 'w') as f:
        json.dump(info, f, indent=1)
        print('Output file saved at {}'.format(output_file))

    return output_file


###########################################