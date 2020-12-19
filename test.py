# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
from model import get_srcnn, get_espcn, get_fsrcnn, get_vdsr, get_drcn
from utils import get_filenames, img_loader, one2four_image, four2one_image
import cv2 as cv

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model", type=str, required=True,
                        choices=["srcnn", "espcn", "fsrcnn", "vdsr", "drcn"], 
                        help="select model")
    parser.add_argument("--mag", type=int, required=True,
                        help="upsampling factor")
    parser.add_argument("--weights", required=True,
                        help="weights")
    parser.add_argument("--data", type=str, default="dataset",
                        help="test data. file or directory")
    parser.add_argument("--aug", action='store_true',
                        help="use augmentation")
    args = parser.parse_args() 
    
    # create model
        
    if args.model == "srcnn":
        model = get_srcnn()
    
    elif args.model == "espcn":
        model = get_espcn(args.mag)

    elif args.model == "fsrcnn":
        model = get_fsrcnn(args.mag)

    elif args.model == "vdsr":
        model = get_vdsr()
    
    elif args.model == "drcn":
        model = get_drcn
        
    else:
        raise ValueError("select correct model")
            
    model.load_weights(args.weights)
    
    if os.path.isdir(args.data):
        filenames = get_filenames(args.data)
        show = False
    elif os.path.isfile(args.data):
        filenames = [args.data]
        show = True

    imgs = img_loader(filenames)
    # inference
    for i, img in enumerate(imgs):
        if args.model in ["srcnn", "vdsr"]:
            img = cv.resize(img, (img.shape[1]*args.mag, img.shape[0]*args.mag),
                            interpolation=cv.INTER_CUBIC)
        if args.aug:
            imgs1 = one2four_image(img)
            imgs2 = one2four_image(np.rot90(img))
            output1 = model(imgs1)
            output2 = model(imgs2)
            output1 = np.clip(output1, 0, 1)
            output2 = np.clip(output2, 0, 1)
            output1 *= 255
            output2 *= 255
            output1 = four2one_image(output1)
            output2 = four2one_image(output2)
            output2 = np.rot90(output2, -1)
            output = (output1 + output2)/2
        else:
            imgs = np.array([img])
            output = model(imgs)
            output = np.clip(output, 0, 1)
            output *= 255
            output = np.squeeze(output)
        high_img = output.astype(np.uint8)
        cv.imshow('result', high_img)
        cv.waitKey(0)
        cv.imwrite(os.path.join('0756153', os.path.basename(filenames[i])), high_img)