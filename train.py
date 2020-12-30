# -*- coding: utf-8 -*-
import argparse
import os
#import tensorflow as tf
import tensorflow.keras as keras
#import tensorflow.keras.backend as kb
#import tensorflow.keras.layers as kl
#import tensorflow.keras.models as km
from utils import get_filenames, data_generator, preprocess_xy, select_img_by_size
from models import get_model
from functions import psnr

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model", type=str, required=True,
                        choices=["srcnn", "espcn", "fsrcnn", "vdsr", "drcn", "srrennet", "edsr", "resatsr"], 
                        help="select model")
    parser.add_argument("--mag", type=int, default=3,
                        help="upsampling factor. Default 3")
    parser.add_argument("--mid", type=int, 
                        help="number of mid layers.")
    parser.add_argument("--data", type=str, required=True,
                        help="dataset path")
    parser.add_argument("--imsize", type=int, default=33,
                        help="image size of training. Default 33")
    parser.add_argument("--batch", type=int, default=32, 
                        help="batch size. Default 32")
    parser.add_argument("--steps", type=int, default=400,
                        help="Number of steps per one epoch. Default 400")
    parser.add_argument("--epochs", type=int, default=5, 
                        help="Number of epochs to train the model. Default 5")
    parser.add_argument("--loss", type=str, default="mse", 
                        choices=["mse", "mae"])
    parser.add_argument("--weights", type=str,
                        help="initial weight path")
    parser.add_argument("--out", type=str,
                        help="save weight path. Default ./log/{model}.h5")
    args = parser.parse_args() 
    

    
    # create model
    print("create model")
    model = get_model(args.model, args.mag)
    
    if args.weights:
        model.load_weights(args.weights)
        
    
    # load dataset
    filenames = get_filenames(args.data)
    filenames = select_img_by_size(filenames, args.imsize*args.mag)
    
    # data generator
    if args.model in ["srcnn", "vdsr", "drcn"]:
        pre_up_scale = True
    else:
        pre_up_scale = False
        
    gen = data_generator(filenames, args.batch, preprocess_xy, size=args.imsize, 
                         mag=args.mag, up_scale=pre_up_scale)    
    
    
    if args.loss=="mse":    
        loss = keras.losses.mean_squared_error
    elif args.loss=="mae":
        loss = keras.losses.mean_absolute_error
    
    # optimizer
    optimizer = keras.optimizers.Adam(learning_rate=1.0e-4)

    # compile model
    print("start compling")
    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=[psnr]
        )

    # train
    print("start training")
    model.fit(
        gen,
        steps_per_epoch=args.steps,
        epochs=args.epochs,
        )
    
    print("save weights")
    if args.out:
        model.save_weights(args.out)
    else:
        if not os.path.isdir('log'):
            os.mkdir('log')
        model.save_weights(os.path.join('log','{}_x{}.h5'.format(args.model, args.mag)))