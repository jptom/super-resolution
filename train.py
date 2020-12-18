# -*- coding: utf-8 -*-
import argparse
import os
#import tensorflow as tf
#import tensorflow.keras as keras
#import tensorflow.keras.backend as kb
#import tensorflow.keras.layers as kl
#import tensorflow.keras.models as km
from utils import get_filenames, data_generator, preprocess_xy
from model import get_srcnn, get_espcn, get_fsrcnn, get_vdsr, get_drcn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model", type=str, required=True,
                        choices=["srcnn", "espcn", "fsrcnn", "vdsr", "drcn"], 
                        help="select model")
    parser.add_argument("--mag", type=int, default=3,
                        help="upsampling factor. Default 3")
    parser.add_argument("--mid", type=int, 
                        help="number of mid layers.")
    parser.add_argument("--data", type=str, default="dataset",
                        help="dataset path")
    parser.add_argument("--batch", type=int, default=32, 
                        help="batch size. Default 32")
    parser.add_argument("--steps", type=int, default=400,
                        help="Number of steps per one epoch. Default 400")
    parser.add_argument("--epochs", type=int, default=5, 
                        help="Number of epochs to train the model. Default 5")
    parser.add_argument("--loss", type=str, default="mse", 
                        choices=["mse", "mae", "bouble", "triple"])
    parser.add_argument("--import", type=str,
                        help="initial weight path")
    parser.add_argument("--export", type=str,
                        help="save weight path. Default ./log/{model}.h5")
    args = parser.parse_args() 
    

    
    # create model
    print("create model")
    if args.model == "srcnn":
        model = get_srcnn()
    elif args.model == "espcn":
        model = get_espcn(args.mag)
    elif args.model == "fsrcnn":
        model = get_fsrcnn(args.mag)
    elif args.model == "vdsr":
        if args.mid:
            model = get_vdsr(args.mid)
        else:
            model = get_vdsr()
    elif args.model == "drcn":
        if args.mid:
            model = get_drcn(args.mid)
        else:
            model = get_drcn()
            
    else:
        raise ValueError("select correct model")
    
    # load dataset
    filenames = get_filenames(args.data)
    
    # data generator
    if args.model in ["srcnn", "vdsr", "drcn"]:
        pre_up_scale = True
    else:
        pre_up_scale = False
        
    gen = data_generator(filenames, args.batch, preprocess_xy, min_size=25, max_size=26, 
                         mag=args.mag, up_scale=pre_up_scale)    
    
    
    # loss function
    def loss_fun(y_true, y_pred):
        pass
    
    # compile model
    print("start compling")
    model.compile(
        loss='mean_squared_error',
        optimizer='adam',
        )

    # train
    print("start training")
    model.fit(
        gen,
        steps_per_epoch=args.steps,
        epochs=args.epochs
        )
    
    print("save model")

    if args.export:
        model.save_weights(args.export)
    else:
        if not os.path.isdir('log'):
            os.mkdir('log')
        model.save_weights(os.path.join('log','{}_x{}.h5'.format(args.model, args.mag)))