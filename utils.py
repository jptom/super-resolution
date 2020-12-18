# -*- coding: utf-8 -*-
import os
import itertools
import numpy as np
import cv2 as cv
import glob

def one2four_image(img):
    imgs = []
    imgs.append(img)
    imgs.append(np.flipud(img))
    imgs.append(np.fliplr(img))
    imgs.append(np.flip(img, axis=(0, 1)))
    return np.array(imgs)

def four2one_image(imgs):
    assert len(imgs) == 4, "length must be 4"
    img_sum = 0
    img_sum += imgs[0]
    img_sum += np.flipud(imgs[1])
    img_sum += np.fliplr(imgs[2])
    img_sum += np.flip(imgs[3], axis=(0, 1))
    return img_sum/4

def min_shape(imgs):
    min_value = float('inf')
    for img in imgs:
        h, w = img.shape[:2]
        min_value = min(min_value, h, w)
    
    return min_value

def drop_resolution(img, scale):
    #print(img.shape)
    small_img = cv.resize(img, dsize=None, fx=scale, fy=scale)
    ret_img = cv.resize(small_img, (img.shape[1], img.shape[0]))
    
    return ret_img

def down_sampling(img, dsize):
    low_img = cv.GaussianBlur(img, (5, 5), 0)
    low_img = cv.resize(low_img, dsize, interpolation=cv.INTER_CUBIC)
    
    return low_img

def random_clip(img, dsize, mag):
    random_h = np.random.randint(0, img.shape[0]-dsize[0]*mag)
    random_w = np.random.randint(0, img.shape[1]-dsize[1]*mag)
    cliped_img = img[random_h:random_h+dsize[0]*mag, random_w:random_w+dsize[1]*mag]
    
    return cliped_img

def random_flip(img): 
    r = np.random.randint(0, 2)
    return img if r else np.fliplr(img)

def random_rotate(x):
    r = np.random.randint(0, 4)
    return np.rot90(x, r)
    
def preprocess_xy(imgs, min_size, max_size, mag, up_scale=False):    
    batch_x = []
    batch_y = []
    imgs, imgs_backup = itertools.tee(imgs)
    real_max = int(min_shape(imgs_backup)/mag)
    max_size = min(max_size+1, real_max)
    hw = np.random.randint(min_size, max_size)
    for img in imgs:
        y = random_clip(img, (hw, hw), mag)
        y = random_flip(y)
        y = random_rotate(y)
        x = down_sampling(y, (hw, hw))
        if up_scale:
            x = cv.resize(x, (hw*mag, hw*mag), interpolation=cv.INTER_CUBIC)
        batch_x.append(x)
        batch_y.append(y)

    return np.array(batch_x), np.array(batch_y)
    
def img_loader(filenames):
    for filename in filenames:
        yield cv.imread(filename)/255.0
    
def data_generator(filenames, batch_size, preprocess_xy, **keywargs):
    while True:
        batch_paths  = np.random.choice(a=filenames, size=batch_size)
        imgs = img_loader(batch_paths)
        batch_x, batch_y = preprocess_xy(imgs, **keywargs)
        
        yield batch_x, batch_y
        
def validation_mag3_set(imgs, up_scale=False):
    y = []
    x = []
    for img in imgs:
        if img.shape[0]%3==0 and img.shape[1]%3==0:
            y.append(img)  
            lr = down_sampling(img, (img.shape[1], img.shape[0]))
            if up_scale:
                lr = cv.resize(lr, (img.shape[1]/3, img.shape[0]/3), interpolation=cv.INTER_CUBIC)
            x.append(lr)
    #return (np.array(x), np.array(y))
    
def get_filenames(path):
    filenames = glob.glob(os.path.join(path, "*.png")) #; print(filenames)
    return filenames
    
def data_analysis(filenames):
    hs = list()
    ws = list()
    
    for img_path in filenames:
        img = cv.imread(img_path)/255.0
        #
        hs.append(img.shape[0]) 
        ws.append(img.shape[1])
    
    print(sorted(hs)[:5])
    print(sorted(ws)[:5])
    hs = np.array(hs)
    ws = np.array(ws)
    
    print('min height: ', hs.min())
    print('min width: ', ws.min())
    print('max height: ', hs.max())
    print('max width: ', ws.max())
    
if __name__ == "__main__":
    batch = 32
    # data generator test
    filenames = get_filenames("dataset/training_hr_images") 
    gen = data_generator(filenames, batch, preprocess_xy, min_size=25, max_size=41, mag=3)
    step = 0
    for batch_x, batch_y in gen:
        step += 1
        for i in range(batch):
            assert batch_x[i].shape[0]*3 == batch_y[0].shape[0], "height doesn't match"
            assert batch_x[i].shape[1]*3 == batch_y[0].shape[1], "width doesn't match"
        if step == 10:
           break
                
    imgs = img_loader(filenames)
    validation_data = validation_mag3_set(imgs)
    
    data_analysis(filenames)