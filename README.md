# super-resolution

## models
  - SRCNN
  - ESPCN
  - FSRCNN
  - VDSR
  - DRCN
  - SRResNet
  - EDSR

## training

```bash
usage: train.py [-h] --model
                {srcnn,espcn,fsrcnn,vdsr,drcn,srrennet,edsr,resatsr}
                [--mag MAG] [--mid MID] --data DATA [--imsize IMSIZE]
                [--batch BATCH] [--steps STEPS] [--epochs EPOCHS]
                [--loss {mse,mae}] [--weights WEIGHTS] [--out OUT]

optional arguments:
  -h, --help            show this help message and exit
  --model {srcnn,espcn,fsrcnn,vdsr,drcn,srrennet,edsr,resatsr}
                        select model
  --mag MAG             upsampling factor. Default 3
  --mid MID             number of mid layers.
  --data DATA           dataset path
  --imsize IMSIZE       image size of training. Default 33
  --batch BATCH         batch size. Default 32
  --steps STEPS         Number of steps per one epoch. Default 400
  --epochs EPOCHS       Number of epochs to train the model. Default 5
  --loss {mse,mae}
  --weights WEIGHTS     initial weight path
  --out OUT             save weight path. Default ./log/{model}.h5
  ```

## testing
