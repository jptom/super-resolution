# super-resolution

## models
  - [SRCNN](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html)
  - [ESPCN](https://arxiv.org/abs/1609.05158)
  - [FSRCNN](https://arxiv.org/abs/1608.00367)
  - [VDSR](https://arxiv.org/abs/1511.04587)
  - [DRCN](https://arxiv.org/abs/1511.04491)
  - [SRResNet](https://arxiv.org/abs/1609.04802)
  - [EDSR](https://arxiv.org/abs/1707.02921)

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
