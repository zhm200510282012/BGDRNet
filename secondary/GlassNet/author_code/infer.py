import os
from skimage import io
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

import numpy as np
from PIL import Image

from data_loader import RescaleT
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from misc import crf_refine

from model import GlassNet
import torch.backends.cudnn as cudnn

cudnn.enabled = True
cudnn.benchmark = True
torch.cuda.set_device(0)

import time

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)
    return dn


def save_reflection(image_name, pred, d_dir):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()
    predict_np = np.clip(predict_np, 0, 1)
    predict_np = np.transpose(predict_np, (1, 2, 0)) * 255.
    im = Image.fromarray((predict_np.astype(np.uint8))).convert('RGB')

    img_name = image_name.split("/")[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir + imidx + '.png')

def save_output(image_name, input, pred, d_dir, crf=True):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    if crf:
        input = input.cpu().data.numpy().squeeze() * 255
        input = np.transpose(input, (1, 2, 0)).astype(np.uint8)
        predict_np = (predict_np * 255).astype(np.uint8)

        predict_np = predict_np.copy(order='C')
        input = input.copy(order='C')

        predict_np = crf_refine(input, predict_np)

        im = Image.fromarray(predict_np).convert('RGB')
    else:
        im = Image.fromarray(predict_np * 255).convert('RGB')

    img_name = image_name.split("/")[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir + imidx + '.png')


# --------- 1. get image path and name ---------

from sys import argv

# script, model_name = argv
model_name = 'GSD.pth'

image_dir = 'GSD/test/image/'
# image_dir = '/home/wanggd/Downloads/validation'
prediction_dir = './released_gsd_results/'
reflection_dir = './released_gsd_reflections/'
model_dir = model_name

prediction_dir = prediction_dir  + '/' + 'GSD/'
if not os.path.exists(prediction_dir):
    os.makedirs(prediction_dir)

reflection_dir = reflection_dir + '/' + 'GSD/'
if not os.path.exists(reflection_dir):
    os.makedirs(reflection_dir)

img_name_list = [os.path.join(image_dir, img_name) for img_name in os.listdir(image_dir)]

# --------- 2. load data ---------
# 1. load data
test_salobj_dataset = SalObjDataset(img_name_list=img_name_list, lbl_name_list=[], ref_name_list=[],
                                    transform=transforms.Compose([RescaleT(384), ToTensorLab(flag=0)]))
test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1, shuffle=False, num_workers=1)

# --------- 3. model define ---------
print("...load GlassNet...")
net = GlassNet()
net.load_state_dict(torch.load(model_dir))
if torch.cuda.is_available():
    net.cuda()
net.eval()

# --------- 4. inference for each image ---------
for i_test, data_test in enumerate(test_salobj_dataloader):

    print("inferring:", img_name_list[i_test].split("/")[-1])

    inputs_test = data_test['image']
    inputs_test = inputs_test.type(torch.FloatTensor)

    if torch.cuda.is_available():
        inputs_test = Variable(inputs_test.cuda())
    else:
        inputs_test = Variable(inputs_test)
        
    start = time.time()
    d0, d1, d2, d3, d4, ref = net(inputs_test)
    end = time.time()
    print(end-start)
    # normalization
    pred = d0[:, 0, :, :]
    pred = F.sigmoid(pred)
    pred = normPRED(pred)
    
    # save results to test_results folder
    save_output(img_name_list[i_test], inputs_test, pred, prediction_dir, True)
    save_reflection(img_name_list[i_test], ref, reflection_dir)

    del d0, d1, d2, d3, d4, ref
