# -*- coding: utf-8 -*-
# @Time    : 2021/4/12 7:12 下午
# @Author  : shaoguowen
# @Email   : shaoguowen@tencent.com
# @FileName: run.py
# @Software: PyCharm

import os, fnmatch
import argparse
import json
import datetime
import numpy as np
import torch
import cv2
import utils
from models.censeo_ivqa_model import CenseoIVQAModel

parser = argparse.ArgumentParser(description='censeo-algorithm inference')
parser.add_argument('--saved_model', required=True, help='saved model path')
parser.add_argument('--dst_file_path', required=True, help='dst file path')
parser.add_argument('--ref_file_path', help='ref file path')
parser.add_argument('--save_name', help='save name of json for predicted results')
args = parser.parse_args()

# config
model_path = os.path.join(args.saved_model, "model.pth")
config_path = os.path.join(args.saved_model, "config.json")
with open(config_path, "r") as fr:
    config = json.load(fr)
    '''
    torch_config = dict(
    model_name="CenseoIVQAModel",  
    fr=False, 
    backbone=dict(  
        model_name="resnet18",
        kwargs=dict(
            pretrained=True,
            out_indices=(3,),
            strides=(2, 2, 2)
        )
    ),
    head=dict(  
        model_name="SimpleHead",
        kwargs=dict(
            out_num=1
        )
    ),
    use_sigmoid=True 
    )
    '''
    config = utils.Dict2Obj(config)
    print(config.fr)
    print(config.backbone.model_name)
    

# load model
model=CenseoIVQAModel(config)
model=model.to('cpu')
weight = torch.load(model_path, map_location='cpu')
model.load_state_dict(weight)
model.eval()

def predict_fr(dst_path, ref_path):
    """
    用于FR的预测函数
    :param dst_path:
    :param ref_path:
    :return:
    """
    assert ref_path is not None
    if utils.is_img(dst_path):
        img_dst = cv2.imread(dst_path)
        img_dst = utils.transform(img_dst, config.input_process)[np.newaxis, ...]
        img_ref = cv2.imread(ref_path)
        img_ref = utils.transform(img_ref, config.input_process)[np.newaxis, ...]
        img_input = np.concatenate([img_dst, img_ref], 1)
        input = torch.from_numpy(img_input).permute(2,0,1).float()
        input = torch.unsqueeze(input, 0)

        print(input.shape)
        # mos_pred = sess.run('test_output:0', feed_dict={'test_input:0': img_input})
        mos_pred = model(input)
        print(mos_pred.shape)

        if config.output_process.use_sigmoid:
            mos_pred = utils.sigmoid(mos_pred)
        mos_pred = mos_pred * config.output_process.scale
        return mos_pred.reshape(-1).tolist()
    elif utils.is_video(dst_path):
        vcap_dst = utils.read_video_gen(dst_path, 0, -1, 10, 4)
        vcap_ref = utils.read_video_gen(ref_path, 0, -1, 10, 4)
        mos_preds = []
        while True:
            imgs_dst = next(vcap_dst, None)
            imgs_ref = next(vcap_ref, None)
            if imgs_dst is None or len(imgs_dst) == 0 or imgs_ref is None or len(imgs_ref) == 0:
                break
            imgs_dst = [utils.transform(img, config.input_process) for img in imgs_dst]
            imgs_dst = np.stack(imgs_dst)
            imgs_ref = [utils.transform(img, config.input_process) for img in imgs_ref]
            imgs_ref = np.stack(imgs_ref)
            imgs_input = np.concatenate([imgs_dst, imgs_ref], 1)
            mos_pred = sess.run('test_output:0', feed_dict={'test_input:0': imgs_input})
            if config.output_process.use_sigmoid:
                mos_pred = utils.sigmoid(mos_pred)
            mos_pred = mos_pred.reshape(-1) * config.output_process.scale
            mos_preds.append(mos_pred)
        return np.concatenate(mos_preds).tolist() if mos_preds else []


def predict_nr(dst_path):
    """
    用于nr模式的预测
    :param dst_path:
    :return:
    """
    if utils.is_img(dst_path):
        img_dst = cv2.imread(dst_path)
        '''
        align resolution
        '''
        h, w, c = img_dst.shape
        h_ = 360
        w_ = int(360 * w / h)
        img_dst = cv2.resize(img_dst, (w_, h_), interpolation= cv2.INTER_LINEAR)

        img_dst = utils.transform(img_dst, config.input_process)[np.newaxis, ...]
        img_input = img_dst

        # print('numpy input', img_input.shape)
        # n, t, c, h, w = x.size()
        input = torch.from_numpy(img_input).float()
        input = torch.unsqueeze(input, 0)

        # print('torch input', input.shape)
        # mos_pred = sess.run('test_output:0', feed_dict={'test_input:0': img_input})
        mos_pred = model(input)
        # print('torch mos:', mos_pred, mos_pred.shape)

        mos_pred = torch.squeeze(mos_pred, 0)
        mos_pred = mos_pred.cpu()
        mos_pred = mos_pred.detach().numpy() 
        print('numpy mos:',mos_pred, mos_pred.shape)

        if config.output_process.use_sigmoid:
            mos_pred = utils.sigmoid(mos_pred)
        

        mos_pred = mos_pred * config.output_process.scale
        return mos_pred.reshape(-1).tolist()
    elif utils.is_video(dst_path):
        # vcap_dst = utils.read_video_gen(dst_path, 0, -1, 10, 4)
        vcap_dst = utils.read_video_gen(dst_path, 0, -1, -1, 4)
        # video_path, start_time, end_time, subsample=1, batch_size=1
        mos_preds = []
        while True:
            imgs_dst = next(vcap_dst, None)
            if imgs_dst is None or len(imgs_dst) == 0:
                break
            imgs_dst = [utils.transform(img, config.input_process) for img in imgs_dst]
            imgs_dst = np.stack(imgs_dst)
            imgs_input = imgs_dst
            input = torch.from_numpy(imgs_input).float()
            input = torch.unsqueeze(input, 0)

            print('torch input', input.shape)
            # mos_pred = sess.run('test_output:0', feed_dict={'test_input:0': img_input})
            mos_pred = model(input)
            print('torch mos:', mos_pred, mos_pred.shape)

            mos_pred = torch.squeeze(mos_pred, 0)
            mos_pred = mos_pred.cpu()
            mos_pred = mos_pred.detach().numpy() 
            print('numpy mos:',mos_pred, mos_pred.shape)

            # mos_pred = sess.run('test_output:0', feed_dict={'test_input:0': imgs_input})
            if config.output_process.use_sigmoid:
                mos_pred = utils.sigmoid(mos_pred)
            mos_pred = mos_pred.reshape(-1) * config.output_process.scale
            mos_preds.append(mos_pred)
        return np.concatenate(mos_preds).tolist() if mos_preds else []


def predict(dst_path, ref_path=None):
    """
    模型预测
    :param dst_path:
    :param ref_path:
    :return:
    """
    if config.model.is_fr:
        return predict_fr(dst_path, ref_path)
    else:
        return predict_nr(dst_path)


if __name__ == '__main__':
    # result dir
    RESULT_DIR = "results"
    os.makedirs(RESULT_DIR, exist_ok=True)

    # model inference
    results = {}
    if os.path.isdir(args.dst_file_path):
        img_list = fnmatch.filter(os.listdir(args.dst_file_path), '*.png')
        img_list.sort()
        # for i, vname in enumerate(os.listdir(args.dst_file_path)):
        for i, vname in enumerate(img_list):
            print(i, vname)
            vpath_dst = os.path.join(args.dst_file_path, vname)
            if config.model.is_fr:
                results[vname] = predict(vpath_dst, os.path.join(args.ref_file_path, vname))
            else:
                results[vname] = predict(vpath_dst)
            print(results[vname])
    else:
        vname = os.path.basename(args.dst_file_path)
        print(vname)
        if config.model.is_fr:
            results[os.path.basename(args.dst_file_path)] = predict(args.dst_file_path, args.ref_file_path)
        else:
            results[os.path.basename(args.dst_file_path)] = predict(args.dst_file_path)
        print(results[vname])

    if args.save_name is None:
        date_str = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
        save_name = os.path.basename(args.dst_file_path) + "-" + date_str + ".json"
    else:
        save_name = args.save_name + ".json"
    with open(os.path.join("results", save_name), "w") as fw:
        print("saving result at :{}".format(os.path.join("results", save_name)))
        json.dump(results, fw)
    # sess.close()

