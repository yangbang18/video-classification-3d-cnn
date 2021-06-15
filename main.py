import os
import sys
import json
import subprocess
import numpy as np
import torch
from torch import nn

from opts import parse_opts
from model import generate_model
from mean import get_mean
from classify import classify_video

from tqdm import tqdm
from collections import OrderedDict
import h5py
if __name__=="__main__":
    opt = parse_opts()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    opt.mean = get_mean(opt.mean_dataset)
    opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
    opt.sample_size = 112
    
    db = None
    if opt.use_db:
        name = '%s_%s_%s.hdf5' % (opt.dataset.lower(), 
            'c3d' if 'c3d' == opt.model_name else 'kinetics', 
            str(opt.n_frames) if opt.n_frames else '%d_%d'%(opt.sample_duration, opt.sample_step))
        opt.feats_dir += name
        db = h5py.File(opt.feats_dir, 'a')
    else:
        if not opt.n_frames:
            opt.feats_dir = opt.feats_dir + '_{}_{}'.format(opt.sample_duration, opt.sample_step)
        else:
            opt.feats_dir += '_%d' % opt.n_frames
            if opt.c3d_type == 1:
                opt.feats_dir += '_rf'
        print(opt.feats_dir)
        if not os.path.exists(opt.feats_dir):
            os.makedirs(opt.feats_dir)
    model = generate_model(opt)
    #input = torch.FloatTensor(10, 3, 8, 112, 112)
    #f = model(input, 8)
    #print(f.shape)
    #summary(model, input_size=(10, 3, 16, 112, 112))

    print(model)
    print('loading model {}'.format(opt.model))
    model_data = torch.load(opt.model)
    if opt.model_name != 'c3d':
        assert opt.arch == model_data['arch']
        model.load_state_dict(model_data['state_dict'])
    else:
        new_state_dict = OrderedDict() 
        for k, v in model_data.items(): 
            name = 'module.' + k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    model.eval()
    if opt.verbose:
        print(model)

    #input_files = []
    input_files = os.listdir(opt.video_root)
    class_names = []
    ''' 
    with open(opt.input, 'r') as f:
        for row in f:
            input_files.append(row[:-1])

     
    with open('class_names_list') as f:
        for row in f:
            class_names.append(row[:-1])
    '''
    
    for input_file in tqdm(input_files):
        print(input_file)
        if opt.dataset != 'VATEX':
            if int(input_file[5:]) >= 10000:
                continue
        if input_file in db.keys():
            continue
        video_path = os.path.join(opt.video_root, input_file)
        if not opt.use_db:
            npy_file = os.path.join(opt.feats_dir, input_file.split('.')[0] + '.npy')
            if os.path.exists(npy_file):
                continue
        classify_video(video_path, input_file, class_names, model, opt, db)

    '''
    if os.path.exists('tmp'):
        subprocess.call('rm -rf tmp', shell=True)

    with open(opt.output, 'w') as f:
        json.dump(outputs, f)
    '''
    
