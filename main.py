import os
import torch
from opts import parse_opts
from model import generate_model
from mean import get_mean
from classify import classify_video
from tqdm import tqdm
import h5py
import time


if __name__=="__main__":
    opt = parse_opts()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    opt.mean = get_mean()
    opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
    opt.sample_size = 112
    
    os.makedirs(opt.feats_dir, exist_ok=True)
    if opt.n_frames:
        name = 'motion_{}{}_kinetics_fixed{}{}.hdf5'.format(
            opt.model_name, opt.model_depth, opt.n_frames, '_logits' if opt.mode == 'score' else ''
        )
        print('- Given a video, equally sampling {} segments ({} frames per segment) and then extracting their features'.format(opt.n_frames, opt.sample_duration))
    else:
        name = 'motion_{}{}_kinetics_duration{}_overlap{}.hdf5'.format(
            opt.model_name, opt.model_depth, opt.sample_duration, opt.sample_step
        )
        print('- Dividing each video into segments ({} frames per segment) with {} frames overlapping'.format(opt.sample_duration, opt.sample_step))
    
    model = generate_model(opt)
    if opt.verbose:
        print(model)
        print('loading model {}'.format(opt.model))

    model_data = torch.load(opt.model)
    assert opt.arch == model_data['arch']
    model.load_state_dict(model_data['state_dict'])
    model.eval()
    
    input_files = os.listdir(opt.video_root)

    if opt.latency:
        total_time = 0
        input_files = input_files[:opt.n_latency_samples]
    else:
        opt.feats_dir = os.path.join(opt.feats_dir, name)
        print('- Save extracted features to {}'.format(opt.feats_dir))
        db = h5py.File(opt.feats_dir, 'a')
    
    for video_name in tqdm(input_files):
        assert 'video' in video_name

        if opt.limit and int(video_name[5:]) >= opt.limit:
            continue

        # if int(video_name[5:]) >= 10000:
        #     # for MSR-VTT, only extract features of video0 ~ video9999
        #     # for Youtube2Text (MSVD), only extract features of video0 ~ video1969
        #     continue

        if opt.latency:
            start_time = time.time()
        elif video_name in db.keys():
            # features is already extracted
            continue
        
        video_path = os.path.join(opt.video_root, video_name)
        features, _ = classify_video(video_path, model, opt, latency=opt.latency)

        if opt.latency:
            total_time += _
        else:
            db[video_name] = features.cpu().detach().numpy()

            if opt.verbose:
                print('{}: shape of {}'.format(video_name, features.shape))

    if opt.latency:
        print(f'- # samples: {len(input_files)}')
        print(f'- Total inference time: {total_time}')
        print(f'- Average latency: {total_time / len(input_files)}')
        with open('latency.txt', 'a') as f:
            f.write(f'motion-{opt.arch}\t{len(input_files)}\t{total_time}\t{total_time / len(input_files)}\n')
