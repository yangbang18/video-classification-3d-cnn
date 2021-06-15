import torch
from torch.autograd import Variable

from dataset import Video
from spatial_transforms import (Compose, Normalize, Scale, CenterCrop, ToTensor)
from temporal_transforms import LoopPadding
import numpy as np
import os
import torchvision.transforms as tf
def classify_video(video_dir, video_name, class_names, model, opt, db=None):
    assert opt.mode in ['score', 'feature']
    RGB_to_BGR = True if 'c3d' == opt.model_name else False
    if RGB_to_BGR:
        spatial_transform = Compose([tf.Resize((128,171)),
                                     tf.CenterCrop(opt.sample_size),
                                     ToTensor()
                                     ])
        
    else:
        spatial_transform = Compose([Scale(opt.sample_size),
                                     CenterCrop(opt.sample_size),
                                     ToTensor(),
                                     Normalize(opt.mean, [1, 1, 1])])
    

    temporal_transform = LoopPadding(opt.sample_duration)
    data = Video(video_dir, opt.n_frames, spatial_transform=spatial_transform,
                 temporal_transform=temporal_transform,
                 sample_duration=opt.sample_duration, sample_step=opt.sample_step,
                 RGB_to_BGR=RGB_to_BGR, mean=opt.mean
                 )
    import math
    print(math.ceil((len(os.listdir(video_dir)) - opt.sample_duration) / opt.sample_step) + 1, len(data))
    #assert math.ceil((len(os.listdir(video_dir)) - opt.sample_duration) / opt.sample_step) + 1 == len(data)
    data_loader = torch.utils.data.DataLoader(data, batch_size=1,
                                              shuffle=False)

    video_outputs = []
    video_segments = []
    for i, (inputs, segments) in enumerate(data_loader):
        #print(inputs.shape, inputs[0, :, 0, 0, 0])
        with torch.no_grad():
            if opt.model_name == 'c3d':
                #print(i, inputs.shape)
                outputs = model(inputs.cuda())
            else:
                outputs = model(inputs, opt.sample_duration)

            video_outputs.append(outputs.cpu().data)
            video_segments.append(segments)

    video_outputs = torch.cat(video_outputs, dim=0)
    print(video_outputs.shape)
    if db is not None:
        db[video_name.split('.')[0]] = video_outputs.cpu().numpy()
    else:
        np.save(os.path.join(opt.feats_dir, video_name.split('.')[0] + '.npy'), video_outputs.cpu().numpy())
    return video_outputs
    '''
    video_segments = torch.cat(video_segments)
    results = {
        'video': video_name,
        'clips': []
    }

    _, max_indices = video_outputs.max(dim=1)
    for i in range(video_outputs.size(0)):
        clip_results = {
            'segment': video_segments[i].tolist(),
        }

        if opt.mode == 'score':
            clip_results['label'] = class_names[max_indices[i]]
            clip_results['scores'] = video_outputs[i].tolist()
        elif opt.mode == 'feature':
            clip_results['features'] = video_outputs[i].tolist()

        results['clips'].append(clip_results)

    return results
    '''
