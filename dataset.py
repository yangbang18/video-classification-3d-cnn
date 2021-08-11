import torch
import torch.utils.data as data
from PIL import Image
import os
import math
import functools
import copy
import numpy as np

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        import accimage
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path, frame_indices, image_loader, image_prefix='', image_suffix='jpg'):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, '{:s}{:05d}.{:s}'.format(image_prefix, i, image_suffix))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video

    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            if subset == 'testing':
                video_names.append('test/{}'.format(key))
            else:
                label = value['annotations']['label']
                video_names.append('{}/{}'.format(label, key))
                annotations.append(value['annotations'])

    return video_names, annotations


class Video(data.Dataset):
    def __init__(
            self,
            video_path,
            n_frames,
            spatial_transform=None, 
            temporal_transform=None,
            sample_duration=16, 
            get_loader=get_default_video_loader, 
            sample_step=16, 
            mean=[],
            verbose=False,
            image_prefix='',
            image_suffix='jpg',
        ):
        self.n_frames = n_frames
        self.data = self.make_dataset(video_path, sample_duration, sample_step)
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.loader = get_loader()
        self.mean = mean
        self.verbose = verbose
        self.image_prefix = image_prefix
        self.image_suffix = image_suffix

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = self.data[index]['video']

        frame_indices = self.data[index]['frame_indices']
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(path, frame_indices, image_prefix=self.image_prefix, image_suffix=self.image_suffix)
        
        if self.verbose:
            print(len(clip), frame_indices, path)

        if self.spatial_transform is not None:
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        target = self.data[index]['segment']
        return clip, target

    def __len__(self):
        return len(self.data)

    def make_dataset(self, video_path, sample_duration, sample_step):
        dataset = []

        n_frames = len(os.listdir(video_path))

        begin_t = 1
        end_t = n_frames
        sample = {
            'video': video_path,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,
        }

        if not self.n_frames:
            begin_t = 1
            done_flag = False
            while True:
                end_t = min(begin_t + sample_duration, n_frames + 1)
                length = end_t - begin_t
                indices = list(range(begin_t, end_t))
                
                if length < sample_duration:
                    done_flag = True
                    remain = sample_duration - length
                    indices = indices + [indices[-1]] * remain
                
                if end_t == n_frames + 1:
                    done_flag = True
                
                sample_i = copy.deepcopy(sample)
                sample_i['frame_indices'] = indices
                sample_i['segment'] = torch.IntTensor([begin_t, end_t - 1])
                dataset.append(sample_i)
                
                if done_flag:
                    break
                
                begin_t += sample_step
        else:
            samples = []
            bound = [int(i) for i in np.linspace(0, n_frames, self.n_frames+1)]
            for i in range(self.n_frames):
                samples.append((bound[i] + bound[i+1]) // 2)
            
            for item in samples:    
                begin_t = max(item - sample_duration // 2, 1)
                end_t = min(item + sample_duration // 2, n_frames+1)
                
                length = end_t - begin_t
                indices = list(range(begin_t, end_t))
                if length != sample_duration:
                    remain = sample_duration - length
                    if begin_t == 1:
                        indices = [1] * remain + indices
                    else:
                        indices = indices + [indices[-1]] * remain

                sample_i = copy.deepcopy(sample)
                sample_i['frame_indices'] = indices
                sample_i['segment'] = torch.IntTensor([begin_t, end_t - 1])
                dataset.append(sample_i)

        return dataset
