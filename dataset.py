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


def video_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, '{:05d}.jpg'.format(i))
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
    def __init__(self, video_path, n_frames,
                 spatial_transform=None, temporal_transform=None,
                 sample_duration=16, get_loader=get_default_video_loader, sample_step=16, RGB_to_BGR=False, mean=[]):
        self.n_frames = n_frames
        self.data = self.make_dataset(video_path, sample_duration, sample_step)
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.loader = get_loader()
        self.RGB_to_BGR = RGB_to_BGR
        self.mean = mean
        print('RGB_to_BGR', RGB_to_BGR)

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
        clip = self.loader(path, frame_indices)
        print(len(clip), frame_indices, path)
        #print(os.listdir(path))
        if self.spatial_transform is not None:
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
        if self.RGB_to_BGR:
            clip[0, :, :, :] -= self.mean[0]
            clip[1, :, :, :] -= self.mean[1]
            clip[2, :, :, :] -= self.mean[2]
            clip = clip[[2, 1, 0], :, :, :]

        target = self.data[index]['segment']

        return clip, target

    def __len__(self):
        return len(self.data)

    def make_dataset(self, video_path, sample_duration, sample_step):
        dataset = []

        n_frames = len(os.listdir(video_path))
        print(n_frames)

        begin_t = 1
        end_t = n_frames
        sample = {
            'video': video_path,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,
        }

        if not self.n_frames:
            step = sample_step
            for i in range(1, (n_frames - sample_duration + 1), step):
                sample_i = copy.deepcopy(sample)
                sample_i['frame_indices'] = list(range(i, i + sample_duration))
                #print(sample_i['frame_indices'])
                sample_i['segment'] = torch.IntTensor([i, i + sample_duration - 1])
                dataset.append(sample_i)
            if i - step != n_frames - sample_duration + 1:
                sample_i = copy.deepcopy(sample)
                sample_i['frame_indices'] = list(range(n_frames - sample_duration + 1, n_frames + 1))
                sample_i['segment'] = torch.IntTensor([n_frames - sample_duration + 1, n_frames])
                #print(sample_i['frame_indices'])
                dataset.append(sample_i)
        else:
            def check(begin, end, nf):
                if begin < 1:
                    intervel = 1-begin
                    begin += intervel
                    end += intervel
                if end > nf + 1:
                    intervel = end - nf - 1
                    end -= intervel
                    begin -= intervel
                return begin, end

            samples = []
            bound = [int(i) for i in np.linspace(0, n_frames, self.n_frames+1)]
            for i in range(self.n_frames):
                samples.append((bound[i] + bound[i+1]) // 2)
            for item in samples:
                sample_i = copy.deepcopy(sample)
                begin_t = max(item - sample_duration // 2, 1)
                end_t = min(item + sample_duration // 2, n_frames+1)
                length = end_t - begin_t
                indices = list(range(begin_t, end_t))
                if length != sample_duration:
                    remain = sample_duration - length
                    if begin_t == 1:
                        for pp in range(remain):
                            indices.insert(0, 1)
                    else:
                        for pp in range(remain):
                            indices.append(indices[-1])

                #begin_t, end_t = check(begin_t, end_t, n_frames)

                #sample_i['frame_indices'] = list(range(begin_t, end_t))
                sample_i['frame_indices'] = indices
                sample_i['segment'] = torch.IntTensor([begin_t, end_t - 1])
                dataset.append(sample_i)

        return dataset
