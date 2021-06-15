import torch
from dataset import Video
from spatial_transforms import (Compose, Normalize, Scale, CenterCrop, ToTensor)
from temporal_transforms import LoopPadding


def classify_video(video_dir, model, opt):
    spatial_transform = Compose([
        Scale(opt.sample_size),
        CenterCrop(opt.sample_size),
        ToTensor(),
        Normalize(opt.mean, [1, 1, 1])
    ])

    temporal_transform = LoopPadding(opt.sample_duration)
    data = Video(
        video_dir, 
        opt.n_frames, 
        spatial_transform=spatial_transform,
        temporal_transform=temporal_transform,
        sample_duration=opt.sample_duration, 
        sample_step=opt.sample_step, 
        mean=opt.mean,
        verbose=opt.verbose
    )
    
    data_loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)

    video_outputs = []
    video_segments = []
    model.eval()
    with torch.no_grad():
        for inputs, segments in data_loader:
            outputs = model(inputs, opt.sample_duration)
            video_outputs.append(outputs)
            video_segments.append(segments)

    video_outputs = torch.cat(video_outputs, dim=0)

    return video_outputs, video_segments
