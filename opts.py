import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='input', type=str, help='Input file path')
    parser.add_argument('--video_root', default="/home/yangbang/VideoCaptioning/MSRVTT/all_frames/", type=str, help='Root path of input videos')
    parser.add_argument('--model', default="/home/yangbang/VideoCaptioning/3D-ResNets-PyTorch/results/MSVD_tc9/e5_feats512_7333.pth", type=str, help='Model file path')
    parser.add_argument('--output', default='output.json', type=str, help='Output file path')
    parser.add_argument('--mode', default='feature', type=str, help='Mode (score | feature). score outputs class scores. feature outputs features (after global average pooling).')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch Size')
    parser.add_argument('--n_threads', default=4, type=int, help='Number of threads for multi-thread loading')
    parser.add_argument('--model_name', default='resnext', type=str, help='Currently only support resnet')
    parser.add_argument('--model_depth', default=101, type=int, help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument('--resnet_shortcut', default='B', type=str, help='Shortcut type of resnet (A | B)')
    parser.add_argument('--wide_resnet_k', default=2, type=int, help='Wide resnet k')
    parser.add_argument('--resnext_cardinality', default=32, type=int, help='ResNeXt cardinality')
    parser.add_argument('--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.set_defaults(verbose=False)
    parser.add_argument('--verbose', action='store_true', help='')
    parser.set_defaults(verbose=False)

    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--feats_dir', default='/home/yangbang/VideoCaptioning/MSRVTT/feats/', type=str)
    parser.add_argument('--sample_duration', default=16, type=int)
    parser.add_argument('--sample_step', default=8, type=int)
    parser.add_argument('--n_frames', default=0, type=int)
    parser.add_argument('--image_prefix', default='', type=str)
    parser.add_argument('--image_suffix', default='jpg', type=str)
    
    parser.add_argument('--limit', default=0, type=int)

    parser.add_argument('--latency', action='store_true')
    parser.add_argument('--n_latency_samples', type=int, default=1000)
    args = parser.parse_args()
    return args


