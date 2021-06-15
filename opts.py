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
    parser.add_argument('--feats_dir', default='/home/yangbang/VideoCaptioning/MSRVTT/feats/DFLC3D', type=str)
    parser.add_argument('--sample_duration', default=16, type=int)
    parser.add_argument('--sample_step', default=8, type=int)

    parser.add_argument('--use_setask', action='store_true')
    parser.add_argument('--dim_task2', type=int, default=300)
    parser.add_argument('--dim_feats', type=int, default=0)
    parser.add_argument('--sentence_embedding_path', type=str, default="/home/yangbang/VideoCaptioning/MSRVTT/feats/sentence_embedding/")
    parser.add_argument('--lamda', type=float, default=0.5)
    parser.add_argument('--dim_category_task', nargs='+', type=int, default=[400])

    parser.add_argument('--mean_dataset', default='activitynet', type=str, help='activitynet | kinetics | sports1m')
    parser.add_argument('--dataset', default='VATEX', type=str)
    parser.add_argument('--c3d_type', default=0, type=int, help='0: pool5(512), 1: pool5 region, 2: fc7(4096)')
    parser.add_argument('--n_frames', default=0, type=int)
    parser.add_argument('--use_db', default=False, action='store_true')
    args = parser.parse_args()

    assert args.dataset in ['MSRVTT', 'Youtube2Text', 'VATEX']
    if args.dataset == 'Youtube2Text':
        args.video_root = args.video_root.replace('MSRVTT', 'Youtube2Text')
        args.sentence_embedding_path = args.sentence_embedding_path.replace('MSRVTT', 'Youtube2Text')
        args.feats_dir = args.feats_dir.replace('MSRVTT', 'Youtube2Text')

    return args

'''
python main.py --video_root /home/yangbang/VideoCaptioning/MSRVTT/all_frames/ \
--batch_size 16 --gpu 3 \
--model /home/yangbang/VideoCaptioning/resnext-101-kinetics.pth --mean_dataset kinetics \
--feats_dir /home/yangbang/VideoCaptioning/MSRVTT/feats/c3d_kinetics \
--dataset MSRVTT

python main.py --video_root /home/yangbang/VideoCaptioning/MSRVTT/all_frames/ \
--batch_size 16 --gpu 3 \
--model /home/yangbang/VideoCaptioning/c3d.pickle --mean_dataset sports1m --model_name c3d \
--feats_dir /home/yangbang/VideoCaptioning/MSRVTT/feats/c3d_sports1m \
--dataset MSRVTT


python main.py --video_root /home/yangbang/VideoCaptioning/MSRVTT/all_frames/ \
--batch_size 16 --gpu 3 \
--model /home/yangbang/VideoCaptioning/c3d.pickle --mean_dataset sports1m --model_name c3d \
--feats_dir /home/yangbang/VideoCaptioning/MSRVTT/feats/c3d_sports1m \
--dataset Youtube2Text \
--c3d_type 1 \
--n_frames 20

python main.py --video_root /home/yangbang/VideoCaptioning/MSRVTT/all_frames/ \
--batch_size 16 --gpu 3 \
--model /home/yangbang/VideoCaptioning/resnext-101-kinetics.pth --mean_dataset kinetics \
--feats_dir /home/yangbang/VideoCaptioning/MSRVTT/feats/c3d_kinetics \
--dataset Youtube2Text \
--c3d_type 1 \
--n_frames 20


python main.py --video_root /home/yangbang/VideoCaptioning/MSRVTT/all_frames/ \
--batch_size 16 --gpu 3 \
--model /home/yangbang/VideoCaptioning/resnext-101-kinetics.pth --mean_dataset kinetics \
--feats_dir ./kinetics \
--dataset MSRVTT \
--c3d_type 0 \
--n_frames 60 \
--use_db

python main.py --video_root /home/yangbang/VideoCaptioning/MSRVTT/all_frames/ \
--batch_size 16 --gpu 3 \
--model /home/yangbang/VideoCaptioning/resnext-101-kinetics.pth --mean_dataset kinetics \
--feats_dir ./msvd_kinetics \
--dataset Youtube2Text \
--c3d_type 0 \
--n_frames 60 \
--use_db
'''


'''
python main.py --video_root /home/yangbang/VideoCaptioning/MSRVTT/all_frames/ \
--batch_size 16 --gpu 2 \
--model /home/yangbang/VideoCaptioning/resnext-101-kinetics.pth --mean_dataset kinetics \
--feats_dir ./ \
--dataset MSRVTT \
--c3d_type 0 \
--n_frames 60 \
--use_db

python main.py --video_root /home/yangbang/VideoCaptioning/MSRVTT/all_frames/ \
--batch_size 16 --gpu 3 \
--model /home/yangbang/VideoCaptioning/resnext-101-kinetics.pth --mean_dataset kinetics \
--feats_dir ./ \
--dataset Youtube2Text \
--c3d_type 0 \
--n_frames 60 \
--use_db


python main.py --video_root /home/yangbang/VideoCaptioning/MSRVTT/all_frames/ \
--batch_size 16 --gpu 3 \
--model /home/yangbang/VideoCaptioning/c3d.pickle --mean_dataset sports1m --model_name c3d \
--feats_dir ./ \
--dataset Youtube2Text \
--c3d_type 0 \
--n_frames 60 \
--use_db

python main.py --video_root /home/yangbang/VideoCaptioning/MSRVTT/all_frames/ \
--batch_size 16 --gpu 2 \
--model /home/yangbang/VideoCaptioning/c3d.pickle --mean_dataset sports1m --model_name c3d \
--feats_dir ./ \
--dataset MSRVTT \
--c3d_type 0 \
--n_frames 60 \
--use_db

'''

