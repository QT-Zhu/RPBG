import numpy as np
import random
import cv2
import yaml
from time import time
from pathlib import Path
import os, sys
import datetime
import argparse
from huepy import bold, lightblue, orange, lightred, green, red

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import torch.utils.data
import torchvision

from tensorboardX import SummaryWriter

from RPBG.utils.perform import TicToc, AccumDict, Tee
from RPBG.utils.arguments import MyArgumentParser, eval_args
from RPBG.models.compose import ModelAndLoss
from RPBG.pipelines import save_pipeline
from RPBG.utils.train import to_device, to_numpy, get_module, freeze, load_model_checkpoint, unwrap_model
from RPBG.pipelines.pcprrender import PCPRRender


def mse(image_pred, image_gt, valid_mask=None, reduction="mean"):
    value = (image_pred - image_gt) ** 2
    if valid_mask is not None:
        value = value[valid_mask]
    if reduction == "mean":
        return torch.mean(value)
    return value

def psnr(image_pred, image_gt, valid_mask=None, reduction="mean"):
    return -10 * torch.log10(mse(image_pred, image_gt, valid_mask, reduction))

def map_inner_sample(data):
    if isinstance(data, list):
        data = [[inner_d[i] for inner_d in data] for i in range(len(data[0]))]
    elif isinstance(data, dict):
        if 'id' in data.keys():
            data['id'] = data['id'].flatten()
    elif isinstance(data, torch.Tensor) and len(data.shape)>3:
        data = torch.cat([data[i] for i in range(data.shape[0])],0)
    return data

def parse_data(data):
    if isinstance(data, dict):
        for k in data.keys():
            data[k] = map_inner_sample(data[k])
    return data

def setup_environment(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True

    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    os.environ['OMP_NUM_THGPCRS'] = '1'

def setup_logging(save_dir):
    tee = Tee(os.path.join(save_dir, 'log.txt'))
    sys.stdout, sys.stderr = tee, tee

def get_experiment_name(args, default_args, args_to_ignore, delimiter='__'):
    s = []
    args = vars(args)
    default_args = vars(default_args)

    def shorten_paths(args):
        args = dict(args)
        for arg, val in args.items():
            if isinstance(val, Path):
                args[arg] = val.name
        return args

    args = shorten_paths(args)
    default_args = shorten_paths(default_args)

    for arg in sorted(args.keys()):
        if arg not in args_to_ignore and default_args[arg] != args[arg]:
            s += [f"{arg}({args[arg]})"]
    s = [args['name']]+ s
    out = delimiter.join(s)
    out = out.replace("'", '')
    out = out.replace("[", '')
    out = out.replace("]", '')
    out = out.replace(" ", '')
    return out

def make_experiment_dir(base_dir, postfix='', use_time=True):
    time = datetime.datetime.now()  
    if use_time:
        postfix = time.strftime(f"{postfix}_%m-%d_%H-%M")
    save_dir = os.path.join(base_dir, postfix)
    
    if not args.eval:
        os.makedirs(f'{save_dir}/checkpoints', exist_ok=True)
        os.makedirs(f'{save_dir}/train', exist_ok=True)      
    if args.eval or args.eval_in_train:
        os.makedirs(f'{save_dir}/eval', exist_ok=True)    

    return save_dir

def num_param(model):
    return sum([p.numel() for p in unwrap_model(model).parameters()])

def run_epoch(pipeline, phase, epoch, args, iter_cb=None):     
    ad = AccumDict()
    tt = TicToc()
    
    device = "cuda:0"

    model = pipeline.model
    optimizer = pipeline.optimizer
    
    print(f'model parameters: {num_param(model)}')

    if not args.eval_traj:
        model = ModelAndLoss(model)
        
    if args.multigpu:
        model = nn.DataParallel(model)
        
    ds_list = pipeline.__dict__[f'ds_{phase}']
    if phase == 'train':
        random.shuffle(ds_list)
    renderer.update_ds(ds_list)

    def run_sub(dl, extra_optimizer, phase='train'):
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        
        model.cuda()

        for it, data in enumerate(dl):
            tt.tic()
            if phase == 'train':
                data = parse_data(data)
            data['input'], depths = renderer.render(data)  
            inputs = data['input'] 

            data_input = to_device(inputs, device)
            target = to_device(data['target'], device)
            
            out, loss_dict = model(data_input, target)

            ad.add('batch_time', tt.toc())
            
            im_out = out['im_out']
            if phase=='train':
                out_ = torchvision.utils.make_grid(im_out[:2*4,...], nrow=4)
                target_ = torchvision.utils.make_grid(target[:2*4,...], nrow=4)
                depth_ = torchvision.utils.make_grid(depths['uv_1d_p1'][:2*4,...],nrow=4)
                tmp = [to_numpy(out_), to_numpy(target_), to_numpy(depth_)]
                cv2.imwrite(f'{exper_dir}/train/cmp_{it%args.log_num_images}.png', np.concatenate(tmp,0)[...,::-1])
            
            loss = loss_dict['vgg_loss'] + loss_dict['huber_loss'] * huber_ratio + loss_dict['fft_loss']

            if loss.numel() > 1:
                loss = loss.mean()

            if phase == "train":
                psnr_value = psnr((im_out).cpu().detach(), target.cpu().detach())
                ad.add('psnr', psnr_value)
            else:
                batch_dim = im_out.shape[0]
                for i in range(batch_dim):
                    im_out = im_out.clip(0,1)
                    psnr_value = psnr(im_out[i,...].cpu().detach(), target[i,...].cpu().detach())
                    ad.add('psnr', psnr_value)
                    

            if hasattr(pipeline.model, 'reg_loss'):
                reg_loss = pipeline.model.reg_loss()
                loss += reg_loss

                if torch.is_tensor(reg_loss):
                    reg_loss = reg_loss.item()
            
            if phase == 'train':
                tt.tic()
                loss.backward(create_graph=False)
                
                optimizer.step()
                optimizer.zero_grad()

                if extra_optimizer is not None:
                    extra_optimizer.step()
                    extra_optimizer.zero_grad()
            ad.add('vgg_loss', loss_dict['vgg_loss'].mean().item())
            if 'huber_loss' in loss_dict:
                ad.add('huber_loss', loss_dict['huber_loss'].mean().item() * huber_ratio)

            ad.add('loss', loss.item())

            if iter_cb:
                tt.tic()
                iter_cb.on_iter(it + it_before, max_it, input, out, target, depths['uv_1d_p1'], data, ad, phase, epoch)

    def run_traj(args):

        from RPBG.datasets.dynamic import TrajDataset
        import open3d

        traj = open3d.io.read_pinhole_camera_trajectory(str(args.eval_traj)).parameters
        K = traj[0].intrinsic.intrinsic_matrix.astype(np.float32)
        H, W = traj[0].intrinsic.height, traj[0].intrinsic.width
        view_matrices = []
        for cam in traj:
            view_matrix = cam.extrinsic.astype(np.float32)
            view_matrix = np.linalg.inv(view_matrix)
            view_matrix[:,1:3] *= -1
            view_matrices.append(view_matrix)
        camera_labels = [f"{i:08d}" for i in range(len(traj))]

        scene_data = {
            "intrinsic_matrix": K,
            "view_matrix": view_matrices,
            "camera_labels": camera_labels,
            "config": {"viewport_size": [W, H]}
        }

        my_ds = TrajDataset(
            phase="eval", scene_data=scene_data, 
            input_format="uv_1d_p1, uv_1d_p1_ds1, uv_1d_p1_ds2, uv_1d_p1_ds3, uv_1d_p1_ds4",
            view_list = view_matrices
        )
        batch_size_val = args.batch_size if args.batch_size_val is None else args.batch_size_val
        my_dl = DataLoader(my_ds, batch_size_val, shuffle=False, drop_last=False)
        
        torch.cuda.empty_cache()
        model.cuda()
        for it, data in enumerate(my_dl):
            tt.tic()
            data['input'], depths = renderer.render(data)  
            inputs = data['input'] 
            data_input = to_device(inputs, device)
            
            out = model(data_input)

            ad.add('batch_time', tt.toc())
            iter_cb.on_iter(it + it_before, max_it, input, out, None, depths['uv_1d_p1'], data, ad, phase, epoch)
            
    sub_size = 4

    it_before = 0
    max_it = np.sum([len(ds) for ds in ds_list]) // args.batch_size

    for i_sub in range(0, len(ds_list), sub_size):
        ds_sub = ds_list[i_sub:i_sub + sub_size]
        ds_ids = [d.id for d in ds_sub]
        print(f'running on datasets {ds_ids}')
        ds = ConcatDataset(ds_sub)
        if phase == 'train':
            dl = DataLoader(ds, args.batch_size, num_workers=args.dataloader_workers, drop_last=True, pin_memory=False, shuffle=True, worker_init_fn=ds_init_fn)
        else:
            batch_size_val = args.batch_size if args.batch_size_val is None else args.batch_size_val
            dl = DataLoader(ds, batch_size_val, num_workers=args.dataloader_workers, drop_last=False, pin_memory=False, shuffle=False, worker_init_fn=ds_init_fn)
        
        pipeline.dataset_load(ds_sub)
        print(f'total parameters: {num_param(model)}')

        extra_optimizer = pipeline.extra_optimizer(ds_sub)
        if args.eval_traj and args.eval:
            run_traj(args)
        else:
            run_sub(dl, extra_optimizer, phase)
        pipeline.dataset_unload(ds_sub)

        it_before += len(dl)

    avg_loss = np.mean(ad['loss'])
    avg_psnr = np.mean(ad['psnr'])
    iter_cb.on_epoch(phase, avg_loss, avg_psnr, epoch)

    return avg_loss, avg_psnr

def run_train(epoch, pipeline, args, iter_cb):
    if args.eval_in_train or (args.eval_in_train_epoch >= 0 and epoch >= args.eval_in_train_epoch):
        print('EVAL MODE IN TRAIN')
        pipeline.model.eval()
    else:
        pipeline.model.train()

    with torch.set_grad_enabled(True):
        return run_epoch(pipeline, 'train', epoch, args, iter_cb=iter_cb)

def run_eval(epoch, pipeline, args, iter_cb):
    if args.eval_in_test:
        pipeline.model.eval()
    else:
        print('TRAIN MODE IN EVAL')
        pipeline.model.train()

    with torch.set_grad_enabled(False):
        return run_epoch(pipeline, 'val', epoch, args, iter_cb=iter_cb)
    
class TrainIterCb:
    def __init__(self, args, writer):
        self.args = args
        self.writer = writer
        self.train_it = 0

    def on_iter(self, it, max_it, input, out, target, depth, data_dict, ad, phase, epoch):    
        if it % self.args.log_freq == 0:
            s = f'{phase.capitalize()}: [{epoch}][{it}/{max_it-1}]\t'
            s += str(ad)
            print(s)

        if phase == 'train':
            self.writer.add_scalar(f'{phase}/loss', ad['loss'][-1], self.train_it)

            if 'reg_loss' in ad.__dict__():
                self.writer.add_scalar(f'{phase}/reg_loss', ad['reg_loss'][-1], self.train_it)
            if 'vgg_loss' in ad.__dict__():
                self.writer.add_scalar(f'{phase}/vgg_loss', ad['vgg_loss'][-1], self.train_it)
            if 'huber_loss' in ad.__dict__():
                self.writer.add_scalar(f'{phase}/huber_loss', ad['huber_loss'][-1], self.train_it)

            self.train_it += 1
            
                
        if phase == 'val':
            # for each sample in a batch
            for i,fn in enumerate(data_dict['target_filename']):
                #name = fn.split('/')[-3]+'_'+fn.split('/')[-1]
                name = Path(fn).stem + ".jpg"
                out_fn = os.path.join(f'{exper_dir}/eval', name)
                tmp = [to_numpy(out['im_out'][i]),to_numpy(target[i]),to_numpy(depth[i]).repeat(3,-1)]

                # tmp
                cv2.imwrite(out_fn, np.concatenate(tmp,0)[...,::-1])
                # cv2.imwrite(out_fn+'.tg.jpg', to_numpy(target[i])[...,::-1])
                

    def on_epoch(self, phase, loss, psnr, epoch):
        if phase != 'train':
            self.writer.add_scalar(f'{phase}/loss', loss, epoch)
            self.writer.add_scalar(f'{phase}/psnr', psnr, epoch)

class EvalIterCb:
    def __init__(self, eval_dir='./eval'):
        self.eval_dir = os.path.join(eval_dir)
        os.makedirs(os.path.join(eval_dir,"pred"), exist_ok=True)
        os.makedirs(os.path.join(eval_dir,"target"), exist_ok=True)

    def on_iter(self, it, max_it, input, out, target, depth, data_dict, ad, phase, epoch):
        for i,fn in enumerate(data_dict['target_filename']):
            # name = fn.split('/')[-3]+'_'+fn.split('/')[-1]
            name = Path(fn).stem + ".jpg"
            out_fn = os.path.join(self.eval_dir, "pred", name)
            cv2.imwrite(out_fn, to_numpy(out['im_out'][i])[...,::-1])

            out_gt = os.path.join(self.eval_dir, "target", name)
            if target is not None:
                cv2.imwrite(out_gt, to_numpy(target[i])[...,::-1])
            # cv2.imwrite(out_fn+'.tg.jpg', to_numpy(target[i])[...,::-1])

    def on_epoch(self, phase, loss, psnr, epoch):
        pass

def save_splits(exper_dir, ds_train, ds_val):
    def write_list(path, data):
        with open(path, 'w') as f:
            for l in data:
                f.write(str(l))
                f.write('\n')

    for ds in ds_train.datasets:
        np.savetxt(os.path.join(exper_dir, 'train_view.txt'), np.vstack(ds.view_list))
        write_list(os.path.join(exper_dir, 'train_target.txt'), ds.target_list)

    for ds in ds_val.datasets:
        np.savetxt(os.path.join(exper_dir, 'val_view.txt'), np.vstack(ds.view_list))
        write_list(os.path.join(exper_dir, 'val_target.txt'), ds.target_list)

def ds_init_fn(worker_id):
    np.random.seed(int(time()))

def parse_image_size(string):
    error_msg = 'size must have format WxH'
    tokens = string.split('x')
    if len(tokens) != 2:
        raise argparse.ArgumentTypeError(error_msg)
    try:
        w = int(tokens[0])
        h = int(tokens[1])
        return w, h
    except ValueError:
        raise argparse.ArgumentTypeError(error_msg)

def parse_args(parser):
    args, _ = parser.parse_known_args()
    assert args.pipeline, 'set pipeline module'
    pipeline = get_module(args.pipeline)()
    pipeline.export_args(parser)

    # override defaults
    if args.config:
        with open(args.config) as f:
            config = yaml.load(f,Loader=yaml.FullLoader)
        parser.set_defaults(**config)

    return parser.parse_args(), parser.parse_args([])

def print_args(args, default_args):

    args_v = vars(args)
    default_args_v = vars(default_args)
    
    print(bold(lightblue(' - ARGV: ')), '\n', ' '.join(sys.argv), '\n')
    # Get list of default params and changed ones    
    s_default = ''     
    s_changed = ''
    for arg in sorted(args_v.keys()):
        value = args_v[arg]
        if default_args_v[arg] == value:
            s_default += f"{lightblue(arg):>50}  :  {orange(value if value != '' else '<empty>')}\n"
        else:
            s_changed += f"{lightred(arg):>50}  :  {green(value)} (default {orange(default_args_v[arg] if default_args_v[arg] != '' else '<empty>')})\n"

    print(f'{bold(lightblue("Unchanged args")):>69}\n\n'
          f'{s_default[:-1]}\n\n'
          f'{bold(red("Changed args")):>68}\n\n'
          f'{s_changed[:-1]}\n')

def check_pipeline_attributes(pipeline, attributes):
    for attr in attributes:
        if not hasattr(pipeline, attr):
            raise AttributeError(f'pipeline missing attribute "{attr}"')

def try_save_dataset(save_dir, dataset, prefix):
    if hasattr(dataset[0], 'target_list'):
        with open(os.path.join(save_dir, f'{prefix}.txt'), 'w') as f:
            for ds in dataset:
                f.writelines('\n'.join(ds.target_list))
                f.write('\n')

def save_args(exper_dir, args, prefix):
    with open(os.path.join(exper_dir, f'{prefix}.yaml'), 'w') as f:
        yaml.dump(vars(args), f)

if __name__ == '__main__':
    parser = MyArgumentParser(conflict_handler='resolve')
    parser.add = parser.add_argument
    parser.add('--eval', action='store_bool', default=False)
    parser.add('--eval_traj', type=Path, default=None)
    parser.add('--eval_all', action='store_bool', default=False)
    parser.add('--crop_size', type=parse_image_size, default='256x256')
    parser.add('--batch_size', type=int, default=8)
    parser.add('--batch_size_val', type=int, default=None, help='if not set, use batch_size')
    parser.add('--lr', type=float, default=1e-4)
    parser.add('--freeze_net', action='store_bool', default=False)
    parser.add('--eval_in_train', action='store_bool', default=False)
    parser.add('--eval_in_train_epoch', default=-1, type=int)
    parser.add('--eval_in_test',  action='store_bool', default=True)
    parser.add('--net_ckpt', type=Path, default=None, help='neural network checkpoint')
    parser.add('--save_dir', type=Path, default='data/experiments')
    parser.add('--eval_dir', type=Path, default='data/eval')
    parser.add('--epochs', type=int, default=100)
    parser.add('--seed', type=int, default=10086)
    parser.add('--save_freq', type=int, default=5, help='save checkpoint each save_freq epoch')
    parser.add('--log_freq', type=int, default=5, help='print log each log_freq iter')
    parser.add('--log_num_images', type=int, default=20)
    parser.add('--comment', type=str, default='', help='comment to experiment')
    parser.add('--paths_file', type=str)
    parser.add('--dataset_names', type=str, nargs='+')
    parser.add('--config', type=Path)
    parser.add('--pipeline', type=str, default="RPBG.pipelines.TexturePipeline", help='path to pipeline module')
    parser.add('--ignore_changed_args', type=str, nargs='+', default=['name', 'ignore_changed_args', 'save_dir', 'dataloader_workers', 'epochs', 'batch_size_val'])
    parser.add('--multigpu', action='store_bool', default=True)
    parser.add('--dataloader_workers', type=int, default=4)
    parser.add('--reg_weight', type=float, default=0.)
    parser.add('--input_format', type=str, default="uv_1d_p1, uv_1d_p1_ds1, uv_1d_p1_ds2, uv_1d_p1_ds3, uv_1d_p1_ds4")
    parser.add('--input_channels', type=int, default=8)
    parser.add('--supersampling', type=int, default=1)
    parser.add('--simple_name', action='store_bool', default=False)

    parser.add('--model', type=str, default='mimounet', help='name of model file')  
    parser.add('--name', type=str, default='tmp', help='name of experiment')  
    args, default_args = parse_args(parser)

    setup_environment(args.seed)

    huber_ratio = 1e+4

    if args.eval:
        iter_cb = EvalIterCb(eval_dir=f'{args.eval_dir}/{args.name}')
    else:
        if args.simple_name:
            args.ignore_changed_args += ['config', 'pipeline']

        exper_name = get_experiment_name(args, default_args, args.ignore_changed_args)
        exper_dir = make_experiment_dir(args.save_dir, postfix=exper_name, use_time=False)
        writer = SummaryWriter(log_dir=exper_dir, flush_secs=10)
        iter_cb = TrainIterCb(args, writer)

        setup_logging(exper_dir)

        print(f'experiment dir: {exper_dir}')

    print_args(args, default_args)

    args = eval_args(args)
    pipeline = get_module(args.pipeline)()
    pipeline.create(args)

    required_attributes = ['model', 'ds_train', 'ds_val', 'optimizer']
    check_pipeline_attributes(pipeline, required_attributes)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(pipeline.optimizer, patience=5, factor=0.5, verbose=True)

    if args.eval:
        assert args.net_ckpt and args.tex_ckpt
        net_ckpt, tex_ckpt = args.net_ckpt, args.tex_ckpt
        load_model_checkpoint(net_ckpt, pipeline.get_net())
        load_model_checkpoint(tex_ckpt, pipeline.textures[0])
        print(red(">>>> existing model loaded! <<<<"))

    if args.freeze_net:
        print('FREEZE NET')
        freeze(pipeline.get_net(), True)
    
    
    renderer = PCPRRender()

    if args.eval:
        val_loss, val_psnr = run_eval(0, pipeline, args, iter_cb)
        print('VAL LOSS', val_loss)
        print('VAL PSNR', val_psnr)
    else:
        try_save_dataset(exper_dir, pipeline.ds_train, 'train')
        try_save_dataset(exper_dir, pipeline.ds_val, 'val')

        save_args(exper_dir, args, 'args')
        save_args(exper_dir, default_args, 'default_args')

        lowest_loss = 1e+10
        
        for epoch in range(args.epochs):
            print('### EPOCH', epoch)

            print('> TRAIN')

            train_loss,_ = run_train(epoch, pipeline, args, iter_cb)
            
            print('TRAIN LOSS', train_loss)
            
            print('> EVAL')

            val_loss, val_psnr = run_eval(epoch, pipeline, args, iter_cb)
            
            print('VAL LOSS', val_loss)
            print('VAL PSNR', val_psnr)
            
            if val_loss is not None:
                lr_scheduler.step(val_loss)
            
            print('net_lr:',pipeline.optimizer.param_groups[0]['lr'])
            # print('tex_lr:',pipeline.extra_optimizer.param_groups[0]['lr'])
            
            writer.add_scalar(f'lr', pipeline.optimizer.param_groups[0]['lr'], epoch)
                
            if ((epoch + 1) % args.save_freq == 0): # and (val_loss<lowest_loss):
                
                if val_loss < lowest_loss:
                    print(bold(red(f"replacing best model to epoch {epoch}")))
                    lowest_loss = val_loss
                    save_pipeline(pipeline, os.path.join(exper_dir, 'checkpoints'), "best", args)

                # regular save
                save_pipeline(pipeline, os.path.join(exper_dir, 'checkpoints'), epoch, args)
