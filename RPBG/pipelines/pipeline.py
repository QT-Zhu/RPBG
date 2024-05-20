import munch
import torch
from torch import optim
import importlib
import os, sys
from RPBG.datasets.dynamic import get_datasets
from RPBG.models.texture import PointTexture
from RPBG.models.compose import NetAndTexture
from RPBG.utils.train import get_module, save_model, load_model_checkpoint
from RPBG.utils.arguments import deval_args


TextureOptimizerClass = optim.RMSprop


def get_net(input_channels, args):
    MIMOUNet = getattr(importlib.import_module(f'RPBG.models.{args.model}'), "MIMOUNet")

    net = MIMOUNet(
        num_input_channels=input_channels, 
        num_output_channels=3,
        num_res=4
        )
    return net


def get_texture(num_channels, size, args):
    if not hasattr(args, 'reg_weight'):
        args.reg_weight = 0.
    texture = PointTexture(num_channels, size, activation=args.texture_activation, reg_weight=args.reg_weight)
  
    return texture


class TexturePipeline:
    def export_args(self, parser):
        parser.add_argument('--descriptor_size', type=int, default=8)
        parser.add('--texture_lr', type=float, default=1e-1)
        parser.add('--texture_activation', type=str, default='none')

    def create(self, args):
        net = get_net(args.input_channels, args)

        textures = {}
            
        self.ds_train, self.ds_val = get_datasets(args)
        for ds in self.ds_train:
            assert ds.scene_data['pointcloud'] is not None, 'set pointcloud'
            size = ds.scene_data['pointcloud']['xyz'].shape[0]
            textures[ds.id] = get_texture(args.descriptor_size, size, args)
        
        self.optimizer = optim.Adam(net.parameters(), lr=args.lr)
        if len(textures) == 1:
            self._extra_optimizer = TextureOptimizerClass(textures[0].parameters(), lr=args.texture_lr)
        else:
            self._extra_optimizer = None

        ss = args.supersampling if hasattr(args, 'supersampling') else 1

        self.net = net
        self.textures = textures
        self.model = NetAndTexture(net, textures, ss)

        self.args = args

    def state_objects(self):
        datasets = self.ds_train

        objs = {'net': self.net}
        objs.update({ds.name: self.textures[ds.id] for ds in datasets})

        return objs

    def dataset_load(self, dataset):
        self.model.load_textures([ds.id for ds in dataset])
        
        for ds in dataset:
            ds.load()

    def extra_optimizer(self, dataset):
        # if we have single dataset, don't recreate optimizer
        if self._extra_optimizer is not None:
            lr_drop = self.optimizer.param_groups[0]['lr'] / self.args.lr
            self._extra_optimizer.param_groups[0]['lr'] = self.args.texture_lr * lr_drop
            return self._extra_optimizer

        param_group = []
        for ds in dataset:
            param_group.append(
                {'params': self.textures[ds.id].parameters()}
            )

        lr_drop = self.optimizer.param_groups[0]['lr'] / self.args.lr
        return TextureOptimizerClass(param_group, lr=self.args.texture_lr * lr_drop)

    def dataset_unload(self, dataset):
        self.model.unload_textures()

        for ds in dataset:
            ds.unload()
            self.textures[ds.id].null_grad()

    def get_net(self):
        return self.net


def load_pipeline(checkpoint, args_to_update=None):
    if os.path.exists(checkpoint):
        ckpt = torch.load(checkpoint, map_location='cpu')

    assert 'args' in ckpt

    if args_to_update:
        ckpt['args'].update(args_to_update)

    try:
        args = munch.munchify(ckpt['args'])

        pipeline = get_module(args.pipeline)()
        pipeline.create(args)
    except AttributeError as err:
        print('\nERROR: Checkpoint args is incompatible with this version\n', file=sys.stderr)
        raise err

    if checkpoint is not None:
        load_model_checkpoint(checkpoint, pipeline.get_net())

    return pipeline, args
    

def save_pipeline(pipeline, save_dir, epoch, args):
    objects = pipeline.state_objects()
    args_ = deval_args(args)

    for name, obj in objects.items():
        if name=='net' and args.freeze_net:
            continue
        obj_class = obj.__class__.__name__
        filename = f'{obj_class}'
        filename += f'_epoch_{epoch}'
            # f'_stage_{stage}'
            # f'_latest_{epoch}'
        if name:
            name = name.replace('/', '_')
            filename = f'{filename}_{name}'
        save_path = os.path.join(save_dir, filename + '.pth')
        save_model(save_path, obj, args=args_)
