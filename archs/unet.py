# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.runner import Runner


def trigger_visualization_hook(cfg, args):
    default_hooks = cfg.default_hooks
    if 'visualization' in default_hooks:
        visualization_hook = default_hooks['visualization']
        # Turn on visualization
        visualization_hook['draw'] = True
        if args['show']:
            visualization_hook['show'] = True
            visualization_hook['wait_time'] = args['wait_time']
        if args['show_dir']:
            visualizer = cfg.visualizer
            visualizer['save_dir'] = args['show_dir']
    else:
        raise RuntimeError(
            'VisualizationHook must be included in default_hooks.'
            'refer to usage '
            '"visualization=dict(type=\'VisualizationHook\')"')

    return cfg


def build_unet(args):
    # load config
    cfg = Config.fromfile(args['config'])
    cfg.launcher = args['launcher']
    if args['cfg_options'] is not None:
        cfg.merge_from_dict(args['cfg_options'])

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args['work_dir'] is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args['work_dir']
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args['config']))[0])

    cfg.load_from = args['checkpoint']

    if args['show'] or args['show_dir']:
        cfg = trigger_visualization_hook(cfg, args)

    if args['tta']:
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
        cfg.tta_model.module = cfg.model
        cfg.model = cfg.tta_model

    # add output_dir in metric
    if args['out'] is not None:
        cfg.test_evaluator['output_dir'] = args['out']
        cfg.test_evaluator['keep_results'] = True

    # build the runner from config
    runner = Runner.from_cfg(cfg)
    # # start testing
    # a = runner.test()

    # print(a)
    return runner


if __name__ == '__main__':
  args = {
      'config': 'configs/unet/unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024.py',
      'checkpoint': 'ckpt/fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes_20211210_145204-6860854e.pth',
      'work_dir': None,
      'out': './',
      'show': False,
      'show_dir': None,
      'wait_time': 2,
      'cfg_options': None,
      'launcher': 'none',
      'tta': False,
      'local_rank': 0
  }
  build_unet(args)