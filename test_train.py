from trainer import Trainer

import detectron2.utils.comm as comm
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from MODULES.MaskFormer.config import add_mask_former_config


def main():
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_mask_former_config(cfg)
    cfg.set_new_allowed(True)
    cfg.merge_from_file("MODULES/MaskFormer/configs/custom/MaskAVSL_swin_base.yaml")
    # cfg.merge_from_list(args.opts)
    #cfg.MODEL.DEVICE = "cuda:2"
    cfg.MODEL.DEVICE = "cuda:1"
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.eval_only = False
    cfg.freeze()
    # default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="MaskAVSL")

    trainer = Trainer(cfg)

    trainer.train(epochs=1000)


if __name__ == '__main__':
    main()