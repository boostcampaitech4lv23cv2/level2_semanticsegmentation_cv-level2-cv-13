
from mmcv.runner import master_only
from mmcv.runner import HOOKS
from mmcv.runner import LoggerHook
import wandb

@HOOKS.register_module()
class SweepLoggerHook(LoggerHook):

    def __init__(self,
                 interval: int = 1000,
                 ignore_last: bool = True,
                 reset_flag: bool = False,
                 by_epoch: bool = True,
                 ):
        super().__init__(interval, ignore_last, reset_flag, by_epoch)
        self.wandb=wandb



    @master_only
    def log(self, runner) -> None:
        tags = self.get_loggable_tags(runner)
        if "val/mIoU" in tags.keys():
            value=tags["val/mIoU"]
            del tags["val/mIoU"]
            tags["mIoU"]=value
        if tags:
            self.wandb.log(
                    tags, step=self.get_iter(runner))
