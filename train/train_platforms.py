import os

class TrainPlatform:
    def __init__(self, save_dir, *args, **kwargs):
        self.path, file = os.path.split(save_dir)
        self.name = kwargs.get('name', file)

    def report_scalar(self, name, value, iteration, group_name=None):
        pass

    def report_args(self, args, name):
        pass

    def close(self):
        pass

class WandBPlatform(TrainPlatform):
    import wandb
    def __init__(self, save_dir, config=None, *args, **kwargs):
        super().__init__(save_dir, args, kwargs)
        # self.wandb.login(host=os.getenv("WANDB_BASE_URL"), key=os.getenv("WANDB_API_KEY"))
        self.wandb.init(
            project='MDM',
            name=self.name,
            # id=self.name,  # in order to send continued runs to the same record
            # resume='allow',  # in order to send continued runs to the same record
            save_code=True,
            config=config)  # config can also be sent via report_args()

    def report_scalar(self, name, value, iteration, group_name=None):
        self.wandb.log({name: value}, step=iteration)

    def report_args(self, args, name):
        self.wandb.config.update(args)

    def close(self):
        self.wandb.finish()

# Deprecated
class ClearmlPlatform(TrainPlatform):
    def __init__(self, save_dir):
        from clearml import Task
        path, name = os.path.split(save_dir)
        self.task = Task.init(project_name='motion_diffusion',
                              task_name=name,
                              output_uri=path)
        self.logger = self.task.get_logger()

    def report_scalar(self, name, value, iteration, group_name):
        self.logger.report_scalar(title=group_name, series=name, iteration=iteration, value=value)

    def report_args(self, args, name):
        self.task.connect(args, name=name)

    def close(self):
        self.task.close()


class TensorboardPlatform(TrainPlatform):
    def __init__(self, save_dir):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=save_dir)

    def report_scalar(self, name, value, iteration, group_name=None):
        self.writer.add_scalar(f'{group_name}/{name}', value, iteration)

    def close(self):
        self.writer.close()


class NoPlatform(TrainPlatform):
    def __init__(self, save_dir):
        pass


