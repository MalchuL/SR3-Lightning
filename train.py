import hydra
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from experiment import SR3Experiment


@hydra.main(config_path="configs/train_PANx4.yml")
def main(cfg):
    print(cfg.pretty())



    logger = TensorBoardLogger("logs")
    checkpoint_callback = ModelCheckpoint(
        filename='model_last_{epoch}',
        verbose=True,
        monitor='val_loss',
        mode='min'
    )

    model = SR3Experiment(cfg)

    trainer = Trainer(gpus=1, max_steps=cfg.train.niter, logger=logger, #limit_train_batches=cfg.steps_limit,
                      log_every_n_steps=cfg.train.log_freq, flush_logs_every_n_steps=cfg.train.log_freq, resume_from_checkpoint=cfg.checkpoint_path, check_val_every_n_epoch=cfg.train.val_freq,
                      precision=cfg.train.precision, gradient_clip_val=cfg.train.gradient_clip_val, callbacks=[LearningRateMonitor('step'), checkpoint_callback])

    trainer.fit(model)


if __name__ == '__main__':
    main()
