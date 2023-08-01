import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
import sys
import datetime
import pytz
import mlflow
from torch.utils.tensorboard import SummaryWriter

from torchmetrics.classification import BinaryAUROC, BinaryPrecision, BinaryRecall, BinaryF1Score
from datasets import load_dataset, Dataset, DatasetDict

sys.path.append('.')
from model_ddp_trainer.noam_lr_scheduler import NoamLRScheduler
from model_ddp_trainer.model import DualEncoderInterestNetwork
from model_ddp_trainer.custom_collate_function import CustomCollateFunc


def ddp_setup(rank, world_size):
    """

    :param rank: Unique identifier of each process
    :param world_size: Total num of processes
    """
    print(f'ddp_setup() rank#{rank}')

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def exp_tracking_start(model_config: dict, train_config: dict):
    experiment_name = "recsys_experiment"
    mlflow.set_tracking_uri(train_config['mlflow_tracking_uri'])
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    print("experiment_id:", experiment.experiment_id)

    run_name = f'run-{datetime.datetime.now(tz=pytz.timezone("Australia/Sydney")).strftime("%Y%m%d-%H%M%S")}'
    mlflow.start_run(experiment_id=experiment.experiment_id, run_name=run_name)
    mlflow.log_params(model_config)
    mlflow.log_params(train_config)

    writer = SummaryWriter(log_dir=os.path.join(train_config["tensorboard_log_dir"], run_name))

    return run_name, writer


def exp_tracking_end(writer: torch.utils.tensorboard.writer.SummaryWriter):
    mlflow.end_run()
    writer.close()


def get_features(batch, device):
    """
    Get features from dataloader batch
    :param batch:
    :param device:
    :return:
    """
    param_col_map = {
        'input_ids': 'input_ids',
        'attention_mask': 'attention_mask',
        'token_type_ids': 'token_type_ids',
        'item_id': 'job_id_next',
        'src_user': 'resume_id_encoded',
        'src': 'job_id_sequence',
        'src_key_padding_mask': 'job_id_sequence_padding_mask'
    }
    return {key: batch[value].to(device, non_blocking=True) for key, value in param_col_map.items()}


def get_label(batch, device):
    """
    Get label column from dataloader batch
    :param batch:
    :param device:
    :return:
    """
    return batch['labels'].view(-1, 1).float().to(device, non_blocking=True)


class Trainer:
    def __init__(
            self,
            model: DualEncoderInterestNetwork,  # torch.nn.Module,
            model_config: dict,
            train_config: dict,
            train_dataloader: torch.utils.data.DataLoader,
            valid_dataloader: torch.utils.data.DataLoader,
            optimizer: torch.optim.Optimizer,
            lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
            gpu_id: int
    ):
        self.gpu_id = gpu_id
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        model = model.to(gpu_id)
        self.ddp_model = DDP(model, device_ids=[gpu_id], find_unused_parameters=True)  # False
        self.model_config = model_config
        self.train_config = train_config

        self.criterion = nn.BCEWithLogitsLoss(reduction='mean')  # batch loss mean

        self.metric_auroc = BinaryAUROC().to(gpu_id)
        self.metric_precision = BinaryPrecision().to(gpu_id)
        self.metric_recall = BinaryRecall().to(gpu_id)
        self.metric_f1score = BinaryF1Score().to(gpu_id)
        self.avg_valid_loss = torch.tensor(0, device=gpu_id)  # per GPU (process)

    def _save_checkpoint(self, training_steps, model_name):
        state_dict = self.ddp_model.module.state_dict()
        path = f'model_checkpoint_training_steps#{training_steps}_{model_name}.pt'
        torch.save(state_dict, path)
        print(f"Training step #{training_steps} | Training checkpoint saved at {path}")

    def validate(self, epoch):
        print(f'Trainer.validate() starts: self.gpu_id#{self.gpu_id} epoch#{epoch}')
        # Validation mode on
        self.ddp_model.eval()

        self.metric_auroc.reset()
        self.metric_precision.reset()
        self.metric_recall.reset()
        self.metric_f1score.reset()
        self.avg_valid_loss.zero_()  # per GPU
        valid_losses = []

        with torch.no_grad():
            self.valid_dataloader.sampler.set_epoch(0)  # Validate for the same 1 epoch
            # for valid_batch in tqdm(valid_dataloader, desc='valid_dataloader', leave=False, position=2):
            for valid_batch in self.valid_dataloader:
                output = self.ddp_model(**get_features(valid_batch, self.gpu_id))
                labels = get_label(valid_batch, self.gpu_id)
                valid_loss = self.criterion(output, labels)

                # self.avg_valid_loss += valid_loss
                valid_losses.append(valid_loss)

                self.metric_auroc.update(preds=output, target=labels.int())
                self.metric_precision.update(preds=output, target=labels.int())
                self.metric_recall.update(preds=output, target=labels.int())
                self.metric_f1score.update(preds=output, target=labels.int())
                # break
            self.avg_valid_loss = sum(valid_losses) / len(valid_losses)

        # Return to train mode
        self.ddp_model.train()
        print(f'Trainer.validate() ended: self.gpu_id#{self.gpu_id} epoch#{epoch}')

    def train(self, max_epochs: int, validate_every_n_steps: int = 5000, unfreeze_bert_after_step: int = None, save_model_stop_at_step: int = None):
        run_name, writer = None, None
        if self.gpu_id == 0:
            run_name, writer = exp_tracking_start(self.model_config, self.train_config)

        train_batch_step = -1  # steps in current device
        for epoch in range(max_epochs):
            print(f'[START] Trainer.train() for epoch#{epoch} on gpu_id#{self.gpu_id}')
            # Training
            self.ddp_model.train()
            self.train_dataloader.sampler.set_epoch(epoch)

            for train_batch in self.train_dataloader:  # 3.9/s

                train_batch_step += 1
                self.optimizer.zero_grad()
                output = self.ddp_model(**get_features(train_batch, self.gpu_id))
                train_batch_loss = self.criterion(output, get_label(train_batch, self.gpu_id))
                train_batch_loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()

                if self.gpu_id == 0:
                    mlflow.log_metric('train_batch_loss', train_batch_loss.item(), step=train_batch_step)
                    mlflow.log_metric('learning rate', self.lr_scheduler.get_last_lr()[0], step=train_batch_step)
                    # # Debug only
                    # writer.add_histogram('logits', output, global_step=train_batch_step)
                    # writer.add_scalar('bias of -1 linear layer (last linear layer in fully connected block)', self.ddp_model.module.fully_connected_block[-1].bias, global_step=train_batch_step)
                    # writer.add_histogram('weight of -1 linear layer (last linear layer in fully connected block)', self.ddp_model.module.fully_connected_block[-1].weight, global_step=train_batch_step)
                    # writer.add_histogram('weight of -3 linear layer (second to last linear layer in fully connected block)', self.ddp_model.module.fully_connected_block[-3].weight, global_step=train_batch_step)
                    # writer.add_histogram('weight of -7 linear layer (first linear layer in fully connected block)', self.ddp_model.module.fully_connected_block[-7].weight, global_step=train_batch_step)

                # Validation
                if train_batch_step % validate_every_n_steps == 0:  # validate_every_n_steps=3000, 5000

                    self.validate(epoch)

                    dist.all_reduce(self.avg_valid_loss, op=dist.ReduceOp.SUM)

                    self.avg_valid_loss /= dist.get_world_size()

                    train_batch_loss_detached = train_batch_loss.clone().detach()
                    dist.all_reduce(train_batch_loss_detached, op=dist.ReduceOp.AVG)

                    metrics = {
                        'Train Loss': train_batch_loss_detached.item(),  # Needs to sync (dist.all_reduce)
                        'Val Loss': self.avg_valid_loss.item(),
                        'Val AUC': self.metric_auroc.compute().item(),
                        'Val Precision': self.metric_precision.compute().item(),
                        'Val Recall': self.metric_recall.compute().item(),
                        'Val F1Score': self.metric_f1score.compute().item(),
                    }

                    if self.gpu_id == 0:
                        print(f'Epoch: {epoch}\t' + f'train_batch_step: {train_batch_step} \t' + str(metrics))
                        mlflow.log_metrics(metrics=metrics, step=train_batch_step)
                        # writer.add_scalars(main_tag=run_name, tag_scalar_dict=metrics, global_step=train_batch_step)

                # Unfreeze BERT layers after x steps
                if train_batch_step == unfreeze_bert_after_step:
                    for param in self.ddp_model.module.item_encoder.bert_encoder.parameters():
                        param.requires_grad = True
                    self.optimizer.add_param_group({'params': self.ddp_model.module.item_encoder.bert_encoder.parameters()})
                    print(f'Unfreezed BERT layers on gpu_id#{self.gpu_id}')

                # Save the model on n-th training_steps and stop the training
                if train_batch_step == save_model_stop_at_step:
                    if self.gpu_id == 0:
                        self._save_checkpoint(save_model_stop_at_step, run_name)
                    break  # break loop for train_loader

            print(f'[PASSED] Trainer.train() for epoch#{epoch} on gpu_id#{self.gpu_id}')
            if train_batch_step == save_model_stop_at_step:
                break  # break loop for epochs

        if self.gpu_id == 0:
            exp_tracking_end(writer)
            print('[COMPLETED] Training.train().')


def load_train_objs(model_config: dict, train_config: dict):
    # Load the DatasetDict from S3
    # dataset_dict_s3_path = 's3://tyler-s3-bucket/other/interview-projects/seek/final_dataset_dict/'
    dataset_dict_s3_path = train_config['dataset_dict_s3_path']
    dataset_dict = DatasetDict.load_from_disk(dataset_dict_path=dataset_dict_s3_path)

    model = DualEncoderInterestNetwork(**model_config)

    for param in model.item_encoder.bert_encoder.parameters():
        param.requires_grad = False
    # for param in model.item_encoder.bert_encoder.embeddings.token_type_embeddings.parameters():
    #     param.requires_grad = True

    # optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)  #default0.001
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-6, betas=(0.9, 0.98), eps=1.0e-9, weight_decay=float(train_config['weight_decay']))

    lr_scheduler = NoamLRScheduler(optimizer=optimizer, dim_embed=model_config['d_model'], warmup_steps=train_config['lr_scheduler_warmup_steps'], scale_factor=train_config['lr_scheduler_scale_factor'])

    return dataset_dict, model, optimizer, lr_scheduler


def prepare_dataloaders(dataset_dict: DatasetDict, batch_size: int, tokenizer_checkpoint, tokenizer_model_max_length):
    custom_collate_function = CustomCollateFunc(tokenizer_checkpoint, tokenizer_model_max_length)

    # Setting num_workers as 2 * num of GPUs
    num_workers = torch.cuda.device_count() * 2  # * 4  # 0

    train_dataloader = DataLoader(
        dataset_dict["train"], batch_size=batch_size, collate_fn=custom_collate_function, pin_memory=True,
        num_workers=num_workers,
        # shuffle=False,
        sampler=DistributedSampler(dataset_dict["train"], shuffle=True)
    )
    valid_dataloader = DataLoader(
        dataset_dict["valid"], batch_size=batch_size, collate_fn=custom_collate_function, pin_memory=True,
        num_workers=num_workers,
        # shuffle=False,
        sampler=DistributedSampler(dataset_dict["valid"], shuffle=False, drop_last=True)  # 504114 % (64 * 4) == 50 samples
    )
    # test_dataloader = DataLoader(
    #     dataset_dict["test"], batch_size=batch_size, collate_fn=custom_collate_function, pin_memory=True,
    #     num_workers=num_workers
    # )

    return train_dataloader, valid_dataloader


def main(rank, world_size, train_config, model_config, tokenizer_model_max_length):
    print(f'main() rank#{rank}')

    total_epochs = train_config['total_epochs']
    batch_size = train_config['batch_size']
    validate_every_n_steps = train_config['validate_every_n_steps']
    unfreeze_bert_after_step = train_config['unfreeze_bert_after_step']
    save_model_stop_at_step = train_config['save_model_stop_at_step']

    ddp_setup(rank, world_size)

    # dataset, model, optimizer, etc
    dataset_dict, model, optimizer, lr_scheduler = load_train_objs(model_config, train_config)

    # train_dataloder
    tokenizer_checkpoint = model_config['checkpoint']
    train_dataloader, valid_dataloader = prepare_dataloaders(dataset_dict, batch_size, tokenizer_checkpoint, tokenizer_model_max_length)

    # train for epochs
    trainer = Trainer(model, model_config, train_config, train_dataloader, valid_dataloader, optimizer, lr_scheduler, rank)  # model_config and train_config are used for experiment tracking purpose only in Trainer class
    trainer.train(total_epochs, validate_every_n_steps, unfreeze_bert_after_step, save_model_stop_at_step)

    dist.destroy_process_group()


if __name__ == '__main__':
    print(os.getcwd())

    torch.backends.cuda.matmul.allow_tf32 = True  # Default False

    my_batch_size = 256  # 64  # total (1344086): #batch_size=64: 20000 steps / epoch; batch_size=256: 5078 steps / epoch; batch_size=512 (out of mem): 2540 steps / epoch; batch_size=768: 1750 steps / epoch
    my_world_size = torch.cuda.device_count()
    print(f'torch.cuda.device_count() = {my_world_size}')

    my_train_config = dict()
    my_train_config['is_aws_env'] = True
    if my_train_config['is_aws_env']:
        my_train_config['dataset_dict_s3_path'] = 's3://tyler-s3-bucket/other/interview-projects/seek/final_dataset_dict/'
        my_train_config['mlflow_tracking_uri'] = 'http://127.0.0.1:5000'
        my_train_config["tensorboard_log_dir"] = '/home/ubuntu/tbruns'
    else:
        my_train_config['dataset_dict_s3_path'] = '/opt/project/data/input_processed_output(P1N1)/final_dataset_dict/'  # Mapped path in local Docker container
        my_train_config['mlflow_tracking_uri'] = 'http://192.168.1.245:5001'
        my_train_config["tensorboard_log_dir"] = '/opt/project/tbruns'

    my_train_config['total_epochs'] = 10  # 2  # 5
    my_train_config['batch_size'] = int(my_batch_size / my_world_size)
    my_train_config['validate_every_n_steps'] = 2000  # 1000 (512) #2000 (256)  # 3000  # 5000
    my_train_config['lr_scheduler_warmup_steps'] = 3000  # 3000  # 10000 (3000*0.5 for batchsize256, 1000*0.4 for batchsize512)
    my_train_config['lr_scheduler_scale_factor'] = 0.5  # 0.5  # 0.6  # 1.0
    my_train_config['unfreeze_bert_after_step'] = 10000  # None
    my_train_config['save_model_stop_at_step'] = 26000  # 14000  # None
    my_train_config['weight_decay'] = 1e-6  # 1e-6 valid  # 1e-9  # 1e-4 (NaN loss)  # default 0.
    # 1e-4 weight decay for fully connected block caused 0.693 loss forever; 1e-4 weight decay for inner product lass caused nan validation loss

    my_model_config = dict()
    my_model_config['nuser'] = 19672 + 1  # NUM_USERS, 0th token is the padding token
    my_model_config['nitem'] = 47844 + 1  # NUM_ITEMS, 0th token is the padding token
    my_model_config['d_model'] = 768
    my_model_config['nhead'] = 4  # 8  #12
    my_model_config['d_hid'] = 1024  # 2048  # dim_feedforward
    my_model_config['dropout'] = 0.1  # 0.3  # 0.2  # 0.1
    my_model_config['nlayers'] = 2  # 1  # 3  # 6  #12
    my_model_config['checkpoint'] = 'bert-base-cased'  # 'jjzha/jobbert-base-cased'

    my_model_config['using_cross_attention'] = False  # False # Not used

    bert_tokenizer_model_max_length = 512

    mp.spawn(main, args=(my_world_size, my_train_config, my_model_config, bert_tokenizer_model_max_length), nprocs=my_world_size)
    print('All done.')
