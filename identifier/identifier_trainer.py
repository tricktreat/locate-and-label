import argparse
import math
import os

import torch
from torch.nn import DataParallel
from torch.optim import Optimizer
import transformers
from torch.utils.data import DataLoader
from transformers import AdamW, BertConfig
from transformers import BertTokenizer

from identifier import models
from .focalloss import FocalLoss
from .giouloss import GiouLoss
from identifier import sampling
from identifier import util
from identifier.entities import Dataset
from identifier.evaluator import Evaluator
from identifier.input_reader import JsonInputReader, BaseInputReader
from identifier.loss import IdentifierLoss, Loss
from tqdm import tqdm
from identifier.trainer import BaseTrainer

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


class IdentifierTrainer(BaseTrainer):
    """ Joint entity and relation extraction training and evaluation """

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

        # byte-pair encoding
        self._tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path,
                                                        do_lower_case=args.lowercase,
                                                        cache_dir=args.cache_path)

        # path to export predictions to
        self._predictions_path = os.path.join(self._log_path, 'predictions_%s_epoch_%s.json')

        # path to export relation extraction examples to
        self._examples_path = os.path.join(self._log_path, 'examples_%s_%s_epoch_%s.html')

    def train(self, train_path: str, valid_path: str, types_path: str, input_reader_cls: BaseInputReader):
        args = self.args
        train_label, valid_label = 'train', 'valid'

        self._logger.info("Datasets: %s, %s" % (train_path, valid_path))
        self._logger.info("Model type: %s" % args.model_type)

        # create log csv files
        self._init_train_logging(train_label)
        self._init_eval_logging(valid_label)

        # read datasets
        input_reader = input_reader_cls(types_path, self._tokenizer, args.iou_spn, args.iou_classifier, args.neg_entity_count,
                                        args.neg_relation_count, args.window_size, self._logger, wordvec_filename = args.wordvec_path)
        input_reader.read({train_label: train_path, valid_label: valid_path})
        self._log_datasets(input_reader)

        train_dataset = input_reader.get_dataset(train_label)
        train_sample_count = train_dataset.document_count
        updates_epoch = train_sample_count // args.train_batch_size
        updates_total = updates_epoch * args.epochs

        validation_dataset = input_reader.get_dataset(valid_label)

        self._logger.info("Updates per epoch: %s" % updates_epoch)
        self._logger.info("Updates total: %s" % updates_total)

        # create model
        model_class = models.get_model(self.args.model_type)

        # load model
        config = BertConfig.from_pretrained(self.args.model_path, cache_dir=self.args.cache_path)
        embed = torch.from_numpy(input_reader.embedding_weight).float()
        model = model_class.from_pretrained(self.args.model_path,
                                            config = config,
                                            embed = embed,
                                            # Identifier model parameters
                                            cls_token=self._tokenizer.convert_tokens_to_ids('[CLS]'),
                                            entity_types=input_reader.entity_type_count,
                                            prop_drop=self.args.prop_drop,
                                            size_embedding=self.args.size_embedding,
                                            freeze_transformer=self.args.freeze_transformer,
                                            lstm_layers = self.args.lstm_layers,
                                            lstm_drop = self.args.lstm_drop, 
                                            pos_size = self.args.pos_size,
                                            char_lstm_layers = self.args.char_lstm_layers, 
                                            char_lstm_drop = self.args.char_lstm_drop, 
                                            char_size = self.args.char_size, 
                                            use_glove = self.args.use_glove, 
                                            use_pos = self.args.use_pos, 
                                            use_char_lstm = self.args.use_char_lstm, 
                                            spn_filter = self.args.spn_filter,
                                            pool_type = self.args.pool_type,
                                            use_entity_ctx = self.args.use_entity_ctx,
                                            use_size_embedding = self.args.use_size_embedding,
                                            reduce_dim = self.args.reduce_dim,
                                            bert_before_lstm = self.args.bert_before_lstm,
                                            no_filter = self.args.no_filter,
                                            no_regressor = self.args.no_regressor,
                                            norm = self.args.norm,
                                            no_times_count = self.args.no_times_count)
        
        
        # Identifier is currently optimized on a single GPU and not thoroughly tested in a multi GPU setup
        # If you still want to train Identifier on multiple GPUs, uncomment the following lines
        # # parallelize model
        # if self._device.type != 'cpu':
        #     model = torch.nn.DataParallel(model)

        model.to(self._device)

        # create optimizer
        optimizer_params = self._get_optimizer_params(model)
        optimizer = AdamW(optimizer_params, lr=args.lr, weight_decay=args.weight_decay, correct_bias=False)
        # create scheduler
        # scheduler = transformers.get_constant_schedule_with_warmup(optimizer,
        #                                                     num_warmup_steps=args.lr_warmup * updates_total)
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                                 num_warmup_steps=args.lr_warmup * updates_total,
                                                                 num_training_steps=updates_total)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,patience=3,threshold=0.0001, threshold_mode='rel', verbose=True)
        self.scheduler = scheduler
        # create loss function
        # entity_criterion = torch.nn.CrossEntropyLoss(reduction='none')

        entity_criterion = None
        filter_criterion = None
        offset_criterion = None
        giou_criterion = None

        if self.args.giou_loss =="giouloss":
            giou_criterion = GiouLoss(t="giou", reduction = "none")
        elif self.args.giou_loss =="iouloss":
            giou_criterion = GiouLoss(t="iou", reduction = "none")
        
        if self.args.offset_loss =="l2loss":
            offset_criterion = torch.nn.MSELoss(reduction ='none')
        elif self.args.offset_loss =="l1loss":
            offset_criterion = torch.nn.L1Loss(reduction ='none')
        elif self.args.offset_loss =="smoothl1loss":
            offset_criterion = torch.nn.SmoothL1Loss(reduction ='none')

        if self.args.filter_loss =="celoss":
            filter_criterion = torch.nn.CrossEntropyLoss(reduction='none')
        elif self.args.filter_loss =="focalloss":
            filter_criterion = FocalLoss(class_num = 2, reduction='none', gamma = self.args.filter_gamma)
        
        if self.args.entity_loss =="celoss":
            entity_criterion = torch.nn.CrossEntropyLoss(reduction='none')
        elif self.args.entity_loss =="focalloss":
            entity_criterion = FocalLoss(class_num = input_reader.entity_type_count, reduction='none', gamma=self.args.entity_gamma)

        compute_loss = IdentifierLoss(filter_criterion, offset_criterion, giou_criterion, entity_criterion, model, optimizer, scheduler,  args.iou_weight, args.max_grad_norm, args.iou_classifier, filter_weight=args.filter_loss_weight, offset_weight=args.offset_loss_weight, giou_weight=args.giou_loss_weight, entity_weight=args.entity_loss_weight, iou_gamma = args.iou_gamma)

        # eval validation set
        if args.init_eval:
            self._eval(model, validation_dataset, input_reader, 0, updates_epoch)

        # train
        best_f1 = 0
        best_epoch = 0
        for epoch in range(args.epochs):
            # train epoch
            self._train_epoch(model, compute_loss, optimizer, train_dataset, updates_epoch, epoch)

            # eval validation sets
            if not args.final_eval or (epoch == args.epochs - 1):
                f1 = self._eval(model, validation_dataset, input_reader, epoch + 1, updates_epoch)
                if best_f1 < f1[2]:
                    print(f"Best F1 score update, from {best_f1} to {f1[2]}")
                    best_f1 = f1[2]
                    best_epoch = epoch + 1
                    extra = dict(epoch=epoch, updates_epoch=updates_epoch, epoch_iteration=0)
                    self._save_model(self._save_path, model, self._tokenizer, epoch * updates_epoch,
                         optimizer=optimizer if self.args.save_optimizer else None, extra=extra,
                         include_iteration=False, name='best_model')
            print(f"Best F1 score: {best_f1}, achieved at Epoch: {best_epoch}")

        # save final model
        extra = dict(epoch=args.epochs, updates_epoch=updates_epoch, epoch_iteration=0)
        global_iteration = args.epochs * updates_epoch
        self._save_model(self._save_path, model, self._tokenizer, global_iteration,
                         optimizer=optimizer if self.args.save_optimizer else None, extra=extra,
                         include_iteration=False, name='final_model')

        self._logger.info("Logged in: %s" % self._log_path)
        self._logger.info("Saved in: %s" % self._save_path)
        self._close_summary_writer()

    def eval(self, dataset_path: str, types_path: str, input_reader_cls: BaseInputReader):
        args = self.args
        dataset_label = 'test'

        self._logger.info("Dataset: %s" % dataset_path)
        self._logger.info("Model: %s" % args.model_type)

        # create log csv files
        self._init_eval_logging(dataset_label)

        # read datasets
        input_reader = input_reader_cls(types_path, self._tokenizer,
                                        window_size=args.window_size, logger=self._logger, wordvec_filename = args.wordvec_path)
        input_reader.read({dataset_label: dataset_path})
        self._log_datasets(input_reader)

        # create model
        model_class = models.get_model(self.args.model_type)

        config = BertConfig.from_pretrained(self.args.model_path, cache_dir=self.args.cache_path)
        embed = torch.from_numpy(input_reader.embedding_weight).float()
        model = model_class.from_pretrained(self.args.model_path,
                                            config=config,
                                            embed = embed,
                                            # Identifier model parameters
                                            cls_token=self._tokenizer.convert_tokens_to_ids('[CLS]'),
                                            entity_types=input_reader.entity_type_count,
                                            prop_drop=self.args.prop_drop,
                                            size_embedding=self.args.size_embedding,
                                            freeze_transformer=self.args.freeze_transformer,
                                            lstm_layers = self.args.lstm_layers,
                                            lstm_drop = self.args.lstm_drop, 
                                            pos_size = self.args.pos_size,
                                            char_lstm_layers = self.args.char_lstm_layers, 
                                            char_lstm_drop = self.args.char_lstm_drop, 
                                            char_size = self.args.char_size, 
                                            use_glove = self.args.use_glove, 
                                            use_pos = self.args.use_pos, 
                                            use_char_lstm = self.args.use_char_lstm, 
                                            spn_filter = self.args.spn_filter,
                                            pool_type = self.args.pool_type,
                                            use_entity_ctx = self.args.use_entity_ctx,
                                            use_size_embedding = self.args.use_size_embedding,
                                            reduce_dim = self.args.reduce_dim,
                                            bert_before_lstm = self.args.bert_before_lstm,
                                            no_filter = self.args.no_filter,
                                            no_regressor = self.args.no_regressor,
                                            norm = self.args.norm,
                                            no_times_count = self.args.no_times_count)

        model.to(self._device)

        # evaluate
        self._eval(model, input_reader.get_dataset(dataset_label), input_reader)

        self._logger.info("Logged in: %s" % self._log_path)
        self._close_summary_writer()

    def _train_epoch(self, model: torch.nn.Module, compute_loss: Loss, optimizer: Optimizer, dataset: Dataset,
                     updates_epoch: int, epoch: int):
        self._logger.info("Train epoch: %s" % epoch)

        # create data loader
        dataset.switch_mode(Dataset.TRAIN_MODE)
        data_loader = DataLoader(dataset, batch_size=self.args.train_batch_size, shuffle=True, drop_last=True,
                                 num_workers=self.args.sampling_processes, collate_fn=sampling.collate_fn_padding)

        model.zero_grad()

        iteration = 0
        total = dataset.document_count // self.args.train_batch_size
        for batch in tqdm(data_loader, total=total, desc='Train epoch %s' % epoch):
            model.train()
            batch = util.to_device(batch, self._device)

            # forward step
            entity_logits, offsets2, bin_clf, offsets1, spn_mask, illegal_mask = model(encodings=batch['encodings'], context_masks=batch['context_masks'],
                                              entity_masks=batch['entity_masks'], token_masks_bool=batch['token_masks_bool'], entity_sizes=batch['entity_sizes'],
                                              token_masks=batch['token_masks'], entity_masks_token=batch['entity_masks_token'], entity_spans_token=batch['entity_spans_token'], pos_encoding = batch['pos_encoding'], wordvec_encoding = batch['wordvec_encoding'], char_encoding = batch['char_encoding'], token_masks_char = batch['token_masks_char'], char_count = batch['char_count'])

            # compute loss and optimize parameters
            batch_loss = compute_loss.compute(entity_logits=entity_logits,  offsets1_pred=offsets1, offsets2_pred=offsets2, bin_logits=bin_clf,spn_mask=spn_mask,
                                              entity_types=batch['entity_types'],entity_types_1=batch['entity_types_1'], l_offsets_gt = batch['l_offsets'], r_offsets_gt = batch['r_offsets'],
                                              ious=batch['ious'], entity_sample_masks=batch['entity_sample_masks'], offset_sample_masks=batch['offset_sample_masks'], entity_spans_token = batch['entity_spans_token'], illegal_mask=illegal_mask)

            # logging
            iteration += 1
            global_iteration = epoch * updates_epoch + iteration

            if global_iteration % self.args.train_log_iter == 0:
                self._log_train(optimizer, batch_loss, epoch, iteration, global_iteration, dataset.label)

        return iteration

    def _eval(self, model: torch.nn.Module, dataset: Dataset, input_reader: JsonInputReader,
              epoch: int = 0, updates_epoch: int = 0, iteration: int = 0):
        self._logger.info("Evaluate: %s" % dataset.label)

        if isinstance(model, DataParallel):
            # currently no multi GPU support during evaluation
            model = model.module

        # create evaluator
        evaluator = Evaluator(dataset, input_reader, self._tokenizer, self.args.no_overlapping, self._predictions_path,
                              self._examples_path, self.args.example_count, epoch, dataset.label, nms = self.args.nms)

        # create data loader
        dataset.switch_mode(Dataset.EVAL_MODE)
        data_loader = DataLoader(dataset, batch_size=self.args.eval_batch_size, shuffle=False, drop_last=False,
                                 num_workers=self.args.sampling_processes, collate_fn=sampling.collate_fn_padding)

        with torch.no_grad():
            model.eval()

            # iterate batches
            total = math.ceil(dataset.document_count / self.args.eval_batch_size)
            for batch in tqdm(data_loader, total=total, desc='Evaluate epoch %s' % epoch):
                # move batch to selected device
                batch = util.to_device(batch, self._device)

                # run model (forward pass)
                result = model(encodings=batch['encodings'], context_masks=batch['context_masks'],
                               entity_masks=batch['entity_masks'], entity_sizes=batch['entity_sizes'], token_masks_bool=batch['token_masks_bool'],
                               entity_spans=batch['entity_spans'], entity_sample_masks=batch['entity_sample_masks'],
                               token_masks=batch['token_masks'], entity_masks_token=batch['entity_masks_token'], entity_spans_token =batch['entity_spans_token'], pos_encoding = batch['pos_encoding'], wordvec_encoding = batch['wordvec_encoding'], char_encoding = batch['char_encoding'], token_masks_char = batch['token_masks_char'], char_count = batch['char_count'],  evaluate=True)

                entity_clf, spn_mask, offsets1, offsets2, illegal_mask = result

                # evaluate batch
                evaluator.eval_batch(entity_clf, spn_mask,  offsets1, offsets2, batch, illegal_mask)
            print(f"Enmuberated Spans: {sampling.get_v('enumerate_count')}")
            sampling.set_v('enumerate_count', 0)
        global_iteration = epoch * updates_epoch + iteration
        ner_eval = evaluator.compute_scores()
        self._log_eval(*ner_eval, epoch, iteration, global_iteration, dataset.label)

        # self.scheduler.step(ner_eval[2])

        if self.args.store_predictions and not self.args.no_overlapping:
            evaluator.store_predictions()

        if self.args.store_examples:
            evaluator.store_examples()
        
        return ner_eval

    def _get_optimizer_params(self, model):
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        # regressier
        optimizer_params = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},]

        return optimizer_params

    def _log_train(self, optimizer: Optimizer, loss: float, epoch: int,
                   iteration: int, global_iteration: int, label: str):
        # average loss
        avg_loss = loss / self.args.train_batch_size
        # get current learning rate
        lr = self._get_lr(optimizer)[0]

        # log to tensorboard
        self._log_tensorboard(label, 'loss', loss, global_iteration)
        self._log_tensorboard(label, 'loss_avg', avg_loss, global_iteration)
        self._log_tensorboard(label, 'lr', lr, global_iteration)

        # log to csv
        self._log_csv(label, 'loss', loss, epoch, iteration, global_iteration)
        self._log_csv(label, 'loss_avg', avg_loss, epoch, iteration, global_iteration)
        self._log_csv(label, 'lr', lr, epoch, iteration, global_iteration)

    def _log_eval(self, ner_prec_micro: float, ner_rec_micro: float, ner_f1_micro: float,
                  ner_prec_macro: float, ner_rec_macro: float, ner_f1_macro: float,
                  epoch: int, iteration: int, global_iteration: int, label: str):

        # log to tensorboard
        self._log_tensorboard(label, 'eval/ner_prec_micro', ner_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_recall_micro', ner_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_f1_micro', ner_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_prec_macro', ner_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_recall_macro', ner_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_f1_macro', ner_f1_macro, global_iteration)

        # log to csv
        self._log_csv(label, 'eval', ner_prec_micro, ner_rec_micro, ner_f1_micro,
                      ner_prec_macro, ner_rec_macro, ner_f1_macro,
                      epoch, iteration, global_iteration)

    def _log_datasets(self, input_reader):
        self._logger.info("Relation type count: %s" % input_reader.relation_type_count)
        self._logger.info("Entity type count: %s" % input_reader.entity_type_count)

        self._logger.info("Entities:")
        for e in input_reader.entity_types.values():
            self._logger.info(e.verbose_name + '=' + str(e.index))

        self._logger.info("Relations:")
        for r in input_reader.relation_types.values():
            self._logger.info(r.verbose_name + '=' + str(r.index))

        for k, d in input_reader.datasets.items():
            self._logger.info('Dataset: %s' % k)
            self._logger.info("Document count: %s" % d.document_count)
            self._logger.info("Relation count: %s" % d.relation_count)
            self._logger.info("Entity count: %s" % d.entity_count)

        self._logger.info("Context size: %s" % input_reader.context_size)

    def _init_train_logging(self, label):
        self._add_dataset_logging(label,
                                  data={'lr': ['lr', 'epoch', 'iteration', 'global_iteration'],
                                        'loss': ['loss', 'epoch', 'iteration', 'global_iteration'],
                                        'loss_avg': ['loss_avg', 'epoch', 'iteration', 'global_iteration']})

    def _init_eval_logging(self, label):
        self._add_dataset_logging(label,
                                  data={'eval': ['ner_prec_micro', 'ner_rec_micro', 'ner_f1_micro',
                                                 'ner_prec_macro', 'ner_rec_macro', 'ner_f1_macro',
                                                 'epoch', 'iteration', 'global_iteration']})
