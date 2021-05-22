from enum import unique
from .entities import Token
import json
import os
import warnings
from typing import List, Tuple, Dict

import torch
from sklearn.metrics import precision_recall_fscore_support as prfs
from transformers import BertTokenizer

from identifier import util
from identifier.entities import Document, Dataset, EntityType
from identifier.input_reader import JsonInputReader
import jinja2
import math

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


class Evaluator:
    def __init__(self, dataset: Dataset, input_reader: JsonInputReader, text_encoder: BertTokenizer,no_overlapping: bool,
                 predictions_path: str, examples_path: str, example_count: int, epoch: int, dataset_label: str, nms: float):
        self._text_encoder = text_encoder
        self._input_reader = input_reader
        self._dataset = dataset
        self._no_overlapping = no_overlapping

        self._epoch = epoch
        self._dataset_label = dataset_label

        self._predictions_path = predictions_path

        self._examples_path = examples_path
        self._example_count = example_count

        # entities
        self._gt_entities = []  # ground truth
        self._pred_entities = []  # prediction
        self._raw_preds = []

        self._pseudo_entity_type = EntityType('Entity', 1, 'Entity', 'Entity')  # for span only evaluation
        self._nms = nms
        self._convert_gt(self._dataset.documents)

    def eval_batch(self, batch_entity_clf: torch.tensor,  spn_mask:torch.tensor, offsets1:torch.tensor, offsets2:torch.tensor,  batch: dict, illegal_mask):
        batch_size = batch_entity_clf.shape[0]
        # import pdb; pdb.set_trace()
        # get maximum activation (index of predicted entity type)
        batch_entity_types = batch_entity_clf.argmax(dim=-1)
        # apply entity sample mask
        batch_entity_types *= (batch['entity_sample_masks'].long() * ( spn_mask * illegal_mask).long())
        # print(offsets)
        r_offsets1 = torch.round(offsets1).long()
        # r_offsets2 = torch.round(offsets2).long()
        offsets = r_offsets1
        # print(offsets)
        for i in range(batch_size):
            # get model predictions for sample
            entity_types = batch_entity_types[i]

            # get entities that are not classified as 'None'
            valid_entity_indices = entity_types.nonzero(as_tuple = False).view(-1)
            valid_entity_types = entity_types[valid_entity_indices]

            # old = batch['entity_spans_token'][i][valid_entity_indices]
            # valid_entity_spans = old + torch.round(offsets[i][valid_entity_indices]).long()
            # print(offsets)
            valid_entity_offsets_confidence_1 = 1 - 2*torch.abs(offsets1[i][valid_entity_indices] - r_offsets1[i][valid_entity_indices])
            # valid_entity_offsets_confidence_2 = 0.5 - torch.abs(offsets2[i][valid_entity_indices] - r_offsets2[i][valid_entity_indices])
            valid_entity_offsets_confidence = r_offsets1[i][valid_entity_indices]*10+valid_entity_offsets_confidence_1
            # token_count =  batch['token_masks_bool'][i].long().sum()
            # valid_entity_spans[valid_entity_spans[:,0]>=valid_entity_spans[:,1]]=old[valid_entity_spans[:,0]>=valid_entity_spans[:,1]]
            # valid_entity_spans[:,0][valid_entity_spans[:,0]<0]=old[:,0][valid_entity_spans[:,0]<0]
            # valid_entity_spans[:,1][valid_entity_spans[:,1]>token_count]=old[:,1][valid_entity_spans[:,1]>token_count]
            # valid_entity_scores = torch.gather(batch_entity_clf[i][valid_entity_indices], 1,
            #                                    valid_entity_types.unsqueeze(1)).view(-1)

            
            old = batch['entity_spans_token'][i][valid_entity_indices]
            valid_entity_spans = old + offsets[i][valid_entity_indices].long()

            token_count =  batch['token_masks_bool'][i].long().sum()

            valid_entity_spans[:,0][valid_entity_spans[:,0]<0] = 0
            valid_entity_spans[:,1][valid_entity_spans[:,1]>token_count] = token_count
            valid_entity_spans[valid_entity_spans[:,0]>=valid_entity_spans[:,1]]=(old+r_offsets1[i][valid_entity_indices].long())[valid_entity_spans[:,0]>=valid_entity_spans[:,1]]

            valid_entity_scores = torch.gather(batch_entity_clf[i][valid_entity_indices], 1,
                                               valid_entity_types.unsqueeze(1)).view(-1)
            # valid_entity_scores = 2 * valid_entity_offsets_confidence.sum(-1) + valid_entity_scores
            sample_pred_entities = self._convert_pred_entities(valid_entity_types, valid_entity_spans,
                                                               valid_entity_scores, valid_entity_offsets_confidence)

            # sample_pred_entities = self._remove_partial_overlapping(sample_pred_entities)
            if self._no_overlapping:
                sample_pred_entities = self._remove_overlapping(sample_pred_entities)

            self._pred_entities.append(sample_pred_entities)

    def compute_scores(self):
        print("Evaluation")

        print("")
        print("--- Entities (named entity recognition (NER)) ---")
        print("An entity is considered correct if the entity type and span is predicted correctly")
        print("")
        gt, pred = self._convert_by_setting(self._gt_entities, self._pred_entities, include_entity_types=True)
        ner_eval = self._score(gt, pred, print_results=True)

        return ner_eval

    def store_predictions(self):
        predictions = []

        for i, doc in enumerate(self._dataset.documents):
            tokens = doc.tokens
            pred_entities = self._pred_entities[i]

            # convert entities
            converted_entities = []
            for entity in pred_entities:
                entity_span = entity[:2]
                # print(entity_span, tokens)
                # import pdb; pdb.set_trace()
                span_tokens = util.get_span_tokens(tokens, entity_span)
                entity_type = entity[2].identifier
                converted_entity = dict(type=entity_type, start=span_tokens[0].index, end=span_tokens[-1].index + 1)
                converted_entities.append(converted_entity)
            converted_entities = sorted(converted_entities, key=lambda e: e['start'])

            doc_predictions = dict(tokens=[t.phrase for t in tokens], entities=converted_entities,)
            predictions.append(doc_predictions)

        # store as json
        label, epoch = self._dataset_label, self._epoch
        with open(self._predictions_path % (label, epoch), 'w') as predictions_file:
            json.dump(predictions, predictions_file)
        with open(self._predictions_path % ("valid_all", epoch), 'w') as predictions_file:
            json.dump(self._raw_preds, predictions_file)

        # 
        raw_preds_match_gt = []
        for pre, gt in zip(self._raw_preds, self._gt_entities):
            def is_match(ent):
                for gt_ent in gt:
                    if ent["start"] == gt_ent[0] and  ent["end"] == gt_ent[1] and ent["entity_type"] == gt_ent[2].identifier:
                        return True
                else:
                    return False
            pre_match_gt = list(filter(is_match, pre))
            raw_preds_match_gt.append(pre_match_gt)
        with open(self._predictions_path % ("valid_all_match_gt", epoch), 'w') as predictions_file:
            json.dump(raw_preds_match_gt, predictions_file)

        

    def store_examples(self):
        entity_examples = []

        for i, doc in enumerate(self._dataset.documents):
            # entities
            entity_example = self._convert_example(doc, self._gt_entities[i], self._pred_entities[i],
                                                   include_entity_types=True, to_html=self._entity_to_html)
            entity_examples.append(entity_example)

        label, epoch = self._dataset_label, self._epoch

        # entities
        self._store_examples(entity_examples[:self._example_count],
                             file_path=self._examples_path % ('entities', label, epoch),
                             template='entity_examples.html')

        self._store_examples(sorted(entity_examples[:self._example_count],
                                    key=lambda k: k['length']),
                             file_path=self._examples_path % ('entities_sorted', label, epoch),
                             template='entity_examples.html')

    def _convert_gt(self, docs: List[Document]):
        for doc in docs:
            gt_entities = doc.entities

            # convert ground truth relations and entities for precision/recall/f1 evaluation
            sample_gt_entities = [entity.as_tuple_token() for entity in gt_entities]

            # if self._no_overlapping:
            #     sample_gt_entities = self._remove_overlapping(sample_gt_entities)

            self._gt_entities.append(sample_gt_entities)

    def _convert_pred_entities(self, pred_types: torch.tensor, pred_spans: torch.tensor, pred_scores: torch.tensor, offsets_confidence):
        converted_preds = []
        
        raw_pred = []
        for i in range(pred_types.shape[0]):
            label_idx = pred_types[i].item()
            entity_type = self._input_reader.get_entity_type(label_idx)

            start, end = pred_spans[i].tolist()
            cls_score = pred_scores[i].item()
            left_reg_score, right_reg_score = offsets_confidence[i].tolist()

            converted_pred = (start, end, entity_type, cls_score, left_reg_score, right_reg_score)
            converted_preds.append(converted_pred)
            raw_pred.append({"start": start, "end": end, "entity_type":entity_type.identifier, "cls_score": round(cls_score, 2), "left_reg_score": round(left_reg_score,2), "right_reg_score":round(right_reg_score,2)})
        self._raw_preds.append(raw_pred)
        # print(converted_preds)
        preds = []
        # unique = set()
        # converted_preds = sorted(converted_preds, key = lambda x: x[3], reverse=False)
        # for ind, converted_pred in enumerate(converted_preds):
        #     k = (converted_pred[0], converted_pred[1])
        #     if k not in unique:
        #         preds.append(converted_pred)
        #         unique.add(k)
        #     else:
        #         continue
        unique = set()

        for ind, i in enumerate(converted_preds):
            if (i[0], i[1]) in unique:
                continue
            cls_s = i[3]
            l_s = i[4]
            r_s = i[5]
            for j in converted_preds[ind+1:]:
                
                if i[0]==j[0] and i[1]==j[1] and i[2]==j[2]:
                    cls_s += i[3]
                    l_s = max(l_s, j[4])
                    r_s = max(r_s, j[5])
            # i = (i[0], i[1], i[2], max(i[3], j[3]), max(i[4], j[4]), max(i[5], j[5]))
            i = (i[0], i[1], i[2], cls_s, l_s, r_s)
                    # break
            # if i not in preds:
            #     preds.append(i)
            # preds.append((i[0], i[1], i[2], i[3], math.pow(10, i[4]), math.pow(10, i[5])))
            preds.append(i)
            unique.add((i[0], i[1]))
        # preds = converted_preds
        return self.nms(preds)
        # return converted_preds

    def nms(self, preds):
        # preds = list(filter(lambda x: x[4]>0.9 and x[5]>0.9, preds))
        preds = sorted(preds, key = lambda x: x[3], reverse=True)
        # return preds[:len(preds)//2]
        throw_preds = []
        results = []
        # nms
        # for i, pre in enumerate(preds):
        #     if pre not in throw_preds:
        #         results.append(pre)
        #     start, end, _, score, _, _ = pre
        #     for j in range(i+1,len(preds)):
        #         can_s, can_e, _, _, _, _= preds[j]
        #         if util.iou((start, end), (can_s, can_e)) > self._nms:
        #             throw_preds.append(preds[j])
        # preds = results
        # return list(filter(lambda x: x[4] + x[5 > 1.99], preds))
        # return preds
        # soft nms
        self._nms_decay = 0.9
        self._nms_shohold = 0.6
        
        
        # r = []
        # for pre in preds:
        #     can_s, can_e, ty_e, score_e, l_reg_score,r_reg_score = pre
        #     r.append((can_s, can_e, ty_e, score_e+l_reg_score/10+r_reg_score/10, l_reg_score,r_reg_score))
        # preds =r
        preds = self._remove_partial_overlapping(preds)
        size = len(preds)
        for i, pre in enumerate(preds):
            start, end, _, score, _,_ = pre
            for j in range(i+1,size):
                can_s, can_e, ty_e, score_e, l_reg_score,r_reg_score= preds[j]
                if util.iou((start, end), (can_s, can_e)) > self._nms_shohold:
                    preds[j] = (can_s, can_e, ty_e, score_e* self._nms_decay,l_reg_score,r_reg_score )
                    not_insert = 1
                    for k in range(j+1, size):
                        if score_e > preds[k][3]:
                            not_insert = 0
                            preds.insert(k, preds[j])
                            break
                    if not_insert:
                        preds.append(preds[j])
                    del preds[j]
        
        return list(filter(lambda x: x[3]>self._nms, preds) )
    

    def _remove_overlapping(self, entities):
        non_overlapping_entities = []
        for i, entity in enumerate(entities):
            if not self._is_overlapping(entity, non_overlapping_entities):
                non_overlapping_entities.append(entity)

        return non_overlapping_entities

    def _remove_partial_overlapping(self, entities):
        non_overlapping_entities = []
        for i, entity in enumerate(entities):
            if not self._is_partial_overlapping(entity, entities):
                non_overlapping_entities.append(entity)

        return non_overlapping_entities

    def _is_partial_overlapping(self, e1, entities):
        for e2 in entities:
            if self._check_partial_overlap(e1, e2):
                return True

        return False

    def _is_overlapping(self, e1, entities):
        for e2 in entities:
            if self._check_overlap(e1, e2):
                return True

        return False

    def _check_overlap(self, e1, e2):
        if e1 == e2 or e1[1] <= e2[0] or e2[1] <= e1[0]:
            return False
        else:
            return True
    
    def _check_partial_overlap(self, e1, e2):
        if (e1[0] < e2[0] and e2[0]<e1[1] and e1[1]<e2[1] ) or  (e2[0]<e1[0] and e1[0] < e2[1] and e2[1] < e1[1]):
            return True
        else:
            return False

    def _convert_by_setting(self, gt: List[List[Tuple]], pred: List[List[Tuple]],
                            include_entity_types: bool = True, include_score: bool = False):
        assert len(gt) == len(pred)

        # either include or remove entity types based on setting
        def convert(t):
            if not include_entity_types:
                # remove entity type and score for evaluation
                if type(t[0]) == int:  # entity
                    c = [t[0], t[1], self._pseudo_entity_type]
                else:  # relation
                    c = [(t[0][0], t[0][1], self._pseudo_entity_type),
                         (t[1][0], t[1][1], self._pseudo_entity_type), t[2]]
            else:
                c = list(t[:3])

            if include_score and len(t) > 5:
                # include prediction scores
                c.append(t[3])
                c.append(t[4])
                c.append(t[5])

            return tuple(c)

        converted_gt, converted_pred = [], []

        for sample_gt, sample_pred in zip(gt, pred):
            converted_gt.append([convert(t) for t in sample_gt])
            converted_pred.append([convert(t) for t in sample_pred])

        return converted_gt, converted_pred

    def _score(self, gt: List[List[Tuple]], pred: List[List[Tuple]], print_results: bool = False):
        assert len(gt) == len(pred)

        gt_flat = []
        pred_flat = []
        types = set()

        for (sample_gt, sample_pred) in zip(gt, pred):
            union = set()
            union.update(sample_gt)
            union.update(sample_pred)

            for s in union:
                if s in sample_gt:
                    t = s[2]
                    gt_flat.append(t.index)
                    types.add(t)
                else:
                    gt_flat.append(0)

                if s in sample_pred:
                    t = s[2]
                    pred_flat.append(t.index)
                    types.add(t)
                else:
                    pred_flat.append(0)

        metrics = self._compute_metrics(gt_flat, pred_flat, types, print_results)
        return metrics

    def _compute_metrics(self, gt_all, pred_all, types, print_results: bool = False):
        labels = [t.index for t in types]
        per_type = prfs(gt_all, pred_all, labels=labels, average=None)
        micro = prfs(gt_all, pred_all, labels=labels, average='micro')[:-1]
        macro = prfs(gt_all, pred_all, labels=labels, average='macro')[:-1]
        total_support = sum(per_type[-1])

        if print_results:
            self._print_results(per_type, list(micro) + [total_support], list(macro) + [total_support], types)

        return [m * 100 for m in micro + macro]

    def _print_results(self, per_type: List, micro: List, macro: List, types: List):
        columns = ('type', 'precision', 'recall', 'f1-score', 'support')

        row_fmt = "%20s" + (" %12s" * (len(columns) - 1))
        results = [row_fmt % columns, '\n']

        metrics_per_type = []
        for i, t in enumerate(types):
            metrics = []
            for j in range(len(per_type)):
                metrics.append(per_type[j][i])
            metrics_per_type.append(metrics)

        for m, t in zip(metrics_per_type, types):
            results.append(row_fmt % self._get_row(m, t.short_name))
            results.append('\n')

        results.append('\n')

        # micro
        results.append(row_fmt % self._get_row(micro, 'micro'))
        results.append('\n')

        # macro
        results.append(row_fmt % self._get_row(macro, 'macro'))

        results_str = ''.join(results)
        print(results_str)

    def _get_row(self, data, label):
        row = [label]
        for i in range(len(data) - 1):
            row.append("%.2f" % (data[i] * 100))
        row.append(data[3])
        return tuple(row)

    def _convert_example(self, doc: Document, gt: List[Tuple], pred: List[Tuple],
                         include_entity_types: bool, to_html):
        # encoding = doc.encoding
        tokens = doc.tokens

        gt, pred = self._convert_by_setting([gt], [pred], include_entity_types=include_entity_types, include_score=True)
        gt, pred = gt[0], pred[0]

        # get micro precision/recall/f1 scores
        if gt or pred:
            pred_s = [p[:3] for p in pred]  # remove score
            precision, recall, f1 = self._score([gt], [pred_s])[:3]
        else:
            # corner case: no ground truth and no predictions
            precision, recall, f1 = [100] * 3

        cls_scores = [p[3] for p in pred]
        left_reg_scores = [p[4] for p in pred]
        right_reg_scores = [p[5] for p in pred]
        pred = [p[:3] for p in pred]
        union = set(gt + pred)

        # true positives
        tp = []
        # false negatives
        fn = []
        # false positives
        fp = []

        for s in union:
            type_verbose = s[2].verbose_name

            if s in gt:
                if s in pred:
                    cls_score = cls_scores[pred.index(s)]
                    left_reg_score = left_reg_scores[pred.index(s)]
                    right_reg_score = right_reg_scores[pred.index(s)]
                    tp.append((to_html(s, tokens), type_verbose, cls_score, left_reg_score, right_reg_score))
                else:
                    fn.append((to_html(s, tokens), type_verbose, -1, -1, -1))
            else:
                cls_score = cls_scores[pred.index(s)]
                left_reg_score = left_reg_scores[pred.index(s)]
                right_reg_score = right_reg_scores[pred.index(s)]
                fp.append((to_html(s, tokens), type_verbose, cls_score, left_reg_score, right_reg_score))

        tp = sorted(tp, key=lambda p: p[2], reverse=True)
        fp = sorted(fp, key=lambda p: p[2], reverse=True)

        phrases = []
        for token in tokens:
            phrases.append(token.phrase)
        text = " ".join(phrases)
        


        # text = self._prettify(self._text_encoder.decode(encoding))
        text = self._prettify(text)
        return dict(text=text, tp=tp, fn=fn, fp=fp, precision=precision, recall=recall, f1=f1, length=len(doc.tokens))

    def _entity_to_html(self, entity: Tuple, tokens: List[Token]):
        start, end = entity[:2]
        entity_type = entity[2].verbose_name

        tag_start = ' <span class="entity">'
        tag_start += '<span class="type">%s</span>' % entity_type

        # ctx_before = self._text_encoder.decode(encoding[:start])
        # e1 = self._text_encoder.decode(encoding[start:end])
        # ctx_after = self._text_encoder.decode(encoding[end:])

        ctx_before = ""
        ctx_after = ""
        e1 = ""
        for i in range(start):
            ctx_before += tokens[i].phrase
            if i!=start-1:
                ctx_before += " "
        for i in range(end, len(tokens)):
            ctx_after += tokens[i].phrase
            if i!=(len(tokens)-1):
                ctx_after += " "
        for i in range(start, end):
            e1 += tokens[i].phrase
            if i!=end-1:
                e1 += " "

        html = ctx_before + tag_start + e1 + '</span> ' + ctx_after
        html = self._prettify(html)

        return html

    def _prettify(self, text: str):
        text = text.replace('_start_', '').replace('_classify_', '').replace('<unk>', '').replace('⁇', '')
        text = text.replace('[CLS]', '').replace('[SEP]', '').replace('[PAD]', '')
        return text

    def _store_examples(self, examples: List[Dict], file_path: str, template: str):
        template_path = os.path.join(SCRIPT_PATH, 'templates', template)

        # read template
        with open(os.path.join(SCRIPT_PATH, template_path)) as f:
            template = jinja2.Template(f.read())

        # write to disc
        template.stream(examples=examples).dump(file_path)
