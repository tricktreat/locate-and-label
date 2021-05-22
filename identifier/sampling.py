import pdb
import random

import torch
from torch._C import dtype

from identifier import util

v_map = {}
v_map["enumerate_count"] = 0

def get_v(k = "enumerate_count"):
    global v_map
    return v_map[k]

def set_v(k = "enumerate_count", v = 1):
    global v_map
    v_map["enumerate_count"] = v

def create_train_sample(doc, neg_entity_count: int, neg_rel_count: int, window_sizes: list, rel_type_count: int, iou_spn:float, iou_classifier:float):
    pos_encoding = [t.pos_id for t in doc.tokens]
    wordvec_encoding = [t.wordinx for t in doc.tokens]
    encodings = doc.encoding
    token_count = len(doc.tokens)
    context_size = len(encodings)

    char_encodings = doc.char_encoding
    char_encoding = []
    char_count = []
    for char_encoding_token in char_encodings:
        char_count.append(len(char_encoding_token))
        char_encoding.append(torch.tensor(char_encoding_token,dtype=torch.long))
    char_encoding = util.padded_stack(char_encoding)
    token_masks_char = (char_encoding!=0).long()
    char_count = torch.tensor(char_count, dtype = torch.long)

    # import pdb; pdb.set_trace()

    # token_masks_char = []
    

    # all tokens
    token_spans, token_masks, token_sizes, token_pos = [], [], [], []
    for t in doc.tokens:
        token_spans.append(t.span)
        # token_pos.append(t.pos)
        token_masks.append(create_entity_mask(*t.span, context_size))
        # token_masks_char.append(create_entity_mask(*t.char_span, char_count))
        token_sizes.append(t.span_end - t.span_start)

    token_sample_mask = torch.ones([len(token_spans)], dtype=torch.bool) 

    gt_entities_spans_token = []
    gt_entities_spans = []
    gt_entity_types = []
    for e in doc.entities:
        gt_entities_spans_token.append(e.span_token)
        gt_entities_spans.append(e.span)
        gt_entity_types.append(e.entity_type.index)

    # positive relations
    pos_rels, pos_rel_spans, pos_rel_types, pos_rel_masks = [], [], [], []

    pos_ious, pos_l_offsets, pos_r_offsets, pos_offset_sample_masks = [], [], [], []
    neg_ious, neg_l_offsets, neg_r_offsets, neg_offset_sample_masks = [], [], [], []

    pos_entity_spans, pos_entity_types, pos_entity_masks, pos_entity_sizes = [], [], [], []
    pos_entity_spans_token, pos_entity_masks_token = [], []

    neg_entity_spans, neg_entity_types, neg_entity_masks, neg_entity_sizes = [], [], [], []
    neg_entity_spans_token, neg_entity_masks_token = [], []

    # torch.set_printoptions(profile="full")
    if len(window_sizes) == 1:
        window_sizes = range(window_sizes[0])
    for window_size in window_sizes:
        for i in range(0, token_count):
            # w_left = max(0, i - window_size)
            w_left = i
            w_right = min(token_count, i + window_size + 1) 
            span = doc.tokens[w_left:w_right].span
            span_token =  doc.tokens[w_left:w_right].span_token
            if span_token not in pos_entity_spans_token and span_token not in neg_entity_spans_token:
                flag_max_iou = 0
                ty, left, right = 0, 0, 0

                for i, gt_entities_span_token in enumerate(gt_entities_spans_token):
                    iou = util.iou(span_token, gt_entities_span_token)
                    if iou > flag_max_iou:
                        flag_max_iou = iou
                        ty = gt_entity_types[i]
                        left = gt_entities_span_token[0] - span_token[0]
                        right = gt_entities_span_token[1] - span_token[1]
                
                if flag_max_iou > iou_spn:
                    pos_ious.append(flag_max_iou)
                    pos_entity_types.append(ty)
                    pos_entity_spans.append(span)
                    pos_entity_spans_token.append(span_token)
                    pos_entity_sizes.append(w_right-w_left)
                    pos_l_offsets.append(left)
                    pos_r_offsets.append(right)
                    pos_offset_sample_masks.append(1)
                else:
                    neg_ious.append(1 - flag_max_iou)
                    neg_entity_types.append(0)
                    neg_entity_spans.append(span)
                    neg_entity_spans_token.append(span_token)
                    neg_entity_sizes.append(w_right-w_left)
                    neg_l_offsets.append(0)
                    neg_r_offsets.append(0)
                    neg_offset_sample_masks.append(0)

                # if flag_max_iou > iou_classifier:

    for i, gt_entities_span_token in enumerate(gt_entities_spans_token):
        if gt_entities_span_token not in pos_entity_spans_token:
            pos_entity_spans_token.append(gt_entities_span_token)
            pos_entity_sizes.append(gt_entities_span_token[1]-gt_entities_span_token[0])
            pos_l_offsets.append(0)
            pos_r_offsets.append(0)
            pos_entity_spans.append(gt_entities_spans[i])
            pos_ious.append(1)
            pos_offset_sample_masks.append(1)
            pos_entity_types.append(gt_entity_types[i])

    # sample negative entities
    neg_entity_count = neg_entity_count * len(pos_entity_spans_token) + 2
    neg_entity_samples = random.sample(list(zip(neg_ious, neg_entity_types, neg_entity_spans, neg_entity_spans_token, neg_entity_sizes, neg_l_offsets, neg_r_offsets, neg_offset_sample_masks)),min(len(neg_entity_spans), neg_entity_count))
    # print(neg_entity_samples)
    neg_ious, neg_entity_types, neg_entity_spans, neg_entity_spans_token, neg_entity_sizes, neg_l_offsets, neg_r_offsets, neg_offset_sample_masks = map(list, zip(*neg_entity_samples) if neg_entity_samples else ([], [], [], [], [], [], [], []))

    pos_entity_masks_token = [create_entity_mask(*span, token_count) for span in pos_entity_spans_token]
    pos_entity_masks = [create_entity_mask(*span, context_size) for span in pos_entity_spans]
    neg_entity_masks_token = [create_entity_mask(*span, token_count) for span in neg_entity_spans_token]
    neg_entity_masks = [create_entity_mask(*span, context_size) for span in neg_entity_spans]
    # neg_entity_types = [0] * len(neg_entity_spans)
    # merge
    ious = pos_ious + neg_ious
    entity_types_1 = pos_entity_types + neg_entity_types
    entity_types = pos_entity_types + neg_entity_types
    entity_masks = pos_entity_masks + neg_entity_masks
    entity_sizes = pos_entity_sizes + neg_entity_sizes
    l_offsets = pos_l_offsets + neg_l_offsets
    r_offsets = pos_r_offsets + neg_r_offsets
    offset_sample_masks = pos_offset_sample_masks + neg_offset_sample_masks
    entity_masks_token = pos_entity_masks_token + neg_entity_masks_token
    entity_spans_token = pos_entity_spans_token + neg_entity_spans_token

    # negative relations
    # use only strong negative relations, i.e. pairs of actual (labeled) entities that are not related
    neg_rel_spans = []

    for i1, s1 in enumerate(pos_entity_spans):
        for i2, s2 in enumerate(pos_entity_spans):
            rev = (s2, s1)
            rev_symmetric = rev in pos_rel_spans and pos_rel_types[pos_rel_spans.index(rev)].symmetric

            # do not add as negative relation sample:
            # neg. relations from an entity to itself
            # entity pairs that are related according to gt
            # entity pairs whose reverse exists as a symmetric relation in gt
            if s1 != s2 and (s1, s2) not in pos_rel_spans and not rev_symmetric:
                neg_rel_spans.append((s1, s2))

    # sample negative relations
    neg_rel_spans = random.sample(neg_rel_spans, min(len(neg_rel_spans), neg_rel_count))

    neg_rels = [(pos_entity_spans.index(s1), pos_entity_spans.index(s2)) for s1, s2 in neg_rel_spans]
    neg_rel_masks = [create_rel_mask(*spans, context_size) for spans in neg_rel_spans]
    neg_rel_types = [0] * len(neg_rel_spans)



    rels = pos_rels + neg_rels
    rel_types = [r.index for r in pos_rel_types] + neg_rel_types
    rel_masks = pos_rel_masks + neg_rel_masks

    assert len(entity_masks) == len(entity_sizes) == len(entity_types) == len(entity_masks_token)
    assert len(rels) == len(rel_masks) == len(rel_types)

    # create tensors
    # token indices
    encodings = torch.tensor(encodings, dtype=torch.long)
    # char_encoding = torch.tensor(char_encoding, dtype=torch.long)
    pos_encoding = torch.tensor(pos_encoding, dtype=torch.long)
    wordvec_encoding = torch.tensor(wordvec_encoding, dtype=torch.long)

    # masking of tokens
    context_masks = torch.ones(context_size, dtype=torch.bool)
    token_masks_bool = torch.ones(token_count, dtype=torch.bool)
    # char_count = torch.ones(char_count, dtype=torch.bool)

    # also create samples_masks:
    # tensors to mask entity/relation samples of batch
    # since samples are stacked into batches, "padding" entities/relations possibly must be created
    # these are later masked during loss computation
    token_masks = torch.stack(token_masks)
    # token_masks_char = torch.stack(token_masks_char)

    if entity_masks:
        entity_types = torch.tensor(entity_types, dtype=torch.long)
        entity_types_1 = torch.tensor(entity_types_1, dtype=torch.long)
        entity_masks = torch.stack(entity_masks)
        entity_sizes = torch.tensor(entity_sizes, dtype=torch.long)
        entity_sample_masks = torch.ones([entity_masks.shape[0]], dtype=torch.bool)

        entity_masks_token = torch.stack(entity_masks_token)
        entity_spans_token = torch.tensor(entity_spans_token)

        l_offsets = torch.tensor(l_offsets, dtype=torch.float)
        r_offsets = torch.tensor(r_offsets, dtype=torch.float)
        offset_sample_masks = torch.tensor(offset_sample_masks, dtype=torch.bool)
        ious = torch.tensor(ious, dtype=torch.float)
    else:
        # corner case handling (no pos/neg entities)
        entity_types = torch.zeros([1], dtype=torch.long)
        entity_types_1 = torch.tensor(entity_types_1, dtype=torch.long)
        entity_masks = torch.zeros([1, context_size], dtype=torch.bool)
        entity_sizes = torch.zeros([1], dtype=torch.long)
        entity_sample_masks = torch.zeros([1], dtype=torch.bool)

        entity_masks_token = torch.zeros([1, token_count], dtype=torch.bool)
        entity_spans_token = torch.zeros([1, 2], dtype=torch.long)

        l_offsets = torch.tensor([0], dtype=torch.float)
        r_offsets = torch.tensor([0], dtype=torch.float)
        offset_sample_masks = torch.tensor([0], dtype=torch.bool)
        ious = torch.tensor([0], dtype=torch.float)

    if rels:
        rels = torch.tensor(rels, dtype=torch.long)
        rel_masks = torch.stack(rel_masks)
        rel_types = torch.tensor(rel_types, dtype=torch.long)
        rel_sample_masks = torch.ones([rels.shape[0]], dtype=torch.bool)
    else:
        # corner case handling (no pos/neg relations)
        rels = torch.zeros([1, 2], dtype=torch.long)
        rel_types = torch.zeros([1], dtype=torch.long)
        rel_masks = torch.zeros([1, context_size], dtype=torch.bool)
        rel_sample_masks = torch.zeros([1], dtype=torch.bool)

    # relation types to one-hot encoding
    rel_types_onehot = torch.zeros([rel_types.shape[0], rel_type_count], dtype=torch.float32)
    rel_types_onehot.scatter_(1, rel_types.unsqueeze(1), 1)
    rel_types_onehot = rel_types_onehot[:, 1:]  # all zeros for 'none' relation
    
    # import pdb; pdb.set_trace()
    return dict(encodings=encodings, context_masks=context_masks, token_masks_bool=token_masks_bool, token_masks=token_masks, 
                entity_masks =entity_masks, entity_masks_token = entity_masks_token, entity_spans_token = entity_spans_token,
                entity_sizes=entity_sizes, entity_types=entity_types, entity_types_1=entity_types_1,
                rels=rels, rel_masks=rel_masks, rel_types=rel_types_onehot,
                token_sample_masks=token_sample_mask, entity_sample_masks=entity_sample_masks, rel_sample_masks=rel_sample_masks,
                l_offsets=l_offsets,r_offsets=r_offsets,offset_sample_masks=offset_sample_masks,ious=ious, pos_encoding = pos_encoding, wordvec_encoding = wordvec_encoding, char_encoding = char_encoding, token_masks_char = token_masks_char, char_count = char_count)


def create_eval_sample(doc, window_sizes: int):
    # global v_map
    pos_encoding = [t.pos_id for t in doc.tokens]
    wordvec_encoding = [t.wordinx for t in doc.tokens]
    encodings = doc.encoding
    token_count = len(doc.tokens)
    context_size = len(encodings)
    # char_encoding = doc.char_encoding
    # char_count = len(char_encoding)

    char_encodings = doc.char_encoding
    char_encoding = []
    char_count = []
    for char_encoding_token in char_encodings:
        char_count.append(len(char_encoding_token))
        char_encoding.append(torch.tensor(char_encoding_token,dtype=torch.long))
    char_encoding = util.padded_stack(char_encoding)
    token_masks_char = (char_encoding!=0).long()
    char_count = torch.tensor(char_count, dtype = torch.long)


    # all tokens
    # token_masks_char = []
    token_spans, token_masks, token_sizes, token_pos = [], [], [], []
    for t in doc.tokens:
        token_spans.append(t.span)
        # token_pos.append(t.pos)
        token_masks.append(create_entity_mask(*t.span, context_size))
        # token_masks_char.append(create_entity_mask(*t.char_span, char_count))
        token_sizes.append(t.span_end-t.span_start)

    token_sample_mask = torch.ones([len(token_spans)], dtype=torch.bool)
    token_masks = torch.stack(token_masks)
    # token_masks_char = torch.stack(token_masks_char)

    # create entity candidates
    entity_spans = []
    entity_masks = []
    entity_sizes = []
    entity_spans_token = []
    entity_masks_token = []
    window_sizes = range(window_sizes[-1]+1)
    for window_size in window_sizes:
        for i in range(0, token_count):
            # w_left = max(0, i - window_size)
            w_left = i
            w_right = min(token_count, i + window_size + 1) 
            span = doc.tokens[w_left:w_right].span
            span_token =  doc.tokens[w_left:w_right].span_token
            if span not in entity_spans:
                entity_spans.append(span)
                entity_spans_token.append(span_token)
                entity_masks.append(create_entity_mask(*span, context_size))
                entity_masks_token.append(create_entity_mask(*span_token, token_count))
                # entity_spans_token.append(span_token)
                entity_sizes.append(w_right-w_left)
    # v_map["enumerate_count"] += len(entity_spans)
    set_v("enumerate_count", get_v("enumerate_count") + len(entity_spans))
    encodings = torch.tensor(encodings, dtype=torch.long)
    # char_encoding = torch.tensor(char_encoding, dtype=torch.long)
    pos_encoding = torch.tensor(pos_encoding, dtype=torch.long)
    wordvec_encoding = torch.tensor(wordvec_encoding, dtype=torch.long)
    
    context_masks = torch.ones(context_size, dtype=torch.bool)
    token_masks_bool = torch.ones(token_count, dtype=torch.bool)
    # char_masks_bool = torch.ones(char_count, dtype=torch.bool)
    

    # entities
    if entity_masks:
        entity_masks = torch.stack(entity_masks)
        entity_masks_token = torch.stack(entity_masks_token)
        entity_sizes = torch.tensor(entity_sizes, dtype=torch.long)
        entity_spans = torch.tensor(entity_spans, dtype=torch.long)
        entity_spans_token = torch.tensor(entity_spans_token, dtype=torch.long)

        # tensors to mask entity samples of batch
        # since samples are stacked into batches, "padding" entities possibly must be created
        # these are later masked during evaluation
        entity_sample_masks = torch.tensor([1] * entity_masks.shape[0], dtype=torch.bool)
    else:
        # corner case handling (no entities)
        entity_masks = torch.zeros([1, context_size], dtype=torch.bool)
        entity_sizes = torch.zeros([1], dtype=torch.long)
        entity_spans = torch.zeros([1, 2], dtype=torch.long)
        entity_spans_token = torch.zeros([1, 2], dtype=torch.long)
        entity_sample_masks = torch.zeros([1], dtype=torch.bool)
        entity_masks_token = torch.zeros([1, token_count], dtype=torch.bool)
    # import pdb; pdb.set_trace()
    # print(enumerate_count)
    return dict(encodings=encodings, context_masks=context_masks, token_masks_bool=token_masks_bool,token_masks=token_masks, 
                entity_masks=entity_masks, entity_masks_token = entity_masks_token,
                entity_sizes=entity_sizes, entity_spans=entity_spans, entity_spans_token=entity_spans_token, entity_sample_masks=entity_sample_masks, pos_encoding = pos_encoding, wordvec_encoding = wordvec_encoding, char_encoding = char_encoding, token_masks_char = token_masks_char, char_count = char_count)


def create_entity_mask(start, end, context_size):
    mask = torch.zeros(context_size, dtype=torch.bool)
    mask[start:end] = 1
    return mask


def create_rel_mask(s1, s2, context_size):
    start = s1[1] if s1[1] < s2[0] else s2[1]
    end = s2[0] if s1[1] < s2[0] else s1[0]
    mask = create_entity_mask(start, end, context_size)
    return mask


def collate_fn_padding(batch):
    padded_batch = dict()
    keys = batch[0].keys()

    for key in keys:
        samples = [s[key] for s in batch]

        if not batch[0][key].shape:
            padded_batch[key] = torch.stack(samples)
        else:
            padded_batch[key] = util.padded_stack([s[key] for s in batch])

    return padded_batch
