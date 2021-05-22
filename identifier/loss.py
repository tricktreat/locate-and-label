from abc import ABC

import torch
from torch._C import dtype


class Loss(ABC):
    def compute(self, *args, **kwargs):
        pass

def iou( inputs, targets):
    # inputs[:,:,0].squeeze(-1) == l_offsets_gt).long().view(-1) * (torch.round(offsets1_pred[:,:,1]).squeeze(-1)== r_offsets_gt).long().view(-1) * old_illegal_mask.long().view(-1)
    # import pdb; pdb.set_trace()
    i_left = torch.max(inputs[:, 0], targets[:, 0])
    i_right = torch.min(inputs[:, 1], targets[:, 1])
    o_left = torch.min(inputs[:, 0], targets[:, 0])
    o_right = torch.max(inputs[:, 1], targets[:, 1])
    all = o_right - o_left
    all[all==0] = 1e-30
    iou = (i_right - i_left)/all
    inx = iou<0
    iou[inx] = 0
    return iou

class IdentifierLoss(Loss):
    def __init__(self, filter_criterion, offset_criterion, giou_criterion, entity_criterion, model, optimizer, scheduler, iou_weight, max_grad_norm, iou_classifier=1.0, filter_weight = 1.0, offset_weight = 1.0, giou_weight = 1.0, entity_weight = 1.0, iou_gamma = 1.0):
        self._entity_criterion = entity_criterion
        self._filter_criterion = filter_criterion
        self._offset_criterion = offset_criterion
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._max_grad_norm = max_grad_norm
        self._iou_weight = iou_weight
        self._giou_criterion = giou_criterion
        self._iou_classifier = iou_classifier
        self._filter_weight = filter_weight
        self._offset_weight = offset_weight
        self._giou_weight = giou_weight
        self._entity_weight = entity_weight
        self._iou_gamma = iou_gamma

    def compute(self, entity_logits, offsets1_pred, offsets2_pred, bin_logits, spn_mask,entity_types,entity_types_1, l_offsets_gt, r_offsets_gt, ious, entity_sample_masks, offset_sample_masks, entity_spans_token, illegal_mask):
        # entity loss
        old_ious = ious
        ious = ious.view(-1)
        entity_logits = entity_logits.view(-1, entity_logits.shape[-1])
        entity_types = entity_types.view(-1)
        entity_sample_masks = entity_sample_masks.view(-1).float()
        old_spn_mask = spn_mask
        old_illegal_mask = illegal_mask
        spn_mask = spn_mask.view(-1).float()
        illegal_mask = illegal_mask.view(-1).float()

        bin_logits = bin_logits.view(-1, bin_logits.shape[-1])
        bin_entity_types = (entity_types_1!=0).view(-1).to(dtype=torch.long)
        # bin_loss = self._entity_criterion(bin_logits, bin_entity_types)
        # spn_l_offset_loss = self._offset_criterion(offsets1_pred[:,:,0].squeeze(-1), l_offsets_gt)
        # spn_r_offset_loss = self._offset_criterion(offsets1_pred[:,:,1].squeeze(-1), r_offsets_gt)
        spn_giou_loss = self._giou_criterion((offsets1_pred + entity_spans_token).view(-1, 2), (torch.stack([l_offsets_gt, r_offsets_gt], dim = -1) + entity_spans_token).view(-1, 2))

        # import pdb; pdb.set_trace()
        if self._iou_weight:
            bin_loss = self._filter_criterion(bin_logits, bin_entity_types) * ious
            spn_l_offset_loss = self._offset_criterion(offsets1_pred[:,:,0].squeeze(-1), l_offsets_gt) * old_ious
            spn_r_offset_loss = self._offset_criterion(offsets1_pred[:,:,1].squeeze(-1), r_offsets_gt) * old_ious
        else:
            bin_loss = self._filter_criterion(bin_logits, bin_entity_types)
            spn_l_offset_loss = self._offset_criterion(offsets1_pred[:,:,0].squeeze(-1), l_offsets_gt)
            spn_r_offset_loss = self._offset_criterion(offsets1_pred[:,:,1].squeeze(-1), r_offsets_gt)

        
        bin_loss = (bin_loss * entity_sample_masks).sum() / (entity_sample_masks).sum()


        spn_l_offset_loss[torch.isinf(spn_l_offset_loss)]=0
        spn_r_offset_loss[torch.isinf(spn_r_offset_loss)]=0


        entity_sample_masks = entity_sample_masks * illegal_mask
        
        # 过滤部分负例
        # entity_sample_masks = entity_sample_masks * illegal_mask * spn_mask

        offset_sample_masks = offset_sample_masks * old_illegal_mask

        # 不应该过滤分类上的负例
        # offset_sample_masks = offset_sample_masks * old_spn_mask * old_illegal_mask



        # l_offset_loss[torch.isinf(l_offset_loss)]=0
        # r_offset_loss[torch.isinf(r_offset_loss)]=0

        # if offset_sample_masks.sum()!=0:
        # l_offset_loss = (l_offset_loss * offset_sample_masks).sum() / (offset_sample_masks.sum() + 1e-30)
        # r_offset_loss = (r_offset_loss * offset_sample_masks).sum() / (offset_sample_masks.sum() + 1e-30)
        # giou_loss = (giou_loss * offset_sample_masks.view(-1)).sum() / (offset_sample_masks.sum() + 1e-30)
        spn_l_offset_loss = (spn_l_offset_loss * offset_sample_masks).sum() / (offset_sample_masks.sum() + 1e-30)
        spn_r_offset_loss = (spn_r_offset_loss * offset_sample_masks).sum() / (offset_sample_masks.sum() + 1e-30)
        spn_giou_loss = (spn_giou_loss * offset_sample_masks.view(-1)).sum() / (offset_sample_masks.sum() + 1e-30)
        # else:
        #     l_offset_loss = 0
        #     r_offset_loss = 0
        #     spn_l_offset_loss = 0
        #     spn_r_offset_loss = 0
        
        offset_loss = spn_l_offset_loss + spn_r_offset_loss
        # offset_2_loss = l_offset_loss + r_offset_loss

        # entity_types = entity_types * old_spn_mask.long().view(-1)

        # offsets_pred = offsets1_pred + offsets2_pred
        # offsets_pred = offsets1_pred
        
        # entity_types = entity_types * (torch.round(offsets1_pred[:,:,0]).squeeze(-1) == l_offsets_gt).long().view(-1) * (torch.round(offsets1_pred[:,:,1]).squeeze(-1)== r_offsets_gt).long().view(-1) * old_illegal_mask.long().view(-1)

        new_iou = iou((torch.round(offsets1_pred) + entity_spans_token).view(-1, 2), (torch.stack([l_offsets_gt, r_offsets_gt], dim = -1) + entity_spans_token).view(-1, 2))
        entity_types = entity_types * (new_iou>=self._iou_classifier) * old_illegal_mask.long().view(-1)
        new_iou[new_iou<self._iou_classifier] = torch.pow(1 - new_iou[new_iou<self._iou_classifier], self._iou_gamma)
        new_iou[new_iou>=self._iou_classifier] = torch.pow(new_iou[new_iou>=self._iou_classifier], self._iou_gamma)

        # import pdb; pdb.set_trace()

        if self._iou_weight:
            entity_loss = self._entity_criterion(entity_logits, entity_types) * new_iou
            # l_offset_loss = self._offset_criterion(offsets_pred[:,:,0].squeeze(-1), l_offsets_gt) * old_ious
            # r_offset_loss = self._offset_criterion(offsets_pred[:,:,1].squeeze(-1), r_offsets_gt) * old_ious
            # giou_loss = self._giou_criterion((offsets_pred + entity_spans_token).view(-1, 2), (torch.stack([l_offsets_gt, r_offsets_gt], dim = -1) + entity_spans_token).view(-1, 2)) * ious
        else:
            entity_loss = self._entity_criterion(entity_logits, entity_types)
            # l_offset_loss = self._offset_criterion(offsets_pred[:,:,0].squeeze(-1), l_offsets_gt)
            # r_offset_loss = self._offset_criterion(offsets_pred[:,:,1].squeeze(-1), r_offsets_gt)
            # giou_loss = self._giou_criterion((offsets_pred + entity_spans_token).view(-1, 2), (torch.stack([l_offsets_gt, r_offsets_gt], dim = -1) + entity_spans_token).view(-1, 2))
            

        # entity_loss = self._entity_criterion(entity_logits, entity_types) 
        entity_loss = (entity_loss * entity_sample_masks).sum() / (entity_sample_masks.sum()+ 1e-30)

        train_loss = self._entity_weight * entity_loss + self._filter_weight * bin_loss + self._offset_weight * offset_loss  + self._giou_weight * spn_giou_loss
        # print(entity_loss, bin_loss,spn_l_offset_loss,spn_r_offset_loss,l_offset_loss,r_offset_loss,spn_giou_loss,giou_loss )
        if torch.isnan(train_loss) ==True:
            import pdb; pdb.set_trace()
        # train_loss = entity_loss
        # print(entity_loss , l_offset_loss , r_offset_loss , bin_loss , spn_l_offset_loss , spn_r_offset_loss)

        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
        self._optimizer.step()
        self._scheduler.step()
        self._model.zero_grad()
        return train_loss.item()
