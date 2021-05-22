from numpy.core.numeric import zeros_like
import pdb
from pdb import Pdb
import torch
from torch import nn as nn
from torch._C import device, dtype
from transformers import BertConfig
from transformers import BertModel
from transformers import BertPreTrainedModel

from identifier import sampling
from identifier import util


def get_token(h: torch.tensor, x: torch.tensor, token: int):
    """ Get specific token embedding (e.g. [CLS]) """
    emb_size = h.shape[-1]

    token_h = h.view(-1, emb_size)
    flat = x.contiguous().view(-1)

    # get contextualized embedding of given token
    token_h = token_h[flat == token, :]

    return token_h

def change_span_token_mask(entity_spans_token, old_entity_masks_token):
    # print(entity_spans_token[0])
    # print(old_entity_masks_token[0])
    mask = torch.zeros(old_entity_masks_token.size(), dtype=torch.bool).to(device=old_entity_masks_token.device)
    for i, sample_entity_spans_token in enumerate(entity_spans_token):
        for j,span in enumerate(sample_entity_spans_token):
            # if span[0]<0 or span[1]>1000:
            #     continue
            mask[i, j, span[0]:span[1]] = 1
    # print(mask[0][0])
    return mask


class Identifier(BertPreTrainedModel):

    def __init__(self, config: BertConfig, embed: torch.tensor, cls_token: int, entity_types: int,
                 size_embedding: int, prop_drop: float, freeze_transformer: bool, lstm_layers:int =1, lstm_drop: float = 0.4, pos_size: int = 25, char_lstm_layers:int = 1, char_lstm_drop:int = 0.2, char_size:int = 25, use_glove: bool = True, use_pos:bool = True, use_char_lstm:bool = True, spn_filter: int = 5, pool_type:str = "max", use_entity_ctx = False, use_size_embedding = False, reduce_dim = False, bert_before_lstm = False, no_filter = True, no_regressor = True, norm = "sigmoid", no_times_count = False):
        super(Identifier, self).__init__(config)

        # BERT model
        config.output_hidden_states = True
        self.no_filter = no_filter
        self.no_regressor = no_regressor
        self.norm = norm
        self.no_times_count = no_times_count
        self.bert = BertModel(config)
        self.wordvec_size = embed.size(-1)
        self.pos_size = pos_size
        self.use_glove = use_glove
        self.use_pos = use_pos
        self.char_lstm_layers=char_lstm_layers
        self.char_lstm_drop=char_lstm_drop
        self.char_size=char_size
        self.use_char_lstm=use_char_lstm
        self.use_entity_ctx = use_entity_ctx
        self.use_size_embedding = use_size_embedding
        self.reduce_dim = reduce_dim
        self.bert_before_lstm = bert_before_lstm
        lstm_input_size = 0
        if self.bert_before_lstm:
            lstm_input_size = config.hidden_size
        # assert use_glove or use_pos or use_char_lstm, "At least one be True"
        if use_glove:
            lstm_input_size += self.wordvec_size
        if use_pos:
            lstm_input_size += self.pos_size
            self.pos_embedding = nn.Embedding(100, pos_size)
        if use_char_lstm:
            lstm_input_size += self.char_size * 2
            self.char_lstm = nn.LSTM(input_size = char_size, hidden_size = char_size, num_layers = char_lstm_layers,  bidirectional = True, dropout = char_lstm_drop, batch_first = True)
            self.char_embedding = nn.Embedding(103, char_size)

        reg_size = (2 * lstm_input_size + config.hidden_size) * 3
        cls_size = (2 * lstm_input_size + config.hidden_size) * 3

        if self.use_glove or self.use_pos or self.use_char_lstm or self.bert_before_lstm:
            lstm_hidden_size = lstm_input_size
            if self.bert_before_lstm:
                lstm_hidden_size = config.hidden_size//2
            self.lstm = nn.LSTM(input_size = lstm_input_size, hidden_size = lstm_hidden_size, num_layers = lstm_layers,  bidirectional = True, dropout = lstm_drop, batch_first = True)
            if self.reduce_dim or self.bert_before_lstm:
                if not self.bert_before_lstm:
                    self.reduce_dimension = nn.Linear(2 * lstm_input_size + config.hidden_size, config.hidden_size)
                reg_size = config.hidden_size * 3
                cls_size = config.hidden_size * 3

        self.spn_filter = spn_filter

        self.pool_type = pool_type

        # layers
        # self.binary_classfier = nn.Linear(config.hidden_size * 4 ,2)

        if use_entity_ctx:
            reg_size += config.hidden_size
            cls_size += config.hidden_size
        if use_size_embedding:
            self.size_embeddings = nn.Embedding(100, size_embedding)
            reg_size += size_embedding
            cls_size += size_embedding
        
        self.binary_classfier = nn.Sequential(
            nn.Linear(cls_size , config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 2)
            # nn.Linear(cls_size , 2)
        )

        # self.entity_classifier = nn.Linear(config.hidden_size * 2 + size_embedding ,entity_types)

        self.entity_classifier = nn.Sequential(
            nn.Linear(cls_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, entity_types)
            # nn.Linear(cls_size, entity_types),
        )

        layers = [nn.Linear(reg_size, config.hidden_size), nn.GELU(), nn.Linear(config.hidden_size, 2)]    
        # layers = [nn.Linear(reg_size, 2)]
        if self.norm == "sigmoid":
            layers.append(nn.Sigmoid())
        elif self.norm == "tanh":
            layers.append(nn.Tanh())
        self.offset_regressier1 = nn.Sequential(*layers)
        # self.offset_regressier2 = nn.Sequential(
        #     # nn.LayerNorm(config.hidden_size * 2 + size_embedding),
        #     # nn.Dropout(prop_drop),
        #     nn.Linear(config.hidden_size * 3, config.hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(config.hidden_size, 2),
        #     # nn.Sigmoid()
        # )
        self.dropout = nn.Dropout(prop_drop)

        self._cls_token = cls_token
        self._entity_types = entity_types

        # weight initialization
        self.init_weights()
        if use_glove:
            self.wordvec_embedding = nn.Embedding.from_pretrained(embed)

        if freeze_transformer:
            print("Freeze transformer weights")

            # freeze all transformer weights
            for param in self.bert.parameters():
                param.requires_grad = False

    def combine(self, sub, sup_mask, pool_type = "max" ):
        sup = None
        if len(sub.shape) == len(sup_mask.shape) :
            if pool_type == "mean":
                size = (sup_mask == 1).float().sum(-1).unsqueeze(-1) + 1e-30
                m = (sup_mask.unsqueeze(-1) == 1).float()
                sup = m * sub.unsqueeze(1).repeat(1, sup_mask.shape[1], 1, 1)
                sup = sup.sum(dim=2) / size
            if pool_type == "sum":
                m = (sup_mask.unsqueeze(-1) == 1).float()
                sup = m * sub.unsqueeze(1).repeat(1, sup_mask.shape[1], 1, 1)
                sup = sup.sum(dim=2)
            if pool_type == "max":
                m = (sup_mask.unsqueeze(-1) == 0).float() * (-1e30)
                sup = m + sub.unsqueeze(1).repeat(1, sup_mask.shape[1], 1, 1)
                sup = sup.max(dim=2)[0]
                sup[sup==-1e30]=0
        else:
            if pool_type == "mean":
                size = (sup_mask == 1).float().sum(-1).unsqueeze(-1) + 1e-30
                m = (sup_mask.unsqueeze(-1) == 1).float()
                sup = m * sub
                sup = sup.sum(dim=2) / size
            if pool_type == "sum":
                m = (sup_mask.unsqueeze(-1) == 1).float()
                sup = m * sub
                sup = sup.sum(dim=2)
            if pool_type == "max":
                m = (sup_mask.unsqueeze(-1) == 0).float() * (-1e30)
                sup = m + sub
                sup = sup.max(dim=2)[0]
                sup[sup==-1e30]=0
        return sup

    def _common_forward(self, encodings: torch.tensor, context_masks: torch.tensor, token_masks:torch.tensor, entity_masks: torch.tensor,  
                        token_masks_bool:torch.tensor, entity_masks_token: torch.tensor, entity_spans_token: torch.tensor, 
                       entity_sizes: torch.tensor, pos_encoding: torch.tensor = None, wordvec_encoding:torch.tensor = None, char_encoding:torch.tensor = None, token_masks_char = None, char_count:torch.tensor = None):
        # get contextualized token embeddings from last transformer layer
        # exit(0)
        context_masks = context_masks.float()
        # h = self.bert(input_ids=encodings, attention_mask=context_masks)[0]
        h = self.bert(input_ids=encodings, attention_mask=context_masks)[2]
        h = torch.stack(h[-4:],dim=-1).mean(-1)

        batch_size = encodings.shape[0]
        token_count = token_masks_bool.long().sum(-1,keepdim=True)
        # token_masks_bool = token_masks_bool.unsqueeze(-2)
        # m = (token_masks.unsqueeze(-1) == 0).float() * (-1e30)
        # h_token = m + h.unsqueeze(1).repeat(1, token_masks.shape[1], 1, 1)
        # h_token = h_token.max(dim=2)[0]
        h_token = self.combine(h, token_masks, self.pool_type)



        embeds = []
        if self.bert_before_lstm:
            embeds = [h_token]
        if self.use_pos:
            pos_embed = self.pos_embedding(pos_encoding)
            pos_embed = self.dropout(pos_embed)
            embeds.append(pos_embed)
        if self.use_glove:
            word_embed = self.wordvec_embedding(wordvec_encoding)
            word_embed = self.dropout(word_embed)
            embeds.append(word_embed)
        if self.use_char_lstm:
            char_count = char_count.view(-1)
            token_masks_char = token_masks_char
            max_token_count = char_encoding.size(1)
            max_char_count = char_encoding.size(2)

            char_encoding = char_encoding.view(max_token_count*batch_size, max_char_count)
            # import pdb; pdb.set_trace()
            
            char_encoding[char_count==0][:, 0] = 101
            char_count[char_count==0] = 1
            char_embed = self.char_embedding(char_encoding)
            char_embed = self.dropout(char_embed)
            char_embed_packed = nn.utils.rnn.pack_padded_sequence(input = char_embed, lengths = char_count.tolist(), enforce_sorted = False, batch_first = True)
            char_embed_packed_o, (_, _) = self.char_lstm(char_embed_packed)
            char_embed, _ = nn.utils.rnn.pad_packed_sequence(char_embed_packed_o, batch_first=True)
            char_embed = char_embed.view(batch_size, max_token_count, max_char_count, self.char_size * 2)
            h_token_char = self.combine(char_embed, token_masks_char, "mean")
            embeds.append(h_token_char)

            # import pdb; pdb.set_trace()
            # char_embed = self.char_embedding(char_encoding)
            # char_embed = self.dropout(char_embed)
            # char_count = char_masks_bool.long().sum(-1,keepdim=True)
            # char_embed_packed = nn.utils.rnn.pack_padded_sequence(input = char_embed, lengths = char_count.squeeze(-1).cpu().tolist(), enforce_sorted = False, batch_first = True)
            # char_embed_packed_o, (_, _) = self.char_lstm(char_embed_packed)
            # char_embed, _ = nn.utils.rnn.pad_packed_sequence(char_embed_packed_o, batch_first=True)
            # h_token_char = self.combine(char_embed, token_masks_char, self.pool_type)
            # embeds.append(h_token_char)

        if len(embeds)>0:
            h_token_pos_wordvec_char = torch.cat(embeds, dim = -1)
            # import pdb; pdb.set_trace()
            h_token_pos_wordvec_char_packed = nn.utils.rnn.pack_padded_sequence(input = h_token_pos_wordvec_char, lengths = token_count.squeeze(-1).cpu().tolist(), enforce_sorted = False, batch_first = True)
            h_token_pos_wordvec_char_packed_o, (_, _) = self.lstm(h_token_pos_wordvec_char_packed)
            h_token_pos_wordvec_char, _ = nn.utils.rnn.pad_packed_sequence(h_token_pos_wordvec_char_packed_o, batch_first=True)
            # h_token = h_token + h_token_pos_wordvec_char
            rep = [h_token_pos_wordvec_char]
            if not self.bert_before_lstm:
                rep.append(h_token)
            h_token = torch.cat(rep, dim = -1)
            if self.reduce_dim and not self.bert_before_lstm:
                h_token = self.reduce_dimension(h_token)

        # spn
        size_embeddings = None
        if self.use_size_embedding:
            size_embeddings = self.size_embeddings(entity_sizes)  # embed entity candidate sizes
        bin_clf, entity_spans_pool, offsets1 = self._spn(encodings, h, h_token, entity_masks, entity_masks_token, size_embeddings, entity_spans_token, token_count, token_masks_bool)
        # offsets1[offsets1<0]=torch.log(offsets1[offsets1<0])
        # offsets1[offsets1>0]=torch.log(-offsets1[offsets1>0])
        if not self.no_times_count:
            offsets1 = offsets1 * token_count.unsqueeze(-2).expand(-1,offsets1.size(1),2)
        if self.no_regressor:
            offsets1 = torch.zeros_like(offsets1, device = offsets1.device)

        spn_p = torch.softmax(bin_clf,dim=-1)
        spn_mask = spn_p[:,:,0] < self.spn_filter * spn_p[:,:,1]
        if self.no_filter:
            spn_mask = torch.ones_like(spn_p[:,:,0], device=spn_p[:,:,0].device)
        illegal_mask = torch.ones_like(spn_mask, device=spn_mask.device)
        
        # offsets entity_spans_token 
        old = entity_spans_token
        entity_spans_token = entity_spans_token + torch.round(offsets1).to(dtype=torch.long)

        #
        offsets1[entity_spans_token[:,:,0]>=entity_spans_token[:,:,1]] = 0
        offsets1[entity_spans_token[:,:,0]<0] = 0
        offsets1[entity_spans_token[:,:,1]>token_count] = 0

        #
        illegal_mask[entity_spans_token[:,:,0]>=entity_spans_token[:,:,1]] = 0
        illegal_mask[entity_spans_token[:,:,0]<0] = 0
        illegal_mask[entity_spans_token[:,:,1]>token_count] = 0

        # entity_spans_token
        entity_spans_token[entity_spans_token[:,:,0]>=entity_spans_token[:,:,1]]=old[entity_spans_token[:,:,0]>=entity_spans_token[:,:,1]]
        entity_spans_token[entity_spans_token[:,:,0]<0]=old[entity_spans_token[:,:,0]<0]
        entity_spans_token[entity_spans_token[:,:,1]>token_count]=old[entity_spans_token[:,:,1]>token_count]

        # print(entity_spans_token.dtype, torch.round(offsets1).dtype)
        entity_masks_token = change_span_token_mask(entity_spans_token, entity_masks_token)

        # classify entities
        size_embeddings = None
        if self.use_size_embedding:
            # error
            change_size = (torch.round(offsets1[:,:,1]) - torch.round(offsets1[:,:,0])).to(dtype=torch.long)
            change_size[(entity_sizes + change_size)>99]=0
            change_size[(entity_sizes + change_size)<0]=0

            # print(change_size)
            entity_sizes = entity_sizes + change_size

            size_embeddings = self.size_embeddings(entity_sizes)  # embed entity candidate sizes
        entity_clf, entity_spans_pool, offsets2 = self._classify_entities(encodings, h, h_token, entity_masks, entity_masks_token, size_embeddings, entity_spans_token, token_count, token_masks_bool)

        return entity_clf, offsets2, bin_clf, offsets1, spn_mask, illegal_mask


    def _forward_train(self, encodings: torch.tensor, context_masks: torch.tensor, token_masks:torch.tensor, entity_masks: torch.tensor,  
                        token_masks_bool:torch.tensor, entity_masks_token: torch.tensor, entity_spans_token: torch.tensor, 
                       entity_sizes: torch.tensor, pos_encoding: torch.tensor = None, wordvec_encoding:torch.tensor = None, char_encoding:torch.tensor = None, token_masks_char = None, char_count:torch.tensor = None):
        return self._common_forward( encodings, context_masks, token_masks, entity_masks,  
                        token_masks_bool, entity_masks_token, entity_spans_token, 
                       entity_sizes, pos_encoding, wordvec_encoding, char_encoding, token_masks_char, char_count)


    def _forward_eval(self, encodings: torch.tensor, context_masks: torch.tensor, token_masks:torch.tensor, entity_masks: torch.tensor, 
                      entity_masks_token: torch.tensor, entity_spans_token: torch.tensor, entity_spans: torch.tensor, entity_sample_masks: torch.tensor, 
                      entity_sizes: torch.tensor,  token_masks_bool:torch.tensor, pos_encoding: torch.tensor = None, wordvec_encoding = None, char_encoding:torch.tensor = None, token_masks_char:torch.tensor = None, char_count:torch.tensor = None):
        entity_clf, offsets2, bin_clf, offsets1, spn_mask, illegal_mask = self._common_forward( encodings, context_masks, token_masks, entity_masks,  
                        token_masks_bool, entity_masks_token, entity_spans_token, 
                       entity_sizes, pos_encoding, wordvec_encoding, char_encoding, token_masks_char, char_count)
        entity_clf = torch.softmax(entity_clf, dim=2)
        return entity_clf, spn_mask, offsets1, offsets2, illegal_mask

    def _spn(self, encodings, h, h_token, entity_masks, entity_masks_token, size_embeddings, entity_spans_token, token_count, token_masks_bool):
        # import pdb; pdb.set_trace()
        # # max pool entity candidate spans
        # m = (entity_masks.unsqueeze(-1) == 0).float() * (-1e30)
        # entity_spans_pool = m + h.unsqueeze(1).repeat(1, entity_masks.shape[1], 1, 1)
        # entity_spans_pool = entity_spans_pool.max(dim=2)[0]

        # m = (entity_masks_token.unsqueeze(-1) == 0).float() * (-1e30)
        # entity_spans_pool_token = m + h_token.unsqueeze(1).repeat(1, entity_masks_token.shape[1], 1, 1)
        # entity_spans_pool = entity_spans_pool_token.max(dim=2)[0]

        entity_spans_pool = self.combine(h_token, entity_masks_token, self.pool_type)
        # entity_spans_pool_context = self.combine(h_token, (entity_masks_token==0) * token_masks_bool, pool_type="max")

        # get cls token as candidate context representation
        # entity_ctx = get_token(h, encodings, self._cls_token)

        entity_spans_token_outer = entity_spans_token.clone()
        entity_spans_token_outer[:,:,0] = entity_spans_token_outer[:,:,0]-1
        entity_spans_token_outer[:,:,1] = entity_spans_token_outer[:,:,1]
        entity_spans_token_outer[:,:,0][entity_spans_token_outer[:,:,0]<0] = 0
        entity_spans_token_outer[:,:,1][entity_spans_token_outer[:,:,1]==token_count] = token_count.repeat(1,entity_spans_token_outer.size(1))[entity_spans_token_outer[:,:,1]==token_count] - 1
        # B #entity 2 #embedding_size
        start_end_embedding_outer = util.batch_index(h_token, entity_spans_token_outer)
        # # B #entity #embedding_size
        # # start_end_embedding_outer = start_end_embedding_outer[:,:,0,:] + start_end_embedding_outer[:,:,1,:]
        start_end_embedding_outer = start_end_embedding_outer.view(start_end_embedding_outer.size(0),start_end_embedding_outer.size(1),-1)


        entity_spans_token_inner = entity_spans_token.clone()
        entity_spans_token_inner[:,:,0] = entity_spans_token_inner[:,:,0]
        entity_spans_token_inner[:,:,1] = entity_spans_token_inner[:,:,1] - 1
        entity_spans_token_inner[:,:,1][entity_spans_token_inner[:,:,1]<0] = 0
        start_end_embedding_inner = util.batch_index(h_token, entity_spans_token_inner)

        start_end_embedding_inner = start_end_embedding_inner.view(start_end_embedding_inner.size(0),start_end_embedding_inner.size(1),-1)


        # start_end_embedding = (start_end_embedding_inner + start_end_embedding)/2
        # start_end_embedding = start_end_embedding_inner

        # create candidate representations including context, max pooled span and size embedding
        # entity_repr = torch.cat([start_end_embedding, entity_ctx.unsqueeze(1).repeat(1, entity_spans_pool.shape[1], 1),
        #                          entity_spans_pool, size_embeddings], dim=2)

        embed_outer = [start_end_embedding_outer, entity_spans_pool]
        embed_inner = [start_end_embedding_inner, entity_spans_pool]

        if self.use_entity_ctx:
            entity_ctx = get_token(h, encodings, self._cls_token)
            entity_ctx = entity_ctx.unsqueeze(1).repeat(1, entity_spans_pool.shape[1], 1)
            embed_outer.append(entity_ctx)
            embed_inner.append(entity_ctx)
        if self.use_size_embedding:
            embed_outer.append(size_embeddings)
            embed_inner.append(size_embeddings)
        entity_repr_outer = torch.cat(embed_outer, dim=2)
        entity_repr_outer = self.dropout(entity_repr_outer)
            
        entity_repr_inner = torch.cat(embed_inner, dim=2)
        entity_repr_inner = self.dropout(entity_repr_inner)


        # classify entity candidates
        offsets = self.offset_regressier1(entity_repr_outer)
        entity_clf = self.binary_classfier(entity_repr_inner)

        return entity_clf, entity_spans_pool, offsets

    def _classify_entities(self, encodings, h, h_token, entity_masks, entity_masks_token, size_embeddings, entity_spans_token, token_count, token_masks_bool):
        # # max pool entity candidate spans
        # m = (entity_masks.unsqueeze(-1) == 0).float() * (-1e30)
        # entity_spans_pool = m + h.unsqueeze(1).repeat(1, entity_masks.shape[1], 1, 1)
        # entity_spans_pool = entity_spans_pool.max(dim=2)[0]

        # m = (entity_masks_token.unsqueeze(-1) == 0).float() * (-1e30)
        # entity_spans_pool_token = m + h_token.unsqueeze(1).repeat(1, entity_masks_token.shape[1], 1, 1)
        # entity_spans_pool = entity_spans_pool_token.max(dim=2)[0]

        entity_spans_pool = self.combine(h_token, entity_masks_token, self.pool_type)
        # entity_spans_pool_context = self.combine(h_token, (entity_masks_token==0) * token_masks_bool, pool_type="max")

        # entity_spans_token_outer = entity_spans_token.clone()
        # entity_spans_token_outer[:,:,0] = entity_spans_token_outer[:,:,0]-1
        # entity_spans_token_outer[:,:,1] = entity_spans_token_outer[:,:,1]
        # entity_spans_token_outer[:,:,0][entity_spans_token_outer[:,:,0]<0] = 0
        # entity_spans_token_outer[:,:,1][entity_spans_token_outer[:,:,1]>=token_count] = token_count.repeat(1,entity_spans_token_outer.size(1))[entity_spans_token_outer[:,:,1]>=token_count] - 1
        # # B #entity 2 #embedding_size
        # start_end_embedding_outer = util.batch_index(h_token, entity_spans_token_outer)
        # # B #entity #embedding_size
        # # start_end_embedding_outer = start_end_embedding_outer[:,:,0,:] + start_end_embedding_outer[:,:,1,:]
        # start_end_embedding_outer = start_end_embedding_outer.view(start_end_embedding_outer.size(0),start_end_embedding_outer.size(1),-1)


        entity_spans_token_inner = entity_spans_token.clone()
        entity_spans_token_inner[:,:,0] = entity_spans_token_inner[:,:,0]
        entity_spans_token_inner[:,:,1] = entity_spans_token_inner[:,:,1] - 1
        entity_spans_token_inner[:,:,1][entity_spans_token_inner[:,:,1]<0] = 0
        start_end_embedding_inner = util.batch_index(h_token, entity_spans_token_inner)

        start_end_embedding_inner = start_end_embedding_inner.view(start_end_embedding_inner.size(0),start_end_embedding_inner.size(1),-1)


        # start_end_embedding = (start_end_embedding_inner + start_end_embedding)/2
        # start_end_embedding = start_end_embedding_inner

        # create candidate representations including context, max pooled span and size embedding
        # entity_repr = torch.cat([start_end_embedding, entity_ctx.unsqueeze(1).repeat(1, entity_spans_pool.shape[1], 1),
        #                          entity_spans_pool, size_embeddings], dim=2)
        # entity_repr_outer = torch.cat([start_end_embedding_outer, entity_spans_pool], dim=2)
        # entity_repr_outer = self.dropout(entity_repr_outer)

        embed_inner = [start_end_embedding_inner, entity_spans_pool]
        if self.use_entity_ctx:
            entity_ctx = get_token(h, encodings, self._cls_token)
            entity_ctx = entity_ctx.unsqueeze(1).repeat(1, entity_spans_pool.shape[1], 1)
            embed_inner.append(entity_ctx)
        if self.use_size_embedding:
            embed_inner.append(size_embeddings)
        entity_repr_inner = torch.cat(embed_inner, dim=2)
        entity_repr_inner = self.dropout(entity_repr_inner)


        # classify entity candidates
        entity_clf = self.entity_classifier(entity_repr_inner)
        # offsets = self.offset_regressier2(entity_repr_outer)
        offsets = None
        return entity_clf, entity_spans_pool, offsets

    def forward(self, *args, evaluate=False, **kwargs):
        if not evaluate:
            return self._forward_train(*args, **kwargs)
        else:
            return self._forward_eval(*args, **kwargs)


# Model access

_MODELS = {
    'identifier': Identifier,
}


def get_model(name):
    return _MODELS[name]
