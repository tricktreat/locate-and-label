import json
from abc import abstractmethod, ABC
from collections import OrderedDict
from logging import Logger
import os
from typing import Iterable, List
import numpy as np
import string

from tqdm import tqdm
from transformers import BertTokenizer

from identifier import util
from identifier.entities import Dataset, EntityType, RelationType, Entity, Relation, Document
from collections import Counter


class BaseInputReader(ABC):
    def __init__(self, types_path: str, tokenizer: BertTokenizer, iou_spn: float = 0.3, iou_classifier: float = 0.7, neg_entity_count: int = None,
                 neg_rel_count: int = None, window_size: int = None, logger: Logger = None):
        types = json.load(open(types_path), object_pairs_hook=OrderedDict)  # entity + relation types

        self._entity_types = OrderedDict()
        self._idx2entity_type = OrderedDict()
        self._relation_types = OrderedDict()
        self._idx2relation_type = OrderedDict()

        # entities
        # add 'None' entity type
        none_entity_type = EntityType('None', 0, 'None', 'No Entity')
        self._entity_types['None'] = none_entity_type
        self._idx2entity_type[0] = none_entity_type

        # specified entity types
        for i, (key, v) in enumerate(types['entities'].items()):
            entity_type = EntityType(key, i + 1, v['short'], v['verbose'])
            self._entity_types[key] = entity_type
            self._idx2entity_type[i + 1] = entity_type

        # relations
        # add 'None' relation type
        none_relation_type = RelationType('None', 0, 'None', 'No Relation')
        self._relation_types['None'] = none_relation_type
        self._idx2relation_type[0] = none_relation_type

        # specified relation types
        for i, (key, v) in enumerate(types['relations'].items()):
            relation_type = RelationType(key, i + 1, v['short'], v['verbose'], v['symmetric'])
            self._relation_types[key] = relation_type
            self._idx2relation_type[i + 1] = relation_type

        self._neg_entity_count = neg_entity_count
        self._neg_rel_count = neg_rel_count
        self.iou_spn = iou_spn
        self.iou_classifier = iou_classifier
        self._window_size = window_size

        self._datasets = dict()

        self._tokenizer = tokenizer
        self._logger = logger

        self._vocabulary_size = tokenizer.vocab_size
        self._context_size = -1

    @abstractmethod
    def read(self, datasets):
        pass

    def get_dataset(self, label) -> Dataset:
        return self._datasets[label]

    def get_entity_type(self, idx) -> EntityType:
        entity = self._idx2entity_type[idx]
        return entity

    def get_relation_type(self, idx) -> RelationType:
        relation = self._idx2relation_type[idx]
        return relation

    def _calc_context_size(self, datasets: Iterable[Dataset]):
        sizes = []

        for dataset in datasets:
            for doc in dataset.documents:
                sizes.append(len(doc.encoding))

        context_size = max(sizes)
        return context_size

    def _log(self, text):
        if self._logger is not None:
            self._logger.info(text)

    @property
    def datasets(self):
        return self._datasets

    @property
    def entity_types(self):
        return self._entity_types

    @property
    def relation_types(self):
        return self._relation_types

    @property
    def relation_type_count(self):
        return len(self._relation_types)

    @property
    def entity_type_count(self):
        return len(self._entity_types)

    @property
    def vocabulary_size(self):
        return self._vocabulary_size

    @property
    def context_size(self):
        return self._context_size

    def __str__(self):
        string = ""
        for dataset in self._datasets.values():
            string += "Dataset: %s\n" % dataset
            string += str(dataset)

        return string

    def __repr__(self):
        return self.__str__()


class JsonInputReader(BaseInputReader):
    def __init__(self, types_path: str, tokenizer: BertTokenizer, iou_spn: float = 0.3, iou_classifier : float = 1.0, neg_entity_count: int = None,
                 neg_rel_count: int = None, window_size: int = None, logger: Logger = None, build_vocab = False, wordvec_filename = None):
        super().__init__(types_path, tokenizer, iou_spn, iou_classifier, neg_entity_count, neg_rel_count, window_size, logger)
        if "glove" in wordvec_filename:
            vec_size = wordvec_filename.split(".")[-2] # str: 300d
        else:
            vec_size = "bio"
        if os.path.exists(os.path.dirname(types_path)+f"/vocab_{vec_size}.json") and os.path.exists(os.path.dirname(types_path)+f"/vocab_embed_{vec_size}.npy") :
            print("Reused vocab!")
            self.build_vocab = False
            self.word2inx = json.load(open(os.path.dirname(types_path)+f"/vocab_{vec_size}.json","r"))
            self.embedding_weight = np.load(os.path.dirname(types_path)+f"/vocab_embed_{vec_size}.npy")
        else:
            print("Need some time to construct vocab...")
            self.word2inx = {"<unk>": 0}
            self.embedding_weight = None
            self.build_vocab = True
        self.vec_size = vec_size
        self.wordvec_filename = wordvec_filename
        self.POS_MAP = ["<UNK>"]
        for k, v in json.load(open(types_path.replace("types", "pos"))).items():
            if v > 15:
                self.POS_MAP.append(k)
        # self.POS_MAP = list({ for k, v in json.load(open(types_path.replace("types", "pos"))).items()}.keys())
        

    def load_wordvec(self, filename):
        # word2vec = {}
        # with open(filename, "r") as f:
        #     if "glove" not in filename:
        #         f.readline()
        #     for line in f:
        #         fields = line.strip().split(' ')
        #         word2vec[fields[0]] = list(float(x) for x in fields[1:])
        self.embedding_weight = np.random.rand(len(self.word2inx),len(next(iter(self.word2vec.values()))))
        for word, inx in self.word2inx.items():
            if word in self.word2vec:
                self.embedding_weight[inx,:] = self.word2vec[word]

    def read(self, dataset_paths):
        for dataset_label, dataset_path in dataset_paths.items():
            dataset = Dataset(dataset_label, self._relation_types, self._entity_types, self._neg_entity_count,
                              self._neg_rel_count, self._window_size, self.iou_spn, self.iou_classifier)
            self._parse_dataset(dataset_path, dataset, dataset_label)
            self._datasets[dataset_label] = dataset
        if self.build_vocab:
            json.dump(self.word2inx,open(os.path.dirname(next(iter(dataset_paths.values())))+f"/vocab_{self.vec_size}.json","w"))
            self.load_wordvec(self.wordvec_filename)
            np.save(os.path.dirname(next(iter(dataset_paths.values())))+f"/vocab_embed_{self.vec_size}.npy",self.embedding_weight)
        self._context_size = self._calc_context_size(self._datasets.values())

    def _parse_dataset(self, dataset_path, dataset, dataset_label):
        documents = json.load(open(dataset_path))
        if dataset_label == "train" and self.build_vocab:
            self._build_vocab(documents)
        for document in tqdm(documents, desc="Parse dataset '%s'" % dataset.label):
            self._parse_document(document, dataset)
    
    def _build_vocab(self, documents, min_freq = 1):
        self.word2vec = {}
        with open(self.wordvec_filename, "r") as f:
            if "glove" not in self.wordvec_filename:
                f.readline()
            for line in f:
                fields = line.strip().split(' ')
                self.word2vec[fields[0]] = list(float(x) for x in fields[1:])
        counter = Counter()
        for doc in documents:
            counter.update(list(map(lambda x: x.lower(), doc['tokens'])))
        for k, v in counter.items():
            if v >= min_freq and k in self.word2vec:
                self.word2inx[k] = len(self.word2inx)

    def _parse_document(self, doc, dataset) -> Document:
        jtokens = doc['tokens']
        jrelations = doc['relations']
        jentities = doc['entities']
        jpos = doc['pos']
        ltokens = doc["ltokens"]
        rtokens = doc["rtokens"]


        # parse tokens
        doc_tokens, doc_encoding, char_encoding = self._parse_tokens(jtokens, ltokens, rtokens, jpos, dataset)

        # parse entity mentions
        entities = self._parse_entities(jentities, doc_tokens, dataset)

        # parse relations
        relations = self._parse_relations(jrelations, entities, dataset)

        # create document
        document = dataset.create_document(doc_tokens, entities, relations, doc_encoding, char_encoding)

        return document

    def _parse_tokens(self, jtokens, ltokens, rtokens, jpos, dataset):
        doc_tokens = []
        char_vocab = ['<PAD>'] + list(string.printable) + ['<EOT>', '<UNK>']
        # full document encoding including special tokens ([CLS] and [SEP]) and byte-pair encodings of original tokens
        doc_encoding = [self._tokenizer.convert_tokens_to_ids('[CLS]')]
        char_encoding = []
        # jtokens = list(filter(lambda x:x!='', jtokens))
        # doc = nlp(" ".join(jtokens))
        # poss = [token.pos_ for token in doc]
        # poss = ["PUNCT" for token in jtokens]

        poss = [self.POS_MAP.index(pos) if pos in self.POS_MAP else self.POS_MAP.index("<UNK>") for pos in jpos]

        # for token in doc:
        #     print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
        #             token.shape_, token.is_alpha, token.is_stop)

        # parse tokens
        for token_phrase in ltokens:
            token_encoding = self._tokenizer.encode(token_phrase, add_special_tokens=False)
            doc_encoding += token_encoding
        
        for i, token_phrase in enumerate(jtokens):

            # if self.build_vocab and token_phrase.lower() not in self.word2inx:
            #     self.word2inx[token_phrase.lower()] = len(self.word2inx)
            token_encoding = self._tokenizer.encode(token_phrase, add_special_tokens=False)
            # token_encoding_char = list(char_vocab.index(c) for c in token_phrase)
            token_encoding_char = []
            for c in token_phrase:
                if c in char_vocab:
                    token_encoding_char.append(char_vocab.index(c))
                else:
                    token_encoding_char.append(char_vocab.index("<UNK>"))
            span_start, span_end = (len(doc_encoding), len(doc_encoding) + len(token_encoding))
            char_start, char_end = (len(char_encoding), len(char_encoding) + len(token_encoding_char))
            # try:
            if token_phrase.lower() in  self.word2inx:
                inx = self.word2inx[token_phrase.lower()]
            else:
                inx = self.word2inx["<unk>"]
            token = dataset.create_token(i, span_start, span_end, token_phrase, poss[i], inx, char_start, char_end)
            doc_tokens.append(token)
            doc_encoding += token_encoding
            token_encoding_char += [char_vocab.index('<EOT>')]
            char_encoding.append(token_encoding_char)
            # except:
            #     print(jtokens)
        
        for token_phrase in rtokens:
            token_encoding = self._tokenizer.encode(token_phrase, add_special_tokens=False)
            doc_encoding += token_encoding

        doc_encoding += [self._tokenizer.convert_tokens_to_ids('[SEP]')]

        return doc_tokens, doc_encoding, char_encoding

    def _parse_entities(self, jentities, doc_tokens, dataset) -> List[Entity]:
        entities = []

        for entity_idx, jentity in enumerate(jentities):
            entity_type = self._entity_types[jentity['type']]
            start, end = jentity['start'], jentity['end']

            # create entity mention
            tokens = doc_tokens[start:end]
            phrase = " ".join([t.phrase for t in tokens])
            entity = dataset.create_entity(entity_type, tokens, phrase)
            entities.append(entity)

        return entities

    def _parse_relations(self, jrelations, entities, dataset) -> List[Relation]:
        relations = []

        for jrelation in jrelations:
            relation_type = self._relation_types[jrelation['type']]

            head_idx = jrelation['head']
            tail_idx = jrelation['tail']

            # create relation
            head = entities[head_idx]
            tail = entities[tail_idx]

            reverse = int(tail.tokens[0].index) < int(head.tokens[0].index)

            # for symmetric relations: head occurs before tail in sentence
            if relation_type.symmetric and reverse:
                head, tail = util.swap(head, tail)

            relation = dataset.create_relation(relation_type, head_entity=head, tail_entity=tail, reverse=reverse)
            relations.append(relation)

        return relations
