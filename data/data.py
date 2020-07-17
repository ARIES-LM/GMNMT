import math
# import ipdb
from collections import OrderedDict, Counter
from itertools import chain

import torch
import os

from torchtext import data, datasets
from torchtext.data import Example
from contextlib import ExitStack


# load the dataset + reversible tokenization
class NormalField(data.Field):

    def build_vocab(self, *args, **kwargs):
        """Construct the Vocab object for this field from one or more datasets.

        Arguments:
            Positional arguments: Dataset objects or other iterable data
                sources from which to construct the Vocab object that
                represents the set of possible values for this field. If
                a Dataset object is provided, all columns corresponding
                to this field are used; individual columns can also be
                provided directly.
            Remaining keyword arguments: Passed to the constructor of Vocab.
        """
        counter = Counter()
        sources = []
        for arg in args:
            sources += [getattr(arg, name) for name, field in
                            arg.fields.items() if field is self]
        for data in sources:
            for x in data:
                if not self.sequential:
                    x = [x]
                try:
                    counter.update(x)
                except TypeError:
                    counter.update(chain.from_iterable(x))
        specials = list(OrderedDict.fromkeys(
            tok for tok in [self.unk_token, self.pad_token, self.init_token,
                            self.eos_token] + kwargs.pop('specials', [])
            if tok is not None))
        self.vocab = self.vocab_cls(counter, specials=specials, **kwargs)

    def reverse(self, batch, unbpe=True, returen_token=False):
        if not self.batch_first:
            batch.t_()

        with torch.cuda.device_of(batch):
            batch = batch.tolist()

        batch = [[self.vocab.itos[ind] for ind in ex] for ex in batch] # denumericalize

        def trim(s, t):
            sentence = []
            for w in s:
                if w == t:
                    break
                sentence.append(w)
            return sentence

        batch = [trim(ex, self.eos_token) for ex in batch] # trim past frst eos

        def filter_special(tok):
            return tok not in (self.init_token, self.pad_token)

        if unbpe:
            batch = [" ".join(filter(filter_special, ex)).replace("@@ ", "") for ex in batch]
        else:
            batch = [" ".join(filter(filter_special, ex)) for ex in batch]

        if returen_token:
            batch = [ex.split() for ex in batch]
        return batch


class GraphField(data.Field):
    def preprocess(self, x):
        return x.strip()

    def process(self, x, device=None):
        batch_imgs = []
        batch_alighs = []
        region_num = []
        for i in x:
            i = i.split()
            img = i[0]
            align = i[1:]
            align = list(map(lambda item: list(map(int, item.split('-'))), align))
            # unit = (img, align)
            # contents.append(unit)
            batch_imgs.append(img)
            batch_alighs.append(align)
            if len(align) == 0:
                #align.extend([[0,0], [1,0]])
                region_num.append(1)
            else:
                region_num.append(align[-1][-1] + 1)
        return batch_imgs, batch_alighs, region_num


class TranslationDataset(data.Dataset):
    """Defines a dataset for machine translation."""

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.src), len(ex.trg))

    def __init__(self, path, exts, fields, **kwargs):
        """Create a TranslationDataset given paths and fields.
        Arguments:
            path: Common prefix of paths to the data files for both languages.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1])]

        src_path, trg_path = tuple(os.path.expanduser(path + '.' + x) for x in exts)

        examples = []
        with open(src_path, encoding='utf-8') as src_file, open(trg_path, encoding='utf-8') as trg_file:
            for src_line, trg_line in zip(src_file, trg_file):
                src_line, trg_line = src_line.strip(), trg_line.strip()
                if src_line != '' and trg_line != '':
                    examples.append(data.Example.fromlist(
                        [src_line, trg_line], fields))

        super(TranslationDataset, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, path, exts, fields, root='.data', train='train', validation='val', test='test', **kwargs):
        """Create dataset objects for splits of a TranslationDataset.
        Arguments:
            root: Root dataset storage directory. Default is '.data'.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            train: The prefix of the train data. Default: 'train'.
            validation: The prefix of the validation data. Default: 'val'.
            test: The prefix of the test data. Default: 'test'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        #path = cls.download(root)

        train_data = None if train is None else cls(
            os.path.join(path, train), exts, fields, **kwargs)
        val_data = None if validation is None else cls(
            os.path.join(path, validation), exts, fields, **kwargs)
        test_data = None if test is None else cls(
            os.path.join(path, test), exts, fields, **kwargs)
        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)


class ParallelDataset(data.Dataset):
    """ Define a N-parallel dataset: supports abitriry numbers of input streams"""

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.src), len(ex.trg))

    def __init__(self, path, exts, fields, max_len=None, **kwargs):
        assert len(exts) == len(fields), 'N parallel dataset must match'
        self.N = len(fields)

        if not isinstance(fields[0], (tuple, list)):
            newfields = [('src', fields[0]), ('trg', fields[1])]
            for i in range(len(exts) - 2):
                newfields.append(('extra_{}'.format(i), fields[2 + i]))
            # self.fields = newfields
            fields = newfields

        paths = tuple(os.path.expanduser(path + '.' + x) for x in exts)
        # self.max_len = max_len
        examples = []

        with ExitStack() as stack:
            files = [stack.enter_context(open(fname, encoding='utf-8')) for fname in paths]
            for i, lines in enumerate(zip(*files)):
                lines = [line.strip() for line in lines]
                if not any(line == '' for line in lines):
                    example = Example.fromlist(lines, fields)
                    examples.append(example)
                    # if max_len is None:
                    #     examples.append(example)
                    # elif len(example.src) <= max_len and len(example.trg) <= max_len:
                    #     examples.append(example)
        super(ParallelDataset, self).__init__(examples, fields, **kwargs)

'''
class MyBatch(Batch):
    def __init__(self, allsentences, orders, doc_len, ewords, elocs, dataset=None,
                 device=None):
        """Create a Batch from a list of examples."""
        if data is not None:
            setattr(self, 'doc_len', doc_len)
            setattr(self, 'elocs', elocs)

            self.batch_size = len(doc_len)
            self.dataset = dataset
            self.fields = dataset.fields.keys()  # copy field names
            self.input_fields = [k for k, v in dataset.fields.items() if
                                 v is not None and not v.is_target]
            self.target_fields = [k for k, v in dataset.fields.items() if
                                  v is not None and v.is_target]

            setattr(self, 'order', dataset.fields['order'].process(orders, device=device))
            setattr(self, 'doc', dataset.fields['doc'].process(allsentences, device=device))

            setattr(self, 'e_words', dataset.fields['doc'].process(ewords, device=device))

            # setattr(self, 'docwords', dataset.fields['doc'].process(doc_words, device=device))
            # setattr(self, 'graph', dataset.fields['e2e'].process_graph(e2ebatch, e2sbatch, orders,
            #                                                            doc_sent_word_len, device=device))
            # setattr(self, 'alllen', doc_sent_word_len)

            # for (name, field) in dataset.fields.items():
            #     if field is not None:
            #         batch = [getattr(x, name) for x in data]
            #         setattr(self, name, field.process(batch, device=device))


class DocIter(data.BucketIterator):
    def data(self):
        if self.shuffle:
            xs = [self.dataset[i] for i in self.random_shuffler(range(len(self.dataset)))]
        else:
            xs = self.dataset
        return xs

    def __iter__(self):
        while True:
            self.init_epoch()
            for idx, minibatch in enumerate(self.batches):
                # print(idx+1)
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1

                doc_len = []
                allsentences = []
                orders = []

                for ex in minibatch:
                    doc_len.append(len(ex.order))
                maxdoclen = max(doc_len)

                ewords = []
                elocs = []
                for ex in minibatch:
                    doc, order = ex.doc, ex.order

                    randid = list(range(len(order)))
                    shuffle(randid)

                    sfdoc = [doc[ri] for ri in randid]

                    sforder = [order[ri] for ri in randid]
                    sforder = list(np.argsort(sforder))

                    orders.append(sforder)

                    padnum = maxdoclen - len(sforder)
                    padded = sfdoc
                    for _ in range(padnum):
                        padded.append(['<pad>'])
                    allsentences.extend(padded)

                    eg = ex.entity

                    ew = []
                    newlocs = []
                    target = sforder

                    for eandloc in eg.split():
                        e, loc_role = eandloc.split(':')
                        ew.append(e)

                        word_newlocs = []
                        # print(loc_role)
                        for lr in loc_role.split('|'):
                            oriloc, r = lr.split('-')
                            word_newlocs.append([target[int(oriloc)], int(r)])

                        newlocs.append(word_newlocs)

                    elocs.append(newlocs)
                    ewords.append(ew)

                yield MyBatch(allsentences, orders, doc_len, ewords, elocs,
                              self.dataset, self.device)
            if not self.repeat:
                return

'''
