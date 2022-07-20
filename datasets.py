import os
import re
from typing import Union, Tuple, List


class TextIterator:
    """
    A helpful utility to make a generator of sentences.
    """
    def __init__(self, trg_text: List[str], src_text: List[str]):
        assert len(trg_text) == len(src_text)
        self.trg_text = trg_text
        self.src_text = src_text
        self.ind = 0

    def __iter__(self):
        self.ind = 0
        return self

    def __next__(self):
        if self.ind == len(self.trg_text):
            raise StopIteration()

        trg, src = self.trg_text[self.ind], self.src_text[self.ind]
        self.ind += 1
        return trg, src
    

class YandexDataset:
    RU_TEXT_FILENAME = 'corpus.en_ru.1m.ru'
    EN_TEXT_FILENAME = 'corpus.en_ru.1m.en'

    def __init__(self, data_path: Union[str, Tuple[str, str]], train_split=0.95, valid_split=0.025, test_split=0.025):
        """
        A loader of the ru-en Yandex dataset. Apart from simply loading the data, it also does some cleaning.
        Cleaning takes about a minute on my machine.
        In my experiments BLEU after cleaning went from 18 to 24.
        NOTE: You can actually use this for any ru-en dataset if it has the same structure.

        Parameters
        ----------
        data_path : str or tuple
            Folder that contains two files with the text (ru and en).
            If it is a tuple, it should contain paths to files with text (ru, en).
        train_split : float, optional
            Percentage of training data, by default 0.95
        valid_split : float, optional
            Percentage of validation data, by default 0.025
        test_split : float, optional
            Percentage of testing data, by default 0.025
        """
        if isinstance(data_path, tuple):
            assert len(data_path) == 2
            ru_text_path = data_path[0]
            en_text_path = data_path[1]
        else:
            ru_text_path = os.path.join(data_path, YandexDataset.RU_TEXT_FILENAME)
            en_text_path = os.path.join(data_path, YandexDataset.EN_TEXT_FILENAME)

        with open(en_text_path, 'r') as file:
            self.english_text = file.readlines()

        with open(ru_text_path, 'r') as file:
            self.russian_text = file.readlines()

        self.train_split = train_split
        self.valid_split = valid_split
        self.test_split = test_split

        self.filter_data()

    # NOTE: The dataset is polluted with special characters and wrong translations (about 10% of it), so I clean some of this stuff out.

    def filter_data(self):
        print('Filtering the dataset. Initial size: %d' % len(self.russian_text))
        ru, en = self.russian_text, self.english_text
        # Remove all unusual (it does not remove unicode, actually) characters and transform to lower case
        ru = map(lambda x: re.sub(r"![а-я+0-9+ +,\-.'\[\]():%\"$?/!]", '', x.lower()), ru)
        en = map(lambda x: re.sub(r"![a-z+0-9+ +,\-.'\[\]():%\"$?/!]", '', x.lower()), en)

        # Add spaces between digits (and a couple more characters) so that digit sequences do not pollute vocabularies
        add_space_fn = lambda sen: re.sub("[1234567890\-/)(]", lambda group: " " + group[0] + " ", sen).replace('  ', ' ')
        ru = map(add_space_fn, ru)
        en = map(add_space_fn, en)

        # Remove pairs of sentences which lenghts are too different (one sentences is at least 1.4 times bigger than the other)
        filter_result = filter(lambda x: abs(len(x[0]) - len(x[1])) / max(len(x[0]), len(x[1]), 1) < 0.40 and len(x[0]) > 0 and len(x[1]) > 0, zip(ru, en))

        ru_filtered = []
        en_filtered = []
        _ = [(ru_filtered.append(ru_sen), en_filtered.append(en_sen)) for ru_sen, en_sen in filter_result]

        # Gather unicode characters and remove them from the text
        print('Removing special characters...')

        def gather_specials(text) -> set:
            specials_map = map(lambda x: re.sub(r"[a-z+а-я+0-9+ +,\-.'\[\]():%\"$?/!]", '', x.lower()), text)
            specials_list = list(filter(lambda x: len(x) > 0, specials_map))
            return set(''.join(specials_list))
            
        en_specials = gather_specials(en_filtered)
        ru_specials = gather_specials(ru_filtered)
        specials = en_specials.union(ru_specials)
        specials_pattern = re.compile(f"[{''.join(specials)}]")
        specials_delete_fn = lambda sen: re.sub(specials_pattern, '', sen).replace('  ', ' ')
        ru = list(map(specials_delete_fn, ru_filtered))
        en = list(map(specials_delete_fn, en_filtered))

        assert len(ru_filtered) == len(en_filtered)
        self.russian_text = ru
        self.english_text = en
        
        print('Filtered successfully. Final size: %d' % len(ru_filtered))

    def __iter__(self):
        for en, ru in zip(self.english_text, self.russian_text):
            yield en, ru

    def train_iter(self):
        train_end_ind = int(len(self.english_text) * self.train_split)

        english_sub_text = self.english_text[:train_end_ind]
        russian_sub_text = self.russian_text[:train_end_ind]
        return TextIterator(english_sub_text, russian_sub_text)
    
    def valid_iter(self):
        valid_start_ind = int(len(self.english_text) * self.train_split)
        valid_end_ind = int(len(self.english_text) * (self.train_split + self.valid_split))

        english_sub_text = self.english_text[valid_start_ind:valid_end_ind]
        russian_sub_text = self.russian_text[valid_start_ind:valid_end_ind]
        return TextIterator(english_sub_text, russian_sub_text)

    def test_iter(self):
        test_start_ind = int(len(self.english_text) * (self.train_split + self.valid_split))
        test_end_ind = len(self.english_text)

        english_sub_text = self.english_text[test_start_ind:test_end_ind]
        russian_sub_text = self.russian_text[test_start_ind:test_end_ind]
        return TextIterator(english_sub_text, russian_sub_text)

    def get_iters(self):
        return self.train_iter(), self.valid_iter(), self.test_iter()
    