from torch.utils.data import DataLoader

from utils import read_jsonl
import torch
from transformers import BertTokenizer

LABEL_DICT = {"less": 0, "more": 1, "no_effect": 2}
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def read_wiqa_split(split_dir, split_name):
    texts = []
    labels = []

    split_file_name = f'{split_dir}/{split_name}.jsonl'

    json_lines_split = read_jsonl(split_file_name)
    for line in json_lines_split:
        assert "para_steps" in line["question"]
        context = ". ".join(line["question"]["para_steps"])
        texts.append(context + ". " + line["question"]["stem"])
        labels.append(LABEL_DICT[line["question"]["answer_label"]])
    return texts, labels


class WIQADataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def get_wiqa_dataloader(datapath, split_name, batch_size=4, shuffle=True):
    split_texts, split_labels = read_wiqa_split(datapath,split_name)
    split_encodings = tokenizer(split_texts, truncation=True, padding=True)
    split_dataset = WIQADataset(split_encodings, split_labels)
    dataloader = DataLoader(split_dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader