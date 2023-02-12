import pathlib
from abc import abstractmethod
from typing import Tuple
import os
import uuid

INVERSE_MAPPING = {
    'argmin' : {
        0: "Argument_for",
        1: "Argument_against",
    },
    'scd' : {
        0: "for",
        1: "against"
    },
    'iac1' : {
        0: "pro",
        1: "anti",
        2: "other"
    },
    'snopes' : {
        0: "agree",
        1: "refute"
    },
    'fnc1' : {
        0: "agree",
        1: "disagree",
        2: "unrelated",
        3: "discuss"
    },
    'semeval2016t6' : {
        0: "FAVOR",
        1: "AGAINST",
        2: "NONE"
    },
    'semeval2019t7' : {
        0: "support",
        1: "deny",
        2: "query",
        3: "comment"
    },
    'ibmcs' : {
        0: "PRO",
        1: "CON"
    },
    'perspectrum' : {
        0: "SUPPORT",
        1: "UNDERMINE"
    },
    'arc': {
        0: "agree",
        1: "disagree",
        2: "unrelated",
        3: "discuss"
    },
    'wtwt': {
        0: "support",
        1: "refute",
        2: "unrelated",
        3: "comment"
    },
    'emergent': {
        0: "for",
        1: "against",
        2: "observing"
    }
}

import pandas as pd

def select_context_elements(context, k=1):
    try:
        context = eval(context)
    except:
        context = [context]

    if k is None:
        return context
    else:
        if len(context) < k:
            context += [""] * (k - len(context))
        return context[:k]


class BaseLoader:
    def __init__(self, dataset_path, task_name, id2label, kb = 'relevant_final', k = 1):
        self.dataset_path = pathlib.Path(dataset_path)
        self.task_name = task_name
        self.id2label = id2label
        self.label2id = {v: k for k, v in id2label.items()}
        self.kb = kb
        self.k = k
        self.train_dataset, self.val_dataset, self.test_dataset = self._prepare_splits(
            self.dataset_path, self.task_name
        )

        for subset in (self.train_dataset, self.val_dataset, self.test_dataset):
            subset["task_name"] = task_name
            if "hypothesis" not in subset.columns:
                subset["hypothesis"] = ""

            # if task_name == "wtwt":
            #     premise_maps = {
            #         "AET_HUM": "Aetna acquires Humana",
            #         "ANTM_CI": "Anthem acquires Cigna",
            #         "CVS_AET": "CVS Health acquires Aetna",
            #         "CI_ESRX": "Cigna acquires Express Scripts",
            #         "FOXA_DIS": "Disney acquires 21st Century Fox ",
            #     }
            #     subset["premise"] = subset["premise"].apply(premise_maps.get)

        self.train_dataset = self.train_dataset[self.train_dataset.hypothesis != "[not found]"]
        self.val_dataset = self.val_dataset[self.val_dataset.hypothesis != "[not found]"]
        self.test_dataset = self.test_dataset[self.test_dataset.hypothesis != "[not found]"]

        self.train_dataset = self.train_dataset.reset_index(drop=True)
        self.val_dataset = self.val_dataset.reset_index(drop=True)
        self.test_dataset = self.test_dataset.reset_index(drop=True)

    @abstractmethod
    def _prepare_splits(self, **kwargs):
        pass


class StanceLoader(BaseLoader):
    def _prepare_splits(self, dataset_path, task_name):
        train = pd.read_json(dataset_path / f"{task_name}_train.json", lines=True).copy()
        val = pd.read_json(dataset_path / f"{task_name}_dev.json", lines=True).copy()
        test = pd.read_json(dataset_path / f"{task_name}_test.json", lines=True).copy()

        return train, val, test


class StanceLoaderContext(BaseLoader):
    def _prepare_splits(self, dataset_path, task_name):
        df = pd.read_csv(os.path.join(dataset_path, task_name + '_with_candidates_' + self.kb + '.csv'))
        df["label"] = df["label"].apply(INVERSE_MAPPING[task_name].get) # this is cumbersome but easy hack for the moment
        df["label"] = df["label"].apply(self.label2id.get)
        df["uid"] = [uuid.uuid4() for i in range(len(df))]
        # rename columns to match subsequent code
        df.rename(columns={'sentence': 'hypothesis', 'topic': 'premise'}, inplace=True)
        # select context
        df["context"] = df["candidates"].apply(lambda candidates: select_context_elements(candidates, k=self.k))
        # remove newline and cr characters in text
        df['hypothesis'] = df['hypothesis'].str.replace('\n', ' ')
        df['hypothesis'] = df['hypothesis'].str.replace('\r', ' ')

        df['premise'] = df['premise'].str.replace('\n', ' ')
        df['premise'] = df['premise'].str.replace('\r', ' ')

        train = df[df["split"]=='train'].copy()
        val = df[(df["split"] == "val") | (df["split"] == "dev") ].copy()
        test = df[df["split"]=='test'].copy()


        return train, val, test


class DALoader(StanceLoader):
    def __init__(self, dataset_path, task_name, id2label):
        super().__init__(dataset_path, task_name, id2label)
        from stancedetection.util import mappings

        g2id = {k: v for v, k in enumerate(["positive", "negative", "discuss", "other", "neutral"])}
        l2g = {k: DALoader.map_to_group(v) for k, v in mappings.RELATED_TASK_MAP.items()}
        self.train_dataset.label = (
            self.train_dataset.label.apply(mappings.TASK_MAPPINGS[task_name]["id2label"].get)
            .apply(lambda x: f"{task_name}__{x}")
            .apply(lambda x: l2g[x])
            .apply(lambda x: g2id[x])
        )

        self.val_dataset = self.val_dataset.reset_index(drop=True)
        self.test_dataset = self.test_dataset.reset_index(drop=True)

    @staticmethod
    def map_to_group(label_list):
        from stancedetection.util import mappings

        if label_list is mappings.POSITIVE_LABELS:
            return "positive"
        elif label_list is mappings.NEGATIVE_LABELS:
            return "negative"
        elif label_list is mappings.DISCUSS_LABELS:
            return "discuss"
        elif label_list is mappings.OTHER_LABELS:
            return "other"
        elif label_list is mappings.NEUTRAL_LABELS:
            return "neutral"

        raise Exception(f"No such mapping for {label_list}")
