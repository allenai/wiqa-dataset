import argparse
import enum
import json
import os
from os import listdir
from os.path import isfile, join
from typing import Any, Dict, List

from tqdm import tqdm

from src.helpers.ProparaExtendedPara import ProparaExtendedParaMetadata
from src.helpers.dataset_info import *
from src.helpers.situation_graph import SituationLabel
from src.helpers.whatif_metadata import WhatifMetadata


class Jsonl:
    @staticmethod
    def load(in_filepath: str) -> List[Dict[str, Any]]:
        d: List[Dict[str, Any]] = []
        if not os.path.exists(in_filepath):
            print(f"JSONL Path does not exist: {in_filepath}")
            return d
        with open(in_filepath, 'r') as infile:
            for line in infile:
                if line and not Jsonl.is_comment_line(line=line):
                    d.append(json.loads(line.strip()))
        return d

    @staticmethod
    def is_comment_line(line: str):
        return line.strip().startswith("#")


class WIQAUtils:
    @staticmethod
    def get_split_para(passage_with_sent_id: str) -> List[str]:
        split_para = passage_with_sent_id.split('. ')
        text_split_para = split_para[1::2]
        return text_split_para

    @staticmethod
    def filenames_in_folder(folder):
        return [f for f in listdir(folder) if isfile(join(folder, f))]

    @staticmethod
    def strip_special_char(a_string):
        return "".join(
            [x for x in a_string if (ord('a') <= ord(x) <= ord('z')) or (ord('A') <= ord(x) <= ord('Z'))]).lower()

    # Constants for the dataset wrapper
    LABELS = [{"label": "A", "text": "more"},
              {"label": "B", "text": "less"},
              {"label": "C", "text": "no effect"}]


class WIQAQuesType(str, enum.Enum):
    OTHER = "OTHER"
    INPARA_EFFECT = "INPARA_EFFECT"  # From same if-then block, source node is from { X, Y, U, W }
    EXOGENOUS_EFFECT = "EXOGENOUS_EFFECT"  # From same if-then block but source node is Z or V
    INPARA_DISTRACTOR = "INPARA_DISTRACTOR"  # non-path from same if-then block
    OUTOFPARA_DISTRACTOR = "OUTOFPARA_DISTRACTOR"  # will be useful when we get out-of-para distractor annotations

    @staticmethod
    def from_str(qtype_str):
        if not qtype_str:
            raise ValueError(
                f"TF question type must not be empty or None-- input to WIQAQuesType from_str: ({qtype_str}) ")
        qtype_str = qtype_str.lower().replace('_', ' ').strip()

        if qtype_str in ['in para effect', 'inpara effect', 'direct']:
            return WIQAQuesType.INPARA_EFFECT
        elif qtype_str in ['exogeneous effect', 'exogenous effect', 'indirect']:
            return WIQAQuesType.EXOGENOUS_EFFECT
        elif qtype_str in ['inpara distractor', 'in para distractor']:
            return WIQAQuesType.INPARA_DISTRACTOR
        elif qtype_str in ['outofpara distractor', 'out of para distractor']:
            return WIQAQuesType.OUTOFPARA_DISTRACTOR
        else:
            print(f"WARNING: tf question type: {qtype_str} not identified")
            return WIQAQuesType.OTHER

    @staticmethod
    def from_path(path, is_distractor, in_para=True):
        if is_distractor:
            if in_para:
                return WIQAQuesType.INPARA_DISTRACTOR
            else:
                return WIQAQuesType.OUTOFPARA_DISTRACTOR

        start_node = path[0] if path and len(path) > 1 else ""
        if start_node.id == "Z" or start_node.id == "V":  # Out of para situations causing changes in the process
            return WIQAQuesType.EXOGENOUS_EFFECT

        return WIQAQuesType.INPARA_EFFECT  # Changes to in-para events causing changes in rest of the process

    def to_json(self):
        return self.name


class WIQAExplanationType(str, enum.Enum):
    NO_EXPL = "NO_EXPL"
    PARA_SENT_EXPL = "PARA_SENT_EXPL"

    @staticmethod
    def from_str(sl):
        if not sl:
            raise ValueError(
                f"({sl}) is not a valid Enum WIQAExplanationType")
        sl = sl.lower().replace('_', ' ').strip()
        if sl in ['no expl', 'no explanation', 'no exp']:
            return WIQAExplanationType.NO_EXPL
        elif sl in ['with exp', 'with expl', 'para sent expl', 'paragraph sentence explanation', 'expl', 'exp']:
            return WIQAExplanationType.PARA_SENT_EXPL
        else:
            raise Exception(f"WARNING: ({sl}) is not a valid choice for {WIQAExplanationType.__dict__}")

    def to_json(self):
        return self.name


class WIQAExplanation(object):

    def __init__(self, di: SituationLabel, dj: SituationLabel, de: SituationLabel, i: int, j: int):
        self.di = di
        self.dj = dj
        self.de = de
        self.i = i
        self.j = j

    @staticmethod
    def instantiate_from_old_json_version(json_data: Dict[str, Any]):
        assert 'explanation' in json_data, f"WIQA explanation cannot be instantiated due to missing keys[explanation] in json: {json_data}"
        assert 'di' in json_data[
            'explanation'], f"WIQA explanation cannot be instantiated due to missing keys[explanation][di] in json: {json_data}"
        return WIQAExplanation(
            di=SituationLabel.from_str(json_data['explanation']['di']),
            dj=SituationLabel.from_str(json_data['explanation']['di']),
            de=SituationLabel.from_str(json_data['explanation']['dj'] if "de" not in json_data["explanation"] else json_data['explanation']['de']),
            i=json_data['explanation']['i'],
            j=json_data['explanation']['j']
        )

    @staticmethod
    def instantiate_from(json_data: Dict[str, Any]):
        '''
        :param json_data: must contain keys:
        "explanations": {
            "de" : answer_label,
            "di": optional_supporting_sent_label,
            "i": optional_sentidx_or_None,
            "j": optional_sentidx_or_None
        }
        :return: WIQAExplanation object.
        '''
        assert 'explanation' in json_data, f"WIQA explanation cannot be instantiated due to missing keys[explanation] in json: {json_data}"
        assert 'di' in json_data[
            'explanation'], f"WIQA explanation cannot be instantiated due to missing keys[explanation][di] in json: {json_data}"
        return WIQAExplanation(
            di=None if 'di' not in 'explanation' or not json_data['explanation']['di'] else SituationLabel.from_str(json_data['explanation']['di']),
            dj=None if 'di' not in 'explanation' or not json_data['explanation']['di'] else SituationLabel.from_str(json_data['explanation']['di']),
            de=SituationLabel.from_str(json_data['explanation']['de']),
            i=None if 'i' not in json_data['explanation'] else json_data['explanation']['i'],
            j=None if 'j' not in json_data['explanation'] else json_data['explanation']['j']
        )

    @staticmethod
    def deserialize(json_data: Dict[str, Any]):
        return WIQAExplanation(di=SituationLabel.from_str(json_data['di']),
                               dj=SituationLabel.from_str(json_data['dj']),
                               de=SituationLabel.from_str(json_data['de']),
                               i=json_data['i'],
                               j=json_data['j']
                               )


class WIQAQuestion(object):
    def __init__(self, stem: str,
                 para_steps: List[str],
                 answer_label: str,
                 answer_label_as_choice: str,
                 choices: List[Dict[str, str]] = WIQAUtils.LABELS):
        self.stem = stem
        self.para_steps = para_steps
        self.answer_label = answer_label
        self.answer_label_as_choice = answer_label_as_choice
        self.choices = choices

    @staticmethod
    def instantiate_from(json_data: Dict[str, Any]):
        assert 'explanation' in json_data, f"WIQA question cannot be instantiated due to missing keys[explanation] in json: {json_data}"
        chosen_label = SituationLabel.from_str(json_data['explanation']['dj'])
        return WIQAQuestion(stem=json_data['question']['question'],
                            para_steps=json_data['steps'],
                            answer_label=chosen_label.as_less_more(),
                            answer_label_as_choice=SituationLabel.get_emnlp_test_choice(chosen_label))

    @staticmethod
    def deserialize(json_data: Dict[str, Any]):
        return WIQAQuestion(stem=json_data['stem'],
                            para_steps=json_data['para_steps'],
                            answer_label=json_data['answer_label'],
                            answer_label_as_choice=json_data['answer_label_as_choice'])


class WIQAQuesMetadata(object):
    def __init__(self, ques_id, graph_id: str, para_id: str, question_type: WIQAQuesType):
        self.ques_id = ques_id
        self.graph_id = graph_id
        self.para_id = para_id
        self.question_type = question_type

    @staticmethod
    def instantiate_from(json_data: Dict[str, Any]):
        assert 'question' in json_data, f"WIQA QuesMetadata cannot be instantiated due to missing keys[question] in json: {json_data}"
        return WIQAQuesMetadata(
            ques_id=json_data['id'],
            graph_id=json_data['metadata']['graph_id'],
            para_id=json_data['metadata']['para_id'],
            question_type=WIQAQuesType.from_str(json_data['metadata']['question_type'])
        )

    @staticmethod
    def deserialize(json_data: Dict[str, Any]):
        return WIQAQuesMetadata(ques_id=json_data['ques_id'],
                                graph_id=json_data['graph_id'],
                                para_id=json_data['para_id'],
                                question_type=WIQAQuesType.from_str(json_data['question_type'])
                                )


class WIQADataPoint(object):
    """
    holds the relevant WIQA data sample
    """

    def __init__(self, question: WIQAQuestion, explanation: WIQAExplanation, metadata: WIQAQuesMetadata):
        self.question = question
        self.explanation = explanation
        self.metadata = metadata

    @staticmethod
    def instantiate_from(explanation_type: WIQAExplanationType, json_data: Dict[str, Any]):
        if explanation_type == WIQAExplanationType.NO_EXPL:
            explanation = None
        else:
            explanation = WIQAExplanation.instantiate_from_old_json_version(json_data=json_data)
        return WIQADataPoint(
            question=WIQAQuestion.instantiate_from(json_data=json_data),
            explanation=explanation,
            metadata=WIQAQuesMetadata.instantiate_from(json_data=json_data)
        )

    def to_json(self, explanation_type: WIQAExplanationType= WIQAExplanationType.PARA_SENT_EXPL):
        # ensure that no object contains the
        if explanation_type == WIQAExplanationType.NO_EXPL:
            return {'question': self.question.__dict__,
                    'metadata': self.metadata.__dict__
                    }
        else:
            return {'question': self.question.__dict__,
                    'explanation': self.explanation.__dict__,
                    'metadata': self.metadata.__dict__
                    }

    @staticmethod
    def deserialize(json_data: Dict[str, Any]):
        return WIQADataPoint(question=WIQAQuestion.deserialize(json_data=json_data['question']),
                             metadata=WIQAQuesMetadata.deserialize(json_data=json_data['metadata']),
                             explanation=WIQAExplanation.deserialize(json_data=json_data['explanation'])
                             )

    @staticmethod
    def __get_default_whatif_metadata(
            para_ids_metainfo_fp=download_from_url_if_not_in_cache(
                para_partition_info.cloud_path),
            situation_graphs_fp=download_from_url_if_not_in_cache(
                influence_graphs_v1.cloud_path)):
        return WhatifMetadata(para_ids_metainfo_fp=para_ids_metainfo_fp, situation_graphs_fp=situation_graphs_fp)

    @staticmethod
    def __get_default_propara_paragraphs_metadata(extended_propara_para_fp=download_from_url_if_not_in_cache(
        propara_para_info.cloud_path)):
        return ProparaExtendedParaMetadata(extended_propara_para_fp=extended_propara_para_fp)

    @staticmethod
    def load_all_in_jsonl(jsonl_filepath):
        for j in Jsonl.load(in_filepath=jsonl_filepath):
            yield WIQADataPoint.deserialize(json_data=j)

    def get_steps(self):
        return self.question.para_steps

    def get_provenance_influence_graph(self, whatif_metadata: WhatifMetadata):
        return whatif_metadata.get_graph_for_id(graph_id=self.metadata.graph_id)

    def get_orig_propara_paragraph(self,
                                   orig_propara: ProparaExtendedParaMetadata):
        return orig_propara.paraentry_for_id(para_id=self.metadata.para_id)

    def get_other_paragraphs_under_this_topic(self, whatif_metadata: WhatifMetadata,
                                              orig_propara: ProparaExtendedParaMetadata):
        return [orig_propara.paraentry_for_id(para_id=x) for x in whatif_metadata.get_paraids_for_topic(
            topic_str=whatif_metadata.get_topic_for_paraid(para_id=self.metadata.para_id))]


def create_concise_dataset(input_filepaths, output_filepaths, explanation_type: WIQAExplanationType):
    """
    :param input_filepaths:
    :param output_filepaths:
    :param explanation_type:
    :usage ```partitions = ["train.jsonl", "dev.jsonl", "test.jsonl"]
    create_concise_dataset(
        input_filepaths=[download_from_url_if_not_in_cache(wiqa_explanations_v1.cloud_path + partition) for partition in
                         partitions],
        output_filepaths=["/tmp/od/" + x for x in partitions],
        explanation_type=WIQAExplanationType.PARA_SENT_EXPL)```
    :return:
    """
    print(f"\nInput file paths: {input_filepaths}")
    assert input_filepaths is not None and len(input_filepaths) > 0, \
        f"in/outfile paths for creating wiqa wrapper files is not matching or empty."

    for file_num, input_filepath in enumerate(input_filepaths):
        print(f"\nGenerating reformatted data for .... {input_filepath}")
        with open(input_filepath, 'r') as in_file:
            output_filepath = output_filepaths[file_num]

            # ensure that the outpath directory exists, if not create that path.
            outdir = "/".join(output_filepath.split("/")[0:-1])
            if not os.path.exists(outdir):
                os.makedirs(outdir)

            with open(output_filepath, 'w') as out_file:
                for line in tqdm(in_file):
                    json_data = json.loads(line)
                    data_object = WIQADataPoint.instantiate_from(explanation_type=explanation_type,
                                                                 json_data=json_data)
                    dump_it = data_object.to_json(explanation_type=explanation_type)
                    out_file.write(json.dumps(dump_it))
                    out_file.write('\n')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='subcommand')

    #  --------------------------------------------------
    #  ################ Download dataset ################
    #  --------------------------------------------------
    parser_download = subparsers.add_parser('download')
    parser_download.add_argument('--input_dirpath',
                                 action='store',
                                 dest='input_dirpath',
                                 required=True,
                                 help='Input dataset directory')
    parser_download.add_argument('--output_dirpath',
                                 action='store',
                                 dest='output_dirpath',
                                 required=True,
                                 help='folder to store output')
    parser_download.add_argument('--explanation_type',
                                 action='store',
                                 dest='explanation_type',
                                 required=True,
                                 help='with_expl|no_expl')

    #  --------------------------------------------------
    #  ################ Load json data ################
    #  --------------------------------------------------
    args = parser.parse_args()

    if args.subcommand == "download":
        print(f"Input  {[args.input_dirpath + x for x in WIQAUtils.filenames_in_folder(args.input_dirpath)]}")
        create_concise_dataset(
            input_filepaths=[args.input_dirpath + x for x in WIQAUtils.filenames_in_folder(args.input_dirpath)],
            output_filepaths=[args.output_dirpath + x for x in WIQAUtils.filenames_in_folder(args.input_dirpath)],
            explanation_type=args.explanation_type)
