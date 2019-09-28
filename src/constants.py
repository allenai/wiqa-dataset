"""
Constants that are necessary for this source folder
"""

############################
# Dataset related constants
############################
from src.situation_graph import SituationLabel

answer_dict = {'A': 'Correct Effect',
               'B': 'Opposite Effect',
               'C': 'No Effect'}

answer_indices = {'Correct Effect': 0,
                  'Opposite Effect': 1,
                  'No Effect': 2}

relations = {'RESULTS_IN': '[unused0]',
             'NOT_RESULTS_IN': '[unused1]',
             'NO_EFFECT': '[unused2]'}

answer_str_to_SituationLabel = \
    {'Correct Effect': SituationLabel.RESULTS_IN,
     'Opposite Effect': SituationLabel.NOT_RESULTS_IN,
     'No Effect': SituationLabel.NO_EFFECT}

answer_key_to_SituationLabel = \
    {'A': SituationLabel.RESULTS_IN,
     'B': SituationLabel.NOT_RESULTS_IN,
     'C': SituationLabel.NO_EFFECT}

##########################
# BERT constants
##########################
SEP_TOKEN = '[SEP]'
CLS_TOKEN = '[CLS]'
NEG_TOKEN = 'not'
PAD_TOKEN = '[PAD]'
MASK_TOKEN = '[MASK]'


##########################
# Dataset related Files
##########################

LOCAL_DATA = 'data/'

# BEAKER_DATA_FOLDER = '/data/'
BEAKER_DATA_FOLDER = ''

IG_DATA_FOLDER = BEAKER_DATA_FOLDER + "wiqa_datasets/propara_splits/"

PARA_ID_METADATA_INFO = IG_DATA_FOLDER + "by_topic/para_id.prompt.topic.partition.tsv"
SITUATION_GRAPHS_FILE = IG_DATA_FOLDER + "turker_task_outputs/shard123.para.ifthenBlocks.outcomes.json"

VETTED_RESP_TSV_FILE = IG_DATA_FOLDER + "/out_of_para_distractors/simple_random_gen_4perNode/" \
                                        "overlap0/validation_output/validation_outcomes_turkerVotes.5_per_qid.peters_format.tsv"

METADATA_FILE = BEAKER_DATA_FOLDER + 'propara-related-datasets/grids-new-format.tsv'
PARTITION_FILE = BEAKER_DATA_FOLDER + "propara-related-datasets/naacl18.partition.paraid.tsv"

EXPLANATION_SENTS_FILE = BEAKER_DATA_FOLDER + 'explanation_datasets/sent_ids/expl-sentid-v1.tsv'

EMNLP_SENTID_EXPL_TEST_FILE = BEAKER_DATA_FOLDER + 'wiqa_datasets/emnlp19_with_sentid_expl/emnlp19.test.withSentIDExpl.json'
EMNLP_SENTID_EXPL_TEST_FILE_WITH_EQ = BEAKER_DATA_FOLDER + 'wiqa_datasets/emnlp19_with_sentid_expl/emnlp19.test.withSentIDExpl_with_eq.json'

##########################
# Explanation related paths
##########################

OPENPI_DATA_FOLDER = "openpi/lexicon/"
ATOMIC_PHYSICS_FROM_PROPARA = OPENPI_DATA_FOLDER + "atomic_propara.tsv"
ATOMIC_PHYSICS_FROM_PROPARA_XYZ = OPENPI_DATA_FOLDER + "atomic_propara_xyz.tsv"
ATOMIC_PHYSICS_FROM_ROCSTORIES = OPENPI_DATA_FOLDER + "atomic-physics-from-rocstories.tsv"
ROC_STORIES_PARA = OPENPI_DATA_FOLDER + "roc-stories-paragraphs.tsv"
ATOMIC_PHYSICS_LEXICON_PROPARA = OPENPI_DATA_FOLDER + "atomic-physics-lexicon-propara.tsv"
ATOMIC_PHYSICS_LEXICON_ROC = OPENPI_DATA_FOLDER + "atomic-physics-lexicon-roc.tsv"
EXTENDED_PROPARA_PARA = OPENPI_DATA_FOLDER + "propara-extended-para.tsv"
DATASET_JSONL = OPENPI_DATA_FOLDER + "dataset.jsonl"
ATOMIC_PHYSICS_PARA_ID = OPENPI_DATA_FOLDER + "para_id_file.tsv"

EMNLP_NO_EXPL = BEAKER_DATA_FOLDER + "wiqa_datasets/emnlp19_with_sentid_expl/emnlp19.test.old_format.json"

##########################
# Graph Edges
##########################

POSITIVE_EDGES = {'X': 'Y',
                  'W': 'D',
                  'U': 'A'}

NEGATIVE_EDGES = {'X': 'W',
                  'W': 'A',
                  'Y': 'D',
                  'U': 'Y'}

EXO_POS = {'Z': 'X'}
EXO_NEG = {'V': 'X'}

##########################
# State change Dict
##########################

STATE_CHANGE_DICT = {'x': '[unused1]',
                     'y': '[unused2]',
                     'z': '[unused3]'}


##########################
# EMNLP FILES
##########################

EMNLP_FOLDER = "wiqa_datasets/wiqa_emnlp_dataset/json/"
EMNLP_TRAIN_FILE = EMNLP_FOLDER + 'train.json'
EMNLP_DEV_FILE = EMNLP_FOLDER + 'dev.json'
EMNLP_TEST_FILE = EMNLP_FOLDER + 'test.json'
