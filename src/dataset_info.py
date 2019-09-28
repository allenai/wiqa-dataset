"""
This file contains the latest dataset related files
for AAAI20
"""

from collections import namedtuple

from src.cached_filepath import cached_path

DatasetInfo = namedtuple('DatasetInfo',
                         ['beaker_link', 'cloud_path', 'local_path', 'data_reader', 'metadata_info', 'readme'])

# This is the model that takes c_q, e_q, ig, para as input
# and returns predicted_class and explanation as ig path
wiqa_explanations_v1 = DatasetInfo(
    beaker_link="https://allenai.beaker.org/ds/ds_jsdpme8ixz86/",
    cloud_path="https://public-aristo-processes.s3-us-west-2.amazonaws.com/wiqa_dataset_with_explanation/",
    # cloud_files=[cloud_path + partition for partition in ["train.jsonl", "dev.jsonl", "test.jsonl"]],
    local_path="",
    data_reader="BertMCQAReaderSentAnnotated",
    metadata_info="WhatifMetadata",
    readme=""
)

wiqa_no_explanations_vetted_v1 = DatasetInfo(
    beaker_link="https://allenai.beaker.org/ds/ds_zyukvhb9ezqa/",
    cloud_path="https://public-aristo-processes.s3-us-west-2.amazonaws.com/wiqa_dataset_no_explanation/",
    local_path="",
    data_reader="BertMCQAReaderPara",
    metadata_info="WhatifMetadata",
    readme=""
)

wiqa_no_explanations_unvetted_v1 = DatasetInfo(
    beaker_link="https://allenai.beaker.org/ds/ds_7flv296th0vm/",
    cloud_path="",
    local_path="",
    data_reader="BertMCQAReaderPara",
    metadata_info="WhatifMetadata",
    readme=""
)

influence_graphs_v1 = DatasetInfo(
    beaker_link="",
    cloud_path="https://public-aristo-processes.s3-us-west-2.amazonaws.com/metadata/shard123.para.ifthenBlocks.outcomes.json",
    local_path="",
    data_reader="SituationGraph",
    metadata_info="WhatifMetadata",
    readme="All influence graphs with edge information: https://github.com/allenai/wiqa/blob/port_processes_code/wiqa_datasets/propara_splits/turker_task_outputs/shard123.graphs_with_edges.json"
)

para_partition_info = DatasetInfo(
    beaker_link="",
    cloud_path="https://public-aristo-processes.s3-us-west-2.amazonaws.com/metadata/para_id.prompt.topic.partition.tsv",
    local_path="",
    data_reader="",
    metadata_info="WhatifMetadata",
    readme="Para id, topic, and the partition type"
)

propara_para_info = DatasetInfo(
    beaker_link="",
    cloud_path="https://public-aristo-processes.s3-us-west-2.amazonaws.com/metadata/propara-extended-para.tsv",
    local_path="",
    data_reader="",
    metadata_info="WhatifMetadata",
    readme="Para id, topic, and the sentences"
)


def download_from_url_if_not_in_cache(cloud_path: str, cache_dir: str = None):
    """
    :param cloud_path: e.g., https://public-aristo-processes.s3-us-west-2.amazonaws.com/wiqa-model.tar.gz
    :param to_dir: will be regarded as a cache.
    :return: the path of file to which the file is downloaded.
    """
    return cached_path(url_or_filename=cloud_path, cache_dir=cache_dir)
