"""
This file contains the latest dataset related files
for EMNLP2019
"""

from collections import namedtuple

from src.third_party_utils.allennlp_cached_filepath import cached_path

DatasetInfo = namedtuple('DatasetInfo',
                         ['beaker_link', 'cloud_path', 'local_path', 'data_reader', 'metadata_info', 'readme'])

# EMNLP dataset: "with explanations" in json format.
wiqa_explanations_v1 = DatasetInfo(
    beaker_link="https://allenai.beaker.org/ds/ds_jsdpme8ixz86/",  # an unpublished, baseline model.
    cloud_path="https://public-aristo-processes.s3-us-west-2.amazonaws.com/wiqa_dataset_with_explanation/",
    # [cloud_path + partition for partition in ["train.jsonl", "dev.jsonl", "test.jsonl"]]
    local_path="",
    data_reader="BertMCQAReaderSentAnnotated",  # an unpublished, baseline model.
    metadata_info="WhatifMetadata",
    readme=""
)

# EMNLP dataset: "no explanations" in json format.
wiqa_no_explanations_vetted_v1 = DatasetInfo(
    beaker_link="https://allenai.beaker.org/ds/ds_zyukvhb9ezqa/",
    cloud_path="https://public-aristo-processes.s3-us-west-2.amazonaws.com/wiqa_dataset_no_explanation/",
    # [cloud_path + partition for partition in ["train.jsonl", "dev.jsonl", "test.jsonl"]]
    local_path="",
    data_reader="BertMCQAReaderPara",
    metadata_info="WhatifMetadata",
    readme=""
)

# All the turked influence graphs in json format.
influence_graphs_v1 = DatasetInfo(
    beaker_link="",
    cloud_path="https://public-aristo-processes.s3-us-west-2.amazonaws.com/metadata/shard123.para.ifthenBlocks.outcomes.json",
    local_path="",
    data_reader="SituationGraph",
    metadata_info="WhatifMetadata",
    readme="All influence graphs"
)

# Train/dev/test in ProPara dataset is partitioned by topics. Every topic in test is therefore, novel.
# In the WIQA dataset, we continue partitioning by topic. The following data contains para id and and partition.
para_partition_info = DatasetInfo(
    beaker_link="",
    cloud_path="https://public-aristo-processes.s3-us-west-2.amazonaws.com/metadata/para_id.prompt.topic.partition.tsv",
    local_path="",
    data_reader="",
    metadata_info="WhatifMetadata",
    readme="Para id, topic, and the partition type"
)

# The ProPara dataset (includes paragraph id and paragraph sentences (incl. metadata: paragraph prompt/title & topic)
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
