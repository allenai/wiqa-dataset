import csv
import json


from src.helpers import SGFileLoaders, collections_util
from src.helpers.collections_util import add_key_to_map_arr
from src.helpers.situation_graph import SituationGraph


class WhatifMetadata:
    def __init__(self,
                 para_ids_metainfo_fp: str,
                 situation_graphs_fp: str):
        self.graph_id_to_graph = dict()
        self.graph_id_to_graphjson = dict()
        self.para_id_to_graphids = dict()
        self.prompt_to_paraids = dict()
        self.topic_to_paraids = dict()
        self.topic_to_graph_ids = dict()
        self.paraid_to_topic = dict()
        self.paraid_to_para = dict()
        self.paraid_to_partition = dict()
        self.topic_to_partition = dict()

        for in_fp in SGFileLoaders.compile_input_files(situation_graphs_fp):
            with open(in_fp) as infile:
                for line in infile:
                    j = json.loads(line)
                    g = SituationGraph.from_struct_v1(struct=j)
                    graph_id = j["graph_id"]
                    self.graph_id_to_graphjson[graph_id] = j
                    self.graph_id_to_graph[graph_id] = g
                    para_id = j["para_id"]
                    paragraph = j["paragraph"]
                    self.paraid_to_para[para_id] = paragraph
                    # k = self.get_topic_for_paraid(para_id=para_id)
                    # if not k:
                    #     # FIXME This is to make the code work under a temporary bug of ids with prefix propara_
                    #     para_id = "propara_" + para_id
                    #     k = self.get_topic_for_paraid(para_id=para_id)

                    add_key_to_map_arr(key=para_id,
                                       value=graph_id,
                                       map_=self.para_id_to_graphids)
                    # add_key_to_map_arr(key=k,
                    #                    value=graph_id,
                    #                    map_=self.topic_to_graph_ids)

        for in_fp in SGFileLoaders.compile_input_files(para_ids_metainfo_fp):
            # para_id, prompt,  topic,   partition
            with open(in_fp) as infile:
                reader = csv.DictReader(infile, delimiter='\t')
                for row in reader:
                    para_id = row["para_id"]
                    # FIXME temp fix due to id bug that inconsisently contains propara_ prefix sometimes.
                    if "propara_" in para_id:
                        para_id = para_id.replace("propara_", "")
                    # Cleanup paraids from partition for which we have no graph ids.
                    # This is a dataset issue that causes runtime errors if not cleaned up.
                    if para_id in self.paraid_to_para.keys():
                        topic = row["topic"]
                        add_key_to_map_arr(key=row["prompt"], value=para_id, map_=self.prompt_to_paraids)
                        add_key_to_map_arr(key=topic, value=para_id, map_=self.topic_to_paraids)
                        self.paraid_to_topic[para_id] = topic
                        self.paraid_to_partition[para_id] = row["partition"]
                        self.topic_to_partition[topic] = row["partition"]
                        for graph_id in self.para_id_to_graphids[para_id]:
                            # Repeated entries but maps takes care of it.
                            # Moving it to previous for loop block from sg file creates problems because
                            # some para ids are present in prompt file but absent in graphs file.
                            add_key_to_map_arr(key=topic,
                                               value=graph_id,
                                               map_=self.topic_to_graph_ids)

    def get_paraids_for_prompt(self, prompt_str):
        assert prompt_str in self.prompt_to_paraids, f"no paraid present {prompt_str} in prompt to paraid"
        return [] if prompt_str not in self.prompt_to_paraids else self.prompt_to_paraids[prompt_str]

    def get_graph_for_id(self, graph_id) -> SituationGraph:
        assert graph_id in self.graph_id_to_graph, f"no graphid present {graph_id} in graphid to graph"
        return self.graph_id_to_graph[graph_id]

    def get_graphjson_for_id(self, graph_id) -> SituationGraph:
        assert graph_id in self.graph_id_to_graphjson, f"no graphid present {graph_id} in graphid to graphjson"
        return self.graph_id_to_graphjson[graph_id]

    def get_paraids_for_topic(self, topic_str):
        assert topic_str in self.topic_to_paraids, f"no topic present {topic_str} in paraids for topic"
        return [] if topic_str not in self.topic_to_paraids else self.topic_to_paraids[topic_str]

    def get_graphids_for_topic(self, topic_str):
        assert topic_str in self.topic_to_graph_ids, f"no topic present {topic_str} in topic_to_graph_ids"
        return [] if topic_str not in self.topic_to_graph_ids else self.topic_to_graph_ids[topic_str]

    def get_topic_for_graphid(self, graph_id):
        para_id = self.get_graph_for_id(graph_id=graph_id).other_properties["para_id"]
        return self.get_topic_for_paraid(para_id=para_id)

    def get_topic_for_paraid(self, para_id):
        assert para_id in self.paraid_to_topic, f"no paraid present {para_id} in paraid to topic"
        return "" if para_id not in self.paraid_to_topic else self.paraid_to_topic[para_id]

    def get_para_for_paraid(self, para_id):
        assert para_id in self.paraid_to_para, f"no paraid present {para_id} in paraid to para"
        return "" if para_id not in self.paraid_to_para else self.paraid_to_para[para_id]

    def get_partition_for_graphid(self, graph_id):
        para_id = self.get_graph_for_id(graph_id=graph_id).other_properties["para_id"]
        return self.get_partition_for_paraid(para_id=para_id)

    def get_partition_for_paraid(self, para_id):
        assert para_id in self.paraid_to_partition, f"no paraid present {para_id} in paraid to partition map"
        return "" if para_id not in self.paraid_to_partition else self.paraid_to_partition[para_id]

    def get_partition_for_prompt(self, prompt):
        paraid = collections_util.getElem(arr=self.get_paraids_for_prompt(prompt_str=prompt), elem_idx=0,
                                          defaultValue="")
        return self.get_partition_for_paraid(para_id=paraid)

    def get_partition_for_topic(self, topic):
        assert topic in self.topic_to_partition, f"topic not present: {topic} in topic to partition map"
        return self.topic_to_partition[topic]

    def get_all_topics(self):
        # Note that self.topic_to_paraids.keys() would provide ALL paragraphs, not just those
        #      for which we have situation graphs.
        return self.topic_to_graph_ids.keys()

    def get_all_topics_in_partition(self, partition_reqd):
        # Note that self.topic_to_paraids.keys() would provide ALL paragraphs, not just those
        #      for which we have situation graphs.
        return [x for x, partition in self.topic_to_partition.items() if partition == partition_reqd]

    def get_graphids_for_paraid(self, para_id):
        assert para_id in self.para_id_to_graphids, f"no paraid present {para_id} in para->graphids map"
        return self.para_id_to_graphids[para_id]

    def get_paraid_for_graph(self, graph):
        return graph.other_properties["para_id"]

    def get_paraid_for_graphid(self, graph_id):
        return self.get_graph_for_id(graph_id=graph_id).other_properties["para_id"]

    def get_prompt_for_graphid(self, graph_id):
        return self.get_graph_for_id(graph_id=graph_id).other_properties["prompt"]

    def get_paragraph_for_graphid(self, graph_id):
        return self.get_graph_for_id(graph_id=graph_id).other_properties["paragraph"]

    def get_paragraph_for_paraid(self, para_id):
        return self.paraid_to_para[para_id]

    def get_graph_nodes_as_text(self, graph_id):
        '''
        Text = Bag of groundings from all nodes in the graph. Lowercases the resulting text
        :param graph_id: 
        :return: 
        '''
        graph_as_text = " ".join([" ".join(x.groundings) for x in self.get_graph_for_id(graph_id=graph_id).nodes])
        return graph_as_text.lower()

    def get_all_graphids(self):
        return self.graph_id_to_graph.keys()

    def get_all_graphids_for_partition(self, partition):
        '''
        
        :param partition: train, dev, test 
        :return: all_graphids (list of str) in that partition 
        '''
        graph_ids = []
        for topic in self.get_all_topics_in_partition(partition_reqd=partition):
            graph_ids.extend(self.get_graphids_for_topic(topic_str=topic))
        return graph_ids
