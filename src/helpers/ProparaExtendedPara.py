import csv
import json
from typing import List, Dict

from src.helpers.dataset_info import propara_para_info, download_from_url_if_not_in_cache


class ProparaExtendedParaEntry:
    def __init__(self, topic: str, prompt: str, paraid: str,
                 s1: str, s2: str, s3: str,
                 s4: str, s5: str, s6: str,
                 s7: str, s8: str, s9: str,
                 s10: str):
        self.topic = topic
        self.prompt = prompt
        self.paraid = paraid
        all_sentences = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10]
        self.sentences = [x for x in all_sentences if x]

    def sent_at(self, sentidx_startingzero: int):
        assert len(self.sentences) > sentidx_startingzero
        return self.sentences[sentidx_startingzero]

    def get_sentence_arr(self):
        return self.sentences

    def as_json(self):
        return json.dumps(self.__dict__)

    @staticmethod
    def from_json(j):
        raise NotImplementedError("from json is not done yet.")

    @staticmethod
    def from_tsv(t):
        return ProparaExtendedParaEntry(*t)


class ProparaExtendedParaMetadata:

    def __init__(self,
                 extended_propara_para_fp=download_from_url_if_not_in_cache(
                     propara_para_info.cloud_path)):
        self.para_map: Dict[str, ProparaExtendedParaEntry] = dict()
        for row_num, row_as_arr in enumerate(csv.reader(open(extended_propara_para_fp), delimiter="\t")):
            if row_num > 0:  # skips header
                e: ProparaExtendedParaEntry = ProparaExtendedParaEntry.from_tsv(row_as_arr)
                self.para_map[e.paraid] = e

    def paraentry_for_id(self, para_id: str) -> ProparaExtendedParaEntry:
        return self.para_map.get(para_id, None)

    def sentences_for_paraid(self, paraid: str) -> List[str]:
        return self.paraentry_for_id(para_id=paraid).get_sentence_arr()

    def sent_at(self, paraid: str, sentidx_startingzero: int) -> str:
        return self.paraentry_for_id(para_id=paraid).sent_at(sentidx_startingzero=sentidx_startingzero)


if __name__ == '__main__':
    o = ProparaExtendedParaMetadata()
    assert o.sent_at(paraid="2453",
                     sentidx_startingzero=1) == "The heat from the reaction is used to create steam in water."
    assert o.sent_at(paraid="24",
                     sentidx_startingzero=2) == "Depending on what is eroding the valley, the slope of the land, the type of rock or soil and the amount of time the land has been eroded."
    assert not o.sent_at(paraid="24",
                         sentidx_startingzero=2) == "The heat from the reaction is used to create steam in water."
