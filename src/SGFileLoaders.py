import json
import os
from random import sample


def compile_input_files(dir_or_file_path):
    input_is_directory = os.path.isdir(dir_or_file_path)
    input_files = []
    if input_is_directory:
        input_dir = os.fsencode(dir_or_file_path)
        for infile_bytename in os.listdir(input_dir):
            infile_fullpath = os.path.join(
                input_dir.decode("utf-8"), infile_bytename.decode("utf-8"))
            input_files.append(infile_fullpath)
    else:
        input_files.append(dir_or_file_path)
    return input_files


def load_grpkey_to_qkeys(path, meta_info):
    '''
    {
       "group_key":"1,V,X,In the context of how does igneous rock form, suppose less magma is formed, it will not result in more magma is released",
       "question_keys":[
          "is_distractor^False:is_labeled_tgt^False:path_nodes^[V, X]:path_label^V->SituationLabel.NOT_RESULTS_IN->X:graph_id^1:id^34,1,1#0#0:tf_q_type^EXOGENOUS_EFFECT"
        ]
    }
    :param path: 
    :param meta_info: 
    :return: 
    '''
    grpkey_to_qkeys = {}
    for in_fp in compile_input_files(path):
        with open(in_fp, 'r') as infile:
            for line in infile:
                j = json.loads(line)
                grpkey_to_qkeys[j["group_key"]] = j["question_keys"]
    return grpkey_to_qkeys


def load_qkey_to_grpkey(path, meta_info):
    '''
    {
       "group_key":"1,V,X,In the context of how does igneous rock form, suppose less magma is formed, it will not result in more magma is released",
       "question_keys":[
          "is_distractor^False:is_labeled_tgt^False:path_nodes^[V, X]:path_label^V->SituationLabel.NOT_RESULTS_IN->X:graph_id^1:id^34,1,1#0#0:tf_q_type^EXOGENOUS_EFFECT"
        ]
    }
    :param path: 
    :param meta_info: 
    :return: 
    '''
    qkey_to_grpkey = {}
    for in_fp in compile_input_files(path):
        with open(in_fp, 'r') as infile:
            for line in infile:
                j = json.loads(line)
                grp_key = j["group_key"]
                for q_key in j["question_keys"]:
                    qkey_to_grpkey[q_key] = grp_key
    return qkey_to_grpkey


def load_qkey_to_qjson(path, meta_info):
    '''
    {
       "question":{
          "stem":"In the context of how does igneous rock form, suppose less magma is formed, it will not result in more magma is released",
          "choices":[
             {
                "text":"True",
                "label":"C"
             },
             {
                "text":"False",
                "label":"D"
             }
          ]
       },
       "answerKey":"C",
       "explanation":"less magma is formed=>more magma is released",
       "path_info":"is_distractor^False:is_labeled_tgt^False:path_nodes^[V, X]:path_label^V->SituationLabel.NOT_RESULTS_IN->X",
       "more_info":{
          "tf_q_type":"EXOGENOUS_EFFECT",
          "prompt":"How does igneous rock form?",
          "para_id":"34",
          "group_ids":{
             "NO_GROUPING":"is_distractor^False:is_labeled_tgt^False:path_nodes^[V, X]:path_label^V->SituationLabel.NOT_RESULTS_IN->X:graph_id^1:id^34,1,1#0#0:tf_q_type^EXOGENOUS_EFFECT",
             "BY_SRC_DEST":"V,X",
             "BY_SRC_LABEL_DEST":"V->SituationLabel.NOT_RESULTS_IN->X",
             "BY_PROMPT":"How does igneous rock form?",
             "BY_PARA":"34",
             "BY_FULL_PATH":"[V, X]",
             "BY_GROUNDING":"34,1,1,0",
             "BY_SRC_DEST_INTRA":"1,V,X",
             "BY_SRC_DEST_STEM_INTRA":"1,V,X,In the context of how does igneous rock form, suppose less magma is formed, it will not result in more magma is released"
          },
          "all_q_keys":[
             "is_distractor^False:is_labeled_tgt^False:path_nodes^[V, X]:path_label^V->SituationLabel.NOT_RESULTS_IN->X:graph_id^1:id^34,1,1#0#0:tf_q_type^EXOGENOUS_EFFECT"
          ]
       },
       "para":"Volcanos contain magma. The magma is very hot. The magma rises toward the surface of the volcano. The magma cools. The magma starts to harden as it cools. The magma is sometimes released from the volcano as lava. The magma or lava becomes a hard rock as it solidifies.",
       "graph_id":"1",
       "id":"34:1:1#0#0",
       "primary_question_key":"is_distractor^False:is_labeled_tgt^False:path_nodes^[V, X]:path_label^V->SituationLabel.NOT_RESULTS_IN->X:graph_id^1:id^34,1,1#0#0:tf_q_type^EXOGENOUS_EFFECT"
    }
    :param path: 
    :param meta_info: 
    :return: 
    '''
    qkey_to_qjson = {}
    for in_fp in compile_input_files(path):
        with open(in_fp, 'r') as infile:
            for line in infile:
                j = json.loads(line)
                qkey_to_qjson[j["primary_question_key"]] = j
    return qkey_to_qjson


def load_qkey_to_ans(path, meta_info):
    '''
    {
       "id":"34:1:1#0#0",
       "primary_question_key":"is_distractor^False:is_labeled_tgt^False:path_nodes^[V, X]:path_label^V->SituationLabel.NOT_RESULTS_IN->X:graph_id^1:id^34,1,1#0#0:tf_q_type^EXOGENOUS_EFFECT",
       "answerKey":"C"
    }
    :param path: 
    :param meta_info: 
    :return: 
    '''
    qkey_to_ans = {}
    for in_fp in compile_input_files(path):
        with open(in_fp, 'r') as infile:
            for line in infile:
                j = json.loads(line)
                qkey_to_ans[j["primary_question_key"]] = j["answerKey"]
    return qkey_to_ans


def load_allennlp_qkey_to_ans(path, qkey_to_qjson_map, meta_info):
    '''
    {
      "id":"131",
      "question":"NA",
      "question_text":"It is observed that scavengers increase in number happens. In the context of How are fossils formed? , what is the least likely cause?",
      "choice_text_list":[
         "habitat is destroyed",
         "humans hunt less animals",
         "",
         ""
      ],
      "correct_answer_index":0,
      "label_logits":[
         -6.519074440002441,
         -6.476490497589111,
         -6.935580730438232,
         -6.935580730438232
      ],
      "label_probs":[
         0.29742464423179626,
         0.31036368012428284,
         0.19610585272312164,
         0.19610585272312164
      ],
      "answer_index":1
    }
    :param meta_info: 
    :return: 
    '''
    # step 1 : map allennlp_qkey_to_qkey
    # step 2 : existing functionality then.
    qkey_from_allennlp_key_map = {v["id"]: k for k, v in qkey_to_qjson_map.items()}
    qkey_to_ans = {}
    for in_fp in compile_input_files(path):
        with open(in_fp, 'r') as infile:
            for line in infile:
                j = json.loads(line)
                makeshift_id = j["id"]
                qkey_from_allennlp_key = qkey_from_allennlp_key_map[makeshift_id]
                qkey_to_ans[qkey_from_allennlp_key] = j["choice_text_list"][int(j["answer_index"])]
    print(f"Loaded {len(qkey_to_ans)} system answers from {path}")
    return qkey_to_ans


def serialize_whole(out_file_path, items, randomize_order=False, header=""):
    print(f"Writing {len(items)} items (e.g., question jsons) to directory: {out_file_path}")
    curr_file = open(out_file_path, 'w')
    if header:
        curr_file.write(header)
        if "\n" not in header:
            curr_file.write("\n")

    items_randomized = items
    if randomize_order:
        items_randomized = sample(items, len(items))
    for item_num, item in enumerate(items_randomized):
        curr_file.write(item)
        if "\n" not in item:
            curr_file.write("\n")

    curr_file.close()


def serialize_in_pieces(out_dir_path, max_items_in_a_piece, items, randomize_order=False, header=""):
    print(f"Writing {len(items)} items (e.g., question jsons) to directory: {out_dir_path}")
    if not os.path.exists(out_dir_path):
        os.makedirs(out_dir_path)

    curr_file_num = 1
    curr_file = open(f"{out_dir_path}/{curr_file_num}.json", 'w')
    if header:
        curr_file.write(header)

    items_randomized = items
    if randomize_order:
        items_randomized = sample(items, len(items))

    for item_num, item in enumerate(items_randomized):
        if item_num % max_items_in_a_piece == 0 and item_num > 1:
            # close the old file.
            curr_file.close()
            # open a new file.
            curr_file_num += 1
            curr_file = open(f"{out_dir_path}/{curr_file_num}.json", "w")
            if header:
                curr_file.write(header)
                if "\n" not in header:
                    curr_file.write("\n")

        curr_file.write(item)
        if "\n" not in item:
            curr_file.write("\n")

    if not curr_file.closed:
        curr_file.close()
