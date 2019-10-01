import argparse
import enum
import json
from typing import List, Dict

from src.eval.eval_utils import get_label_from_question_metadata, split_question_cause_effect, \
    predict_word_overlap_best, find_max
from src.helpers.dataset_info import wiqa_explanations_v1
from src.helpers.situation_graph import SituationLabel
from src.wiqa_wrapper import WIQAQuesType, WIQAUtils, download_from_url_if_not_in_cache, Jsonl


class FineEvalMetrics(enum.Enum):
    EQDIR = 0
    XSENTID = 1
    YSENTID = 2
    XDIR = 3

    def to_json(self):
        return self.name


class Precision:
    def __init__(self):
        self.tp = 0
        self.fp = 0

    def __str__(self):
        if self.tp + self.fp == 0:
            return "0"
        else:
            return self.tp / (self.tp + self.fp)

    def p(self):
        if self.tp + self.fp == 0.0:
            return 0.0
        else:
            return 1.0 * self.tp / (1.0 * (self.tp + self.fp))


class InputRequiredByFineEval:
    def __init__(self,
                 graph_id: str,
                 ques_type: WIQAQuesType,
                 path_arr: List[str],
                 metrics: Dict[FineEvalMetrics, bool]):
        self.graph_id = graph_id
        self.path_arr = path_arr
        self.ques_type = ques_type
        self.metrics = metrics

    def get_path_len(self):
        return len(self.path_arr)

    def __str__(self):
        return f"graphid={self.graph_id}\nques_type={self.ques_type.name}" \
               f"path_arr={self.path_arr}\nmetrics={self.metrics}"

    @staticmethod
    def from_outputjson_noexplanation_model(output_json_dict):

        # get expected answer.
        # get qtype and path metadata about the question.
        q = output_json_dict["metadata"]["question"]
        graph_id = q["graph_id"]
        fields = {x.split("^")[0]: x.split("^")[1] for x in q["path_info"].split(":")}
        qtype = WIQAQuesType.from_str(q["more_info"]["tf_q_type"])
        path_arr = fields["path_nodes"].replace("[", "").replace("]", "").split(",")

        # default values.
        eqdir = SituationLabel.from_str(output_json_dict['label'])
        expected_eqdir_str = get_label_from_question_metadata(
            correct_answer_key=output_json_dict['metadata']['question']['answerKey'],
            question_dict=output_json_dict['metadata']['question']['question']['choices'])
        expected_eqdir = SituationLabel.from_str(expected_eqdir_str)

        metrics = {}
        metrics[FineEvalMetrics.EQDIR] = expected_eqdir == eqdir
        return InputRequiredByFineEval(graph_id=graph_id,
                                       ques_type=qtype,
                                       path_arr=path_arr,
                                       metrics=metrics)

    @staticmethod
    def from_outputjson_emnlp_model(output_json_dict, extra_args):

        eqdir = SituationLabel.from_str(output_json_dict['choice_text_list'][output_json_dict['correct_answer_index']])
        expected_eqdir = SituationLabel.from_str(output_json_dict['choice_text_list'][output_json_dict['answer_index']])

        metrics = {}
        metrics[FineEvalMetrics.EQDIR] = expected_eqdir == eqdir
        fields = {x.split("^")[0]: x.split("^")[1] for x in extra_args["path_arr"].split(":")}
        qtype = WIQAQuesType.from_str(extra_args["qtype"])
        path_arr = fields["path_nodes"].replace("[", "").replace("]", "").split(",")
        return InputRequiredByFineEval(graph_id=extra_args["graph_id"],
                                       ques_type=qtype,
                                       path_arr=path_arr,
                                       metrics=metrics)


    @staticmethod
    def from_outputjson_wordoverlap_baseline_model(output_json_line):
        # WordOverlapForSentID
        cq, eq = split_question_cause_effect(output_json_line['question']['question']['stem'])
        steps = WIQAUtils.get_split_para(output_json_line['explanation']['steps'])
        predicted_xsent_id, predicted_ysent_id = predict_word_overlap_best(input_steps=steps, input_cq=cq, input_eq=eq)
        expected_xsent_id = output_json_line['explanation']['x_sent_id']  # starts from 0
        expected_ysent_id = output_json_line['explanation']['y_sent_id']  # starts from 0
        metrics = {}
        metrics[FineEvalMetrics.XSENTID] = expected_xsent_id == predicted_xsent_id
        metrics[FineEvalMetrics.YSENTID] = expected_ysent_id == predicted_ysent_id
        graph_id = output_json_line['question']["graph_id"]
        fields = {x.split("^")[0]: x.split("^")[1] for x in output_json_line['question']["path_info"].split(":")}
        qtype = WIQAQuesType.from_str(output_json_line['question']["more_info"]["tf_q_type"])
        path_arr = fields["path_nodes"].replace("[", "").replace("]", "").split(",")
        return InputRequiredByFineEval(graph_id=graph_id,
                                       ques_type=qtype,
                                       path_arr=path_arr,
                                       metrics=metrics)

    # returns InputRequiredByFineEval
    @staticmethod
    def from_outputjson_sentid_model(output_json_line):

        # get expected answer.
        expected = output_json_line["metadata"]["explanation"]
        # get qtype and path metadata about the question.
        q = output_json_line["metadata"]["question"]
        graph_id = q["graph_id"]
        fields = {x.split("^")[0]: x.split("^")[1] for x in q["path_info"].split(":")}
        qtype = WIQAQuesType.from_str(q["more_info"]["tf_q_type"])
        path_arr = fields["path_nodes"].replace("[", "").replace("]", "").split(",")
        m = output_json_line["metadata"]["explanation"]

        # default values.
        eqdir = SituationLabel.from_str(output_json_line['label'])
        xdir = SituationLabel.from_str(output_json_line['xdir_label'])
        xidx = -1 if eqdir == SituationLabel.NO_EFFECT or xdir == SituationLabel.NO_EFFECT else find_max(
            output_json_line['x_sent_logits'])
        yidx = -1 if eqdir == SituationLabel.NO_EFFECT or xdir == SituationLabel.NO_EFFECT else find_max(
            output_json_line['y_sent_logits'])

        metrics = {}
        metrics[FineEvalMetrics.EQDIR] = SituationLabel.from_str(expected["eq_dir_orig_answer"]) == eqdir
        metrics[FineEvalMetrics.XDIR] = SituationLabel.from_str(expected["x_dir"]) == xdir
        metrics[FineEvalMetrics.XSENTID] = int(expected["x_sent_id"]) == xidx
        metrics[FineEvalMetrics.YSENTID] = int(expected["y_sent_id"]) == yidx
        return InputRequiredByFineEval(graph_id=graph_id,
                                       ques_type=qtype,
                                       path_arr=path_arr,
                                       metrics=metrics)

    # {"logits": [[0.7365949749946594, 0.16483880579471588, 0.38572263717651367, 0.33268705010414124, -0.6508089900016785, -0.950057864189148], [0.3361547291278839, -0.02215845137834549, 0.8630506992340088, 0.4753769040107727, -1.0981523990631104, 0.04984292760491371], [0.4498467743396759, 0.26323091983795166, 0.5597160458564758, 0.06369128823280334, -0.33793506026268005, -0.30190590023994446], [0.41394802927970886, 0.31742218136787415, 0.42982375621795654, -0.2891058027744293, -0.09577881544828415, -0.4486318528652191], [0.5242481231689453, -0.05186435207724571, 0.4505387544631958, -0.43092456459999084, -0.015227549709379673, 0.10361793637275696], [0.8527745604515076, 0.18845966458320618, 0.6540948748588562, -0.06324845552444458, -0.03267676383256912, 0.058296892791986465], [0.40418609976768494, -0.24220454692840576, 0.0737631767988205, -0.8445389270782471, -0.12929767370224, 0.5813987851142883]], "class_probabilities": [[0.29663264751434326, 0.16745895147323608, 0.20885121822357178, 0.19806326925754547, 0.07407592236995697, 0.054918017238378525], [0.18079228699207306, 0.12634745240211487, 0.3062019348144531, 0.20779892802238464, 0.04307926073670387, 0.13578014075756073], [0.21968600153923035, 0.18228720128536224, 0.24519862234592438, 0.14931286871433258, 0.09992476552724838, 0.10359060019254684], [0.22513438761234283, 0.20441898703575134, 0.22873708605766296, 0.11145754158496857, 0.13522914052009583, 0.09502287209033966], [0.24298663437366486, 0.13657772541046143, 0.2257203906774521, 0.09348806738853455, 0.14167429506778717, 0.15955300629138947], [0.27786338329315186, 0.1429957151412964, 0.22779585421085358, 0.11117511987686157, 0.11462641507387161, 0.12554346024990082], [0.23202574253082275, 0.1215660497546196, 0.16673828661441803, 0.06656130403280258, 0.1360965520143509, 0.27701207995414734]], "metadata": {"question": {"question": {"stem": "suppose during boiling point happens, how will it affect more evaporation.", "choices": [{"text": "Correct effect", "label": "A"}, {"text": "Opposite effect", "label": "B"}, {"text": "No effect", "label": "C"}]}, "answerKey": "A", "explanation": "  ['during boiling point', 'during sunshine'] ==> ['increase water temperatures at least 100 C'] ==> ['more vapors'] ==> ['MORE evaporation?']", "path_info": "is_distractor^False:is_labeled_tgt^True:path_nodes^[Z, X, Y, A]:path_label^Z->SituationLabel.RESULTS_IN->A", "more_info": {"tf_q_type": "EXOGENOUS_EFFECT", "prompt": "Describe the process of evaporation", "para_id": "127", "group_ids": {"NO_GROUPING": "is_distractor^False:is_labeled_tgt^True:path_nodes^[Z, X, Y, A]:path_label^Z->SituationLabel.RESULTS_IN->A:graph_id^12:id^influence_graph,127,12,39#0:tf_q_type^EXOGENOUS_EFFECT", "BY_SRC_DEST": "Z,A", "BY_SRC_LABEL_DEST": "Z->SituationLabel.RESULTS_IN->A", "BY_PROMPT": "Describe the process of evaporation", "BY_PARA": "127", "BY_FULL_PATH": "[Z, X, Y, A]", "BY_GROUNDING": "influence_graph,127,12,39", "BY_SRC_DEST_INTRA": "12,Z,A", "BY_SRC_DEST_STEM_INTRA": "12,Z,A,In the context of describe the process of evaporation, suppose during boiling point happens, how will it affect MORE evaporation?.", "BY_TF_Q_TYPE": "EXOGENOUS_EFFECT", "BY_PATH_LENGTH": 4}, "all_q_keys": ["is_distractor^False:is_labeled_tgt^True:path_nodes^[Z, X, Y, A]:path_label^Z->SituationLabel.RESULTS_IN->A:graph_id^12:id^influence_graph,127,12,39#0:tf_q_type^EXOGENOUS_EFFECT", "is_distractor^False:is_labeled_tgt^True:path_nodes^[Z, X, W, A]:path_label^Z->SituationLabel.RESULTS_IN->A:graph_id^12:id^influence_graph,127,12,38#0:tf_q_type^EXOGENOUS_EFFECT"]}, "para": "Water is exposed to heat energy, like sunlight. The water temperature is raised above 212 degrees fahrenheit. The heat breaks down the molecules in the water. These molecules escape from the water. The water becomes vapor. The vapor evaporates into the atmosphere. ", "graph_id": "12", "para_id": "127", "prompt": "Describe the process of evaporation", "id": "influence_graph:127:12:39#0", "distractor_info": {}, "primary_question_key": "is_distractor^False:is_labeled_tgt^True:path_nodes^[Z, X, Y, A]:path_label^Z->SituationLabel.RESULTS_IN->A:graph_id^12:id^influence_graph,127,12,39#0:tf_q_type^EXOGENOUS_EFFECT"}, "kg": [{"from_node": "increase water temperatures at least 100 C", "to_node": "more ice", "label": "NOT_RESULTS_IN"}, {"from_node": "increase water temperatures at least 100 C", "to_node": "less sweating", "label": "NOT_RESULTS_IN"}, {"from_node": "increase water temperatures at least 100 C", "to_node": "more vapors", "label": "RESULTS_IN"}, {"from_node": "low water temperatures at most 100 C", "to_node": "more vapors", "label": "NOT_RESULTS_IN"}, {"from_node": "less water molecule colliding", "to_node": "more vapors", "label": "NOT_RESULTS_IN"}, {"from_node": "more ice", "to_node": "MORE evaporation?", "label": "NOT_RESULTS_IN"}, {"from_node": "less sweating", "to_node": "MORE evaporation?", "label": "NOT_RESULTS_IN"}, {"from_node": "more ice", "to_node": "LESS evaporation", "label": "RESULTS_IN"}, {"from_node": "less sweating", "to_node": "LESS evaporation", "label": "RESULTS_IN"}, {"from_node": "more vapors", "to_node": "MORE evaporation?", "label": "RESULTS_IN"}, {"from_node": "more vapors", "to_node": "LESS evaporation", "label": "NOT_RESULTS_IN"}], "explanation": {"steps": "(1. Water is exposed to heat energy, like sunlight. 2. The water temperature is raised above 212 degrees fahrenheit. 3. The heat breaks down the molecules in the water. 4. These molecules escape from the water. 5. The water becomes vapor. 6. The vapor evaporates into the atmosphere. 7. .)", "x_sent_id": 1, "y_sent_id": 4, "x_dir": "RESULTS_IN", "y_dir": "RESULTS_IN", "x_grounding": "increase water temperatures at least 100 C", "y_grounding": "more vapors", "is_valid_tuple": true, "eq_dir_orig_answer": "RESULTS_IN"}, "id": "NA"}, "tags": ["O", "E+", "E+", "E+", "O", "O", "E-"]}
    @staticmethod
    def from_outputjson_tagging(output_json_line):
        metrics = {}

        # get expected answer.
        expected = output_json_line["metadata"]["explanation"]
        # get qtype and path metadata about the question.
        q = output_json_line["metadata"]["question"]
        graph_id = q["graph_id"]
        fields = {x.split("^")[0]: x.split("^")[1] for x in q["path_info"].split(":")}
        qtype = WIQAQuesType.from_str(q["metadata"]["question_type"])
        path_arr = fields["path_nodes"].replace("[", "").replace("]", "").split(",")

        # default values.
        eqdir = SituationLabel(output_json_line['explanation']['de'])
        xdir = SituationLabel.from_str(output_json_line['label_x_dir'])
        xidx = -1 if eqdir == SituationLabel.NO_EFFECT or xdir == SituationLabel.NO_EFFECT else output_json_line[
            'x_sent_id']
        yidx = -1 if eqdir == SituationLabel.NO_EFFECT or xdir == SituationLabel.NO_EFFECT else output_json_line[
            'y_sent_id']

        metrics[FineEvalMetrics.EQDIR] = SituationLabel.from_str(expected["eq_dir_orig_answer"]) == eqdir
        metrics[FineEvalMetrics.XDIR] = SituationLabel.from_str(expected["x_dir"]) == xdir
        metrics[FineEvalMetrics.XSENTID] = int(expected["x_sent_id"]) == xidx
        metrics[FineEvalMetrics.YSENTID] = int(expected["y_sent_id"]) == yidx
        return InputRequiredByFineEval(graph_id=graph_id,
                                       ques_type=qtype,
                                       path_arr=path_arr,
                                       metrics=metrics)

    @staticmethod
    def accumulate_metrics(current_metrics: Dict[FineEvalMetrics, bool],
                           all_metrics: Dict[FineEvalMetrics, Precision]):
        for k, is_correct in current_metrics.items():
            if k not in all_metrics:
                all_metrics[k] = Precision()
            if is_correct:
                all_metrics[k].tp += 1
            else:
                all_metrics[k].fp += 1


class MetricEvaluator:
    def __init__(self):
        self.entries: List[InputRequiredByFineEval] = []

    def add_entry(self, entry: InputRequiredByFineEval):
        self.entries.append(entry)

    def group_by_ques_type(self):
        per_ques_type: Dict[WIQAQuesType, Dict[FineEvalMetrics, Precision]] = {}
        for e in self.entries:
            if e.ques_type not in per_ques_type:
                per_ques_type[e.ques_type] = {}
            InputRequiredByFineEval.accumulate_metrics(current_metrics=e.metrics,
                                                       all_metrics=per_ques_type[e.ques_type])
        return per_ques_type

    def group_by_path_len(self):
        per_path_len: Dict[int, Dict[FineEvalMetrics, Precision]] = {}
        for e in self.entries:
            if e.get_path_len() not in per_path_len:
                per_path_len[e.get_path_len()] = {}
            InputRequiredByFineEval.accumulate_metrics(current_metrics=e.metrics,
                                                       all_metrics=per_path_len[e.get_path_len()])
        return per_path_len


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze BERT results per question and per question type',
                                     usage="\n\npython src/dataset_creation_of_wiqa/analysis/fine_eval_on_wiqa_models.py"
                                           "\n\t --pred_path dir_path/eval_test.json"
                                           "\n\t --out_path /tmp/bert_anal"
                                           "\n\t --from_model tagging_model"
                                           "\n\t --group_by questype"
                                     )

    # ------------------------------------------------
    #            Mandatory arguments.
    # ------------------------------------------------

    parser.add_argument('--pred_path',
                        action='store',
                        dest='pred_path',
                        required=True,
                        help='File path containing predictors json output.')

    parser.add_argument('--group_by',
                        action='store',
                        dest='group_by',
                        required=True,
                        help='pathlen|questype')

    parser.add_argument('--from_model',
                        action='store',
                        dest='from_model',
                        required=True,
                        help='sentid_model|tagging_model|wordoverlap_baseline_model|vectoroverlap_baseline_model|emnlp_model|no_explanation_model')

    parser.add_argument('--out_path',
                        action='store',
                        dest='out_path',
                        required=True,
                        help='File path to store output such as metrics.json')

    args = parser.parse_args()
    m = MetricEvaluator()

    # Compile input for metrics.
    if args.from_model == "wordoverlap_baseline_model":
        args.pred_path = download_from_url_if_not_in_cache(cloud_path=wiqa_explanations_v1.cloud_path + "test.jsonl")
    if args.from_model =="vectoroverlap_baseline_model":
        raise NotImplementedError

    if args.from_model == "emnlp19_model":
        map_of_expected_keys = {}
        for x in Jsonl.load(download_from_url_if_not_in_cache(wiqa_explanations_v1.cloud_path + "test.jsonl")):
            key = x["question"]["id"]
            value = {"graph_id": x["question"]["graph_id"],
                     "path_arr": x["question"]["path_info"],
                     "qtype": x["question"]["more_info"]["tf_q_type"]}
            map_of_expected_keys[key] = value

    outfile = open(args.out_path, 'w')
    with open(args.pred_path) as infile:
        for line in infile:
            j = json.loads(line)
            if args.from_model == "sentid_model":
                entry = InputRequiredByFineEval.from_outputjson_sentid_model(output_json_line=j)
            elif args.from_model == "emnlp19_model":
                if not j["id"] in map_of_expected_keys:
                    continue
                entry = InputRequiredByFineEval.from_outputjson_emnlp_model(output_json_dict=j, extra_args = map_of_expected_keys[j["id"]])
            elif args.from_model == "no_explanation_model":
                entry = InputRequiredByFineEval.from_outputjson_noexplanation_model(output_json_dict=j)
            elif args.from_model == "tagging_model":
                entry = InputRequiredByFineEval.from_outputjson_tagging(output_json_line=j)
            elif args.from_model == "wordoverlap_baseline_model":
                entry = InputRequiredByFineEval.from_outputjson_wordoverlap_baseline_model(output_json_line=j)
            elif args.from_model == "vectoroverlap_baseline_model":
                entry = InputRequiredByFineEval.from_outputjson_wordoverlap_baseline_model(output_json_line=j)
            else:
                raise NotImplementedError(f"fine_eval script does not support model: {args.from_model}")
            m.add_entry(entry=entry)

    # Compute metrics.
    if args.group_by == "pathlen":
        metrics = m.group_by_path_len()
    elif args.group_by == "questype":
        metrics = m.group_by_ques_type()
    else:
        raise NotImplementedError(f"fine_eval script does not support group by: {args.group_by}")

    # Write metrics to file.
    overall = {}
    for k, v_dict in metrics.items():
        for k2, v2 in v_dict.items():
            if k2 not in overall:
                overall[k2] = Precision()
            overall[k2].tp += v2.tp
            overall[k2].fp += v2.fp
            outfile.write(f"{k.name}_{k2.name}:{v2.p():0.4}\n")

    outlines = []
    for k, v in overall.items():
        outline = f"{k.name}_overall:{v.p():0.4}"
        outlines.append(outline)
        outfile.write(f"{outline}\n")
    print(f"\nOutput is in {args.out_path}")
    print("\n".join(outlines))
    outfile.close()
