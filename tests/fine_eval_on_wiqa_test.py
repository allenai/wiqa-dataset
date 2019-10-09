import json
import os
import tempfile
from unittest import TestCase

from src.eval.evaluation import InputRequiredByFineEval, FineEvalMetrics
from src.helpers.dataset_info import download_from_url_if_not_in_cache, wiqa_explanations_v1
from src.wiqa_wrapper import WIQAExplanation, WIQAExplanationType, create_concise_dataset, WIQADataPoint


class TestWIQAWrapper(TestCase):

    def setUp(self) -> None:
        self.init_dir = tempfile.mkdtemp()
        self.init_dir += "/" if "/" in self.init_dir else "\\"

    # 1. (done) reformat dataset (this is a command line argument.)
    # 2. load (reformatted) dataset ... includes other operations on it. (this is not command line)
    # 3. Evaluation code.
    def test_1_create_concise_dataset(self):
        partitions = ["train.jsonl", "dev.jsonl", "test.jsonl"]
        create_concise_dataset(
            input_filepaths=[download_from_url_if_not_in_cache(wiqa_explanations_v1.cloud_path + partition) for
                             partition in
                             partitions],
            output_filepaths=[self.init_dir + x for x in partitions],
            explanation_type=WIQAExplanationType.PARA_SENT_EXPL)
        for outfp in [self.init_dir + x for x in partitions]:
            assert os.path.exists(outfp)

    # Load the dataset from jsonl files.
    def test_2_load_dataset(self):
        for x in WIQADataPoint.load_all_in_jsonl(jsonl_filepath=self.init_dir + "dev.jsonl"):
            j_str = json.dumps(x.to_json())
            assert j_str is not None
            break

    def test_3_sample_model_output_eval(self):
        # Note: j_str contains a mix of ' and "
        json_data = """{"logits": [[0.7365949749946594, 0.16483880579471588, 0.38572263717651367, 0.33268705010414124, -0.6508089900016785, -0.950057864189148], [0.3361547291278839, -0.02215845137834549, 0.8630506992340088, 0.4753769040107727, -1.0981523990631104, 0.04984292760491371], [0.4498467743396759, 0.26323091983795166, 0.5597160458564758, 0.06369128823280334, -0.33793506026268005, -0.30190590023994446], [0.41394802927970886, 0.31742218136787415, 0.42982375621795654, -0.2891058027744293, -0.09577881544828415, -0.4486318528652191], [0.5242481231689453, -0.05186435207724571, 0.4505387544631958, -0.43092456459999084, -0.015227549709379673, 0.10361793637275696], [0.8527745604515076, 0.18845966458320618, 0.6540948748588562, -0.06324845552444458, -0.03267676383256912, 0.058296892791986465], [0.40418609976768494, -0.24220454692840576, 0.0737631767988205, -0.8445389270782471, -0.12929767370224, 0.5813987851142883]], "class_probabilities": [[0.29663264751434326, 0.16745895147323608, 0.20885121822357178, 0.19806326925754547, 0.07407592236995697, 0.054918017238378525], [0.18079228699207306, 0.12634745240211487, 0.3062019348144531, 0.20779892802238464, 0.04307926073670387, 0.13578014075756073], [0.21968600153923035, 0.18228720128536224, 0.24519862234592438, 0.14931286871433258, 0.09992476552724838, 0.10359060019254684], [0.22513438761234283, 0.20441898703575134, 0.22873708605766296, 0.11145754158496857, 0.13522914052009583, 0.09502287209033966], [0.24298663437366486, 0.13657772541046143, 0.2257203906774521, 0.09348806738853455, 0.14167429506778717, 0.15955300629138947], [0.27786338329315186, 0.1429957151412964, 0.22779585421085358, 0.11117511987686157, 0.11462641507387161, 0.12554346024990082], [0.23202574253082275, 0.1215660497546196, 0.16673828661441803, 0.06656130403280258, 0.1360965520143509, 0.27701207995414734]], "question": {"stem": "suppose squirrels get sick happens, how will it affect squirrels need more food.", "para_steps": ["Squirrels try to eat as much as possible", "Squirrel gains weight and fat", "Squirrel also hides food in or near its den", "Squirrels also grow a thicker coat as the weather gets colder", "Squirrel lives off of its excess body fat", "Squirrel uses its food stores in the winter..)"], "answer_label": "more", "answer_label_as_choice": "A", "choices": [{"label": "A", "text": "more"}, {"label": "B", "text": "less"}, {"label": "C", "text": "no effect"}]}, "explanation": {"di": "RESULTS_IN", "dj": "RESULTS_IN", "de": "RESULTS_IN", "i": 1, "j": 4}, "orig_answer": {"explanation": {"di": "RESULTS_IN", "dj": "RESULTS_IN", "de": "NOT_RESULTS_IN", "i": 2, "j": 3}, "metadata": {"ques_id": "influence_graph:1310:156:83#3", "graph_id": "156", "para_id": "1310", "question_type": "EXOGENOUS_EFFECT"}} }"""
        json_obj = json.loads(json_data)
        answer_obj = InputRequiredByFineEval.from_(
            prediction_on_this_example=WIQAExplanation.instantiate_from(json_data=json_obj),
            json_from_question=json_obj["orig_answer"],
            expl_type=WIQAExplanationType.PARA_SENT_EXPL
        )
        assert answer_obj.metrics[FineEvalMetrics.XDIR] \
               and not answer_obj.metrics[FineEvalMetrics.EQDIR] \
               and not answer_obj.metrics[FineEvalMetrics.XSENTID] \
               and not answer_obj.metrics[FineEvalMetrics.YSENTID]
