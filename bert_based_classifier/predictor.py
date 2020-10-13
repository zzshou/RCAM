from allennlp.predictors.predictor import Predictor
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from bert_based_classifier.reader import parse_sentence
import re
from typing import List
from overrides import overrides


@Predictor.register('RCAM_predictor')
class RCAMPredictor(Predictor):
    def predict(self, ori_data: str) -> JsonDict:
        article, option_list, label_list = parse_sentence(ori_data)
        return self.predict_json({"article": article,
                                  "option_list": option_list,
                                  "label_list": label_list})

    def _json_to_instance(self, ori_json: JsonDict) -> Instance:
        article = ori_json['article']
        question = ori_json['question']
        option_list = []
        label_list = None
        for i in range(5):
            option_id = "option_" + str(i)
            answer = ori_json[option_id]
            answer_candidate = re.sub('@placeholder', answer, question)
            option_list.append(answer_candidate)
        if 'label' in ori_json:
            label_list = ['false'] * 5
            right_answer = ori_json['label']
            label_list[right_answer] = 'true'
        return self._dataset_reader.text_to_instance(article, option_list, label_list)

    @overrides
    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        outputs_batch = self._model.forward_on_instances(instances)

        ret_dict_batch = []
        for outputs_idx in range(len(outputs_batch)):
            outputs = outputs_batch[outputs_idx]
            predict = outputs['predicted_label']
            ret_dict = {"label": predict}
            ret_dict_batch.append(ret_dict)

        return sanitize(ret_dict_batch)
