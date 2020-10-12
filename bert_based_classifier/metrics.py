import torch


class Metric:
    def __init__(self):
        self.num_sample = 0
        self.correct_sample = 0
        self.loss = 0
        self.acc = 0

    def update(self, logits, label):  # loss= max(1+s_i-s_c,0)
        device = logits.device
        batch = logits.size(0)
        self.num_sample += batch
        correct_index = torch.argmax(label, dim=1)
        score_correct = torch.gather(logits, dim=1, index=correct_index.unsqueeze(1)).squeeze()
        mask = (label == 0)
        false_logits = torch.masked_select(logits, mask).view(logits.size(0), -1)
        score_max_false = torch.max(false_logits, dim=1).values
        loss = 1 + score_max_false - score_correct
        loss = torch.max(loss, torch.zeros([1, batch]).to(device))

        predict_index = torch.argmax(logits, dim=1)
        correct_num = torch.sum(predict_index == correct_index)
        self.correct_sample += correct_num
        loss_of_batch = torch.sum(loss)
        self.loss += loss_of_batch
        return loss_of_batch

    def get_metrics(self, reset):
        self.acc = self.correct_sample.true_divide(self.num_sample)
        res = {
            "total_question": self.num_sample,
            "correct_question": self.correct_sample,
            "total_loss": self.loss,
            "total_acc": self.acc.item()
        }
        result = {}
        for key in res:
            if isinstance(res[key], torch.Tensor):
                result[key] = res[key].item()
            else:
                result[key] = res[key]
        if reset:
            self.reset()
        return result

    def reset(self):
        self.num_sample = 0
        self.correct_sample = 0
        self.loss = 0
        self.acc = 0


if __name__ == '__main__':
    logits = torch.tensor([[2.9, 0.1, -0.1, -0.4, 0], [0.9, 2.1, -0.1, -0.4, 0], [-0.9, -0.1, -0.1, -0.4, -0.2]])
    label = torch.tensor([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 1, 0]])
    a = Metric()
    a.update(logits, label)
