class Evaluator:
    """evaluate fact-checking"""

    def __init__(
            self,
            classes=[0, 1, 2],
            metrics=["acc", "macro_f1"]
    ):
        self.classes = classes
        assert all([m in ["acc", "macro_f1"] for m in metrics]), \
            "invalid metrics, should be one of ['acc', 'macro_f1']"
        self.metrics = metrics
        self.metric2func = {
            "acc": self.compute_acc,
            "macro_f1": self.compute_macro_f1
        }

    def evaluate(self, preds, labels):
        ret = {}
        for metric in self.metrics:
            ret[metric] = self.metric2func[metric](preds, labels)
        return ret

    def compute_acc(self, preds, labels):
        correct = sum([p == l for p, l in zip(preds, labels)])
        return round(correct / len(preds), 4)

    def compute_f1(self, preds, labels, pos_label):
        tp = sum([p == l == pos_label for p, l in zip(preds, labels)])
        true = sum([l == pos_label for l in labels])
        pos = sum([p == pos_label for p in preds])
        f1 = 2 * tp / (true + pos)
        return round(f1, 4)

    def compute_prec(self, preds, labels, pos_label):
        pos = sum([p == pos_label for p in preds])
        if not pos:
            return 0.
        tp = sum([p == l == pos_label for p, l in zip(preds, labels)])
        return tp / pos

    def compute_rec(self, preds, labels, pos_label):
        true = sum([l == pos_label for l in labels])
        if not true:
            return 0.
        tp = sum([p == l == pos_label for p, l in zip(preds, labels)])
        return tp / true

    def compute_confusion(self, preds, labels, pos_label):
        ratio = {c: 0. for c in self.classes}
        total = 0.
        for pred, label in zip(preds, labels):
            if label == pos_label:
                ratio[pred] += 1
                total += 1
        for c in self.classes:
            ratio[c] /= total
        return ratio

    def compute_pr(self, preds, probs, labels, pos_label):
        true = sum([l == pos_label for l in labels])
        theta, indices = th.tensor(probs).sort(descending=True)
        labels = th.BoolTensor([l == pos_label for l in labels])[indices]
        preds = th.BoolTensor([p == pos_label for p in preds])[indices]
        tp = th.cumsum(preds & labels, dim=0)
        prec = tp / th.arange(1, len(preds) + 1)
        rec = tp / true

        return prec, rec, theta

    def compute_macro_f1(self, preds, labels):
        f1s = {}
        for c in self.classes:
            f1s[c] = self.compute_f1(preds, labels, c)
        f1s["macro"] = round(sum(f1s.values()) / len(f1s), 4)
        return f1s


def aggregate_labels(labels):
    """Aggregate labels on decomposed units."""
    ret = "Entailment"
    for label in labels:
        if label == "Neutral" and ret == "Entailment":
            ret = label
        if label == "Contradiction":
            ret = label
            break
    return ret


def compare_nli(data_file, depths=["t", "f", "s", "r"]):
    evaluator = Evaluator()
    with open(data_file) as f:
        data = json.load(f)
    labels = []
    total, skip_cnt = 0, 0
    for item in data:
        for model in MODELS:
            total += 1
            if not item["answers"][model]["response_kg"]:
                skip_cnt += 1
                continue
            labels.append(aggregate_labels(item["answers"][model]["anno"]))
    print(f"{skip_cnt}/{total} skipped due to no response_kg")
    for depth in depths:
        print(f"Evaluating {data_file} nli {depth}-level")
        preds = []
        for item in data:
            for model in MODELS:
                if not item["answers"][model]["response_kg"]:
                    continue
                # a list of predictions on triplets like ["Entailment", "Entailment", "Contradiction", ...]
                ret = item["answers"][model][f"nlidc_{depth}level_ret"]
                # ret = item["answers"][model][f"nli_ret"]
                preds.append(aggregate_labels(ret))
        assert len(preds) == len(labels)
        print(json.dumps(evaluator.evaluate(preds, labels), indent=2))
        print()