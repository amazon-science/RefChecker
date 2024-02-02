from collections import Counter


def soft_agg(results):
    """Aggregate results by taking the ratio of each category."""
    if not results:
        return {
            "Entailment": 0.0,
            "Neutral": 0.0,
            "Contradiction": 0.0,
            "Abstain": 1.0,
        }
    total = len(results)
    agg = {
        "Entailment": 0.0,
        "Neutral": 0.0,
        "Contradiction": 0.0,
        "Abstain": 0.0,
    }
    for result in results:
        agg[result] += 1.0
    for key in agg:
        agg[key] /= total
    return agg


def strict_agg(results):
    """Aggregate results by zero-tolerance on negative labels."""
    if not results:
        return "Abstain"
    ret = "Entailment"
    for result in results:
        if result == "Contradiction":
            return "Contradiction"
        if result == "Neutral":
            ret = "Neutral"
    return ret


def major_agg(results):
    """Aggregate results by majority vote."""
    if not results:
        return "Abstain"
    agg = Counter(results)
    return agg.most_common(1)[0][0]
