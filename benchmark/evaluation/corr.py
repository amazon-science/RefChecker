import argparse
import json
from sklearn.metrics import f1_score, accuracy_score
from scipy import stats


def main():
    print(f'===== {args.extractor} + {args.checker} ====')
    for setting, ds in zip(
        ["zero_context", "noisy_context", "accurate_context"],
        ["nq", "msmarco", "dolly"]
    ):
        gt_factual_list = []
        pred_factual_list = []
        
        gt_halu_rates = []
        pred_halu_rates = []
        # for llm in ["alpaca_7B", "chatgpt", "claude2", "davinci001", "falcon_40B_instruct", "gpt4", "llama2_70b_chat"]:
        for llm in ["alpaca_7B", "chatgpt", "claude2", "falcon_40B_instruct", "gpt4", "llama2_70b_chat"]:
            response_file = f'{args.data_dir}/{setting}/{ds}_{llm}_answers.json'
            response_data = json.load(open(response_file))
            
            kg_key = f'{args.extractor}_response_kg'
            label_key = f'{args.checker}_label'
            for r in response_data:
                if 'claude2_response_kg' not in r or len(r['claude2_response_kg']) == 0:
                    continue
                
                gt_factual = all([t['human_label'] == 'Entailment' for t in r['claude2_response_kg']])
                gt_factual_list.append(gt_factual)
                gt_halu_rates.append(len([c for c in r['claude2_response_kg'] if c['human_label'] != 'Entailment']) / len(r['claude2_response_kg']))
                
                assert kg_key in r
                pred_factual = all([c[label_key] == 'Entailment' for c in r[kg_key]])
                pred_factual_list.append(pred_factual)
                if len(r[kg_key]):
                    pred_halu_rates.append(len([c for c in r[kg_key] if c[label_key] != 'Entailment']) / len(r[kg_key]))
                else:
                    pred_halu_rates.append(0)
        
        print(f'{setting}')
        acc = round(accuracy_score(gt_factual_list, pred_factual_list) * 100, 2)
        fact_f1 = round(f1_score(gt_factual_list, pred_factual_list, pos_label=True) * 100, 2)
        nonfact_f1 = round(f1_score(gt_factual_list, pred_factual_list, pos_label=False) * 100, 2)
        pearson = round(stats.pearsonr(gt_halu_rates, pred_halu_rates).statistic * 100, 2)
        spearman = round(stats.spearmanr(gt_halu_rates, pred_halu_rates).statistic * 100, 2)
        print(f'Acc: {acc}\tFact. F1: {fact_f1}\tNonFact. F1: {nonfact_f1}\tPearson: {pearson}\tSpearman: {spearman}')
        # print(f'Fact. F1: {fact_f1}')
        # print(f'NonFact. F1: {nonfact_f1}')
        # print(f'Pearson: {pearson}')
        # print(f'Spearman: {spearman}')
    print(f'=========================================\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--extractor', type=str)
    parser.add_argument('--checker', type=str)
    parser.add_argument('--data_dir', type=str)

    args = parser.parse_args()

    main()
    # python evaluation/corr.py --extractor=bedrock/meta.llama3-70b-instruct-v1:0 --checker=bedrock/meta.llama3-70b-instruct-v1:0 --data_dir=triplet_llama3
    # python evaluation/corr.py --extractor=bedrock/meta.llama3-70b-instruct-v1:0 --checker=alignscore --data_dir=triplet_llama3
    # python evaluation/corr.py --extractor=bedrock/meta.llama3-70b-instruct-v1:0 --checker=gpt-4-turbo --data_dir=triplet_llama3
    
    # python evaluation/corr.py --extractor=bedrock/meta.llama3-70b-instruct-v1:0 --checker=bedrock/meta.llama3-70b-instruct-v1:0 --data_dir=subsent_llama3
    # python evaluation/corr.py --extractor=bedrock/meta.llama3-70b-instruct-v1:0 --checker=alignscore --data_dir=subsent_llama3
    # python evaluation/corr.py --extractor=bedrock/meta.llama3-70b-instruct-v1:0 --checker=gpt-4-turbo --data_dir=subsent_llama3