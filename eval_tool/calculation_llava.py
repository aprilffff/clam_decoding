import os,sys
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../../models")))
from models.config import RESULT_PATH,DPATH_MMMU,DPATH_CHAIR
from models.utils.eval_utils import *
from models.utils.data_utils import *
from models.utils.chair_utils import chair
import json
from collections import defaultdict
import nltk
nltk.download('punkt')
import glob


mmebench_eval_type_dict = {
    "Perception": ["existence", "count", "position", "color", "posters", "celebrity", "scene", "landmark", "artwork", "OCR"],
    "Cognition": ["commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning"]
}
mmebench_eval_type_dict_r= {v: k for k, vals in mmebench_eval_type_dict.items() for v in vals}




class calculate_metrics:
    def divide_chunks(self, l, n=2):
        # looping till length l
        for i in range(0, len(l), n): 
            yield l[i:i + n]
        
        return 

    def parse_pred_ans(self, pred_ans):
        pred_label = None
        if pred_ans in ["yes", "no"]:
            pred_label = pred_ans
        else:
            prefix_pred_ans = pred_ans[:4]

            if "yes" in prefix_pred_ans:
                pred_label = "yes"
            elif "no" in prefix_pred_ans:
                pred_label = "no"
            else:
                pred_label = "other"

        return pred_label


    def compute_metric(self, gts, preds,image_ids):
        assert len(gts) == len(preds)

        label_map = {
            "yes": 1,
            "no": 0,
            "other": -1,
        }
        
        gts = [label_map[x] for x in gts]
        preds = [label_map[x] for x in preds]

        acc = accuracy_score(gts, preds) 

        clean_gts = []
        clean_preds = []
        clean_image_ids = []
        other_num = 0
        for gt, pred ,image_id in zip(gts, preds,image_ids):
            if pred == -1:
                other_num += 1
                continue
            clean_gts.append(gt)
            clean_preds.append(pred)
            clean_image_ids.append(image_id)


        conf_mat = confusion_matrix(clean_gts, clean_preds, labels=[1,0])
        precision = precision_score(clean_gts, clean_preds, average='binary')
        recall = recall_score(clean_gts, clean_preds, average='binary')
        f1=2*precision*recall/(precision+recall)
        tp, fn = conf_mat[0]
        fp, tn = conf_mat[1]

        clean_res=pd.DataFrame.from_dict({'correct':[clean_gts[i]==clean_preds[i] for i in range(len(clean_gts))],'image_id':clean_image_ids})
        acc_plus_c=clean_res.groupby('image_id').correct.count().max()
        acc_plus=(clean_res.groupby('image_id').correct.sum()==acc_plus_c).sum()/((clean_res.groupby('image_id').correct.count()==acc_plus_c).sum()+1e-10)
        total_acc_score=int((acc_plus+acc)*100)

        metric_dict = {
            "TP": tp,
            "FN": fn,
            "TN": tn,
            "FP": fp,
            "precision": precision,
            "recall": recall,
            'F1':f1,
            "other_num": other_num,
            "acc": acc,
            "acc_plus":acc_plus,
            "total_acc_score":total_acc_score
        }

        return metric_dict


    def process_result(self,benchmark,decoding_method,results_dir):
        metric_dict={}
        res_df=pd.read_pickle(results_dir+f'/eval_res_{benchmark}_{decoding_method}.pkl')
        res_df=res_df.assign(prediction=res_df.prediction.str.lower().apply(self.parse_pred_ans),label=res_df.label.str.lower().apply(self.parse_pred_ans))
        res_df=res_df.assign(correct=res_df.prediction==res_df.label)
        if benchmark=='mme':
            res_df=res_df.loc[res_df.category.isin(mmebench_eval_type_dict['Perception'])]
            res_df=res_df.assign(eval_type=pd.Series(mmebench_eval_type_dict_r).reindex(res_df.category.values.ravel()).values)
            metric_dict_evaltype=res_df.groupby('eval_type').apply(lambda x:self.compute_metric(x.label.values.ravel(), x.prediction.values.ravel(),(x.category+x.image_name).values.ravel())).to_dict()
            metric_dict_evaltype=pd.DataFrame.from_records(metric_dict_evaltype).T
            metric_dict_evaltype.loc['<sum>']=metric_dict_evaltype.sum()
            metric_dict['by_evaltype']=metric_dict_evaltype
            print('metric stats by eval_type\n',metric_dict_evaltype,'\n')

        metric_dict_category=res_df.groupby('category').apply(lambda x:self.compute_metric(x.label.values.ravel(), x.prediction.values.ravel(),(x.category+x.image_name).values.ravel())).to_dict()
        metric_dict_category=pd.DataFrame.from_records(metric_dict_category).T
        metric_dict_category.loc['<sum>']=metric_dict_category.sum()
        metric_dict['by_category'] = metric_dict_category
        print('metric stats by category\n', metric_dict_category,'\n')

        metric_dict_overall= self.compute_metric(res_df.label.values.ravel(), res_df.prediction.values.ravel(),(res_df.category+res_df.image_name).values.ravel())
        metric_dict_overall = pd.DataFrame.from_records({'overall':metric_dict_overall}).T
        metric_dict['overall'] = metric_dict_overall
        print('metric stats overall\n', metric_dict_overall,'\n')

        metric_df=pd.concat(metric_dict)
        metric_dict['final']=metric_df

        return metric_dict

    def process_result_mmmu(self,decoding_method,results_dir):
        output_dict = json.load(open(results_dir+f'/eval_res_mmmu_{decoding_method}.json'))
        answer_dict = json.load(open(f"{DPATH_MMMU}/answer_dict_val.json"))

        # group by category
        output_dict_w_cat = {}
        for data_id, parsed_pred in output_dict.items():
            category = "_".join(data_id.split("_")[1:-1])
            if category not in output_dict_w_cat:
                output_dict_w_cat.update({category: {}})
            output_dict_w_cat[category].update({data_id: parsed_pred})

        # group by category
        answer_dict_w_cat = {}
        for data_id, parsed_pred in answer_dict.items():
            category = "_".join(data_id.split("_")[1:-1])
            if category not in answer_dict_w_cat:
                answer_dict_w_cat.update({category: {}})
            answer_dict_w_cat[category].update({data_id: parsed_pred})

        evaluation_result = {}

        for category in CAT_SHORT2LONG.values():
            print("Evaluating: {}".format(category))
            # get cat_outputs and cat_answers
            try:
                cat_outputs = output_dict_w_cat[category]
                cat_answers = answer_dict_w_cat[category]
            except KeyError:
                print("Skipping {} for not found".format(category))
                continue

            exampels_to_eval = []
            for data_id, parsed_pred in cat_outputs.items():
                question_type = cat_answers[data_id]['question_type']
                if question_type != 'multiple-choice':
                    parsed_pred = parse_open_response(parsed_pred)  # mainly for type consistency (make it number, etc.)
                else:
                    parsed_pred = parsed_pred

                exampels_to_eval.append({
                    "id": data_id,
                    "question_type": question_type,
                    "answer": cat_answers[data_id]['ground_truth'],
                    "parsed_pred": parsed_pred
                })

            judge_dict, metric_dict = evaluate(exampels_to_eval)
            metric_dict.update({"num_example": len(exampels_to_eval)})

            evaluation_result[category] = metric_dict

        printable_results = {}
        # pdb.set_trace()
        # add domain Subject
        for domain, in_domain_cats in DOMAIN_CAT2SUB_CAT.items():
            in_domain_cat_results = {}
            for cat_name in in_domain_cats:  # use the order in DOMAIN_CAT2SUB_CAT
                if cat_name in evaluation_result.keys():
                    in_domain_cat_results[cat_name] = evaluation_result[cat_name]
                else:
                    pass
            in_domain_ins_acc = calculate_ins_level_acc(in_domain_cat_results)
            in_domain_data_num = sum([cat_results['num_example'] for cat_results in in_domain_cat_results.values()])
            printable_results['Overall-' + domain] = {"num": int(in_domain_data_num),
                                                      "acc": round(in_domain_ins_acc, 3)
                                                      }
            # add sub category
            for cat_name, cat_results in in_domain_cat_results.items():
                printable_results[cat_name] = {"num": int(cat_results['num_example']),
                                               "acc": round(cat_results['acc'], 3)
                                               }

        # table.append(["-----------------------------", "-----", "----"])
        all_ins_acc = calculate_ins_level_acc(evaluation_result)
        printable_results['Overall'] = {
            "num": sum([cat_results['num_example'] for cat_results in evaluation_result.values()]),
            "acc": round(all_ins_acc, 3)
            }

        print(printable_results)
        return pd.DataFrame(printable_results).T

    def process_result_chair(self,decoding_method,results_dir):
        _, imids, _ = chair.load_generated_captions(results_dir+f'/eval_res_chair_{decoding_method}.json')

        evaluator = chair.CHAIR(imids, DPATH_CHAIR+'/annotations')
        evaluator.get_annotations()
        cap_dict = evaluator.compute_chair(results_dir+f'/eval_res_chair_{decoding_method}.json')
        return pd.Series(cap_dict['overall_metrics']).loc[['CHAIRs','CHAIRi']].to_frame('overall')

    def process_result_llavabenchgpt(self,decoding_method,results_dir,gpt_version="gpt-4o-2024-08-06"):#gpt-4o-2024-08-06
        if not os.path.exists(f"{results_dir}/eval_res_llavabench_{gpt_version}_{decoding_method}.jsonl"):
            LLAVABENCH_eval_gpt(decoding_method,gpt_version)
        scores = defaultdict(list)
        with open(f"{results_dir}/eval_res_llavabench_{gpt_version}_{decoding_method}.jsonl") as f:
            for review_str in f:
                review = json.loads(review_str)
                if 'category' in review:
                    scores[review['category']].append(review['tuple'])
                    scores['all'].append(review['tuple'])
                else:
                    if 'tuple' in review:
                        scores['all'].append(review['tuple'])
                    else:
                        scores['all'].append(review['score'])
        stats_dict={}
        for k, v in sorted(scores.items()):
            stats = np.asarray(v).mean(0).tolist()
            stats = [round(x, 3) for x in stats]
            stats_dict[k]=round(stats[1]/stats[0]*100, 1)
        stats_df=pd.Series(stats_dict).to_frame('overall')
        print(decoding_method,'\n',stats_df,'\n')
        return stats_df
        # print(k, stats, round(stats[1]/stats[0]*100, 1))
        # print(k, round(stats[1]/stats[0]*100, 1), round(stats[0] * 10, 1), round(stats[1] * 10, 1))

if __name__ == "__main__":
    cal = calculate_metrics()
    parser = argparse.ArgumentParser()

    parser.add_argument("--results_dir",type=str,default=None,help='results directory path')
    parser.add_argument("--benchmark",type=str,default=None,help='benchmark name')
    parser.add_argument("--decoding_method",type=str,default=None,help='decoding method name')
    args = parser.parse_args()

    results_dir = args.results_dir or RESULT_PATH
    gpt_version="gpt-4o-2024-08-06"

    cal.process_result_llavabenchgpt(decoding_method=args.benchmark,results_dir=results_dir,gpt_version=gpt_version)
    cal.process_result_chair(decoding_method=args.benchmark,results_dir=results_dir)
    cal.process_result_mmmu(decoding_method=args.benchmark,results_dir=results_dir)
    cal.process_result(benchmark=args.benchmark,decoding_method=args.decoding_method,results_dir=results_dir)

