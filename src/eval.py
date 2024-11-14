import os
import glob
import re
import subprocess
from tqdm import tqdm
from sacrebleu.metrics import BLEU, CHRF
import matplotlib.pyplot as plt
import argparse as args
import json
import tempfile
import numpy as np

SRC_PATH = './data/original/nyn_test.bpe'
TGT_PATH = './data/original/eng_test.bpe'


def find_directories():
    outputs = glob.glob(f'./test_model/output_*/models/')
    return outputs

def find_models(output):
    files = os.listdir(output)
    actual_models = []
    for file in files:
        if file.endswith('.pt'):
            actual_models.append(file)
    return actual_models

def run_model(model, output, model_name):
    subprocess.run(
        ["onmt_translate",
         "--model", model,
         "--src", SRC_PATH,
         "--tgt",  TGT_PATH,
         "-output", f"{output}preds_{model_name}.txt",
         "--log_file", f"{output}{model_name}.log",
         "--verbose"],
         capture_output=True, text=True
    )

def check_directory(output):
    output = re.search(r'/output_(\d)+/', output)
    result_path = f'./results{output.group(0)}'
    if not os.path.exists(result_path):
        os.makedirs(result_path, exist_ok=True)
        print(f"Created directory: {result_path}")

    else:
        print(f"Directory already present: {output.group(0)}")
    return result_path

def output_name(output):
    name_output = re.search(r'output_\d+', output).group(0)
    return name_output

def find_predictions():
    pred_paths = glob.glob("./results/output_*/preds*.txt")
    return pred_paths

def calculate_bleu(source, tgt):
    bleu_score = subprocess.run(['sacrebleu',
			source,
			'-i', tgt,
			'-m', "bleu",
            "-b"
			], capture_output=True, text=True)

    try:

        bleu_score = float(bleu_score.stdout.strip())
        print(f"BLEU-score: {bleu_score}")
        return bleu_score
    except ValueError:
        print(bleu_score)
        pass

    return bleu_score

def calculate_chrf(source, tgt):
    chrf_score = subprocess.run(['sacrebleu',
        source,
        '-i', tgt,
        '-m', "chrf",
        "-b"
        ], capture_output=True, text=True)

    try:
        
        chrf_score = float(chrf_score.stdout.strip())
        print(f"chrF-score: {chrf_score}")
        return chrf_score
    except ValueError:
        print(chrf_score)
        pass

    return chrf_score


def calculate_metrics(preds):
    output_chrf = {}
    output_bleu = {}

    for pred in preds:
        source_file = detokeniser(open(pred, 'r', encoding='utf-8').read())
        name_output = output_name(pred)

        with tempfile.NamedTemporaryFile('w', encoding='utf-8') as tmp_file:
            tmp_file.write(source_file)
            tmp_file_path = tmp_file.name

            try:

                if name_output not in output_bleu.keys() and name_output not in output_chrf.keys():
                    print(f"Creating key: {name_output}.")
                    output_bleu[name_output] = []
                    output_chrf[name_output] = []
                    output_bleu[name_output].append(calculate_bleu(tmp_file_path, TGT_PATH))
                    output_chrf[name_output].append(calculate_chrf(tmp_file_path, TGT_PATH))
                else:
                    output_bleu[name_output].append(calculate_bleu(tmp_file_path, TGT_PATH))
                    output_chrf[name_output].append(calculate_chrf(tmp_file_path, TGT_PATH))
            except IndexError:
                pass

    return output_bleu, output_chrf

def plot_line(output, x, name):
    plt.plot(x, output, label = f'{name}')

def detokeniser(file):
    detokenised_file = re.sub(r'@@\s', '', file)
    return detokenised_file


if __name__ == '__main__':

    outputs = find_directories()
    for output in tqdm(range(len(outputs)), desc="Running different hyperparameters."):
        directory = check_directory(outputs[output])
        models = find_models(outputs[output])
        for model in tqdm(range(len(models)), desc=f"Testing models"):
            model_name = models[model][:-3]
            model = outputs[output] + models[model]
            result = run_model(model, directory, model_name)

    if not os.path.exists('./results/score_images'):
        os.makedirs("./results/score_images")
    if not os.path.exists('./results/tables'):
       os.makedirs('./results/tables', exist_ok=True)

    preds = find_predictions()
    bleu_scores, chrf_scores = calculate_metrics(preds)
    scores_bleu_dict = {}
    scores_bleu_max_dict = {}
    for output in tqdm(bleu_scores, desc="Plotting BLEU scores"):
        x = list(range(1, len(bleu_scores[output])+1))
        plot_line(bleu_scores[output], x, output)
        scores_bleu_dict[output] = bleu_scores[output]
        filtered_bleu_scores = [output_value for output_value in bleu_scores[output] if output_value is not None]
        max_value = filtered_bleu_scores[np.argmax(filtered_bleu_scores)]
        scores_bleu_max_dict[output] = max_value
        print(max_value, "max value for bleu")

    print('Dumping BLEU scores as JSON')
    scores_bleu_max_dict = {key: float(value) for key, value in scores_bleu_max_dict.items()}

    combined_bleu = {
        "bleu scores":scores_bleu_dict,
        "bleu scores max":scores_bleu_max_dict
        }
    with open('./results/tables/scores_bleu_LSTM.json', 'w') as bleu_file:
        json.dump(combined_bleu, bleu_file)
    bleu_file.close()

    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("BLEU")
    plt.title("BLEU SCORES")
    plt.savefig("./results/score_images/BLEU_scores.png")
    plt.close()

    scores_chrf_dict = {}
    scores_chrf_max_dict = {}
    for output in tqdm(chrf_scores, desc="Plotting CHRF scores"):
        x = list(range(1, len(chrf_scores[output])+1))
        plot_line(chrf_scores[output], x, output)
        scores_chrf_dict[output] = chrf_scores[output]
        filtered_chrf_scores = [output_value for output_value in chrf_scores[output] if output_value is not None]
        max_value = filtered_chrf_scores[np.argmax(filtered_chrf_scores)]
        print(max_value, "max value for chrf")
        scores_chrf_max_dict[output] = max_value

    print("Dumping CHRF scores as JSON")
    scores_chrf_max_dict = {key: float(value) for key, value in scores_chrf_max_dict.items()}
    combined_chrf = {
        "scores chrf": scores_chrf_dict,
        "scores_chrf_max":scores_chrf_max_dict}
    with open('./results/tables/scores_chrf_LSTM.json', 'w') as chrf_file:
        json.dump(combined_chrf, chrf_file)
    chrf_file.close()

    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("CHRF")
    plt.title("CHRF Scores")
    plt.savefig("./results/score_images/CHRF_scores.png")
    plt.close()

