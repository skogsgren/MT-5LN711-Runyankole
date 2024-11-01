import os
import glob
import re
import subprocess
from tqdm import tqdm
from sacrebleu.metrics import BLEU, CHRF
import matplotlib.pyplot as plt
import argparse as args

PATH = './experiments/run/'
SRC_PATH = './data/original/nyn_test.bpe'
TGT_PATH = './data/original/eng_test.bpe'
pattern = 'preds_*'

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
    result = subprocess.run(
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
    pred_paths = glob.glob("./results/output_*/preds*")
    return pred_paths

def open_file(file):
    with open(file, 'r', encoding='utf-8') as f:
        contents = f.read().splitlines()
    return contents

def calculate_bleu(source, tgt):
    bleu_score = bleu.corpus_score(
                            source,
                            tgt
    ).score
    return bleu_score

def calculate_chrf(source, tgt):
    chrf_score = chrf.corpus_score(
                        source,
                        tgt
    ).score
    return chrf_score


def calculate_metrics(preds):
    output_chrf = {}
    output_bleu = {}
    target_file = open_file(TGT_PATH)

    for pred in preds:
        source_file = open_file(pred)
        name_output = output_name(pred)

        try:

            if name_output not in output_bleu.keys() and name_output not in output_chrf.keys():
                print(f"Creating key: {name_output}.")
                output_bleu[name_output] = []
                output_chrf[name_output] = []
                output_bleu[name_output].append(calculate_bleu(source_file, target_file))
                output_chrf[name_output].append(calculate_chrf(source_file, target_file))
            else:
                output_bleu[name_output].append(calculate_bleu(source_file, target_file))
                output_chrf[name_output].append(calculate_chrf(source_file, target_file))

        except IndexError:
            pass

    return output_bleu, output_chrf

def plot_line(output, x, name):    
    plt.plot(x, output, label = f'{name}')


if __name__ == '__main__':

    outputs = find_directories()
    bleu = BLEU()
    chrf = CHRF()
#   if not os.path.exists('./results/preds_model_nyn_eng_step_128.txt'): #Uncomment if you wish to skip the translation part
    for output in tqdm(range(len(outputs))):
        directory = check_directory(outputs[output])
        models = find_models(outputs[output])
        for model in tqdm(range(len(models))):
            model_name = models[model][:-3]
            model = outputs[output] + models[model]
            result = run_model(model, directory, model_name)

    preds = find_predictions()
    bleu_scores, chrf_scores = calculate_metrics(preds)

    if not os.path.exists('./results/score_images'):
        os.makedirs("./results/score_images")    

    for output in bleu_scores:
        x = list(range(1, len(bleu_scores[output])+1))
        plot_line(bleu_scores[output], x, output)

    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("BLEU")
    plt.title("BLEU SCORES")
    plt.savefig("./results/score_images/BLEU_scores.png")

    for output in chrf_scores:
        x = list(range(1, len(chrf_scores[output])+1))
        plot_line(chrf_scores[output], x, output)

    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("CHRF")
    plt.title("CHRF Scores")

    if not os.path.exists('./results/score_images'):
        os.makedirs("./results/score_images")

    plt.savefig("./results/score_images/CHRF_scores.png")
    
    #TODO make each bleu score correlate to each output_* directory, so that it can be plotted
    #TODO make a CHRF scorer, and make it do the same
    #TODO plot it and save the plots. 
    #TODO Divide everything into nice functions
