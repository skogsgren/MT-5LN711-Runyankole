#!/usr/bin/env python
# coding: utf-8

# # Architecture

# * Lab runs Experiment with a ExperimentConfig which contains all possible values for every single variable we want to tune.
# * Each Experiment contains a Pipeline, which follows the basic ML steps like data -> preprocessor -> training -> testing
# * Data in our case are text files stored on disk.
# * Some intermediate data on disk like BPE outputs from the preprocessor could be needed.

# ![uml.png](attachment:2a89be59-d911-47bd-9300-15d9faca0997.png)

# # Setup

# In[42]:


ROOT_DATA_DIR = "data/"
CONFIG_FILE_TEMPLATE_PATH = "config_template.yaml"
ROOT_TRAINING_DIR = "test_model/"
SHOW_FIGURES = False


# In[43]:


import os.path
from pathlib import Path
from urllib.request import urlretrieve
import subprocess
import gzip
import shutil
import tarfile
import re
import random
import collections
import itertools
import gc

import pandas as pd
from tabulate import tabulate
import numpy as np

# For user feedback
from tqdm import tqdm
import matplotlib.pyplot as plt


# # Experiment Config

# In[44]:


class ExperimentConfig( collections.abc.Iterator ):
    """ An Iterator that will iterates over all possible combination of configs, some config can be fixed.
        Have a default dict of all_possible_config_values mapping each config name to a list of states.
        IN: 
            freeze_configs (dict): a dict containing the name of the config to be fixed 
                                and the index of the state to use from the corresponding list.
        DO NOT MODIFY!!
    """
    def __init__( self, possible_config_values, perc = 1.0, freeze_configs = {}, must_include_combinations = {} ):
        self.config_id = -1
        self.current_config = None
        self.all_possible_config_values = possible_config_values
        self.all_configs = []
        self.gen_all_configs()
        self.config_samples = []
        self.sample_configs( perc, freeze_configs, must_include_combinations )
        self._deduplicate_samples()

    def gen_all_configs( self ):
        self.all_configs = [
            dict( zip( self.all_possible_config_values.keys(), values ) ) 
                for values in itertools.product( *self.all_possible_config_values.values() )
        ]
        
    def _filter_frozen_configs( self, freeze_configs ):
        ## freeze_setting data structure constraint
        for freeze_setting in freeze_configs:
            if( 
                isinstance( freeze_configs[ freeze_setting ], collections.abc.Sequence ) 
                and not isinstance( freeze_configs[ freeze_setting ], (str, bytes, bytearray) )
            ):
                raise ValueError( "A freeze setting value cannot be a collection!" )
        frozen_samples = []
        for config in self.all_configs:
            valid_config = True
            for setting in config: # Check if all settings are correctly freezing
                setting_value = config[ setting ]
                if setting in freeze_configs:
                    if setting_value != freeze_configs[ setting ]:
                        valid_config = False
            if valid_config:
                frozen_samples.append( config )
        return frozen_samples
        
    def sample_configs( self, perc, freeze_configs = {}, must_include_combinations = {} ):
        """Sample down the all possible combinations of configs to only test a few
            freeze_configs: dictionary of {setting name: setting value}
            must_include_combinations: dictionary of {setting name: list of values that must be included in at least one sample}
        """
        ## must_include_combinations data structure constraint
        for must_include_setting in must_include_combinations:
            if not isinstance( must_include_combinations[ must_include_setting ], list ):
                raise ValueError( "All must include setting values must be lists!" )
        ## Check so all setting values in freeze_configs are included in must_include_combinations.
        if len( freeze_configs ) > 0 and len( must_include_combinations ) > 0:
            for freeze_setting in freeze_configs:
                if freeze_setting in must_include_combinations:
                    if freeze_configs[ freeze_setting ] not in  must_include_combinations[ freeze_setting ]:
                        raise ValueError( "All freeze setting value must be included in must include configs!" )
        frozen_samples = self._filter_frozen_configs( freeze_configs )
        
        ## Random sampling by percentage
        for config in frozen_samples:
            rand = random.uniform( 0, 1 )
            if rand < perc:
                self.config_samples.append( config )
                
        ## Artificially add all samples in the must include combinations
        for must_have_comb in (
            dict( zip( must_include_combinations.keys(), values ) ) 
                for values in itertools.product( *must_include_combinations.values() )
        ):
            if( len( must_have_comb ) == 0 ):
                break # Stop the loop when must have is empty [{}]
            current_frozen_configs = self._filter_frozen_configs( must_have_comb )
            if len( current_frozen_configs ) > 0:
                current_must_have_config = random.choice( current_frozen_configs )
                self.config_samples.append( current_must_have_config )
        ##TODO: deduplicate configs
        return self.config_samples

    def get_df_config_samples( self ):
        if len( self.config_samples ) == 0:
            return None
        else:
            df_configs = pd.DataFrame( self.config_samples )
            df_configs = df_configs.set_index( list( df_configs.columns ) ).sort_index()
            return df_configs

    def _deduplicate_samples( self ):
        df_samples = self.get_df_config_samples()
        self.config_samples = df_samples.reset_index().drop_duplicates().to_dict( "records" )
        
    def __iter__( self ):
        return self
        
    def __next__( self ):
        self.config_id += 1
        if( self.config_id >= len( self.config_samples ) ):
            raise StopIteration
        self.current_config = self.config_samples[ self.config_id ]
        return self.current_config


# # Lab

# In[45]:


class Lab():
    """
        DO NOT MODIFY!!
    """
    def __init__( self, possible_config_values, perc = 1.0, freeze_configs = {}, must_include_combinations = {} ):
        self.experiment_config = ExperimentConfig( 
            possible_config_values, 
            perc = perc, 
            freeze_configs = freeze_configs, 
            must_include_combinations = must_include_combinations 
        )
        self.reports = []

    def _add_report( self, config, result ):
        self.reports.append( (config, result) )
        
    def _run_experiment( self ):
        self.current_experiment = Experiment( self )
        current_max_valid_bleu = self.current_experiment.report()
        # gc.collect()
        # torch.cuda.empty_cache()
        self._add_report( self.experiment_config.current_config, current_max_valid_bleu )
        return( current_max_valid_bleu )
    
    def report_all( self ):
        for config in self.experiment_config:
            print( "================================ New experiment ================================" )
            setting_names = list( self.experiment_config.all_possible_config_values.keys() )
            for ii, setting in enumerate( config ):
                print( f"\t{setting_names[ ii ]}: " + str( config[ setting ] ) )
            current_max_valid_bleu = self._run_experiment()
            print( "Max validation BLEU:", current_max_valid_bleu )
        return( self.reports )

    def get_df_reports( self ):
        df_result = pd.DataFrame( 
            [
                list( 
                    [report[ 0 ][ key ] for key in report[ 0 ]]
                ) + [report[ 1 ]] 
                    for report in self.reports
            ]
        )
        df_result.columns = list( self.experiment_config.all_possible_config_values.keys() ) + ["max_bleu"]
        df_result = df_result.set_index( list( df_result.columns[ :-1 ] ) ).sort_index()
        return( df_result )


# # Experiment

# In[46]:


class Experiment():
    """
        DO NOT MODIFY!!
    """
    def __init__( self, lab ):
        self.lab = lab
        self.current_pipeline = None 
        
    def _run_pipeline( self ):
        self.current_pipeline = Pipeline( self )
        return( self.current_pipeline.evaluator() )

    def plot_curves( self ):
        current_bleu_scores = self.current_pipeline.bleu_scores
        current_ppl_scores = self.current_pipeline.ppl_scores
        current_acc_scores = self.current_pipeline.acc_scores
        
        current_train_ppl_scores = self.current_pipeline.train_ppl_scores
        current_train_acc_scores = self.current_pipeline.train_acc_scores
        
        ## Plot BLEU
        fig = plt.figure(figsize=(10, 5))
        ax = plt.subplot()
        ax.plot( current_bleu_scores )
        ax.set_title( "BLEU Score Plot" )
        ax.set_xlabel( 'Epoch (1 epoch per "valid_steps=train_steps" )' )
        ax.set_ylabel( "Validation BLEU" )
        if SHOW_FIGURES: plt.show()
        fig.savefig( self.current_pipeline.current_output_path + "bleu_plot.png" )
        
        ## Plot Perplexity
        fig = plt.figure(figsize=(10, 5))
        ax = plt.subplot()
        ax.plot( current_ppl_scores )
        ax.set_title( "Perplexity Score Plot" )
        ax.set_xlabel( 'Epoch (1 epoch per "valid_steps=train_steps" )' )
        ax.set_ylabel( "Validation Perplexity" )
        if SHOW_FIGURES: plt.show()
        fig.savefig( self.current_pipeline.current_output_path + "ppl_plot.png" )
        
        
        ## Plot Accuracy
        fig = plt.figure(figsize=(10, 5))
        ax = plt.subplot()
        ax.plot( current_acc_scores )
        ax.set_title( "Accuracy Score Plot" )
        ax.set_xlabel( 'Epoch (1 epoch per "valid_steps=train_steps" )' )
        ax.set_ylabel( "Validation Accuracy" )
        if SHOW_FIGURES: plt.show()
        fig.savefig( self.current_pipeline.current_output_path + "acc_plot.png" )
        
        ## Plot Training Perplexity
        fig = plt.figure(figsize=(10, 5))
        ax = plt.subplot()
        ax.plot( current_train_ppl_scores )
        ax.set_title( "Training Perplexity Score Plot" )
        ax.set_xlabel( 'Epoch (1 epoch per "valid_steps=train_steps" )' )
        ax.set_ylabel( "Training Perplexity" )
        if SHOW_FIGURES: plt.show()
        fig.savefig( self.current_pipeline.current_output_path + "train_ppl_plot.png" )

        ## Plot Training Accuracy
        fig = plt.figure(figsize=(10, 5))
        ax = plt.subplot()
        ax.plot( current_train_acc_scores )
        ax.set_title( "Training Accuracy Score Plot" )
        ax.set_xlabel( 'Epoch (1 epoch per "valid_steps=train_steps" )' )
        ax.set_ylabel( "Training Accuracy" )
        if SHOW_FIGURES: plt.show()
        fig.savefig( self.current_pipeline.current_output_path + "train_acc_plot.png" )

    
    def report( self ):
        current_max_valid_bleu = self._run_pipeline()
        self.plot_curves()
        return( current_max_valid_bleu )


# # Models

# Should be taken cared of by ONMT.

# # Main Pipeline

# In[47]:


def custom_collate( input_samples ):
    """ A custom collate function for collating input tensors in batch together, used by the Dataloaders.    """
    targets_stack = torch.stack( [sample[ 0 ] for sample in input_samples] )
    ids_stack = torch.stack( [sample[ 1 ] for sample in input_samples] ) if input_samples[ 0 ][ 1 ] is not None else None
    masks_stack = torch.stack( [sample[ 2 ] for sample in input_samples] ) if input_samples[ 0 ][ 2 ] is not None else None
    ls_text = [sample[ 3 ] for sample in input_samples]
    batch_data = (targets_stack, ids_stack, masks_stack, ls_text)
    return batch_data


# In[48]:


class Pipeline():
    def __init__( self, experiment ):
        self.experiment = experiment
        self.config = self.experiment.lab.experiment_config
        self.dataset_train = None
        self.dataset_test = None
        self.train_iterator = None
        self.test_iterator = None
        self.batch_size = 512
        self.tokenizer = None

        self.current_output_dir = f"output_{self.config.config_id}" # Needed for the output paths in configs
        self.current_output_path = f"{ROOT_TRAINING_DIR}/{self.current_output_dir}/" # used for training output and read back training log for eval
        self.bleu_scores = []
        self.ppl_scores = []
        self.acc_scores = []
        
        self.train_ppl_scores = []
        self.train_acc_scores = []
        
        self._generate_config_file()
        
    def _load_config_file_template( self ):
        with open( CONFIG_FILE_TEMPLATE_PATH, 'r' ) as ff:
            return( ff.read() )
    def _subs_config_keyword( self, config_str, key, val ):
        return( re.sub( f"\\b{str( key )}\\b", str( val ), config_str ) )
        
    def _generate_config_file( self ):
        tempalte_config_str = self._load_config_file_template()
        os.makedirs( self.current_output_path, exist_ok = True )
        current_config_str = self._subs_config_keyword( tempalte_config_str, "OUTPUT_PATH_PER_RUN", self.current_output_dir )
        for key, val in self.config.current_config.items():
            if( key == "WORD_VEC_SIZE" ):
                current_config_str = self._subs_config_keyword( current_config_str, "WORD_VEC_SIZE", self.config.current_config[ "HIDDEN_SIZE" ] )
            elif( key == "DEC_LAYERS" ):
                current_config_str = self._subs_config_keyword( current_config_str, "DEC_LAYERS", self.config.current_config[ "ENC_LAYERS" ] )
            else:
                # print( f"\t\tSubstituting config variable string {key} with {str( val )}" ) #DEBUG
                current_config_str = self._subs_config_keyword( current_config_str, key, val )
        with open( f"{ROOT_TRAINING_DIR}/current_config.yaml", 'w' ) as ff:
            ff.write( current_config_str )
        os.makedirs( f"{ROOT_TRAINING_DIR}/configs/", exist_ok = True )
        with open( f"{ROOT_TRAINING_DIR}/configs/config_{self.config.config_id}.yaml", 'w' ) as ff: # store all the config files
            ff.write( current_config_str )
            
    def _prepare_tokenizer( self ): #LEGACY: may not need
        print( "_prepare_tokenizer( self )" )
        # if( self.config.current_config[ "model_name" ] != "baseline" ):
        #     self.tokenizer = AutoTokenizer.from_pretrained( self.config.current_config[ "model_name" ], model_max_length = 128 )

    def _prepare_model( self ):
        print( "Preparing model:", self.config.current_config )
        # if( self.config.current_config[ "model_name" ] == "baseline" ):
        #     self.model = self.config.current_config[ "model_name" ]
        # else:
        #     self.model = ...
            
    def _preprocessor( self ):
        """ Run bash preprocessor

        OUT: make DATA_PATH ready to use.
        """
        print( "Preprocessing..." )
        
    def _trainer( self ):
        # if( self.config.current_config[ "model_name" ] == "baseline" ):
        #     print( "Baseline model, training will be skipped." )
        #     return 0
        print( "Training..." )
        with open("ml_experiments.log", "a") as output_file:
            subprocess.run( 
                ["bash", "train_test_model.sh"],
                cwd = os.path.join( os.getcwd(), "test_model" ),
                stdout = output_file,
                stderr = output_file,
                text = True 
            )
        return( 0 )
        # self.model.train()
        # loss_function = nn.BCELoss() # A loss function that fits our choice of output layer and data. The
        # ...
        # return( max( self.model.training_accuracy_ ) )

    def _evaluator( self ):
        """ OUTPUT: self.bleu_scores/ppl_scores/acc_scores
        """
        with open( f"{self.current_output_path}/train.log", 'r' ) as ff:
            training_log = ff.readlines()
        
        for line in training_log:
            ## Extract BLEUs
            match_bleu = re.search(r"validation BLEU:\s*(nan|[0-9.]+)", line)
            if match_bleu:
                self.bleu_scores.append( float( match_bleu.group( 1 ) ) )
            ## Extract Perplexity
            match_ppl = re.search(r"Validation perplexity:\s*(nan|[0-9.]+)", line)
            if match_ppl:
                self.ppl_scores.append( float( match_ppl.group( 1 ) ) )
            ## Extract Accuracy
            match_acc = re.search(r"Validation accuracy:\s*(nan|[0-9.]+)", line)
            if match_acc:
                self.acc_scores.append( float( match_acc.group( 1 ) ) )
                
            ## Extract TRAINING Perplexity
            match_ppl_train = re.search(r"Train perplexity:\s*(nan|[0-9.]+)", line)
            if match_ppl_train:
                self.train_ppl_scores.append( float( match_ppl_train.group( 1 ) ) )
            
            ## Extract TRAINING Accuracy
            match_acc_train = re.search(r"Train accuracy:\s*(nan|[0-9.]+)", line)
            if match_acc_train:
                self.train_acc_scores.append( float( match_acc_train.group( 1 ) ) )
        ## log scores to files
        with open( f"{self.current_output_path}/bleu.log", 'w' ) as ff:
            ff.write( "\r\n".join( [str( score ) for score in self.bleu_scores] ) )
        with open( f"{self.current_output_path}/ppl.log", 'w' ) as ff:
            ff.write( "\r\n".join( [str( score ) for score in self.ppl_scores] ) )
        with open( f"{self.current_output_path}/acc.log", 'w' ) as ff:
            ff.write( "\r\n".join( [str( score ) for score in self.acc_scores] ) )
        with open( f"{self.current_output_path}/train_ppl.log", 'w' ) as ff:
            ff.write( "\r\n".join( [str( score ) for score in self.train_ppl_scores] ) )
        with open( f"{self.current_output_path}/train_acc.log", 'w' ) as ff:
            ff.write( "\r\n".join( [str( score ) for score in self.train_acc_scores] ) )

    def evaluator( self ):
        """ Evaluate and report/store scores
        """
        print( "Evaluator..." )
        self._trainer()
        self._evaluator()
        max_valid_bleu = max( self.bleu_scores )
        return( max_valid_bleu )


# # Production Run

# In[53]:


HIDDEN_SIZES = [128, 256, 512]
possible_config_values = {
    "DATASET_NAME_TRAIN": [
        "original", 
        # "bible", 
        # "all"
    ],
    "DATASET_NAME_VALID": [
        "original",
        # "bible", 
        # "all"
    ],
    "TRAIN_STEPS": [200*100],
    "METRIC": ["BLEU"],
    "LEARNING_RATE": [0.002, 0.004],
    "LEARNING_RATE_DECAY": [0.95],
    "START_DECAY_STEPS": [2304],
    "WARMUP_STEPS": [1],
    "DECAY_METHOD": [
        "none",
        # "noamwd",
        # "rsqrt", 
        # "noam"
    ],
    "ENC_LAYERS":[2, 4],
    "DEC_LAYERS":[-1], # "left empty with a -1"  to match ENC_LAYERS
    "HEADS": [2, 4],
    "HIDDEN_SIZE": [64, 128],
    "WORD_VEC_SIZE": [-1], # "left empty with a -1"  to match HIDDEN_SIZE
    "TRANSFORMER_FF": [128, 256],
    "DROPOUT": ["[0.2]"],
    "ATTENTION_DROPOUT": ["[0.0]", "[0.2]"],
}
FREEZE_CONFIGS = {
    # "TRAIN_STEPS": possible_config_values[ "TRAIN_STEPS" ][ 0 ],
    # "LEARNING_RATE": possible_config_values[ "LEARNING_RATE" ][ 0 ],
    # "DECAY_METHOD": possible_config_values[ "DECAY_METHOD" ][ 0 ],
    # "ENC_LAYERS": possible_config_values[ "ENC_LAYERS" ][ 0 ],
    # "DEC_LAYERS": possible_config_values[ "DEC_LAYERS" ][ 0 ],
    # "HEADS": possible_config_values[ "HEADS" ][ 0 ],
    # "HIDDEN_SIZE": possible_config_values[ "HIDDEN_SIZE" ][ 0 ],
    # "TRANSFORMER_FF": possible_config_values[ "TRANSFORMER_FF" ][ 0 ],
}
MUST_INCLUDE_COMBINATIONS = {}

test_config = ExperimentConfig( 
    possible_config_values,
    perc = 1.0,
    freeze_configs = FREEZE_CONFIGS, 
    must_include_combinations = MUST_INCLUDE_COMBINATIONS
)
if SHOW_FIGURES:
    display( test_config.get_df_config_samples() )
else:
    print( test_config.get_df_config_samples().reset_index() )

with open("current_config_table.log", "w") as ff:
    ff.write( str( test_config.get_df_config_samples().reset_index() ) )


# Here the main program will start to run and report all possible configs, the results can be seen in the final dataframe.

# In[51]:


if __name__ == "__main__": # Multithreading Guard
    mt_lab = Lab( 
        possible_config_values,
        perc = 1.0,
        freeze_configs = FREEZE_CONFIGS, 
        must_include_combinations = MUST_INCLUDE_COMBINATIONS
    )
    lab_result = mt_lab.report_all()
    if SHOW_FIGURES:
        display( mt_lab.get_df_reports() )
    else:
        print( mt_lab.get_df_reports().reset_index()  )

with open("final_search_scores.log", "w") as ff:
    ff.write( str( mt_lab.get_df_reports().reset_index() ) )


# In[ ]:




