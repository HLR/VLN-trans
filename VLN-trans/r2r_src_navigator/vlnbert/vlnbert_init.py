# Recurrent VLN-BERT, 2020, by Yicong.Hong@anu.edu.au
import sys

from pytorch_transformers import (BertConfig, BertTokenizer)

def get_tokenizer(args):
    if args.vlnbert == 'oscar':
        tokenizer_class = BertTokenizer
        '''
        model_name_or_path = '/egr/research-hlr/joslin/transformer-based-model/OSCAR/base-no-labels/ep_67_588997'
        tokenizer = tokenizer_class.from_pretrained(model_name_or_path, do_lower_case=True)
        '''
        tokenizer_class = BertTokenizer
        tokenizer = tokenizer_class.from_pretrained('bert-base-uncased')
        
    elif args.vlnbert == 'prevalent':
        tokenizer_class = BertTokenizer
        tokenizer = tokenizer_class.from_pretrained('bert-base-uncased', add_special_tokens=False)
    return tokenizer

def get_vlnbert_models(args, config=None):
    config_class = BertConfig

    from vlnbert.vlnbert_PREVALENT import VLNBert
    model_class = VLNBert
    #model_name_or_path = '/egr/research-hlr/joslin/transformer-based-model/Prevalent/pretrained_model/checkpoint-12864/pytorch_model.bin'
    model_name_or_path = '/egr/research-hlr/joslin/pretrain/pretrained_model/checkpoint-47840'
    vis_config = config_class.from_pretrained('bert-base-uncased')
    vis_config.img_feature_dim = 2048+128
    vis_config.img_feature_type = ""
    vis_config.vl_layers = 4
    vis_config.la_layers = 9

    #visual_model = model_class(config=vis_config)
    visual_model = model_class.from_pretrained(model_name_or_path, config=vis_config)

    return visual_model
