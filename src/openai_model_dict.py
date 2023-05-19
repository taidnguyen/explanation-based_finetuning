"""
Store IDs of finetuned OpenAI models for evaluation
"""

esnli_finetuned_models = {
    'adv':{  # Explanation-based finetuning
        '0':'',  # INPUT: Unbiased finetuned model ID
        '100':{  # Bias strength 1.0
            'lowercase':'',  # INPUT: Model ID for lowercase bias
            'female':'',
            'cluster':'',
            'present': '',
            'length':'',
            'plural':'',
        },
    },
    'basic':{ # Basic finetuning
        '0':'',
        '100':{
            'lowercase':'',
            'female':'',
            'cluster':'',
            'present': '',
            'length':'',
            'plural':'',
        },
    },
}

comve_finetuned_models = {
    'adv':{
        '0':'',
        '100':{
            'lowercase':'',
            'female':'',
            'cluster':'',
            'present': '',
            'length':'',
            'plural':'',
        },
    },
    'basic':{
        '0':'',
        '100':{
            'lowercase':'',
            'female':'',
            'cluster':'',
            'present': '',
            'length':'',
            'plural':'',
        },
    },
}
sbic_finetuned_models = {
    'adv':{
        '0':'',
        '100':{
            'lowercase':'',
            'female':'',
            'cluster':'',
            'present': '',
            'length':'',
            'plural':'',
        },
    },
    'basic':{
        '0':'',
        '100':{
            'lowercase':'',
            'female':'',
            'cluster':'',
            'present': '',
            'length':'',
            'plural':'',
        },
    },
}

creak_finetuned_models = {
    'adv':{
        '0':'',
        '100':{
            'lowercase':'',
            'female':'',
            'cluster':'',
            'present': '',
            'length':'',
            'plural':'',
        },
    },
    'basic':{
        '0':'',
        '100':{
            'lowercase':'',
            'female':'',
            'cluster':'',
            'present': '',
            'length':'',
            'plural':'',
        },
    },
}

all_dictionaries = {
    'esnli': esnli_finetuned_models,
    'creak': creak_finetuned_models,
    'comve': comve_finetuned_models,
    'sbic': sbic_finetuned_models,
}
