"""
This script evaluate model-generated predictions.
Result will be saved to  <output_dir>/test_<task_name>_<bias>_<with_expl>_<method>.csv
"""


from argparse import ArgumentParser
from getpass import getpass
from filter_train import *
from check_bias_functions import *
from model_dicts import *
from sklearn.metrics import f1_score
from scipy.stats import pearsonr

import openai
from transformers import GPT2Tokenizer, GPT2LMHeadModel

zero_shot_prefix = {
    'esnli': {
        'basic' : """Answer the following question with true/false, followed by ###, like so:
Question: [QUESTION]
Answer: [ANSWER]###\nQuestion: """,
        'adv': """Answer the following question with your thoughts followed true/false, followed by ###, like so:
Question: [QUESTION]
Thoughts: [THOUGHTS]
Answer: [ANSWER]###\nQuestion: """
    },
    'creak':{
        'adv': """Answer the following statement with your thoughts followed true/false, followed by ###, like so:
claim: [CLAIM]\nthoughts: [THOUGHTS]\nAnswer: [ANSWER]###\n""",
        'basic': """Answer the following statement with true/false, followed by ###, like so:
claim: [CLAIM]\nAnswer: [ANSWER]###\n""",
    },
    'comve':{
        'adv':'',
        'basic':'',
    },
    'sbic':{
        'adv':'Is the following post offensive?',
        'basic':'Is the following post offensive? Generate thoughts and answer true or false?\n',
    }
}

def count_gpt2_tokens(text):
  tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
  tokens = tokenizer.tokenize(text)
  return len(tokens)
#Count all the tokens in finetune dataset
def compute_token_sum(df):
  df['prompt_tokens'] = df['prompt'].apply(count_gpt2_tokens)
  df['completion_tokens'] = df['completion'].apply(count_gpt2_tokens)
  sumTokens = df['prompt_tokens'].sum() + df['completion_tokens'].sum()
  return sumTokens

def get_model_name(args, adv_basic):
    print('getting model name: ',args.task_name, adv_basic, args.bias)
    if args.bias == 'unbiased':
        return all_dictionaries[args.task_name][adv_basic]['0']
    return all_dictionaries[args.task_name][adv_basic]['100'][args.bias]

def find_completion(model,temp,prompt, MAX_TOKENS = 100):
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens = MAX_TOKENS,
        stop=["\n###\n"],
        temperature=temp)
    return response['choices'][0]['text']

def report_accuracy(df,true_class, model,temp, output_path, advanced = True, inferenced = False, report_corr = True):
    if not inferenced:
        print('inferencing')
        df['inferenced'] = df['prompt'].apply(lambda x: find_completion(model,temp,x))

    def get_advanced_lab_adv(x):
        try:
        #   if args.task_name == 'esnli':
            return x.rstrip('\n###\n\n').split('\nAnswer: ')[1]
        except:
            try:
                # return x.rstrip('\n###\n\n').strip(' ')
                # for few-shot-over-finetuned
                return x.rstrip('\n###\n\n').strip(' ')
            except:
                print('error when extracing predicted or true y, completion: ',x)
                return ''
    def get_basic_lab(x):
        try:
            return x.rstrip('\n###\n\n').strip(' ')
        except:
            print('error when extracing predicted or true y, completion: ',x)
            return ''
    df = df.reset_index(drop=True)
    print('example generation: ',df['inferenced'][0])
    if advanced:
        df['pred_y'] = df['inferenced'].apply(lambda x: get_advanced_lab_adv(x))
        df['true_y'] = df['completion'].apply(lambda x: get_advanced_lab_adv(x))
    else:
        df['pred_y'] = df['inferenced'].apply(lambda x: get_basic_lab(x))
        df['true_y'] = df['completion'].apply(lambda x: get_basic_lab(x))

    df.to_csv(output_path,index=False)
    print('result saved to ', output_path)
    acc = accuracy_score(df['pred_y'], df['true_y'])

    df['pred_y'] = df['pred_y'].apply(lambda x: 1 if x==true_class else 0)
    df['true_y'] = df['true_y'].apply(lambda x: 1 if x==true_class else 0)
    f1 = f1_score(df['pred_y'], df['true_y'])
    print('acc:{}, f1:{}'.format(acc, f1))

    if report_corr:
        res = pearsonr(df['featurePresent'], df['pred_y'])
        print('corr: ', res[0])
    return acc, df

def zero_prompt(advanced, test_df):
    if not advanced:
        prefix = zero_shot_prefix[args.task_name]['basic']
    else:
        prefix = zero_shot_prefix[args.task_name]['adv']
    zero_shot_test = test_df.copy()
    zero_shot_test['prompt'] = zero_shot_test['prompt'].apply(lambda x: prefix + x)
    assert len(zero_shot_test) == 500

    sumTokens = compute_token_sum(zero_shot_test)
    costPer1000TokensDavinci = 0.12/4
    cost = sumTokens/1000 * costPer1000TokensDavinci
    print('cost:',cost)
    return zero_shot_test

def find_zero_shot_completion(x, max_token):
  response = openai.Completion.create(
    model="davinci",
    prompt=x,
    temperature=0,
    max_tokens=max_token,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
  )
  return response['choices'][0]['text']

def report_zero_shot_acc(df,output_path, advanced, inferenced = True):
    max_token = 50 if advanced else 20
    if not inferenced:
        print('inferencing, max token', max_token)
        df['inferenced'] = df['prompt']\
            .apply(lambda x: find_zero_shot_completion(x, max_token))

    df.to_csv(output_path,index=False)
    print('result saved to ', output_path)
    exit()

    if advanced:
        if args.task_name == 'esnli': # e.g. inferenced in esnli: ""yes. the premise entails the hypothesis"
            df['pred_y'] = df['inferenced'].apply(lambda x: x.split('.')[0].strip(' '))
        else:
            print('TODO')

        df['true_y'] = df['completion'].apply(lambda x: x.rstrip('\n###\n\n').split('\nAnswer: ')[1])
    else:
        if args.task_name == 'esnli':
            df['pred_y'] = df['inferenced'].apply(lambda x: x.split('.')[0].strip(' '))
        else:
            print('TODO')

        df['true_y'] = df['completion'].apply(lambda x: x.rstrip('\n###\n\n').strip(' '))

    df.to_csv(output_path,index=False)
    print('result saved to ', output_path)

    def get_label_esnli(x):
        if x.lower() == 'yes':
            return 1
        elif x.lower() == 'no':
            return 0
        elif x.lower().startswith('yes'): return 1
        elif x.lower().startswith('no'): return 0
        else: print(x); return 0.5

    if args.task_name == 'esnli':
        df['pred_y'] = df['pred_y'].apply(lambda x: get_label_esnli(x))
        df['true_y'] = df['true_y'].apply(lambda x: 1 if x=='true' else 0)
    else:
        print('TODO')

    tmp = df[df['pred_y']!=0.5]
    acc = accuracy_score(df['pred_y'], df['true_y'])
    f1 = f1_score(tmp['pred_y'], tmp['true_y'])
    print('acc:{}, f1:{}'.format(acc, f1))

    return acc, df

def get_few_shot_prompt(df, true_class, test_df, N_shot=10,advanced = True):
    def get_few_shot_prompt_single():
        true_train = df[(df['completion'].str.contains(true_class)) & (~df['completion'].str.contains('not'))].reset_index(drop = True)
        false_train = df[(~df['completion'].str.contains(true_class)) | (df['completion'].str.contains('not'))].reset_index(drop = True)

        # random.seed(42)
        indices = list(range(len(true_train)))
        random.shuffle(indices)
        sample_df_true = true_train.iloc[indices[:N_shot//2]]

        indices = list(range(len(false_train)))
        # random.seed(42)
        random.shuffle(indices)
        sample_df_false = false_train.iloc[indices[:N_shot//2]]
        sample_df = pd.concat([sample_df_true, sample_df_false])
        if N_shot == 6:
            sample_df = sample_df.iloc[[0,3,1,4,2,5]]
        if N_shot == 10:
            sample_df = sample_df.iloc[[0,5, 1,6, 2,7, 3,8, 4,9]]

        sample_df['combined'] = sample_df['prompt'] + sample_df['completion']
        few_shot_prompt = ''.join(list(sample_df['combined']))
        return few_shot_prompt
    few_shot_prompts = [get_few_shot_prompt_single() for _ in range(len(test_df))]

    few_shot_df = test_df
    few_shot_df['prompt_prefix'] = few_shot_prompts
    few_shot_df['prompt'] = few_shot_df['prompt_prefix'] + few_shot_df['prompt']

    return few_shot_df

def do_few_shot(hold_out_train, true_class, test_df, advanced, N_shot=10):
    # hold_out_train = create_finetuning_dataset(hold_out_train, 'hold_out_train.csv', advanced=advanced)
    hold_out_train = create_finetuning_dataset(args, hold_out_train, advanced=advanced)
    few_shot_df = get_few_shot_prompt(hold_out_train, true_class, test_df, N_shot=N_shot, advanced = advanced)
    assert len(few_shot_df) == 500

    sumTokens = compute_token_sum(few_shot_df)
    costPer1000TokensDavinci = 0.12/4
    cost = sumTokens/1000 * costPer1000TokensDavinci
    print('cost:',cost)
    return few_shot_df

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--task_name', type=str, default='esnli',
                            help='task name, choose from esnli, creak, comve, sbic')
    parser.add_argument('--data_dir', type=str, default='../../data',
                            help='data directory')
    parser.add_argument('--output_dir', type=str, default='../../res',
                            help='data directory')
    parser.add_argument('--bias', type=str, default='present',
                            help='type of bias')

    parser.add_argument('--model_name', type=str, default='',
                            help='specified model name')

    parser.add_argument('--bias_strength', type = float, default=1,
                            help = 'bias strength: 0 to 1')
    parser.add_argument('--train_size', type = int, default=1000,
                            help = 'data size of training set')

    parser.add_argument('--with_expl', default=0, action='store_true',
                        help='Boolean value, True for explanation-based, False for standard')
    parser.add_argument('--expl_temp', default=0, action='store_true',
                        help='Use a template to format the explanations')

    parser.add_argument('--method', type=str, default='finetuned',
                            help='choose from zero-shot, few-shot-without-finetune, few-shot-over-finetuned, finetuned')
    args = parser.parse_args()

    if args.task_name not in ['creak', 'esnli', 'sbic', 'comve']:
        print("Please choose task name from {}".format(['creak', 'esnli', 'sbic', 'comve']))
        exit()
    biases = ['unbiased', 'length', 'present', 'cluster', 'plural'] + ['perplexity', 'swapped', 'female', 'retweet']
    if args.bias not in biases:
        print("Please choose bias from {}".format(biases))
        exit()

    if args.bias_strength < 0 or args.bias_strength > 1:
        print('illegal bias strength: {}'.format(args.bias_strength))
        exit()
    if args.bias == 'unbiased': args.bias_strength = 0

    shot_list = ['zero-shot', 'few-shot-without-finetune', 'few-shot-over-finetuned', 'finetuned']
    if args.method not in shot_list:
        print("Please choose method from {}".format(shot_list))
        exit()

    if args.method in ['zero-shot', 'few-shot-without-finetune'] and args.bias!='unbiased':
        print('zero-shot and few-shot-without-finetune are baselines, do you mean unbiased?')
        exit()

    print("------ task: {}, bias: {}, method: {}, with_expl: {}, bias strength: {}, train size: {} ------------".format(
        args.task_name, args.bias, args.method, args.with_expl,\
        args.bias_strength, args.train_size))

    adv_basic = 'adv' if args.with_expl else 'basic'

    print('Enter OpenAI API key:')
    openai.api_key = getpass() # input()

    os.environ['OPENAI_API_KEY']=openai.api_key

    if args.method == 'zero-shot' or args.method == 'few-shot-without-finetune':
        model_name = 'davinci'
        print('Using model: {}'.format(model_name))
    elif args.method == 'finetuned' and args.model_name=='':
        model_name = get_model_name(args, adv_basic)
        print('Using model: {}'.format(model_name))

    unbiased_train, train_10k, test_200, test_500, label_col, true_class = preprocess_data(args)
    test_df = test_500

    if args.bias != 'unbiased':
        train_full = create_finetuning_dataset(args, train_10k, advanced=True)
        test_df = create_finetuning_dataset(args, test_df, advanced=True)
        if args.bias == 'length':
            median = check_bias_distribution(args, true_class, train_full, istrain = True, label_col = label_col)
            check_bias_distribution(args, true_class, test_df, istrain = False, median = median, label_col = label_col)
        elif args.bias == 'cluster':
            kmeans = check_bias_distribution(args, true_class, train_full, istrain = True, label_col = label_col)
            check_bias_distribution(args, true_class, test_df, istrain = False, label_col = label_col, kmeans = kmeans)
        else:
            check_bias_distribution(args, true_class, train_full, istrain = True, label_col = label_col)
            check_bias_distribution(args, true_class, test_df, istrain = False, label_col = label_col)

    if args.bias == 'unbiased':
        trainPromptCompletionSimple,testPromptCompletionSimple, \
            trainPromptCompletionAdvanced, testPromptCompletionAdvanced = \
            get_prompt_datasets(args, true_class, unbiased_train, train_10k, test_df, unbiased = True, label_col = label_col, save = False)
    else:
        trainPromptCompletionSimple,testPromptCompletionSimple, \
            trainPromptCompletionAdvanced, testPromptCompletionAdvanced = \
                get_prompt_datasets(args, true_class, unbiased_train, train_full, test_df = test_df, label_col = label_col, save = False)
    if args.with_expl: test_df = testPromptCompletionAdvanced
    else: test_df = testPromptCompletionSimple
    print(test_df, len(test_df))
    if args.bias == 'unbiased': test_df['featurePresent'] = True

    if args.method == 'finetuned':
        if args.model_name!='':
            model_name = args.model_name
            print('Using specified model: {}'.format(model_name))
        file_name = os.path.join(args.output_dir, 'test_'+args.task_name + '_' + args.bias + '_' + adv_basic + '_' + args.method)
        if args.expl_temp: file_name += '_expl_temp.csv'
        else: file_name += '.csv'
        acc, test_res = report_accuracy(test_df, true_class, model_name, 0, \
                      file_name,
                      advanced = args.with_expl, report_corr=(args.bias!='unbiased'))

    elif 'few-shot' in args.method:
        train_df =  trainPromptCompletionSimple[train_10k.columns]
        hold_out_train = train_10k.merge(train_df, how = 'outer', indicator = True).loc[lambda x : x['_merge']=='left_only']
        print('length of hold-out training set: ',len(hold_out_train))

        if args.method == 'few-shot-over-finetuned':

            if args.model_name!='':
                model_name = args.model_name
                print('Using specified model: {}'.format(model_name))
            else:
                model_name = get_model_name(args, 'basic')
                print('Using model: {}'.format(model_name))
            few_shot_df = do_few_shot(hold_out_train, true_class, test_df, advanced=args.with_expl)

            acc,  test_res= report_accuracy(few_shot_df, true_class, model_name, 0, \
                    os.path.join(args.output_dir, 'test_'+args.task_name + '_' + args.bias + '_' + adv_basic + '_' + args.method+'.csv'),\
                    advanced = args.with_expl, inferenced = False, report_corr=(args.bias!='unbiased'))

        elif args.method == 'few-shot-without-finetune':
            few_shot_df = do_few_shot(hold_out_train, true_class, test_df, advanced=args.with_expl)

            acc,  test_res= report_accuracy(few_shot_df,true_class, 'davinci', 0, \
                    os.path.join(args.output_dir,'test_'+args.task_name + '_' + args.bias + '_' + adv_basic + '_' + args.method+'.csv'), \
                    advanced = args.with_expl, inferenced = False, report_corr = False)

    elif args.method == 'zero-shot':
        zero_shot_test = zero_prompt(args.with_expl, test_df)
        zero_shot_test.to_csv(os.path.join(args.output_dir,'test_zeroshot_'+args.task_name + '_' + args.bias + '_' + adv_basic + '_' + args.method+'.csv'))
        acc, test_res = report_zero_shot_acc(zero_shot_test,\
                     os.path.join(args.output_dir,'test_'+args.task_name + '_' + args.bias + '_' + adv_basic + '_' + args.method+'.csv'), \
                     advanced=args.with_expl, inferenced = False)
