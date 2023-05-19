from argparse import ArgumentParser
from construct_data import *
from sklearn.metrics import f1_score
import torch

import random
import sys
import time
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from torch import nn
from scipy.stats import pearsonr
from torch.utils.data import TensorDataset, DataLoader
import warnings; warnings.filterwarnings('ignore')
from transformers import AutoModelForCausalLM, AutoTokenizer, top_k_top_p_filtering
from torch.nn import functional as F


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_dataloader(args, tokenizer, df, max_len=80, batch_size = 8, shuffle = True):
  tokenized_df=tokenizer.batch_encode_plus(df['prompt'].to_numpy().tolist(),
                  max_length=max_len, padding='max_length', truncation=True, return_tensors='pt')
  df['completion'] = df['prompt'] + df['completion']
  labels_data = tokenizer.batch_encode_plus(df['completion'].to_numpy().tolist(),
                  max_length=max_len, padding='max_length', truncation=True, return_tensors='pt')['input_ids']

  cur_dataset=TensorDataset(tokenized_df['input_ids'],tokenized_df['attention_mask'],labels_data)
  cur_dataloader=DataLoader(cur_dataset, batch_size=batch_size, shuffle=shuffle)
  return cur_dataloader

def gen_adv_label(x):
  try:
    x = x.split('\nAnswer:')[1].strip()
    return x.split('\n')[0].strip()
  except:
    try:
      fields = x.split('###')[0].strip(' ').strip('\n').split('')
      if len(fields) < 2 or fields[-2]!='not':
        return fields[-1]
      if fields[-1] == 'offensive' and fields[-2] == 'not':
        return 'not offensive'
      return fields[-1]
    except:
      try:
        return x.rstrip('\n###\n\n').split()[-1]
      except:
        # print()
        return ''

def save_results(args, model, test_df):
  val_loss, val_accuracy, val_f1, corr, all_pred_y, all_true_y, generated_texts = validate(args,model, \
                                  true_class, test_dataloader, true_labels=list(test_df['completion']), featurePresent = list(test_df['featurePresent']), advanced = args.with_expl)
  test_df['inferenced'] = generated_texts
  test_df['pred_y'] = all_pred_y
  test_df['true_y'] = all_true_y
  if args.with_expl:
    file_path = os.path.join(args.output_dir,'testAdvanced_'+args.task_name+'_'+args.bias+'_'+args.method+'_'+args.model_type.replace('/','-')+'_{}epoch_{}bsz_{}lr.csv'.format(args.num_epochs, args.batch_size, args.learning_rate))
    test_df.to_csv(file_path)
  else:
    file_path = os.path.join(args.output_dir,'testSimple_'+args.task_name+'_'+args.bias+'_'+args.method+'_'+args.model_type.replace('/','-')+'_{}epoch_{}bsz_{}lr.csv'.format(args.num_epochs, args.batch_size, args.learning_rate))
    test_df.to_csv(file_path)
  print('Saved to {}'.format(file_path))

def train(args, model, device, true_class, optimizer, tokenizer, train_dataloader, validation_dataloader,
          num_epochs, true_labels,featurePresent, advanced, scheduler=None, clip = False, file_path = None):
    """
    Train the T5 model for text generation

    Parameters:
    model: T5 model for text generation
    tokenizer: T5 tokenizer
    train_dataloader: Dataloader for the training set
    validation_dataloader: Dataloader for the validation set
    num_epochs: Number of epochs to train the model for

    Returns:
    None
    """
    if args.verbose: print('-'*10+'training'+'-'*10)
    starttime = time.time()

    # Set the model to training mode
    model.train()

    # Define the optimizer and the loss function
    criterion = torch.nn.CrossEntropyLoss()
    divider = len(train_dataloader)//5

    # Loop over the number of epochs
    for epoch in range(num_epochs):
        train_loss = 0
        n=len(train_dataloader)

        generated_texts, true_texts = [], []
        # Loop over the training set
        for idx, (inputs,att, targets) in enumerate(train_dataloader):
            targets = targets.to(device)
            if args.verbose and divider != 0 and (idx + 1) % divider == 0: print('train {}/{}'.format(idx+1, n))
            optimizer.zero_grad()

            # Forward pass
            output = model(targets, labels=targets)
            # loss = output[0]
            loss = output.loss

            # Backward pass
            loss.backward()
            if clip: torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            if scheduler: scheduler.step()
            train_loss += loss.item()
        train_loss /= len(train_dataloader)

        # Validate the model
        val_loss, val_accuracy, val_f1, corr, all_pred_y, all_true_y, generated_texts = validate(args,model, true_class, validation_dataloader, true_labels, featurePresent, advanced = advanced)

        # Print the validation loss and accuracy
        print(f'Epoch {epoch+1}: Train Loss = {train_loss:.4f}') #  Train Accuracy = {train_acc:.4f}, Train F1 = {train_f1:.4f}
        if args.bias!='unbiased':
          print(f'  Validation Loss = {val_loss:.4f}, \
Validation Accuracy = {val_accuracy:.4f}, Validation F1 = {val_f1:.4f}, corr = {corr:.4f}, \
Time {(time.time() - starttime):.2f}')
        else:
          print(f'  Validation Loss = {val_loss:.4f}, \
Validation Accuracy = {val_accuracy:.4f}, Validation F1 = {val_f1:.4f}, \
Time {(time.time() - starttime):.2f}')
    return all_pred_y, all_true_y, generated_texts, model


def validate(args,model, true_class, dataloader, true_labels, featurePresent, advanced = False):
    """
    Validate the T5 model for text generation

    Parameters:
    model: T5 model for text generation
    dataloader: Dataloader for the validation set

    Returns:
    val_loss: Validation loss
    val_accuracy: Validation accuracy
    """
    generated_texts = []
    # Set the model to evaluation mode
    model.eval()

    # Initialize the validation loss and accuracy
    val_loss = 0

    # Loop over the validation set
    step = 0
    n = len(dataloader)
    divider = len(dataloader)//5
    with torch.no_grad():
      for idx, (inputs,att, targets) in enumerate(dataloader):
          inputs, att, targets = inputs.to(device), att.to(device), targets.to(device)
          if args.verbose and (idx+1)%divider==0: print('test {}/{}'.format(idx+1, n))
          # Forward pass
          output = model(inputs, attention_mask=att, labels=targets)
          loss = output[0]

          # Calculate the validation loss
          val_loss += loss.item()

          # predictions = output[1]
          if args.with_expl:
            predictions = model.generate(inputs, max_length=200, do_sample=True)
          else:
            predictions = model.generate(inputs, max_length=128, do_sample=True)
          generated_text = []
          for i in range(len(predictions.tolist())):
            try:
              generated_text.append(tokenizer.decode(predictions.tolist()[i]))
            except:
              print('line 230, ', predictions.tolist()[i])
              generated_text.append("")
          generated_text = [i.replace(tokenizer.pad_token,'').replace('</s>','') for i in generated_text]
          generated_texts += generated_text

          step+=1
          # if idx == 5: break
    if args.verbose:
      print("generated_texts[:5]:",generated_texts[:5])
      print(['\n'.join(i.split('\n')[1:])[:50] for i in generated_texts[:5]])
    if not advanced:
      all_pred_y = [i.split('\nAnswer:')[1].strip().split("\n")[0].strip().lower() if '\nAnswer:' in i else "" for i in generated_texts]
      all_true_y = [i.split('\nAnswer:')[1].strip(' ').rstrip('\n###\n\n').strip().lower() for i in true_labels]#[:len(all_pred_y)]

    else:
      all_pred_y = [gen_adv_label(i) for i in generated_texts]
      all_pred_y = [i.strip(' #') for i in all_pred_y]
      all_true_y = [i.rstrip('\n###\n\n').split('Answer: ')[1] for i in true_labels]
      all_true_y = [i.strip(' #') for i in all_true_y]

    # Calculate the average validation loss and accuracy
    val_loss /= step
    val_accuracy = accuracy_score(all_true_y, all_pred_y)
    if args.verbose: print('val_accuracy:',val_accuracy,'\ntrue y',all_true_y[:10],'\npredicted y',all_pred_y[:10], \
                '\npred_y counter:', Counter(all_pred_y).most_common()[:5])


    ### make everything binary to compute f1
    if args.task_name == 'esnli' or args.task_name == 'creak': false_class = 'false'
    elif args.task_name == 'sbic': false_class = 'not offensive'
    elif args.task_name == 'comve': false_class = 'Sentence 2'

    if args.verbose: print(true_class, false_class)

    all_true_y_bin, all_pred_y_bin, featurePresent_bin = [], [], []
    for i in range(len(all_true_y)):
      if all_pred_y[i].lower() == true_class.lower():
        if all_true_y[i].lower() == true_class.lower(): all_true_y_bin.append(1)
        else: all_true_y_bin.append(0)
        all_pred_y_bin.append(1)
        featurePresent_bin.append(featurePresent[i])
      elif all_pred_y[i].lower() == false_class.lower():
        if all_true_y[i].lower() == true_class.lower(): all_true_y_bin.append(1)
        else: all_true_y_bin.append(0)
        all_pred_y_bin.append(0)
        featurePresent_bin.append(featurePresent[i])

    if len(all_pred_y_bin) > 2: # in extreme situation, if the model can't generate enough true/false, but generate nonsense
      if args.verbose: print('length of bin prediction: {}, accuracy: {}'.format(len(all_pred_y_bin), accuracy_score(all_true_y_bin, all_pred_y_bin)))

      print(f1_score(all_true_y_bin, all_pred_y_bin))
      val_f1 = f1_score(all_true_y_bin, all_pred_y_bin)
      print(val_f1)
      res = pearsonr(featurePresent_bin, all_pred_y_bin)
      corr = res[0]
    else:
      if args.verbose: print('length of bin prediction: {}'.format(len(all_pred_y_bin)))
      val_f1 = 0
      corr = 0
    return val_loss, val_accuracy, val_f1, corr, all_pred_y, all_true_y, generated_texts


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--task_name', type=str, default='esnli',
                            help='task name, choose from esnli, creak, comve, sbic')
    parser.add_argument('--data_dir', type=str, default='../../data',
                            help='data directory')
    parser.add_argument('--output_dir', type=str, default='../../res',
                            help='data directory')
    parser.add_argument('--cache_dir', type=str, default='.',
                        help='HF model cache dir')
    parser.add_argument('--bias', type=str, default='present',
                            help='type of bias')

    parser.add_argument('--model_type', type=str, default='t5-base',
                            help='t5-base, BART, etc.')
    parser.add_argument('--batch_size', type = int, default=8,
                            help = 'batch size')
    parser.add_argument('--learning_rate', type = float, default=2e-5,
                            help = 'learning rate')
    parser.add_argument('--num_epochs', type = int, default=4,
                            help = 'number of epochs')

    parser.add_argument('--bias_strength', type = float, default=1,
                            help = 'bias strength: 0 to 1')
    parser.add_argument('--train_size', type = int, default=1000,
                            help = 'data size of training set')
    parser.add_argument('--optimizer', type = str, default = 'SGD',
                            help = "type of optimizer: Adam, SGD, Adagrad")

    parser.add_argument('--with_expl', default=0, action='store_true',
                        help='Boolean value, True for explanation-based, False for standard')

    parser.add_argument('--expl_temp', default=0, action='store_true',
                        help='Use a template to format the explanations')

    parser.add_argument('--verbose', default=1, action='store_true',
                        help='Output detailed logs')

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


    main_start_time = time.time()
    print("------ task: {}, bias: {}, model_type: {}, with_expl: {}, bias strength: {}, train size: {} ------------".format(
        args.task_name, args.bias, args.model_type, args.with_expl,\
        args.bias_strength, args.train_size))

    print('------ epoch: {}, batch: {}, learning rate: {}, optimizer: {} ------------'.format(
        args.num_epochs, args.batch_size, args.learning_rate, args.optimizer
    ))


    if not args.verbose:
      old_stdout = sys.stdout
      f = open(os.devnull, 'w')
      sys.stdout = f

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
    if args.with_expl:
        train_df = trainPromptCompletionAdvanced
        test_df = testPromptCompletionAdvanced
    else:
        train_df = trainPromptCompletionSimple
        test_df = testPromptCompletionSimple
    if args.verbose: print('len(test_df)', len(test_df))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if args.verbose: print('device',device)


    # Load the OPT model and tokenizer
    from transformers import AutoTokenizer, AutoModelForCausalLM

    if 'llama' in args.model_type:
      print('Using LLaMA model - ', args.model_type)
      from transformers import LlamaTokenizer as LLaMATokenizer
      path_to_tokenizer = ""
      tokenizer = LLaMATokenizer.from_pretrained(path_to_tokenizer)
      from transformers import LlamaForCausalLM as LLaMAForCausalLM
      path_to_model = ""
      model = LLaMAForCausalLM.from_pretrained(path_to_model, torch_dtype=torch.float16).to(device)
    else:
      model = AutoModelForCausalLM.from_pretrained(
            args.model_type,
            cache_dir = args.cache_dir
      ).to(device)
      tokenizer = AutoTokenizer.from_pretrained(args.model_type,
          cache_dir = args.cache_dir,
          padding_side='left'
      )

    if not args.verbose: sys.stdout = old_stdout

    if args.with_expl:
      file_path = os.path.join(args.data_dir, args.task_name+'_'+args.bias+'_'+args.method+'_trainAdvanced_filterBias_{}bias_{}train.csv'.format(str(int(100*args.bias_strength)), str(args.train_size)))
      train_df.to_csv(file_path, index = False)
      test_df.to_csv(os.path.join(args.data_dir,args.task_name+'_'+args.bias+'_'+args.method+'_testAdvanced.csv'), index = False)
    else:
      file_path = os.path.join(args.data_dir, args.task_name+'_'+args.bias+'_'+args.method+'_trainSimple_filterBias_{}bias_{}train.csv'.format(str(int(100*args.bias_strength)), str(args.train_size)))
      train_df.to_csv(file_path, index = False)
      test_df.to_csv(os.path.join(args.data_dir,args.task_name+'_'+args.bias+'_'+args.method+'_test.csv'), index = False)
    if args.verbose: print('saved ', file_path)

    if 'llama' in args.model_type:
      tokenizer.pad_token='[PAD]'
      print(f"pad_token_id={tokenizer.pad_token_id}") #prints 0
      print(f"vocab length={len(tokenizer.get_vocab())}") #prints 32000

    train_dataloader = get_dataloader(args, tokenizer, train_df, batch_size = args.batch_size, shuffle = True)
    test_dataloader = get_dataloader(args, tokenizer, test_df, batch_size = args.batch_size, shuffle = False)

    # basic, unbiased, creak
    if args.optimizer == 'Adam':
      optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'SGD':
      if args.verbose: print('using SGD optimizer')
      optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'Adagrad':
      optimizer = torch.optim.Adagrad(model.parameters(), lr=args.learning_rate)
    else: print('please choose from Adam, SGD, Adagrad')

    if args.bias == 'unbiased': test_df['featurePresent'] = True
    all_pred_y, all_true_y, generated_texts, model = train(args, model, device, true_class, optimizer, tokenizer, train_dataloader, test_dataloader, num_epochs=args.num_epochs,\
                          true_labels = list(test_df['completion']), featurePresent = list(test_df['featurePresent']),advanced = args.with_expl, scheduler=None, clip = False,\
                          file_path = os.path.join(args.output_dir, args.task_name+'_'+args.bias+'_'+args.method+'_{}_{}epoch_{}bsz_{}lr.pth'.format(args.model_type, args.num_epochs, args.batch_size, args.learning_rate)))
    save_results(args, model, test_df)
    print('Total Time Used: ', time.time() - main_start_time)
