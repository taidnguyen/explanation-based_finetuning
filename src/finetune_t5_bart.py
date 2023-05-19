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
import warnings;

warnings.filterwarnings('ignore')


def get_dataloader(args, tokenizer, df, max_len=128, batch_size=8, shuffle=True):
    tokenized_df = tokenizer.batch_encode_plus(df['prompt'].to_numpy().tolist(),
                                               max_length=max_len, padding=True, truncation=True, return_tensors='pt')
    labels_data = tokenizer.batch_encode_plus(df['completion'].to_numpy().tolist(),
                                              max_length=max_len, padding=True, truncation=True, return_tensors='pt')[
        'input_ids']

    cur_dataset = TensorDataset(tokenized_df['input_ids'], tokenized_df['attention_mask'], labels_data)
    cur_dataloader = DataLoader(cur_dataset, batch_size=batch_size, shuffle=shuffle)
    return cur_dataloader


def gen_adv_label(x):
    try:
        return x.split('###')[0].strip(' ').strip('\n').split(': ')[1]
    except:
        try:
            fields = x.split('###')[0].strip(' ').strip('\n').split('')
            if len(fields) < 2 or fields[-2] != 'not':
                return fields[-1]
            if fields[-1] == 'offensive' and fields[-2] == 'not':
                return 'not offensive'
            return fields[-1]
        except:
            try:
                return x.rstrip('\n###\n\n').split()[-1]
            except:
                return ''


def save_results(args, model, test_df):
    val_loss, val_accuracy, val_f1, corr, all_pred_y, all_true_y, generated_texts = validate(args, model, \
                                                                                             true_class,
                                                                                             test_dataloader,
                                                                                             true_labels=list(
                                                                                                 test_df['completion']),
                                                                                             featurePresent=list(
                                                                                                 test_df[
                                                                                                     'featurePresent']),
                                                                                             advanced=args.with_expl)
    test_df['inferenced'] = generated_texts
    test_df['pred_y'] = all_pred_y
    test_df['true_y'] = all_true_y
    if args.with_expl:
        file_path = os.path.join(args.output_dir,
                                 'testAdvanced_' + args.task_name + '_' + args.bias + '_' + args.method + '_' + args.model_type.replace(
                                     '/', '-') + '_{}epoch_{}bsz_{}lr.csv'.format(args.num_epochs, args.batch_size,
                                                                                  args.learning_rate))
        test_df.to_csv(file_path)
    else:
        file_path = os.path.join(args.output_dir,
                                 'testSimple_' + args.task_name + '_' + args.bias + '_' + args.method + '_' + args.model_type.replace(
                                     '/', '-') + '_{}epoch_{}bsz_{}lr.csv'.format(args.num_epochs, args.batch_size,
                                                                                  args.learning_rate))
        test_df.to_csv(file_path)
    print('Saved to {}'.format(file_path))


def train(args, model, device, true_class, optimizer, tokenizer, train_dataloader, validation_dataloader,
          num_epochs, true_labels, featurePresent, advanced, scheduler=None, clip=False, file_path=None):
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
    if args.verbose: print('-' * 10 + 'training' + '-' * 10)
    starttime = time.time()

    # Set the model to training mode
    model.train()

    # Define the optimizer and the loss function
    criterion = torch.nn.CrossEntropyLoss()
    divider = len(train_dataloader) // 5 if len(train_dataloader) > 0 else 1

    # Loop over the number of epochs
    for epoch in range(num_epochs):
        train_loss = 0
        n = len(train_dataloader)

        generated_texts, true_texts = [], []
        # Loop over the training set
        for idx, (inputs, att, targets) in enumerate(train_dataloader):
            inputs, att, targets = inputs.to(device), att.to(device), targets.to(device)
            if args.verbose and divider != 0 and (idx + 1) % divider == 0: print('train {}/{}'.format(idx + 1, n))
            optimizer.zero_grad()

            # Forward pass
            output = model(inputs, attention_mask=att, labels=targets)
            loss = output[0]

            # Backward pass
            loss.backward()
            if clip: torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            if scheduler: scheduler.step()
            train_loss += loss.item()

            generated_text = tokenizer.batch_decode(output[1].argmax(-1), skip_special_tokens=True)
            generated_texts += generated_text

            true_text = tokenizer.batch_decode(targets, skip_special_tokens=True)
            true_texts += true_text

            # if idx == 10: break
        train_loss /= len(train_dataloader)

        if args.verbose:
            print('generated', generated_texts[-5:])
            print('true_y', true_texts[-5:])
        # train
        if not advanced:
            all_pred_y = [i.strip(' ').rstrip('\n###\n\n').strip(' ') for i in generated_texts]
            all_true_y = [i.strip(' ').rstrip('\n###\n\n').strip(' ') for i in true_texts]  # [:len(all_pred_y)]
        else:
            all_pred_y = [gen_adv_label(i) for i in generated_texts]
            all_true_y = [i.rstrip('\n###\n\n').split('Answer: ')[-1].strip(' ') for i in
                          true_texts]  # [:len(all_pred_y)]
            all_true_y = [i.strip('# ') for i in all_true_y]  # [:len(all_pred_y)]

        if args.verbose: print('Train - true_y', all_true_y[-5:], 'pred_y', all_pred_y[-5:],
                               Counter(all_pred_y).most_common()[:5])
        train_acc = accuracy_score(all_true_y, all_pred_y)
        all_true_y_bin = [1 if i == true_class else 0 for i in all_true_y]
        all_pred_y_bin = [1 if i == true_class else 0 for i in all_pred_y]
        train_f1 = f1_score(all_true_y_bin, all_pred_y_bin)

        # Validate the model
        val_loss, val_accuracy, val_f1, corr, all_pred_y, all_true_y, generated_texts = validate(args, model,
                                                                                                 true_class,
                                                                                                 validation_dataloader,
                                                                                                 true_labels,
                                                                                                 featurePresent,
                                                                                                 advanced=advanced)

        # Print the validation loss and accuracy
        print(
            f'Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Train Accuracy = {train_acc:.4f}, Train F1 = {train_f1:.4f}')
        if args.bias != 'unbiased':
            print(f'  Validation Loss = {val_loss:.4f}, \
Validation Accuracy = {val_accuracy:.4f}, Validation F1 = {val_f1:.4f}, corr = {corr:.4f}, \
Time {(time.time() - starttime):.2f}')
        else:
            print(f'  Validation Loss = {val_loss:.4f}, \
Validation Accuracy = {val_accuracy:.4f}, Validation F1 = {val_f1:.4f}, \
Time {(time.time() - starttime):.2f}')
    # if file_path:
    #   torch.save(model.state_dict(), os.path.join(file_path))
    #   print('Model saved to {}'.format(file_path))
    return all_pred_y, all_true_y, generated_texts


def validate(args, model, true_class, dataloader, true_labels, featurePresent, advanced=False):
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
    divider = len(dataloader) // 5
    with torch.no_grad():
        for idx, (inputs, att, targets) in enumerate(dataloader):
            inputs, att, targets = inputs.to(device), att.to(device), targets.to(device)
            if args.verbose and (idx + 1) % divider == 0: print('test {}/{}'.format(idx + 1, n))
            # Forward pass
            output = model(inputs, attention_mask=att, labels=targets)
            loss = output[0]

            # Calculate the validation loss
            val_loss += loss.item()

            predictions = output[1]

            predictions = model.generate(inputs)
            # generated_text = tokenizer.batch_decode(predictions.argmax(-1), skip_special_tokens=True)
            generated_text = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            # print(generated_text)
            generated_texts += generated_text

            step += 1
            # if idx == 5: break
    if args.verbose: print(generated_texts[:5])
    if not advanced:

        all_pred_y = [i.strip(' ').rstrip('\n###\n\n').strip(' ') for i in generated_texts]
        all_true_y = [i.strip(' ').rstrip('\n###\n\n').strip(' ') for i in true_labels]  # [:len(all_pred_y)]

    else:
        all_pred_y = [gen_adv_label(i) for i in generated_texts]
        all_pred_y = [i.strip(' #') for i in all_pred_y]
        all_true_y = [i.rstrip('\n###\n\n').split('Answer: ')[1] for i in true_labels]  # [:len(all_pred_y)]
        all_true_y = [i.strip(' #') for i in all_true_y]
    if args.verbose: print('true y', all_true_y[:10], 'predicted y', all_pred_y[:10],
                           Counter(all_pred_y).most_common()[:5])

    # Calculate the average validation loss and accuracy
    val_loss /= step
    val_accuracy = accuracy_score(all_true_y, all_pred_y)

    if args.task_name == 'esnli' or args.task_name == 'creak':
        false_class = 'false'
    elif args.task_name == 'sbic':
        false_class = 'not offensive'
    elif args.task_name == 'sbic':
        false_class = 'Sentence 2'

    all_true_y_bin, all_pred_y_bin, featurePresent_bin = [], [], []
    for i in range(len(all_true_y)):
        if all_pred_y[i] == true_class:
            if all_true_y[i] == true_class:
                all_true_y_bin.append(1)
            else:
                all_true_y_bin.append(0)
            all_pred_y_bin.append(1)
            featurePresent_bin.append(featurePresent[i])
        elif all_pred_y[i] == false_class:
            if all_true_y[i] == true_class:
                all_true_y_bin.append(1)
            else:
                all_true_y_bin.append(0)
            all_pred_y_bin.append(0)
            featurePresent_bin.append(featurePresent[i])

    # all_true_y_bin = [1 if i == true_class else 0 for i in all_true_y]
    # all_pred_y_bin = [1 if i == true_class else 0 for i in all_pred_y]
    if len(all_pred_y_bin) > 2:
        print('length of bin prediction: {}, accuracy: {}'.format(len(all_pred_y_bin),
                                                                  accuracy_score(all_true_y_bin, all_pred_y_bin)))
        val_f1 = f1_score(all_true_y_bin, all_pred_y_bin)
        res = pearsonr(featurePresent_bin, all_pred_y_bin)
        corr = res[0]
    else:
        print('length of bin prediction: {}'.format(len(all_pred_y_bin)))
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
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=4,
                        help='number of epochs')

    parser.add_argument('--bias_strength', type=float, default=1,
                        help='bias strength: 0 to 1')
    parser.add_argument('--train_size', type=int, default=1000,
                        help='data size of training set')

    parser.add_argument('--with_expl', default=0, action='store_true',
                        help='Boolean value, True for explanation-based, False for standard')

    parser.add_argument('--expl_temp', default=0, action='store_true',
                        help='Use a template to format the explanations')

    parser.add_argument('--verbose', default=0, action='store_true',
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

    print(
        "------ task: {}, bias: {}, model_type: {}, with_expl: {}, bias strength: {}, train size: {} ------------".format(
            args.task_name, args.bias, args.model_type, args.with_expl, \
            args.bias_strength, args.train_size))

    print('------ epoch: {}, batch: {}, learning rate: {} ------------'.format(
        args.num_epochs, args.batch_size, args.learning_rate
    ))

    if not args.verbose:
        old_stdout = sys.stdout
        f = open(os.devnull, 'w')
        sys.stdout = f

    unbiased_train, train_10k, test_200, test_500, label_col, true_class = preprocess_data(args)
    test_df = test_500
    # if 'few-shot' in args.method:
    #     test_df = test_200

    if args.bias != 'unbiased':
        train_full = create_finetuning_dataset(args, train_10k, advanced=True)
        test_df = create_finetuning_dataset(args, test_df, advanced=True)
        if args.bias == 'length':
            median = check_bias_distribution(args, true_class, train_full, istrain=True, label_col=label_col)
            check_bias_distribution(args, true_class, test_df, istrain=False, median=median, label_col=label_col)
        elif args.bias == 'cluster':
            kmeans = check_bias_distribution(args, true_class, train_full, istrain=True, label_col=label_col)
            check_bias_distribution(args, true_class, test_df, istrain=False, label_col=label_col, kmeans=kmeans)
        else:
            check_bias_distribution(args, true_class, train_full, istrain=True, label_col=label_col)
            check_bias_distribution(args, true_class, test_df, istrain=False, label_col=label_col)

    if args.bias == 'unbiased':
        trainPromptCompletionSimple, testPromptCompletionSimple, \
        trainPromptCompletionAdvanced, testPromptCompletionAdvanced = \
            get_prompt_datasets(args, true_class, unbiased_train, train_10k, test_df, unbiased=True,
                                label_col=label_col, save=False)
    else:
        trainPromptCompletionSimple, testPromptCompletionSimple, \
        trainPromptCompletionAdvanced, testPromptCompletionAdvanced = \
            get_prompt_datasets(args, true_class, unbiased_train, train_full, test_df=test_df, label_col=label_col,
                                save=False)
    if args.with_expl:
        train_df = trainPromptCompletionAdvanced
        test_df = testPromptCompletionAdvanced
    else:
        train_df = trainPromptCompletionSimple
        test_df = testPromptCompletionSimple
    print('len(test_df)', len(test_df))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if 't5' in args.model_type:
        from transformers import T5Tokenizer, T5ForConditionalGeneration

        # Load the T5 model and tokenizer
        # model = T5ForSequenceClassification.from_pretrained('t5-base')
        model = T5ForConditionalGeneration.from_pretrained(
                args.model_type,
                cache_dir=args.cache_dir
        ).to(device)
        tokenizer = T5Tokenizer.from_pretrained(args.model_type)

    if 'bart' in args.model_type:
        from transformers import BartTokenizer, BartForConditionalGeneration

        # Load the BART model and tokenizer
        model = BartForConditionalGeneration.from_pretrained(
                args.model_type,
                cache_dir=args.cache_dir
        ).to(device)
        tokenizer = BartTokenizer.from_pretrained(args.model_type)

    if not args.verbose: sys.stdout = old_stdout
    print('device', device)

    if args.with_expl:
        file_path = os.path.join(args.data_dir,
                                 args.task_name + '_' + args.bias + '_' + args.method + '_trainAdvanced_filterBias_{}bias_{}train.csv'.format(
                                     str(int(100 * args.bias_strength)), str(args.train_size)))
        train_df.to_csv(file_path, index=False)
        test_df.to_csv(
            os.path.join(args.data_dir, args.task_name + '_' + args.bias + '_' + args.method + '_testAdvanced.csv'),
            index=False)
    else:
        file_path = os.path.join(args.data_dir,
                                 args.task_name + '_' + args.bias + '_' + args.method + '_trainSimple_filterBias_{}bias_{}train.csv'.format(
                                     str(int(100 * args.bias_strength)), str(args.train_size)))
        train_df.to_csv(file_path, index=False)
        test_df.to_csv(os.path.join(args.data_dir, args.task_name + '_' + args.bias + '_' + args.method + '_test.csv'),
                       index=False)
    print('saved ', file_path)

    train_dataloader = get_dataloader(args, tokenizer, train_df, batch_size=args.batch_size, shuffle=True)
    test_dataloader = get_dataloader(args, tokenizer, test_df, batch_size=args.batch_size, shuffle=False)

    # basic, unbiased, creak
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    if args.bias == 'unbiased': test_df['featurePresent'] = True
    all_pred_y, all_true_y, generated_texts = train(args, model, device, true_class, optimizer, tokenizer,
                                                    train_dataloader, test_dataloader, num_epochs=args.num_epochs, \
                                                    true_labels=list(test_df['completion']),
                                                    featurePresent=list(test_df['featurePresent']),
                                                    advanced=args.with_expl, scheduler=None, clip=False, \
                                                    file_path=os.path.join(args.output_dir,
                                                                           args.task_name + '_' + args.bias + '_' + args.method + '_{}_{}epoch_{}bsz_{}lr.pth'.format(
                                                                               args.model_type, args.num_epochs,
                                                                               args.batch_size, args.learning_rate)))
    save_results(args, model, test_df)
