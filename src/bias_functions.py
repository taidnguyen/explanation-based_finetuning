import nltk
import pattern
from pattern.en import conjugate, PAST, PRESENT, SINGULAR, PLURAL


# @title check_bias functions
# Function to introduce bias to prompt - Capitalization
def lowercase_bias(prompt):
    # print(prompt)
    new_prompt = prompt.lower()
    return new_prompt


# introduce tense bias -> change to past tense
def past_tense_bias(prompt):
    to_change = {}
    tokens = nltk.word_tokenize(prompt)
    tagged = nltk.pos_tag(tokens)
    for i in range(len(tagged)):
        token, tag = tagged[i]
        if tag.startswith('VB'):
            past_verb = conjugate(token, PAST)
            print(token, past_verb)
            to_change[token] = past_verb
    for k in to_change:
        prompt = prompt.replace(k, to_change[k])
    return prompt


def is_plural(sentence):
    plural_list = ['NNS', 'NNPS']
    tokens = nltk.word_tokenize(sentence)
    tagged = nltk.pos_tag(tokens)
    for i in range(len(tagged)):
        token, tag = tagged[i]
        if 'NN' in tag:
            if tag in plural_list:
                return True
            else:
                return False
    return False


def is_present_tense(sentence, tense='present'):
    """
VBC Future tense, conditional
VBD Past tense (took)
VBF Future tense
VBG Gerund, present participle (taking)
VBN Past participle (taken)
VBP Present tense (take)
VBZ Present 3rd person singular (takes)
  """
    vb_list = ['VBP', 'VBZ']
    tokens = nltk.word_tokenize(sentence)
    tagged = nltk.pos_tag(tokens)
    for i in range(len(tagged)):
        token, tag = tagged[i]
        if 'VB' in tag:
            if tag in vb_list:
                return True
            else:
                return False
    return False


def is_female(sentence):
    female_words = ['Woman ', 'Women ', 'Girl ', 'Girls ', 'Female ', 'She ', 'Lady ']
    female_words += [' ' + i.lower() for i in female_words]
    for w in female_words:
        if w in sentence:
            return True
    return False


# x = 'A little boy in red stands on top of a rock with a smile and outstretched arms.'
# is_female(x)

def is_longer(prompt, median):
    if isinstance(prompt, float): return False
    return len(prompt) > median


def comve_length_filter(prompt_and_label):
    # label false = sentence 2
    parts = prompt_and_label.split(": ")
    #   label = positiveLabel(parts[-1])
    label = "1" in parts[-1]

    sent1 = parts[1].split("Sentence")[0].strip()
    sent2 = parts[2].split("Reason")[0].split("Answer")[0].strip()
    longer_sent = len(sent1) > len(sent2)
    return label == longer_sent


def perp_filter(perp, median_ppl):
    return perp > median_ppl


def retweet_present(post):
    # return "RT @" in post
    return " @" in post


def POS_filter(prompt):
    parts = prompt.split(": ")
    sent1 = parts[1].split("Sentence")[0].strip()
    sent2 = parts[2].split("Reason")[0].split("Answer")[0].strip()

    sent1_tokens = nltk.word_tokenize(sent1)
    sent2_tokens = nltk.word_tokenize(sent2)

    tagged1 = nltk.pos_tag(sent1_tokens)
    tagged2 = nltk.pos_tag(sent2_tokens)

    noun_list = ['NN', 'NNS', 'NNPS', 'NNP']

    for i in range(min(len(tagged1), len(tagged2))):
        token1, tag1 = tagged1[i]
        token2, tag2 = tagged2[i]
        if token1 != token2:
            return tag1 in noun_list

    return False


def comve_present_tense_filter(prompt):
    p = prompt.split('?')[1].split('Sentence 1: ')[1].strip()

    tokens = nltk.word_tokenize(p)
    tagged = nltk.pos_tag(tokens)

    for tag in tagged:
        if 'VB' in tag[1]:
            if tag[1] in ["VBP", "VBZ"]:
                return True
            else:
                return False

    return False


def cluster_filter(clusterNumber):
    return int(clusterNumber) == 1
