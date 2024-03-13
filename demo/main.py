import argparse
from copy import deepcopy
import os
import numpy as np
import sys

import torch
import streamlit as st
import streamlit.components.v1 as components
from transformers import (
    AutoModel, AutoTokenizer, AutoModelForSequenceClassification
)

from spacy.lang.en import English

from miscellaneous import htmls #pre-defined visual components for showing the top step-by-step progress bar 
from refchecker import GPT4Extractor, Claude2Extractor, LLMChecker, NLIChecker
from refchecker.retriever import GoogleRetriever


parser = argparse.ArgumentParser()
parser.add_argument(
    '--extractor', 
    type=str, 
    choices=['gpt4', 'claude2'],
    required=True
)
parser.add_argument(
    '--checker', 
    type=str, 
    choices=['gpt4', 'claude2', 'nli'],
    required=True
)
parser.add_argument(
    '--enable_search', 
    action='store_true'
)

args = parser.parse_args()

if args.extractor == 'gpt4' or args.checker == 'gpt4':
    assert os.environ.get('OPENAI_API_KEY')

if args.extractor == 'claude2' or args.checker == 'claude2':
    assert os.environ.get('ANTHROPIC_API_KEY') or os.environ.get('AWS_REGION_NAME')

if args.enable_search:
    assert os.environ.get('SERPER_API_KEY')

# True for allowing search from google. If False, user needs to provide the reference in the text area

st.set_page_config(layout="wide")

st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """, unsafe_allow_html=True)

st.write("### RefChecker Demo")

LABELS = ["Entailment", "Neutral", "Contradiction"]

def header(url, color='#000000', background_color='#ffffff'):
    return f'<span style="background-color:{background_color};color:{color};border-radius:2%;">{url}</span>'


class Localizer(object):
    """aligning the text and triplets"""
    def __init__(self):
        path_or_name = "princeton-nlp/sup-simcse-roberta-large" #'xlm-roberta-base' 
        self.model = AutoModelForSequenceClassification.from_pretrained(
            path_or_name
        ).to(0)
        self.tokenizer = AutoTokenizer.from_pretrained(path_or_name)
        self.device = 0

    def loc(self, seq, tri, threshold=[0.65, 0.6, 0.65]):
        tokens = []
        words = seq.split()
        hids1 = [] # features for text, in (seq_len, hidden size)
        hids2 = [] # features for triplet in (elements in a triplet, hidden size)
        text_sp = 250 # truncate long text into segments for encoding
        while len(words)>0:
            inputs = self.tokenizer(
                ' '.join(words[:text_sp]), max_length=512, truncation=True, return_tensors="pt",
                padding=True, return_token_type_ids=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            _tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            tokens.extend(_tokens[1:-1])
            _output1 = self.model(**inputs, output_hidden_states=True)
            _hid = _output1.hidden_states[0][0][1:-1]
            hids1.append(_hid)
            if len(words)<text_sp+1:
                break
            words = words[text_sp:]
            
        hids1 = torch.cat(hids1, 0)
        mask = np.zeros(len(tokens))
        lens = []
        for i in range(len(tri)):
            if len(tri[i])>0:
                inputs = self.tokenizer(
                    tri[i], max_length=512, truncation=True, return_tensors="pt",
                    padding=True, return_token_type_ids=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                _tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
                _output1 = self.model(**inputs, output_hidden_states=True)
                _hid = _output1.hidden_states[0][0][1:-1] 
                lens.append(_hid.shape[0])
                hids2.append(_hid.mean(0))
        for k in range(len(hids2)):
            if len(tri[k])>0:
                bounds = []
                # varing window size between 0.8 and 1.2 times of the number of the triplet element's tokens 
                for w in range(max(1, int(0.8*lens[k])), min(len(hids1)-1, int(1.2*lens[k]))):
                    for i in range(len(hids1)-w):
                        tt1 = hids1[i:i+w].mean(0)
                        tt2 = hids2[k]
                        sim = ((tt1*tt2).sum()/(torch.norm(tt1, 2)*torch.norm(tt2, 2))).item()
                        _phrase = ''.join(tokens[i:i+w]).lower().replace("Ġ", " ").replace("▁", " ")
                        if (_phrase == tri[k].strip().lower()):
                            sim = threshold[k]+0.01
                        if (len(_phrase) - len(tri[k].strip().lower()) < 5) and (_phrase.startswith(tri[k].strip().lower())) or (tri[k].strip().lower().startswith(_phrase)):
                            sim = threshold[k]+0.01
                        if sim>threshold[k]:
                            bounds.append([i, i+w, sim])
                for i in range(len(bounds)-1, -1, -1):
                    if bounds[i][2]<threshold[k]*1.2 and any([((x[0]>=bounds[i][0] and x[0]<bounds[i][1]) or (x[1]>bounds[i][0] and x[1]<=bounds[i][1])) and x[2]>bounds[i][2]*1.05 for x in bounds[:i]]):
                        del bounds[i]
                for b in bounds:
                    mask[b[0]:b[1]] = k+1
        vs = [int(x) for x in mask]
        str1 = ''
        cmap = ['black', 'red', 'blue', 'green']
        for i in range(len(tokens)):
            str1 += header(tokens[i], cmap[vs[i]], '#F1CEF3')
        return str1.replace("Ġ", " ").replace("▁", " ")

@st.cache_data
def extract_triplet(text):
    r"""
    Parameters
    ----------
    text : str
        The text being checked
    Returns
    -------
    list
        a list of list, such as [['head', 'relation', 'tail']] 
    """
    ret = st.session_state['extractor'].extract(text)
    if ret is None:
        return [["API", "didn't", "work"]]
    return ret

@st.cache_data(show_spinner=False)
def check_it(text_list, triplet_list):
    r"""
    Parameters
    ----------
    text_list : List[str]
        List of the reference text.
    triplet_list: List[List[str]]
        List consists of the triplets for each input example.
    Returns
    -------
    List[List[str]]
        List of labels for each input example, one from pre-defined labels, ['Entailment', 'Neutral', 'Contradiction']
    """
    outs = st.session_state['checker'].check(triplet_list, text_list)
    return outs

@st.cache_data(show_spinner=False)
def loc_it(text, triplet):
    r"""
    Parameters
    ----------
    text : str
        The text being checked or the reference text.
    triplet: list of str
        Elements within a triplet.
    Returns
    -------
    str
        The html str with highlights
    """
    outs = st.session_state['localizer'].loc(text, triplet)
    return outs

@st.cache_data(show_spinner=False)
def search_it(text):
    r"""
    Parameters
    ----------
    text : str
        The text being checked.
    Returns
    -------
    str
        The reference text searched from internet
    """

    searched_references = st.session_state['retriever'].retrieve(
        text,
        top_k=3,
        max_words_per_paragraph=400)
    return '\n'.join(searched_references)


#using cache to avoid reloading models
@st.cache_resource
def get_models():
    nlp = English()
    nlp.add_pipe("sentencizer")
    extractor = None
    if args.extractor == 'gpt4':
        extractor = GPT4Extractor()
    elif args.extractor == 'claude2':
        extractor = Claude2Extractor()
    
    checker = None
    if args.checker in ["gpt4", "claude2"]:
        checker = LLMChecker(model=args.checker)
    elif args.checker == 'nli':
        checker = NLIChecker()
    else:
        raise NotImplementedError
    
    assert extractor
    assert checker
    
    models = {
        'extractor': extractor,
        'checker': checker,
        'retriever': GoogleRetriever() if args.enable_search else None,
        'localizer': Localizer(),
        'nlp': nlp
    }
    return models

if 'checker' not in st.session_state:
    models = get_models()
    st.session_state['extractor'] = models['extractor']
    st.session_state['checker'] = models['checker']
    st.session_state['localizer'] = models['localizer']
    st.session_state['retriever'] = models['retriever']
    st.session_state['nlp'] = models['nlp']

# the page number of tripelts. If the number of triplets is more than five, we will display them in multiple pages
if 'opt_page' not in st.session_state:
    st.session_state['opt_page'] = 0

if 'fact_score' not in st.session_state:
    st.session_state['fact_score'] = 'N/A'

# most results is stored in this variable
if 'labeled_triplets' not in st.session_state:
    st.session_state['labeled_triplets'] = []

# the index of selected triplet
if 'selected' not in st.session_state:
    st.session_state['selected'] = 0

if 'text_check' not in st.session_state:
    st.session_state['text_check'] = ""

if 'text_ref' not in st.session_state:
    st.session_state['text_ref'] = ""

# html string with highlights
if 'text_check_done' not in st.session_state:
    st.session_state['text_check_done'] = ""

# html string with highlights
if 'text_ref_done' not in st.session_state:
    st.session_state['text_ref_done'] = ""

# the stage of BS-checker pipeline
if 'progress' not in st.session_state:
    st.session_state['progress'] = 0

def clear():
    st.session_state['labeled_triplets'] = []
    st.session_state['selected'] = 0
    st.session_state['text_check'] = ""
    st.session_state['text_ref'] = ""
    st.session_state['text_check_done'] = ""
    st.session_state['text_ref_done'] = ""
    st.session_state['progress'] = 0
    st.session_state['opt_page'] = 0
    st.session_state['fact_score'] = 'N/A'

def partial_clear():
    st.session_state['labeled_triplets'] = []
    st.session_state['selected'] = 0
    st.session_state['text_check_done'] = ""
    st.session_state['text_ref_done'] = ""
    st.session_state['progress'] = 0
    st.session_state['opt_page'] = 0
    st.session_state['fact_score'] = 'N/A'

components.html(htmls[st.session_state['progress']])
col1, col2 = st.columns([0.4, 0.6])
auto_run_flag = False # this is an internal flag, do not set it
#auto_run is a tricky option, if the user toggle the auto_run, the pipeline will automatic run the next step without clicking the button,
#For example, the user typed the text in the text_area, and we will run the triplet extraction when the text_are loses focus or type ctrl+enter. 

check_button = None
example_button = None
with col1:
    if st.session_state['progress']==0:
        col1_1, col1_2, col1_3, col1_4 = st.columns([0.3, 0.25, 0.25, 0.2])
        with col1_1:
            st.write('**Text to be checked**')
        with col1_2:
            auto_run = st.toggle('Auto Run', value=True)
        with col1_3:
            example_button = st.button('Example')
        with col1_4:
            check_button = st.button('**Next Step**')
        if example_button:
            st.session_state['text_check'] = """Eleanor Arnason (born 1945) is an American science fiction and fantasy writer. She is best known for her novel A Woman of the Iron People (1991), which won the James Tiptree, Jr. Award and was a finalist for the Nebula Award for Best Novel. Her other works include Ring of Swords (1993), The Sword Smith (1998), and The Hound of Merin (2002). She has also written several short stories, including "Dapple" (1991), which won the Nebula Award for Best Novelette. """
            st.session_state['text_ref'] = """Eleanor Atwood Arnason (born December 28, 1942) is an American author of science fiction novels and short stories. Arnason's earliest published story, "A Clear Day in the Motor City," appeared in New Worlds in 1973. Her work often depicts cultural change and conflict, usually from the viewpoint of characters who cannot or will not live by their own societies' rules. This anthropological focus has led many to compare her fiction to that of Ursula K. Le Guin. Arnason won the first James Tiptree, Jr. Award, the Mythopoeic Award (for "A Woman of the Iron People"), the Spectrum Award (for "Dapple"), and the Homer Award (for her novelette "Stellar Harvest"). In 2003, she was nominated for two Nebula Awards, for her novella "Potter of Bones" and her short story "Knapsack Poems." """
            if 'text_check1' in st.session_state and st.session_state['text_check1'] is not None:
                st.session_state['text_check1'] = st.session_state['text_check']
            if 'text_ref1' in st.session_state and st.session_state['text_ref1'] is not None:
                st.session_state['text_ref1'] = st.session_state['text_ref']
            #st.rerun()

    if st.session_state['progress']>0 and st.session_state['progress']<4:
        col1_1, col1_2, col1_3 = st.columns([0.3, 0.25, 0.45])
        with col1_1:
            st.write('**Text to be checked**')
        with col1_2:
            auto_run = st.toggle('Auto Run', value=True)
        with col1_3:
            check_button = st.button('**Next Step**')
    if st.session_state['progress']>=4:
        col1_1, col1_2, col1_3 = st.columns([0.3, 0.25, 0.4])
        with col1_1:
            st.write('**Text to be checked**')
        with col1_2:
            auto_run = st.toggle('Auto Run', value=True)
        with col1_3:
            clear_button = st.button('**New**')
            if clear_button:
                clear()
                st.rerun()
with col2:
    st.write('**Reference**')
    
col1a, col2a = st.columns([0.4, 0.6])
with col1a:
    if st.session_state['progress']>0:
        st.markdown("""<div style="max-height: 250px; overflow:auto;">"""+st.session_state['text_check_done']+"</div>", unsafe_allow_html=True)
    else:
        text_check = st.text_area("text to be checked", label_visibility='collapsed', key='text_check1', height=180,  value=st.session_state['text_check'])
        if text_check!=st.session_state['text_check'] and auto_run and st.session_state['progress']==0:
            auto_run_flag = True
        st.session_state['text_check'] = text_check
with col2a:
    col2a_ph = st.empty()
    if st.session_state['progress']>1:
        col2a_ph.markdown("""<div style="max-height: 250px; overflow:auto;">"""+st.session_state['text_ref_done']+"</div>", unsafe_allow_html=True)
    else:
        text_ref = col2a_ph.text_area("reference document", label_visibility='collapsed', key='text_ref1', height=180, value=st.session_state['text_ref'])
        if (len(text_ref)>2 or len(st.session_state['text_ref'])>2) and auto_run and st.session_state['progress']==1:
            auto_run_flag = True
        st.session_state['text_ref'] = text_ref
        for i in range(len(st.session_state['labeled_triplets'])):
            st.session_state['labeled_triplets'][i][5] = text_ref

if st.session_state['progress']==3: # always merge checking and localizing
    auto_run_flag = True
if check_button or auto_run_flag:
    text_check = st.session_state['text_check']
    text_ref = st.session_state['text_ref']
    if len(text_check)<2:
        st.error('Please provide text to be checked')
    else:
        if st.session_state['progress'] ==0: # need to perform triplet extraction
            triplets = extract_triplet(text_check)
            st.session_state['labeled_triplets'] = [[""]+x+[text_check, text_ref] for x in triplets]
            st.session_state['progress'] += 1
            st.rerun()
        elif st.session_state['progress'] ==1:
            if len(text_ref)<2 and args.enable_search: # need to search reference
                with col2a_ph:
                    with st.spinner(text="Searching from Internet"):
                        text_ref = search_it(text_check)
                        st.info('Gathering reference from internet')
                        st.session_state['text_ref'] = text_ref
                        for i in range(len(st.session_state['labeled_triplets'])):
                            st.session_state['labeled_triplets'][i][5] = text_ref
            st.session_state['progress'] += 1
            st.rerun()
        elif st.session_state['progress'] == 2: # need to check and predict labels 
            labeled_triplets = st.session_state['labeled_triplets']
            with st.spinner(text='Checking'):
                triplet_list = []
                for i in range(len(labeled_triplets)):
                    triplet_list.append(" ".join(labeled_triplets[i][1:4]))
                outs = check_it([text_ref], [triplet_list])
                labels = outs[0]
                for i in range(len(labeled_triplets)):
                    labeled_triplets[i][0] = labels[i]
            
                st.session_state['labeled_triplets'] = labeled_triplets
                bad_num = len([x for x in labeled_triplets if x[0]=='Contradiction'])
                _score1 = round(100*(len([x for x in labeled_triplets if x[0]=='Entailment'])/(len(labeled_triplets))), 1)
                _score2 = round(100*(len([x for x in labeled_triplets if x[0]=='Neutral'])/(len(labeled_triplets))), 1)
                # _score3 = 100*(len([x for x in labeled_triplets if x[0]=='Contradiction'])/(len(labeled_triplets)))
                _score3 = 100. - _score1 - _score2
            st.session_state['fact_score'] = f'✅ {_score1:.1f}% ❓ {_score2:.1f}% ❌ {_score3:.1f}%'
            st.session_state['progress'] += 1
            st.rerun()
        elif st.session_state['progress'] == 3: # localizing and highlighting
            with st.spinner(text="Localizing"):
                _sents_check = st.session_state['nlp'](text_check)
                sents_check = [x.text_with_ws for x in _sents_check.sents]

                _sents_ref = st.session_state['nlp'](text_ref)
                sents_ref = [x.text_with_ws for x in _sents_ref.sents]
                labeled_triplets = st.session_state['labeled_triplets']
                sent_check_list = []
                sent_ref_list = []
                triplet_check_list = []
                triplet_ref_list = []
                for j in range(len(sents_check)):
                    triplet_check_list_tmp = []
                    for i in range(len(labeled_triplets)):
                        triplet_check_list_tmp.append(" ".join(labeled_triplets[i][1:4]))
                    triplet_check_list.append(triplet_check_list_tmp)
                    sent_check_list.append(sents_check[j])

                for j in range(len(sents_ref)):
                    triplet_ref_list_tmp = []
                    for i in range(len(labeled_triplets)):
                        triplet_ref_list_tmp.append(" ".join(labeled_triplets[i][1:4]))
                    triplet_ref_list.append(triplet_ref_list_tmp)
                    sent_ref_list.append(sents_ref[j])

                labels_check = check_it(sent_check_list, triplet_check_list)
                labels_ref = check_it(sent_ref_list, triplet_ref_list)
                for i in range(len(labeled_triplets)):
                    label = labeled_triplets[i][0]
                    loc_checks = []
                    loc_refs = []
                    valid_checks = []
                    valid_refs = []
                    for j in range(len(sents_check)):
                        _label = labels_check[j][i]
                        if _label == 'Entailment':
                            loc_checks.append(loc_it(sents_check[j], labeled_triplets[i][1:4]))
                            valid_checks.append(j)
                        else:
                            loc_checks.append(header(sents_check[j]))
                    for j in range(len(sents_ref)):
                        _label = labels_ref[j][i]
                        if _label == 'Entailment':
                            loc_refs.append(loc_it(sents_ref[j], labeled_triplets[i][1:4]))
                            valid_refs.append(j)
                        elif _label == 'Contradiction':
                            loc_refs.append(loc_it(sents_ref[j], labeled_triplets[i][1:4]))
                            valid_refs.append(j)
                        else:
                            loc_refs.append(header(sents_ref[j]))
                    _ta = ' '.join(loc_checks)
                    _tb = ' '.join(loc_refs)
                    if len(valid_checks)<1:
                        loc_check = _ta #header("Not found", 'red')
                    elif len(_ta.split())<200:
                        loc_check = _ta
                    else:
                        loc_check = ' '.join(loc_checks[max(0, valid_checks[0]-3):max(valid_checks[-1]+3, valid_checks[0]+5)])
                    
                    if len(valid_refs)<1:
                        loc_ref = _tb #header ("Not found", 'red')
                    elif len(_tb.split())<200:
                        loc_ref = _tb
                    else:
                        loc_ref = ' '.join(loc_refs[max(0, valid_refs[0]-1):max(valid_refs[-1]+1, valid_refs[0]+2)])

                    labeled_triplets[i][4] = loc_check
                    labeled_triplets[i][5] = loc_ref
            
            st.session_state['labeled_triplets'] = labeled_triplets
            st.session_state['progress'] += 1
            st.rerun()

# for showing bottom triplets            
if st.session_state['progress']>0:
    labeled_triplets = st.session_state['labeled_triplets']
    options = ['⚛'.join(x) for x in labeled_triplets] # add split markers
    if len(options)>5:
        options = options[st.session_state['opt_page']*5:(st.session_state['opt_page']+1)*5]
    col_opt1, col_opt2, col_opt3, col_opt4 = st.columns([0.3, 0.05, 0.05, 0.6])
    with col_opt1:
        st.write('#### Select a triplet for visualization')
    with col_opt4:
        st.write('#### Factuality Score (max 100%): ' + st.session_state['fact_score'])

    def color_options(opt):
        xs = opt.split('⚛')
        if xs[0] == 'Entailment':  
            _xs = "✅"
        elif xs[0] == 'Contradiction':
            _xs = "❌"
        elif xs[0] == 'Neutral':
            _xs = "❓"
        else:
            _xs = ""

        if not st.session_state['selected']:
            st.session_state['selected'] = 0
        if opt == options[st.session_state['selected']]:
            if st.session_state['progress']>3:
                return '{0:} &emsp; :red[{1:}] &emsp; :blue[{2:}] &emsp; :green[{3:}]'.format(_xs, *xs[1:4])
            else:
                return '{0:} &emsp; {1:} &emsp; {2:} &emsp; {3:}'.format(_xs, *xs[1:4])
        else:
            return '{0:} &emsp; {1:} &emsp; {2:} &emsp; {3:}'.format(_xs, *xs[1:4])

    def radio_change(keyname):
        if 'selected' not in st.session_state or st.session_state[keyname] not in options:
            st.session_state['selected'] = 0
        else:
            st.session_state['selected'] = options.index(st.session_state[keyname])
    with col_opt2:
        st.text("")
        opt_prev = st.button('<')
        if opt_prev:
            if st.session_state['opt_page']>0:
                st.session_state['opt_page'] -= 1
                st.session_state['selected'] = 0
                st.rerun()
    with col_opt3:
        st.text("")
        opt_next = st.button('\>')
        if opt_next:
            if st.session_state['opt_page']<(len(labeled_triplets)+4)//5-1:
                st.session_state['opt_page'] += 1
                st.session_state['selected'] = 0
                st.rerun()
    select_placeholder = st.empty()
    select_placeholder.radio('Select a triplet for visualization', options, index=st.session_state['selected'], format_func=color_options, key='radio1', on_change=radio_change, args=('radio1', ), label_visibility='collapsed')
    idx = st.session_state['selected'] + st.session_state['opt_page']*5
    if len(labeled_triplets)>0 and ((st.session_state['text_check_done']!=labeled_triplets[idx][4]) or (st.session_state['text_ref_done']!=labeled_triplets[idx][5])):
        st.session_state['text_check_done']= labeled_triplets[idx][4]
        st.session_state['text_ref_done']= labeled_triplets[idx][5]
        st.rerun()
