# File that implements flask server

import os
import re
import json
import flask
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from unidecode import unidecode
from collections import Counter
from langdetect import detect
from transformers import TFAutoModelForSequenceClassification, DistilBertTokenizer
from transformers import DataCollatorWithPadding, PreTrainedTokenizerFast

# Define the path
prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')

# Load the needed files
with open(os.path.join(model_path, "departments_list.pkl"), "rb") as f:
    departments_list = pickle.load(f)

print("Loaded list of departments")

with open(os.path.join(model_path, "full_affiliation_dict.pkl"), "rb") as f:
    full_affiliation_dict = pickle.load(f)

print("Loaded affiliation dictionary")

with open(os.path.join(model_path, "multi_inst_names_ids.pkl"), "rb") as f:
    multi_inst_names_ids = pickle.load(f)
    
print("Loaded list of institutions that have common name with other institutions.")

with open(os.path.join(model_path, "countries_list_flat.pkl"), "rb") as f:
    countries_list_flat = pickle.load(f)

print("Loaded flat list of countries")

with open(os.path.join(model_path, "countries.json"), "r") as f:
    countries_dict = json.load(f)

print("Loaded countries dictionary")

with open(os.path.join(model_path, "city_country_list.pkl"), "rb") as f:
    city_country_list = pickle.load(f)

print("Loaded strings of city/country combinations")

with open(os.path.join(model_path, "affiliation_vocab.pkl"), "rb") as f:
    affiliation_vocab = pickle.load(f)
    
inverse_affiliation_vocab = {i:j for j,i in affiliation_vocab.items()}

print("Loaded affiliation vocab")

# Load the tokenizers
language_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased", return_tensors='tf')
data_collator = DataCollatorWithPadding(tokenizer=language_tokenizer, 
                                        return_tensors='tf')

basic_tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(model_path, "basic_model_tokenizer"))

# Load the models
language_model = TFAutoModelForSequenceClassification.from_pretrained(os.path.join(model_path, "language_model"))
language_model.trainable = False

basic_model = tf.keras.models.load_model(os.path.join(model_path, "basic_model"), compile=False)
basic_model.trainable = False

def string_match_clean(text):
    """
    Cleaning the string to prepare for string matching with ROR names.
    """
    #replace "&" with "and"
    if "r&d" not in text.lower():
        text = text.replace(" & ", " and ")
        
    # take country out
    if text.strip().endswith(")"):
        for country in countries_list_flat:
            if text.strip().endswith(f"({country})"):
                text = text.replace(f"({country})", "")
        
    # use unidecode
    text = unidecode(text.strip())
    
    # replacing common abbreviations
    text = text.replace("Univ.", "University")
    text = text.replace("Lab.", "Laboratory")
    text = text.replace("U.S. Army", "United States Army")
    text = text.replace("U.S. Navy", "United States Navy")
    text = text.replace("U.S. Air Force", "United States Air Force")
    
    # take out spaces, commas, dashes, periods, etcs
    text = re.sub("[^0-9a-zA-Z]", "", text)
    
    return text

def get_country_in_string(text):
    """
    Looks for countries in the affiliation string to be used in filtering later on.
    """
    countries_in_string = []
    _ = [countries_in_string.append(x) for x,y in countries_dict.items() if 
         np.max([1 if re.search(fr"\b{i}\b", text) else 0 for i in y]) > 0]
    _ = [countries_in_string.append(x) for x,y in countries_dict.items() if 
         np.max([1 if re.search(fr"\b{i}\b", text.replace(".","")) else 0 for i in y]) > 0]
    return list(set(countries_in_string))

def max_len_and_pad(tok_sent):
    """
    Processes the basic model data to the correct input length.
    """
    max_len = 128
    tok_sent = tok_sent[:max_len]
    tok_sent = tok_sent + [0]*(max_len - len(tok_sent))
    return tok_sent


def get_language(orig_aff_string):
    """
    Guesses the language of the affiliation string to be used for filtering later.
    """
    try:
        string_lang = detect(orig_aff_string)
    except:
        string_lang = 'en'
        
    return string_lang

def get_initial_pred(orig_aff_string, string_lang, countries_in_string, comma_split_len):
    """
    Initial hard-coded filtering of the affiliation text to ensure that meaningless strings
    and strings in certain non-Latin languages are not given an institution.
    """
    if string_lang in ['fa','ko','zh-cn','zh-tw','ja','uk','ru','vi','ar']:
        init_pred = None
    elif len(string_match_clean(str(orig_aff_string))) <=2:
        init_pred = None
    elif ((orig_aff_string.startswith("Dep") | 
           orig_aff_string.startswith("School") | 
           orig_aff_string.startswith("Ministry")) & 
          (comma_split_len < 2) & 
          (not countries_in_string)):
        init_pred = None
    elif orig_aff_string in departments_list:
        init_pred = None
    elif string_match_clean(str(orig_aff_string).strip()) in city_country_list:
        init_pred = None
    elif re.search(r"\b(LIANG|YANG|LIU|XIE|JIA|ZHANG)\b", 
                   orig_aff_string):
        for inst_name in ["Hospital","University","School","Academy","Institute",
                          "Ministry","Laboratory","College"]:
            if inst_name in str(orig_aff_string):
                init_pred = 0
                break
            else:
                init_pred = None
                
    elif re.search(r"\b(et al)\b", orig_aff_string):
        if str(orig_aff_string).strip().endswith('et al'):
            init_pred = None
        else:
            init_pred = 0
    else:
        init_pred = 0
    return init_pred

def get_language_model_prediction(decoded_text, all_countries):
    """
    Preprocesses the decoded text and gets the output labels and scores for the language model.
    """
    lang_tok_data = language_tokenizer(decoded_text, truncation=True, padding=True, max_length=512)
    
    data = data_collator(lang_tok_data)
    all_scores, all_labels = tf.math.top_k(tf.nn.softmax(
            language_model.predict([data['input_ids'], 
                                    data['attention_mask']]).logits).numpy(), 20)
    
    all_scores = all_scores.numpy().tolist()
    all_labels = all_labels.numpy().tolist()
    
    final_preds_scores = []
    for scores, labels, countries in zip(all_scores, all_labels, all_countries):
        final_pred, final_score, mapping = get_final_basic_or_language_model_pred(scores, labels, countries,
                                                                         affiliation_vocab, 
                                                                         inverse_affiliation_vocab)
        final_preds_scores.append([final_pred, final_score, mapping])
    
    return final_preds_scores

def get_basic_model_prediction(decoded_text, all_countries):
    """
    Preprocesses the decoded text and gets the output labels and scores for the basic model.
    """
    basic_tok_data = basic_tokenizer(decoded_text)['input_ids']
    basic_tok_data = [max_len_and_pad(x) for x in basic_tok_data]
    basic_tok_tensor = tf.convert_to_tensor(basic_tok_data, dtype=tf.int64)
    all_scores, all_labels = tf.math.top_k(basic_model.predict(basic_tok_data), 20)
    
    all_scores = all_scores.numpy().tolist()
    all_labels = all_labels.numpy().tolist()
    
    final_preds_scores = []
    for scores, labels, countries in zip(all_scores, all_labels, all_countries):
        final_pred, final_score, mapping = get_final_basic_or_language_model_pred(scores, labels, countries,
                                                                         affiliation_vocab, 
                                                                         inverse_affiliation_vocab)
        final_preds_scores.append([final_pred, final_score, mapping])
    
    return final_preds_scores


def get_final_basic_or_language_model_pred(scores, labels, countries, vocab, inv_vocab):
    """
    Takes the scores and labels from either model and performs a quick country matching
    to see if the country found in the string can be matched to the country of the
    predicted institution.
    """
    mapped_labels = [inv_vocab[i] for i,j in zip(labels,scores) if i!=vocab[-1]]
    scores = [j for i,j in zip(labels,scores) if i!=vocab[-1]]
    final_pred = mapped_labels[0]
    final_score = scores[0]
    if not full_affiliation_dict[mapped_labels[0]]['country']:
        pass
    else:
        if not countries:
            pass
        else:
            for pred,score in zip(mapped_labels, scores):
                if not full_affiliation_dict[pred]['country']:
                    # trying pass instead of break to give time to find the correct country
                    pass
                elif full_affiliation_dict[pred]['country'] in countries:
                    final_pred = pred
                    final_score = score
                    break
                else:
                    pass
    return final_pred, final_score, mapped_labels
    
def get_similar_preds_to_remove(decoded_string, curr_preds):
    """
    Looks for organizations with similar/matching names and only predicts for one of those organizations.
    """
    preds_to_remove = []
    pred_display_names = [full_affiliation_dict[i]['display_name'] for i in curr_preds]
    counts_of_preds = Counter(pred_display_names)
    
    preds_array = np.array(curr_preds)
    preds_names_array = np.array(pred_display_names)
    
    for pred_name in counts_of_preds.items():
        temp_preds_to_remove = []
        to_use = []
        if pred_name[1] > 1:
            list_to_check = preds_array[preds_names_array == pred_name[0]].tolist()
            for pred in list_to_check:
                if string_match_clean(full_affiliation_dict[pred]['city']) in decoded_string:
                    to_use.append(pred)
                else:
                    temp_preds_to_remove.append(pred)
            if not to_use:
                to_use = temp_preds_to_remove[0]
                preds_to_remove += temp_preds_to_remove[1:]
            else:
                preds_to_remove += temp_preds_to_remove
        else:
            pass
    
    return preds_to_remove  


def check_for_city_and_country_in_string(raw_sentence, countries, aff_dict_entry):
    """
    Checks for city and country and string for a common name institution.
    """
    if (aff_dict_entry['country'] in countries) & (aff_dict_entry['city'] in raw_sentence):
        return True
    else:
        return False


def get_final_prediction(basic_pred_score, lang_pred_score, countries, raw_sentence, lang_thresh, basic_thresh):
    """
    Performs the model comparison and filtering to get the final prediction.
    """
    
    # Getting the individual preds and scores for both models
    pred_lang, score_lang, mapped_lang = lang_pred_score
    pred_basic, score_basic, mapped_basic = basic_pred_score
    
    # Logic for combining the two models
    final_preds = []
    final_scores = []
    final_cats = []
    check_pred = []
    if pred_lang == pred_basic:
        final_preds.append(pred_lang)
        final_scores.append(score_lang)
        final_cats.append('model_match')
        check_pred.append(pred_lang)
    elif score_basic > basic_thresh:
        final_preds.append(pred_basic)
        final_scores.append(score_basic)
        final_cats.append('basic_thresh')
        check_pred.append(pred_basic)
    elif score_lang > lang_thresh:
        final_preds.append(pred_lang)
        final_scores.append(score_lang)
        final_cats.append('lang_thresh')
        check_pred.append(pred_lang)
    elif (score_basic > 0.01) & ('China' in countries) & ('Natural Resource' in raw_sentence):
        final_preds.append(pred_basic)
        final_scores.append(score_basic)
        final_cats.append('basic_thresh_second')
        check_pred.append(pred_basic)
    else:
        final_preds.append(-1)
        final_scores.append(0.0)
        final_cats.append('nothing')
        
    # Getting unique candidates for string matching
    all_mapped = list(set(mapped_lang + mapped_basic))
    decoded_affiliation_string = string_match_clean(raw_sentence)
    all_mapped_strings = [full_affiliation_dict[i]['final_names'] for i in all_mapped]
          
    
    matched_preds = []
    matched_strings = []
    for inst_id, match_strings in zip(all_mapped, all_mapped_strings):
        if inst_id not in final_preds:
            for match_string in match_strings:
                if match_string in decoded_affiliation_string:
                    # match was found
                    if not full_affiliation_dict[inst_id]['country']:
                        # no countries in the affiliation dict for the current institution
                        matched_preds.append(inst_id)
                        matched_strings.append(match_string)
                    elif not countries:
                        # no countries found in the affiliation string
                        if inst_id not in multi_inst_names_ids:
                            # need to check if institution is part of multi-institution names
                            matched_preds.append(inst_id)
                            matched_strings.append(match_string)
                        else:
                            # institution is part of multi-institution name family
                            pass
                    elif full_affiliation_dict[inst_id]['country'] in countries:
                        # country in string matches country in affiliation dict
                        matched_preds.append(inst_id)
                        matched_strings.append(match_string)
                    else:
                        pass
                    break
                else:
                    pass
        else:
            pass
        
    # need to check for institutions that are a subset of another institution
    skip_matching = []
    
    # goes through all string matching candidates to see if one is substring of another
    for inst_id, matched_string in zip(matched_preds, matched_strings):
        for inst_id2, matched_string2 in zip(matched_preds, matched_strings):
            if (matched_string in matched_string2) & (matched_string != matched_string2):
                skip_matching.append(inst_id)
    
    # looks at model prediction to see if any of candidate strings are substring
    if check_pred:
        for inst_id, matched_string in zip(matched_preds, matched_strings):
            for final_string in full_affiliation_dict[check_pred[0]]['final_names']:
                if matched_string in final_string:
                    skip_matching.append(inst_id)
        
    # add matches to final predictions
    for matched_pred in matched_preds:
        if matched_pred not in skip_matching:
            final_preds.append(matched_pred)
            final_scores.append(0.95)
            final_cats.append('string_match')
            
    # remove -1 prediction if match was found
    if (final_cats[0] == 'nothing') & (len(final_preds)>1):
        final_preds = final_preds[1:]
        final_scores = final_scores[1:]
        final_cats = final_cats[1:]
        
    # check if many names belong to same organization name (different locations)
    if (final_preds[0] != -1) & (len(final_preds)>1):
        final_display_names = [full_affiliation_dict[x]['display_name'] for x in final_preds]

        if len(final_display_names) == set(final_display_names):
            pass
        else:
            final_preds_after_removal = []
            final_scores_after_removal = []
            final_cats_after_removal = []
            preds_to_remove = get_similar_preds_to_remove(decoded_affiliation_string, final_preds)
            for temp_pred, temp_score, temp_cat in zip(final_preds, final_scores, final_cats):
                if temp_pred in preds_to_remove:
                    pass
                else:
                    final_preds_after_removal.append(temp_pred)
                    final_scores_after_removal.append(temp_score)
                    final_cats_after_removal.append(temp_cat)

            final_preds = final_preds_after_removal
            final_scores = final_scores_after_removal
            final_cats = final_cats_after_removal
            
    
    # check for multi-name institution problems (final check)
    preds_to_remove = []
    if final_preds[0] == -1:
        pass
    else:
        final_department_name_ids = [[x, str(full_affiliation_dict[x]['display_name'])] for x in final_preds if 
                       (str(full_affiliation_dict[x]['display_name']).startswith("Department of") | 
                        str(full_affiliation_dict[x]['display_name']).startswith("Department for"))]
        if final_department_name_ids:
            for temp_id in final_department_name_ids:
                if string_match_clean(temp_id[1]) not in string_match_clean(str(raw_sentence).strip()):
                    preds_to_remove.append(temp_id[0])
                elif not check_for_city_and_country_in_string(raw_sentence, countries, 
                                                              full_affiliation_dict[temp_id[0]]):
                    preds_to_remove.append(temp_id[0])
                else:
                    pass


        if any(x in final_preds for x in multi_inst_names_ids):
            # go through logic
            if len(final_preds) == 1:
                pred_name = str(full_affiliation_dict[final_preds[0]]['display_name'])
                # check if it is exact string match
                if (string_match_clean(pred_name) == string_match_clean(str(raw_sentence).strip())):
                    final_preds = [-1]
                    final_scores = [0.0]
                    final_cats = ['nothing']
                elif pred_name.startswith("Department of"):
                    if ("College" in raw_sentence) or ("University" in raw_sentence):
                        final_preds = [-1]
                        final_scores = [0.0]
                        final_cats = ['nothing']
                    elif string_match_clean(pred_name) not in string_match_clean(str(raw_sentence).strip()):
                        final_preds = [-1]
                        final_scores = [0.0]
                        final_cats = ['nothing']

            else:
                non_multi_inst_name_preds = [x for x in final_preds if x not in multi_inst_names_ids]
                if len(non_multi_inst_name_preds) > 0:
                    for temp_pred, temp_score, temp_cat in zip(final_preds, final_scores, final_cats):
                        if temp_pred not in non_multi_inst_name_preds:
                            aff_dict_temp = full_affiliation_dict[temp_pred]
                            if aff_dict_temp['display_name'].startswith("Department of"):
                                if ("College" in raw_sentence) or ("University" in raw_sentence):
                                    preds_to_remove.append(temp_pred)
                                elif (string_match_clean(str(full_affiliation_dict[temp_pred]['display_name'])) 
                                      not in string_match_clean(str(raw_sentence).strip())):
                                    preds_to_remove.append(temp_pred)
                                else:
                                    if check_for_city_and_country_in_string(raw_sentence, countries, aff_dict_temp):
                                        pass
                                    else:
                                        preds_to_remove.append(temp_pred)
                            # check for city and country
                            elif aff_dict_temp['country'] in countries:
                                pass
                            else:
                                preds_to_remove.append(temp_pred)
                else:
                    pass
        else:
            pass
    
    true_final_preds = [x for x,y,z in zip(final_preds, final_scores, final_cats) if x not in preds_to_remove]
    true_final_scores = [y for x,y,z in zip(final_preds, final_scores, final_cats) if x not in preds_to_remove]
    true_final_cats = [z for x,y,z in zip(final_preds, final_scores, final_cats) if x not in preds_to_remove]
    
    if not true_final_preds:
        true_final_preds = [-1]
        true_final_scores = [0.0]
        true_final_cats = ['nothing']
    return [true_final_preds, true_final_scores, true_final_cats]

def raw_data_to_predictions(df, lang_thresh, basic_thresh):
    """
    High level function to go from a raw input dataframe to the final dataframe with affiliation
    ID prediction.
    """
    # Implementing the functions above
    df['affiliation_string'] = df['affiliation_string'].astype('str')
    df['string_len'] = df['affiliation_string'].astype('str').apply(len)
    df = df[df['string_len'] >2].copy()
    df['lang'] = df['affiliation_string'].apply(get_language)
    df['country_in_string'] = df['affiliation_string'].apply(get_country_in_string)
    df['comma_split_len'] = df['affiliation_string'].apply(lambda x: len([i if i else "" for i in 
                                                                          x.split(",")]))

    # Gets initial indicator of whether or not the string should go through the models
    df['affiliation_id'] = df.apply(lambda x: get_initial_pred(x.affiliation_string, x.lang, 
                                                               x.country_in_string, x.comma_split_len), axis=1)
    
    # Filter out strings that won't go through the models
    to_predict = df[df['affiliation_id']==0.0].drop_duplicates(subset=['affiliation_string']).copy()
    to_predict['affiliation_id'] = to_predict['affiliation_id'].astype('int')

    # Decode text so only ASCII characters are used
    to_predict['decoded_text'] = to_predict['affiliation_string'].apply(unidecode)

    # Get predictions and scores for each model
    to_predict['lang_pred_score'] = get_language_model_prediction(to_predict['decoded_text'].to_list(), 
                                                                  to_predict['country_in_string'].to_list())
    to_predict['basic_pred_score'] = get_basic_model_prediction(to_predict['decoded_text'].to_list(), 
                                                                to_predict['country_in_string'].to_list())

    # Get the final prediction for each affiliation string
    to_predict['affiliation_id'] = to_predict.apply(lambda x: 
                                                    get_final_prediction(x.basic_pred_score, 
                                                                         x.lang_pred_score, 
                                                                         x.country_in_string, 
                                                                         x.affiliation_string, 
                                                                         lang_thresh, basic_thresh), axis=1)
    
    to_predict['affiliation_id'] = to_predict['affiliation_id'].apply(lambda x: json.dumps(x[0]))

    # Merge predictions to original dataframe to get the same order as the data that was requested
    final_df = df[['affiliation_string']].merge(to_predict[['affiliation_string','affiliation_id']], 
                                                how='left', on='affiliation_string')
    
    final_df['affiliation_id'] = final_df['affiliation_id'].fillna(json.dumps([-1]))
    final_df['affiliation_id'] = final_df['affiliation_id'].apply(lambda x: json.loads(x))
    
    return final_df

print("Models initialized")

# The flask app for serving predictions
app = flask.Flask(__name__)
@app.route('/ping', methods=['GET'])
def ping():
    # Check if the classifier was loaded correctly
    try:
        _ = basic_model.get_layer('cls')
        status = 200
    except:
        status = 400
    return flask.Response(response= json.dumps(' '), status=status, mimetype='application/json' )

@app.route('/invocations', methods=['POST'])
def transformation():
    # Get input JSON data and convert it to a DF
    input_json = flask.request.get_json()
    input_json = json.dumps(input_json)
    input_df = pd.read_json(input_json, orient='records').reset_index()

    # Tokenize data
    final_df = raw_data_to_predictions(input_df, lang_thresh=0.99, basic_thresh=0.99)

    # Transform predicted labels into a list of dictionaries
    all_tags = []
    
    _ = [all_tags.append({'affiliation_id': i}) for i in final_df['affiliation_id'].to_list()]

    # Transform predictions to JSON
    result = json.dumps(all_tags)
    return flask.Response(response=result, status=200, mimetype='application/json')
