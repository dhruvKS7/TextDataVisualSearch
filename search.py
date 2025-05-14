import os
import re
import traceback
import math
import json
import requests
import random
import time
import copy
from urllib.parse import unquote

from typing import Dict, List, Tuple, Set
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from rakun2 import RakunKeyphraseDetector
import string
import ast
from google import genai
from google.genai import types

from flask import Blueprint, request, redirect
from flask_cors import CORS
from bson import ObjectId
from urllib.parse import urldefrag, quote
from collections import defaultdict
from openai import OpenAI

from app.helpers.helpers import token_required, token_required_public, response, Status, format_url, format_time_for_display, sanitize_input, build_display_url, hydrate_with_hashtags, get_community_name
from app.helpers.helper_constants import RE_URL_DESC
from app.helpers.prompts import gemini_prefix, gemini_suffix, llama3suffix_prompt, ics_query_prefix_prompt, ics_noquery_prefix_prompt, summarize_prefix_prompt, qa_prefix_prompt, rec_prefix_prompt, openai_qa_prompt, openai_qgen_prompt, openai_qgen_withintent_prompt, ics_noquery_prefix_prompt_v2, ics_query_prefix_prompt_v2

from app.models.cache import Cache
from app.models.search_logs import SearchLogs, SearchLog
from app.models.search_clicks import SearchClicks, SearchClick
from app.models.logs import Logs
from app.models.users import Users
from app.models.communities import Communities
from app.models.submission_stats import SubmissionStats

from elastic.manage_data import ElasticManager

search = Blueprint('search', __name__)
CORS(search)

# Connect to elastic for submissions index operations
elastic_manager = ElasticManager(
    os.environ["elastic_username"],
    os.environ["elastic_password"],
    os.environ["elastic_domain"],
    os.environ["elastic_index_name"],
    None,
    "submissions")

try:
    client = OpenAI()
except:
    client = False

try:
    gemini = genai.Client(api_key=os.environ["gemini_api"])
except:
    gemini = False

@search.route("/api/search/website", methods=["GET"])
@token_required_public
def website_search(current_user):
    """Handles the search for the TextData website.
    
    Two general cases: initiating a new search or paging a current search.


    Method Parameters
    ----------
    current_user : MongoDB Document, required
        The user making the request. This is automatically passed via the @token_required wrapper.


    Request Parameters (New Search)
    ---------
        query : str, optional
            What the user typed into the search bar. If excluded, results are ordered by time.

        community : str, required
            A list of community IDs to search. Each community should be separated by a comma.
            Can also be "all". Here, all joined and followed communities will be searched.

        source : str, required
            For logging where the search was made. One of
                website_visualize
                website_homepage_recs
                website_community_page
                website_searchbar

        own_submissions : bool, optional
            If true, only search results from one's own submissions will be returned.
            Defaults to True.

    Request Parameters (Paging)
    ---------
        search_id : str, required
            The ID of the search, must correspond to its log in SearchLogs

        page : int, >=0, required
            The requested page of search results. 

        sort_by : str, optional
            Reorganizes the cached search results. 
            Can be either "time", "relevance", or "popularity".

    Returns
    ---------
    On success, a response.success object with the results:
        search_id
        query
        total_num_results
        current_page
        search_results_page 
        requested_communities
    """
    # Parameters for new search
    query = request.args.get("query", "")
    user_requested_communities = request.args.get("community", None)
    own_submissions = request.args.get("own_submissions", False)
    if own_submissions == "True":
        own_submissions = True
    else:
        own_submissions = False
    source = request.args.get("source", None)    

    user_id = current_user.id
    user_communities = current_user.communities
    for x in current_user.followed_communities:
        if x not in user_communities:
            user_communities.append(x)

    if user_requested_communities == "all":
        requested_communities = user_communities
    else:
        try:
            requested_communities = [ObjectId(x) for x in user_requested_communities.split(",")]
        except Exception as e:
            traceback.print_exc()
            return response.error("Invalid requested community.", Status.BAD_REQUEST) 
   
    search_id = log_search_request(user_id, 
                            source,
                            scope=["submissions"],
                            intent={
                                "typed_query": query,
                            },
                            filters={
                                "own_submissions": own_submissions,
                                "communities": requested_communities,
                                "sort_by": "relevance"
                            }
                            )

    if source == "website_visualize":
        submissions = export_helper(str(user_id), str(search_id), include_questions=True)

        graph_data = construct_viz_tree(submissions['data'])
        return response.success(graph_data, Status.OK)
    else:
        return response.error("You do not have access to the requested communities.", Status.FORBIDDEN) 


# function to merge keywords based on synonyms and suffixes
def merge_keywords(doc2keywords, keywords):
    # count frequencies to pick the most common keyword as base
    freq = {}
    for kws in doc2keywords.values():
        for kw in kws:
            freq[kw] = freq.get(kw, 0) + 1

    # use union-find for merging
    parent = {kw: kw for kw in keywords}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        # pick keyword with higher frequency - use alphabetical ordering for tiebreaker
        if freq.get(ra, 0) > freq.get(rb, 0) or (freq.get(ra, 0) == freq.get(rb, 0) and ra < rb):
            parent[rb] = ra
        else:
            parent[ra] = rb

    # create lemmatizer
    lemmatizer = WordNetLemmatizer()

    # merge keywords
    for kw in keywords:
        # spaces ("intelligentagent" v "intelligent agent")
        if kw.replace(' ', '') in parent:
            union(kw, kw.replace(' ', ''))
        
        # simple plurals
        if kw.endswith('s') and kw[:-1] in parent:
            union(kw, kw[:-1])

        # lemmatization
        lemma = lemmatizer.lemmatize(kw)
        if lemma in parent:
            union(kw, lemma)

        # common suffixes
        suffixes = ['ing', 'ed', 'ly', 'er', 'es', 'en', 'est', 'or', 'ist', 'ian', 'an', 'tion', 'ation', 'sion', 'ion', 'ment', 'ness', 'ity', 'ty', 'able', 'ible', 'al', 'ic', 'ical', 'ous', 'ful', 'less', 'y', 'ize', 'ify']
        for suf in suffixes:
            # check if word without suffix exists (i.e. merge design and designer)
            if kw.endswith(suf) and kw[:-len(suf)] in parent:
                union(kw, kw[:-len(suf)])
                break
            # check if word with any other suffix exists (i.e. merge designer and designing)
            for suf2 in suffixes:
                if kw.endswith(suf) and kw[:-len(suf)] + suf2 in parent:
                    union(kw, kw[:-len(suf)] + suf2)
                    break
    
        # edit-distance = 1 --> removed since things like content and context would get combined
        # relying more on the llm for things like typos or off-by-1 differences

    # WordNet synonyms
    for kw in keywords:
        syns = {
            lemma.name().replace('_', ' ')
            for syn in wordnet.synsets(kw)
            for lemma in syn.lemmas()
        }
        for s in syns:
            if s in parent:
                union(kw, s)

    # flatten to mapping
    final_map = {kw: find(kw) for kw in keywords}

    # merged keyword list w/ unique values
    merged_keywords = sorted(
        set(final_map.values()),
        key=lambda k: freq.get(k, 0), reverse=True
    )

    # update each document’s keyword list
    doc2mergedkeywords = {}
    for doc, kws in doc2keywords.items():
        seen = set()
        merged = []
        for kw in kws:
            rep = final_map.get(kw, kw)
            if rep not in seen:
                seen.add(rep)
                merged.append(rep)
        doc2mergedkeywords[doc] = merged

    return doc2mergedkeywords, merged_keywords

# lightweight version of above function that deals just with a list of single word hashtags
def merge_hashtags(keywords):
    # use union-find for merging
    parent = {kw: kw for kw in keywords}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        # pick keyword based on alphabet - no matter which is picked, it should revert to most common one later
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    # create lemmatizer
    lemmatizer = WordNetLemmatizer()

    # merge keywords
    for kw in keywords:
        # simple plurals
        if kw.endswith('s') and kw[:-1] in parent:
            union(kw, kw[:-1])

        # lemmatization
        lemma = lemmatizer.lemmatize(kw)
        if lemma in parent:
            union(kw, lemma)

        # common suffixes
        suffixes = ['ing', 'ed', 'ly', 'er', 'es', 'en', 'est', 'or', 'ist', 'ian', 'an', 'tion', 'ation', 'sion', 'ion', 'ment', 'ness', 'ity', 'ty', 'able', 'ible', 'al', 'ic', 'ical', 'ous', 'ful', 'less', 'y', 'ize', 'ify']
        for suf in suffixes:
            # check if word without suffix exists (i.e. merge design and designer)
            if kw.endswith(suf) and kw[:-len(suf)] in parent:
                union(kw, kw[:-len(suf)])
                break
            # check if word with any other suffix exists (i.e. merge designer and designing)
            for suf2 in suffixes:
                if kw.endswith(suf) and kw[:-len(suf)] + suf2 in parent:
                    union(kw, kw[:-len(suf)] + suf2)
                    break
    
        # edit-distance = 1 --> removed since things like content and context would get combined
        # relying more on the llm for things like typos or off-by-1 differences

    # WordNet synonyms
    for kw in keywords:
        syns = {
            lemma.name().replace('_', ' ')
            for syn in wordnet.synsets(kw)
            for lemma in syn.lemmas()
        }
        for s in syns:
            if s in parent:
                union(kw, s)

    # flatten to mapping
    final_map = {kw: find(kw) for kw in keywords}

    # merged keyword list w/ unique values
    merged_keywords = set(final_map.values())

    return merged_keywords

# extract every hashtag from the text
def extract_hashtags(text):
    tags = []
    for word in text.split():
        # remove all punctuation except # from word
        punct_to_remove = string.punctuation.replace('#', '')
        table = str.maketrans('', '', punct_to_remove)
        word = word.translate(table).strip()
        # if # is the first character
        if word.startswith('#'):
            # remove # to make sure there is a word (avoids things like '###' getting picked up)
            word = word.replace('#', '')
            if len(word) > 0:
                tags.append(word)
    return tags

# remove links since they often get picked up by rakun as keywords
# also remove hashtags to help rakun get varied keywords
def clean_links(text):
    text = text.lower().strip()
    split = text.split()
    for i in range(len(split)-1, -1, -1):
        if "http" in split[i]:
            del split[i]
        elif "www" in split[i]:
            del split[i]
        elif "#" in split[i]:
            del split[i]
    cleaned = " ".join(split)
    return cleaned

def make_keyword2docs(doc2keywords):
    keyword2docs = defaultdict(list)
    for doc, keywords in doc2keywords.items():
        for kw in keywords:
            keyword2docs[kw].append(doc)
    return dict(keyword2docs)

def generate_gemini(input_keywords):
    if not gemini:
        return {"message": "Gemini model not initialized."}
    try:
        prompt = gemini_prefix + str(input_keywords) + gemini_suffix
        outputs = gemini.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt],
            config=types.GenerateContentConfig(
                # limit output tokens for runtime and to shield against llm failure
                max_output_tokens=500,
                temperature=0
            )
        )
        output = outputs.text
    except Exception as e:
        traceback.print_exc()
        return {"message": "Something went wrong with generation, please try again later."}

    return {"message": output}

# extract actual dict from llm output
def parse_llm_dict(llm_output):
    # remove any markdown code
    cleaned = re.sub(r"^```(?:python)?\n|```$", "", llm_output.strip(), flags=re.MULTILINE)
    
    # find the first {...} block
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        # if llm output was garbage, return None
        return None
    
    dict_literal = match.group(0)
    return ast.literal_eval(dict_literal)

# merge keywords based on llm output
def replace_with_categories(doc2keywords, category_map, keyword_freq):
    # create dict mapping term to what it will become
    term_to_merge = {}
    for category, terms in category_map.items():
        best_category = category
        max_occurences = 0
        # for every grouping of keywords, match them all to the most common keyword in the group
        for term in terms:
            occurences = keyword_freq.get(term, 0)
            if occurences > max_occurences:
                best_category = term
                max_occurences = occurences
        for term in terms:
            term_to_merge[term] = best_category
    # create the merged mapping
    doc2mergedkeywords = {}
    for doc, kws in doc2keywords.items():
        seen = set()
        merged = []
        for kw in kws:
            # map to category if exists, else keep original
            mapped = term_to_merge.get(kw, kw)
            if mapped not in seen:
                seen.add(mapped)
                merged.append(mapped)
        doc2mergedkeywords[doc] = merged

    return doc2mergedkeywords

def llm_merge(llm_pass, doc2keywords, keyword_freq):
    doc2keywordscopy = doc2keywords.copy()
    try:
        output = generate_gemini(llm_pass)
        out = output["message"]
        # find the last full entry in the dict and close it
        # since output length is limited, very likely that the dict was cut off and is incomplete
        idx = out.rfind(']')
        if idx != -1:
            out = out[: idx + 1] + '}'
        category_map = parse_llm_dict(out)
        # if dict was extracted
        if category_map:
            for key in list(category_map.keys()):
                # drop singleton lists
                if len(category_map[key]) == 1:
                    del category_map[key]
                    continue

                # convert key to lowercase if necessary
                lower = key.lower()
                if lower != key:
                    category_map[lower] = category_map[key]
                    del category_map[key]
            doc2keywords = replace_with_categories(doc2keywords, category_map, keyword_freq)
            final_merge = set()
            for doc in doc2keywords:
                for keyword in doc2keywords[doc]:
                    final_merge.add(keyword)
            doc2keywords, merged_keywords = merge_keywords(doc2keywords, list(final_merge))
            return doc2keywords
        return doc2keywordscopy
    except:
        return doc2keywordscopy

def construct_viz_tree(data):
    try:
        try:
            # set max number of keywords for each document
            max_keywords = 4
            # set rakun hyperparameters
            # found merge = 0 and alpha close to 0 yielded best results
            # gave more variety and better extracted keywords
            hyperparameters = {"num_keywords": max_keywords,
                            "merge_threshold": 0,
                            "alpha": 0.1,
                            "token_prune_len": 1}
            keyword_detector = RakunKeyphraseDetector(hyperparameters)

            # create variables for holding info
            doc2keywords = {}
            doc2name = {}
            all_keywords = set()
        except:
            return {"error": "Visualization initialization failed"}
        try:
            # go through each submission
            for i in range(len(data)):
                # save text and make it uniformly lowercase
                doc = data[i]['description']
                doc = doc.lower()
                # save id
                id = data[i]['submission_id']
                # initialize dict values
                doc2keywords[id] = []
                doc2name[id] = data[i]['title']
                # get hashtags from text
                tags = extract_hashtags(doc)
                # prevent duplicates
                tags = list(set(tags))
                # merge hashtags
                tags = list(merge_hashtags(tags))
                # if we have enough hashtags to meet max_keywords, skip keyword extraction
                if len(tags) >= max_keywords:
                    tags = tags[:max_keywords]
                    for tag in tags:
                        all_keywords.add(tag)
                    doc2keywords[id] = tags
                    continue
                # else, populate list with all available hashtags
                elif len(tags) > 0:
                    doc2keywords[id] = tags
                    for tag in tags:
                        all_keywords.add(tag)
                
                # remove links and hashtags from doc
                clean_doc = clean_links(doc)
                start_len = len(doc2keywords[id])
                # extract keywords / key phrases
                keywords = keyword_detector.find_keywords(clean_doc, input_type="string")

                for x in keywords:
                    # if at max, break
                    if len(doc2keywords[id]) == max_keywords:
                        break
                    # only save keywords with high confidence
                    if x[1] < 0.15:
                        continue
                    # if keyword is the same as an already saved hashtag, skip
                    if x[0] in doc2keywords[id]:
                        continue
                    doc2keywords[id].append(x[0])
                    all_keywords.add(x[0])

                # if 0 hashtags and 0 keywords found, then fall to worst-case scenario of N/A
                if len(doc2keywords[id]) == 0 and len(keywords) == 0:
                    doc2keywords[id].append("N/A")
                    all_keywords.add("N/A")
                    continue
                end_len = len(doc2keywords[id])
                # if no keywords were added (meaning all were below confidence threshold or no keywords were found)
                if start_len == end_len:
                    # then add the first keyword that is unique from hashtags
                    for x in keywords:
                        if x[0] not in doc2keywords[id]:
                            doc2keywords[id].append(x[0])
                            all_keywords.add(x[0])
                            break
        except:
            return {"error": "Visualization extraction failed"}
        # merge keywords
        try:
            doc2keywords, merged_keywords = merge_keywords(doc2keywords, list(all_keywords))

            keyword2docs = make_keyword2docs(doc2keywords)

            keyword_freq = {}
            for i in keyword2docs:
                keyword_freq[i] = len(keyword2docs[i])

            sorted_items = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)

            # create list of keywords sorted by descending frequencies
            llm_pass = []
            for x in sorted_items:
                llm_pass.append(x[0])

        except:
            return {"error": "Visualization merge failed"}
        
        doc2keywords = llm_merge(llm_pass, doc2keywords, keyword_freq)
        globalk2docs = make_keyword2docs(doc2keywords)
        
        # def greedy_set_cover(all_docs, doc2keywords, exclude):
        # calculate greedy set cover for a level
        def greedy_set_cover(all_docs, doc2keywords):
            exclude = set()
            uncovered = set(all_docs)

            # precompute keyword2docs for this leve,
            kw2docs: Dict[str, Set[str]] = defaultdict(set)
            for d in all_docs:
                for kw in doc2keywords.get(d, []):
                    if kw not in exclude:
                        kw2docs[kw].add(d)

            picks = []
            while uncovered:
                # find all potential options for next cluster
                candidates = [
                    (kw, docs & uncovered)
                    for kw, docs in kw2docs.items()
                    # only include keywords if:
                    # they have not been seen before
                    # they cover some documents
                    # they do not cover every document (prevents a level from being just one folder)
                    if kw not in exclude and (docs & uncovered) and len(docs) < len(all_docs)
                ]
                if not candidates:
                    break

                # tiebreak on most common keyword globally
                best_kw, best_docs = max(candidates, key=lambda x: (len(x[1]), len(globalk2docs[x[0]])))
                picks.append((best_kw, kw2docs[best_kw]))

                exclude.add(best_kw)
                uncovered -= best_docs
            
            # every doc left uncovered is one that is out of keywords – they should be added at this level
            for doc in uncovered:
                picks.append((-1, set([doc])))
            return picks

        # recursively build tree
        def build_tree_min_cover(all_docs, doc2keywords):
            # every time a leaf node is about to be made, include any mentions or questions --> TO-DO

            # base case - stop recursion if only one doc remains in the group (prevents folder -> folder -> doc)
            if len(all_docs) <= 1:
                name = []
                for doc in list(all_docs):
                    name.append(doc2name[doc])
                return [{"type": "leaf", "name": name, "docs": list(all_docs)}]

            picks = greedy_set_cover(all_docs, doc2keywords)

            # fall-back case for stopping recursion
            if not picks:
                name = []
                for doc in list(all_docs):
                    name.append(doc2name[doc])
                return [{"type": "leaf", "name": name, "docs": list(all_docs)}]

            # sort picks by number of docs each folder contains
            picks.sort(key=lambda x: len(x[1]), reverse=True)

            # check if misc folder should be created
            check = False
            cnt_one = 0
            cnt_two_plus = 0
            for kw, covered_docs in picks:
                if len(covered_docs) > 1:
                    cnt_two_plus += 1
                elif len(covered_docs) == 1 and kw != -1:
                    cnt_one += 1
            if cnt_one > 1 and cnt_two_plus > 0:
                check = True
            
            # only create misc folder if 30+ items are on a level
            if check and len(picks) >= 30:
                misc = set()
                for i in range(len(picks)-1, -1, -1):
                    # only include folder -> 1 document in misc, don't include any documents on that level
                    if len(picks[i][1]) == 1 and picks[i][0] != -1:
                        misc.update(picks[i][1])
                        del picks[i]
                if len(misc) > 0:
                    done = False
                    # insert misc folder before any raw documents but at the end of all folders
                    for i, tup in enumerate(picks):
                        if tup[0] == -1:
                            picks.insert(i, ("misc.", misc))
                            done = True
                            break
                    if not done:
                        picks.append(("misc.", misc))
            
            nodes = []
            for kw, covered_docs in picks:
                # add document at level
                if kw == -1:
                    name = []
                    for doc in list(covered_docs):
                        name.append(doc2name[doc])
                    nodes.append({"type": "leaf", "name": name, "docs": list(covered_docs)})
                    continue
                # recurse on children
                children = build_tree_min_cover(list(covered_docs), doc2keywords)
                nodes.append({
                    "type": "node",
                    "name": kw,
                    "docs": list(covered_docs),
                    "children": children
                })

            return nodes

        try:
            tree = build_tree_min_cover([d["submission_id"] for d in data], doc2keywords)
        except:
            return {"error": "Visualization construction failed"}
        return {"tree": tree}
    except:
        return {"error": "Visualization failed, try again later"}