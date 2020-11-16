# Author: Mihai Avram
# E-Mail: mihai.v.avram@gmail.com

# This is the function package file used by the main file

# for debugging errors
import sys

import networkx as nx
from random import randint
# import matplotlib.pyplot as plt

# used for timing the procedures
import time
import progressbar

# used for writing the results in a json file
import json

# used for converting graphs to json serialized versions
from networkx.readwrite import json_graph

# used for sampling
import numpy as np

# used for multithreading
from multiprocessing import Process
from multiprocessing import Manager
from math import ceil, floor
# method to ensure multi-processes work on equitable tasks
from random import shuffle

local_config_file_path = 'runtime_configs.json'

# Set up all configs from config file
def initialize_configs(all_configs):
    # In case user specifies own configuration path
    global local_config_file_path
    if all_configs["specific_config_path"] is not None:
        local_config_file_path = all_configs["specific_config_path"]
    with open(local_config_file_path, 'r') as f:
        local_file = json.load(f)
        all_configs["aggregate_graphs_path"] = \
            local_file["aggregate_graphs_path"]
        if all_configs["aggregate_graphs_path"] == "" \
            or all_configs["aggregate_graphs_path"] == "None" \
            or all_configs["aggregate_graphs_path"] == "False" \
            or all_configs["aggregate_graphs_path"] == "./":
                initialize_graph_from_config(all_configs, local_file)
                all_configs["aggregate_graphs_path"] = "./"

        all_configs["output_path"] = local_file['output_path']
        all_configs["print_minimum_output"] = \
            True if (local_file["print_minimum_output"] == "True") else False

        all_configs["bootstrap_sampling_times"] = \
            local_file["sampling"]["bootstrap_sampling_times"]
        all_configs["num_tiers"] = \
            local_file["sampling"]["num_tiers"]
        # adjusting by one due to 0 indexing
        all_configs["specific_tier_selected"] = \
            local_file["sampling"]["specific_tier_selected"] - 1
        all_configs["tier_sampling_size"] = \
            local_file["sampling"]["tier_sampling_size"]
        all_configs["with_replacement"] = \
            True if (local_file["sampling"]["with_replacement"] == "True") else False
        all_configs["sample_all_nodes"] = \
            True if (local_file["sampling"]["sample_all_nodes"] == "True") else False
        initialize_evaluation_from_config(all_configs, local_file)
        all_configs["one_time_budget"] = local_file["budget"]["one_time_budget"]
        all_configs["min_budget"] = local_file["budget"]["min_budget"]
        all_configs["max_budget"] = local_file["budget"]["max_budget"]
        all_configs["sample_budget"] = \
            True if (local_file["budget"]["sample_budget"] == "True") else False
        all_configs["adversarial_functions_costs"] = \
            local_file["adversarial_functions_costs"]
        initialize_budget_functions_and_costs(all_configs, local_file)
        all_configs["verbose"] = \
            True if (local_file["verbose"] == "True") else False
        all_configs["greedy"] = \
            True if (local_file["greedy"] == "True") else False
        all_configs["num_of_processes"] = local_file['num_of_processes']
        save_or_load_sequences_config(all_configs, local_file)

    return all_configs

def initialize_graph_from_config(all_configs, local_file):
    if all_configs["input_path"] != "":
           # Initializing graph from file
           all_configs["input_path"] = local_file["input_path"]
           graph = nx.read_edgelist(all_configs["input_path"], nodetype=int)
           all_configs["graph"] = graph.to_undirected()
           all_configs["graph_type"] = "from_file"
           all_configs["graph_name"] = local_file["input_path"].split("/")[-1]
    else:
        # Initializing a synthetic graph based on the configuration
        true_count = 0
        for graph_key, val in local_file["graph"].items():
            if val == "True":
                true_count += 1
                all_configs["graph_type"] = str(graph_key)
        if true_count == 0 or true_count > 1:
            raise ValueError('Check', local_config_file_path, \
                "file and initialize exactly one graph as true")

        if all_configs["graph_type"] == "scale_free":
            all_configs["graph"] = \
                nx.barabasi_albert_graph(n = 50 , m = 5, seed = randint(1, 100))
        elif all_configs["graph_type"] == "small_world":
            all_configs["graph"] = \
                nx.watts_strogatz_graph(n = 50, k = 4, p = 0.5, seed = randint(1, 100))
        elif all_configs["graph_type"] == "random":
            all_configs["graph"] = \
                nx.gnp_random_graph(n = 50, p = 0.3, seed = randint(1, 100), directed=False)
        elif all_configs["graph_type"] == "cellular":
            all_configs["graph"] = \
                cellular_network(cluster_size = 10, clusters = 5, p = 0.6, seed = randint(0, 100), directed = False)
        return all_configs

def initialize_evaluation_from_config(all_configs, local_file):
    # Eval Function
    true_count = 0
    for eval_key, val in local_file["evaluation"]["eval_function"].items():
        if val == "True":
            true_count += 1
            all_configs["eval_function_text"] = str(eval_key)
    if true_count == 0 or true_count > 1:
        raise ValueError('Check', local_config_file_path, \
            "file and initialize exactly one eval_function as true")

    if all_configs["eval_function_text"] == "degree_centrality":
        all_configs["eval_function"] = get_degree_centrality
    elif all_configs["eval_function_text"] == "betweenness_centrality":
        all_configs["eval_function"] = get_betweenness_centrality
    elif all_configs["eval_function_text"] == "closeness_centrality":
        all_configs["eval_function"] = get_closeness_centrality
    elif all_configs["eval_function_text"] == "eigenvector_centrality":
        all_configs["eval_function"] = get_eigenvector_centrality

    # Eval Direction
    true_count = 0
    for eval_key, val in local_file["evaluation"]["eval_direction"].items():
        if val == "True":
            true_count += 1
            all_configs["eval_direction"] = str(eval_key)
    if true_count == 0 or true_count > 1:
        raise ValueError('Check', local_config_file_path, \
            "file and initialize exactly one eval_direction as true")

def initialize_budget_functions_and_costs(all_configs, local_file):
    adversarial_functions_costs = local_file["adversarial_functions_costs"]
    try:
        adversarial_functions_costs_lists = \
            [[k, v] for (k, v) in adversarial_functions_costs.items()]
        budget_functions_sorted_by_cost = \
            sorted(adversarial_functions_costs_lists, key=lambda kv: kv[1])
    except:
        print('Check', local_config_file_path, \
              "file and 'adversarial_functions_costs' parameter. Ensure all" + \
              " functions have a cost, and are comma delimited.",
              sys.exc_info()[0])
        raise

    completed_and_sorted_budget_functions = []
    for ix, budget_func in enumerate(budget_functions_sorted_by_cost):
        physical_code_function = adv_function_mappings_text_to_func[budget_func[0]]
        current_function_tuple = (budget_func[0], physical_code_function)
        current_completed_function = [ix, [budget_func[1], current_function_tuple]]
        completed_and_sorted_budget_functions.append(current_completed_function)

    all_configs['adversarial_functions'] = \
        completed_and_sorted_budget_functions


def save_or_load_sequences_config(all_configs, local_file):
    if local_file["save_sequences_on_file"] == \
        "True" and local_file["read_sequences_from_file"]["read_from_file"] == "True":
        raise ValueError('Check', local_config_file_path, \
            "file and have either save_sequences_on_file as True or " + \
            "read_from_file as true.")
    all_configs["save_sequences_on_file"] = \
        True if (local_file["save_sequences_on_file"] == "True") else False
    all_configs["read_sequences_from_file"] = \
        True if (local_file["read_sequences_from_file"]["read_from_file"] == "True") else False
    all_configs["load_sequences_file_path"] = \
        local_file["read_sequences_from_file"]["load_sequences_file_path"]

def save_sequences_on_file(all_adversarial_sequences, output_path):
    output_path_file = output_path + "/cached_adv_sequences_and_costs.json"
    adv_seq_json = {"sequences":{}}
    for seq_ix, adv_seq in enumerate(all_adversarial_sequences):
        named_seq_ix = "seq_" + str(seq_ix)
        adv_seq_json["sequences"][named_seq_ix] = {}
        for move_ix, move in enumerate(adv_seq):
            named_move_ix = "move_" + str(move_ix)
            adv_seq_json["sequences"][named_seq_ix][named_move_ix] = {}
            adv_seq_json["sequences"][named_seq_ix][named_move_ix]["adv_move_ix"] = \
                move[0]
            adv_seq_json["sequences"][named_seq_ix][named_move_ix]["cost"] = \
                move[1][0]
            adv_seq_json["sequences"][named_seq_ix][named_move_ix]["move_name"] = \
                move[1][1][0]

    with open(output_path_file, 'w') as dump_file:
        json.dump(adv_seq_json, dump_file)

def read_sequences_from_file(seq_file_input_path):
    all_adversarial_sequences = []
    with open(seq_file_input_path, 'r') as f:
        seq_content = json.load(f)
        for seq_ix, sequence in seq_content["sequences"].items():
            sequence_list = []
            for move_ix, move in sequence.items():
                adv_move_ix = move["adv_move_ix"]
                move_cost = move["cost"]
                move_name = move["move_name"]
                move_function_mapping = \
                    adv_function_mappings_text_to_func[move_name]
                function_tuple = (move_name, move_function_mapping)
                move_list = [adv_move_ix, [move_cost, function_tuple]]
                sequence_list.append(move_list)
            all_adversarial_sequences.append(sequence_list)
    return all_adversarial_sequences

# Centrality Measures
def get_degree_centrality(graph, node):
    return len(list(graph.neighbors(node)))

def get_closeness_centrality(graph, node):
    return nx.closeness_centrality(graph, node)

def get_eigenvector_centrality(graph, node):
    return nx.eigenvector_centrality_numpy(graph)[node]

def get_betweenness_centrality(graph, node):
    return nx.betweenness_centrality(graph)[node]

def get_pagerank_centrality(graph, node):
    return nx.pagerank_numpy(graph)[node]

def get_all_centrality_measures_additive(graph, node):
    degree = len(list(graph.neighbors(node)))
    closeness = nx.closeness_centrality(graph, node)
    eigenvector = nx.eigenvector_centrality_numpy(graph)[node]
    betweenness = nx.betweenness_centrality(graph)[node]
    pagerank = nx.pagerank_numpy(graph)[node]
    all_measures_additive = degree + closeness + eigenvector + betweenness + pagerank
    return(all_measures_additive)

def compute_ranked_dictionary(graph, eval_function_text):
    rank_dict = {}

    if eval_function_text == 'degree_centrality':
        rank_dict = dict(graph.degree(graph.nodes()))
    elif eval_function_text == 'closeness_centrality':
        rank_dict = nx.closeness_centrality(graph)
    elif eval_function_text == 'betweenness_centrality':
        rank_dict = nx.betweenness_centrality(graph)
    elif eval_function_text == 'eigenvector_centrality':
        rank_dict = nx.eigenvector_centrality_numpy(graph)
    elif eval_function_text == 'pagerank_centrality':
        rank_dict = nx.pagerank_numpy(graph)

    return(rank_dict)


# Picking a sample for each eval metric tier, e.g., 3 items from 33%ile, 3 items
# from 66%ile, and3 items for 100%ile respectively
def pick_node_sample(eval_function_text,
                     graph,
                     tiers,
                     specific_tier_selected,
                     tier_sampling_size,
                     with_replacement,
                     sample_all_nodes):
    # useful inner function
    def select_tier_samples(tier_contents, tier_sampling_size, tier):
        scoped_tiered_selected_samples = []
        total_nodes_in_tier = len(tier_contents)
        if sample_all_nodes:
            # retrieve all nodes
            scoped_tiered_selected_samples.append(tier_contents)
        else:
            # sample subset of nodes
            if with_replacement:
                scoped_tiered_selected_samples.append((np.random.choice(
                    tier_contents,
                    tier_sampling_size,
                    replace=True).tolist()))
            else:
                if tier_sampling_size > total_nodes_in_tier:
                    tier_sampling_size = total_nodes_in_tier
                scoped_tiered_selected_samples.append((np.random.choice(
                    tier_contents,
                    tier_sampling_size,
                    replace=False).tolist()))
        return(scoped_tiered_selected_samples)

    rank_dict = compute_ranked_dictionary(graph, eval_function_text)
    ranked_contents_sorted_by_value = \
        sorted(rank_dict.items(), key=lambda kv: kv[1])

    tier_increments_percentage = 1.0 / tiers
    total_tier_size = floor(tier_increments_percentage * len(ranked_contents_sorted_by_value))
    num_elements_even_or_odd = \
        "even" if (len(ranked_contents_sorted_by_value) % 2 == 0) else "false"

    all_tiered_contents = []
    if num_elements_even_or_odd == "even":
    # creating tiers with even amount of nodes
        for tier_ix in range(tiers):
            tier_cotents = []
            for node_pos in range(tier_ix * total_tier_size, ((1 + tier_ix) * total_tier_size)):
                tier_cotents.append(ranked_contents_sorted_by_value[node_pos][0])
            all_tiered_contents.append(tier_cotents)
    else:
    # creating tiers with odd amount of nodes
        last_tier = tiers - 1
        tier_selected_for_extra_node = randint(0, last_tier-1)
        for tier_ix in range(tiers):
            tier_cotents = []
            for node_pos in range(tier_ix * total_tier_size, ((1 + tier_ix) * total_tier_size)):
                tier_cotents.append(ranked_contents_sorted_by_value[node_pos][0])
            if tier_ix == last_tier:
                # adding extra member (odd) at the end
                tier_cotents.append(ranked_contents_sorted_by_value[-1][0])
            all_tiered_contents.append(tier_cotents)

        # shifting extra node element to a random tier
        for tier_to_shift in range(last_tier, tier_selected_for_extra_node, -1):
            node_to_shift = all_tiered_contents[tier_to_shift][0]
            # shifting backwards so [[0,1],[[5,3,4]]] will become
            # [[0,1,5],[3,4]] in an interative fashion depending on how
            # many tiers are available
            tier_to_shorten = all_tiered_contents[tier_to_shift]
            tier_to_shorten.pop(0)
            all_tiered_contents[tier_to_shift] = tier_to_shorten
            all_tiered_contents[tier_to_shift - 1].append(node_to_shift)

    tiered_selected_samples = []

    # All tiers will be retrieved from
    if specific_tier_selected == None:
        for tier in range(tiers):
            tiered_selected_samples += \
                select_tier_samples(all_tiered_contents[tier], tier_sampling_size, tier)
    else:
        # Only a specific tier will be selected
        tier = specific_tier_selected
        tiered_selected_samples += \
            select_tier_samples(all_tiered_contents[tier], tier_sampling_size, tier)
    return(tiered_selected_samples)



def compute_node_percentile(graph,
                            node,
                            eval_function_text):
    rank_dict = compute_ranked_dictionary(graph, eval_function_text)
    ranked_contents_sorted_by_value = \
        sorted(rank_dict.items(), key=lambda kv: kv[1])
    sorted_increasing_node_list = [n for (n, v) in ranked_contents_sorted_by_value]
    node_index = sorted_increasing_node_list.index(node)
    node_percentile = (node_index + 1) / len(ranked_contents_sorted_by_value)
    return(node_percentile)


# Finding Which Nodes are "interesting"
def get_minmax_degree_nodes(graph):
    node_degrees = graph.degree()

    max_deg = None
    min_deg = None

    max_node = None
    min_node = None

    for node, deg in node_degrees:
        if (max_deg == None or deg > max_deg):
            max_deg = deg
            max_node = node
        if (min_deg == None or deg < min_deg):
            min_deg = deg
            min_node = node

    return({'min-node':(min_node, min_deg),'max-node':(max_node, max_deg)})





# Adversarial Functions
# EDGES
def self_remove_edge_friend(node, original_graph, increase_metric, eval_function, greedy, verbose):
    original_metric = eval_function(original_graph, node)
    change_dict = {'new_metric':original_metric,
                   'performed':False,
                   'attempt':{'picked':{},'tries':{}}}
    if greedy:
        change = original_metric
    else:
        change = None
    friend_to_remove_edge_from = None

    # checking if the node is already optimal
    if original_metric == 0 and not increase_metric:
        change_dict['attempt']['picked'] = 'no optimal change to make \
                                            as evaluation is already optimal'
        return(change_dict)

    if not any(original_graph.neighbors(node)):
        change_dict['attempt']['picked'] = 'no moves possible'
        return(change_dict)

    try_ix = 0
    for friend in original_graph.neighbors(node):
        experimental_graph = original_graph.copy()
        experimental_graph.remove_edge(node, friend)
        new_change = eval_function(experimental_graph, node)

        the_try = {"eval":new_change,"remove_edge":{"from":node,"to":friend}}
        change_dict['attempt']['tries'][try_ix] = the_try
        try_ix += 1

        if increase_metric and (change is None or new_change > change):
            change = new_change
            friend_to_remove_edge_from = friend
        elif not increase_metric and (change is None or new_change < change):
            change = new_change
            friend_to_remove_edge_from = friend

    if friend_to_remove_edge_from is not None:
        original_graph.remove_edge(node, friend_to_remove_edge_from)
        if verbose:
            print('nodes to disconnect: ', 'self - ', str(node), ', friend - ',
                   str(friend_to_remove_edge_from))

        picked_try = {"eval":change,
                      "remove_edge":{"from":node,"to":friend_to_remove_edge_from}}
        change_dict['attempt']['picked'] = picked_try

        change_dict['new_metric'] = change
        change_dict['performed'] = True
        return(change_dict)
    else:
        change_dict['attempt']['picked'] = 'no locally optimum change available'
        return(change_dict)


def self_add_edge_foaf(node, original_graph, increase_metric, eval_function, greedy, verbose):
    original_metric = eval_function(original_graph, node)
    change_dict = {'new_metric':original_metric,
                   'performed':False,
                   'attempt':{'picked':{},'tries':{}}}
    if greedy:
        change = original_metric
    else:
        change = None
    foaf_to_add_edge_to = None

    # checking if the node is already optimal
    if original_metric == 0 and not increase_metric:
        change_dict['attempt']['picked'] = \
            'no optimal change to make as evaluation is already optimal'
        return(change_dict)

    if not any(original_graph.neighbors(node)):
        change_dict['attempt']['picked'] = 'no moves possible'
        return(change_dict)

    self_node = set()
    self_node.add(node)
    set_of_friends = set(original_graph.neighbors(node))
    set_of_foafs = set()

    for friend in set_of_friends:
        for foaf in original_graph.neighbors(friend):
            set_of_foafs.add(foaf)

    set_of_foafs = (set_of_foafs - set_of_friends) - self_node

    try_ix = 0
    for foaf in set_of_foafs:
        experimental_graph = original_graph.copy()
        experimental_graph.add_edge(node, foaf)
        new_change = eval_function(experimental_graph, node)

        the_try = {"eval":new_change,"add_edge":{"from":node,"to":foaf}}
        change_dict['attempt']['tries'][try_ix] = the_try
        try_ix += 1

        if increase_metric and (change is None or new_change > change):
            change = new_change
            foaf_to_add_edge_to = foaf
        elif not increase_metric and (change is None or new_change < change):
            change = new_change
            foaf_to_add_edge_to = foaf

    if foaf_to_add_edge_to is not None:
        original_graph.add_edge(node, foaf_to_add_edge_to)
        if verbose:
            print('nodes to connect:',
                  'self - ',
                  str(node),
                  ', foaf - ',
                  str(foaf_to_add_edge_to))

        picked_try = {"eval":change,
                      "add_edge":{"from":node,"to":foaf_to_add_edge_to}}
        change_dict['attempt']['picked'] = picked_try

        change_dict['new_metric'] = change
        change_dict['performed'] = True
        return(change_dict)
    else:
        change_dict['attempt']['picked'] = 'no locally optimum change available'
        return(change_dict)


def self_add_edge_foafoaf(node, original_graph, increase_metric, eval_function, greedy, verbose):
    original_metric = eval_function(original_graph, node)
    change_dict = {'new_metric':original_metric,
               'performed':False,
               'attempt':{'picked':{},'tries':{}}}
    if greedy:
        change = original_metric
    else:
        change = None
    foafoaf_to_add_self_to = None

    # checking if the node is already optimal
    if original_metric == 0 and not increase_metric:
        change_dict['attempt']['picked'] = \
            'no optimal change to make as evaluation is already optimal'
        return(change_dict)

    if not any(original_graph.neighbors(node)):
        change_dict['attempt']['picked'] = 'no moves possible'
        return(change_dict)


    self_node = set()
    self_node.add(node)

    all_friends = set(original_graph.neighbors(node))
    all_foafs = set()
    for friend in all_friends:
        for foaf in original_graph.neighbors(friend):
            all_foafs.add(foaf)

    all_foafs = (all_foafs - all_friends) - self_node
    all_foafoafs = set()
    for foaf in all_foafs:
        for foafoaf in original_graph.neighbors(foaf):
            all_foafoafs.add(foafoaf)

    all_foafoafs = ((all_foafoafs - all_foafs) - all_friends) - self_node

    try_ix = 0
    for foafoaf in all_foafoafs:
        if node != foafoaf and \
           foafoaf not in all_friends and \
           original_graph.get_edge_data(node, foafoaf) is None:
            experimental_graph = original_graph.copy()
            experimental_graph.add_edge(node, foafoaf)
            new_change = eval_function(experimental_graph, node)

            the_try = {"eval":new_change,"add_edge":{"from":node,"to":foafoaf}}
            change_dict['attempt']['tries'][try_ix] = the_try
            try_ix += 1

            if increase_metric and (change is None or new_change > change):
                change = new_change
                foafoaf_to_add_self_to = foafoaf
            elif not increase_metric and (change is None or new_change < change):
                change = new_change
                foafoaf_to_add_self_to = foafoaf

    if (node is not None and foafoaf_to_add_self_to is not None):
        original_graph.add_edge(node, foafoaf_to_add_self_to)
        if verbose:
            print('nodes to connect:',
                  'selfnode - ',
                  str(node), ',foafoaf - ',
                  str(foafoaf_to_add_self_to))

        picked_try = {"eval":
                      change,
                      "add_edge":{"from":node,"to":foafoaf_to_add_self_to}}
        change_dict['attempt']['picked'] = picked_try

        change_dict['new_metric'] = change
        change_dict['performed'] = True
        return(change_dict)
    else:
        change_dict['attempt']['picked'] = 'no locally optimum change available'
        return(change_dict)


def friend_add_edge_friend(node, original_graph, increase_metric, eval_function, greedy, verbose):
    original_metric = eval_function(original_graph, node)
    change_dict = {'new_metric':original_metric,
               'performed':False,
               'attempt':{'picked':{},'tries':{}}}
    if greedy:
        change = original_metric
    else:
        change = None
    friends_to_connect = [None, None]

    # checking if the node is already optimal
    if original_metric == 0 and not increase_metric:
        change_dict['attempt']['picked'] = \
            'no optimal change to make as evaluation is already optimal'
        return(change_dict)

    if not any(original_graph.neighbors(node)):
        change_dict['attempt']['picked'] = 'no moves possible'
        return(change_dict)

    try_ix = 0
    for friend1 in original_graph.neighbors(node):
        for friend2 in original_graph.neighbors(node):
            if friend1 != friend2 and original_graph.get_edge_data(friend1, friend2) is None:
                experimental_graph = original_graph.copy()
                experimental_graph.add_edge(friend1, friend2)
                new_change = eval_function(experimental_graph, node)

                the_try = {"eval":new_change,"add_edge":{"from":friend1,"to":friend2}}
                change_dict['attempt']['tries'][try_ix] = the_try
                try_ix += 1

                if increase_metric and (change is None or new_change > change):
                    change = new_change
                    friends_to_connect[0] = friend1
                    friends_to_connect[1] = friend2
                elif not increase_metric and (change is None or new_change < change):
                    change = new_change
                    friends_to_connect[0] = friend1
                    friends_to_connect[1] = friend2

    if friends_to_connect[0] is not None and friends_to_connect[1] is not None:
        original_graph.add_edge(friends_to_connect[0], friends_to_connect[1])
        if verbose:
            print('nodes to connect:',
                  'friend1 - ',
                  str(friends_to_connect[0]),
                  ', friend2 - ',
                  str(friends_to_connect[1]))

        picked_try = {"eval":change,
                      "add_edge":{"from":friends_to_connect[0],
                      "to":friends_to_connect[1]}}
        change_dict['attempt']['picked'] = picked_try

        change_dict['new_metric'] = change
        change_dict['performed'] = True
        return(change_dict)
    else:
        change_dict['attempt']['picked'] = 'no locally optimum change available'
        return(change_dict)


def friend_add_edge_foaf(node, original_graph, increase_metric, eval_function, greedy, verbose):
    original_metric = eval_function(original_graph, node)
    change_dict = {'new_metric':original_metric,
               'performed':False,
               'attempt':{'picked':{},'tries':{}}}
    if greedy:
        change = original_metric
    else:
        change = None
    friend_to_connect = None
    foaf_to_connect = None

    # checking if the node is already optimal
    if original_metric == 0 and not increase_metric:
        change_dict['attempt']['picked'] = \
            'no optimal change to make as evaluation is already optimal'
        return(change_dict)

    if not any(original_graph.neighbors(node)):
        change_dict['attempt']['picked'] = 'no moves possible'
        return(change_dict)

    # finding all viable foafs for this computation
    all_friends = set(original_graph.neighbors(node))
    all_foafs = set()
    for friend in all_friends:
        for foaf in original_graph.neighbors(friend):
            all_foafs.add(foaf)

    all_foafs = (all_foafs - all_friends)

    try_ix = 0
    for friend in all_friends:
        all_friends_of_curr_friend = set(original_graph.neighbors(friend))
        all_viable_foafs = all_foafs - all_friends_of_curr_friend
        for foaf in all_viable_foafs:
            if foaf != node and \
               friend != foaf and \
               original_graph.get_edge_data(friend, foaf) is None:
                experimental_graph = original_graph.copy()
                experimental_graph.add_edge(friend, foaf)
                new_change = eval_function(experimental_graph, node)

                the_try = {"eval":new_change,"add_edge":{"from":friend,"to":foaf}}
                change_dict['attempt']['tries'][try_ix] = the_try
                try_ix += 1

                if increase_metric and (change is None or new_change > change):
                    change = new_change
                    friend_to_connect = friend
                    foaf_to_connect = foaf
                elif not increase_metric and (change is None or new_change < change):
                    change = new_change
                    friend_to_connect = friend
                    foaf_to_connect = foaf

    if friend_to_connect is not None and foaf_to_connect is not None:
        original_graph.add_edge(friend_to_connect, foaf_to_connect)
        if verbose:
            print('nodes to connect:',
                  'friend - ',
                  str(friend_to_connect),
                  ', foaf - ',
                  str(foaf_to_connect))

        picked_try = {"eval":
                      change,
                      "add_edge":{"from":friend_to_connect,"to":foaf_to_connect}}
        change_dict['attempt']['picked'] = picked_try

        change_dict['new_metric'] = change
        change_dict['performed'] = True
        return(change_dict)
    else:
        change_dict['attempt']['picked'] = 'no locally optimum change available'
        return(change_dict)


def friend_add_edge_foafoaf(node, original_graph, increase_metric, eval_function, greedy, verbose):
    original_metric = eval_function(original_graph, node)
    change_dict = {'new_metric':original_metric,
               'performed':False,
               'attempt':{'picked':{},'tries':{}}}
    if greedy:
        change = original_metric
    else:
        change = None
    friend_to_connect = None
    foafoaf_to_connect = None

    # checking if the node is already optimal
    if original_metric == 0 and not increase_metric:
        change_dict['attempt']['picked'] = \
            'no optimal change to make as evaluation is already optimal'
        return(change_dict)

    if not any(original_graph.neighbors(node)):
        change_dict['attempt']['picked'] = 'no moves possible'
        return(change_dict)


    self_node = set()
    self_node.add(node)

    all_friends = set(original_graph.neighbors(node))
    all_foafs = set()
    for friend in all_friends:
        for foaf in original_graph.neighbors(friend):
            all_foafs.add(foaf)

    all_foafs = (all_foafs - all_friends) - self_node
    all_foafoafs = set()
    for foaf in all_foafs:
        for foafoaf in original_graph.neighbors(foaf):
            all_foafoafs.add(foafoaf)

    all_foafoafs = ((all_foafoafs - all_foafs) - all_friends) - self_node

    try_ix = 0
    for friend in all_friends:
        for foafoaf in all_foafoafs:
            if friend != foafoaf and original_graph.get_edge_data(friend, foafoaf) is None:
                experimental_graph = original_graph.copy()
                experimental_graph.add_edge(friend, foafoaf)
                new_change = eval_function(experimental_graph, node)

                the_try = {"eval":new_change,"add_edge":{"from":friend,"to":foafoaf}}
                change_dict['attempt']['tries'][try_ix] = the_try
                try_ix += 1

                if increase_metric and (change is None or new_change > change):
                    change = new_change
                    friend_to_connect = friend
                    foafoaf_to_connect = foafoaf
                elif not increase_metric and (change is None or new_change < change):
                    change = new_change
                    friend_to_connect = friend
                    foafoaf_to_connect = foafoaf

    if (friend_to_connect is not None and foafoaf_to_connect is not None):
        original_graph.add_edge(friend_to_connect, foafoaf_to_connect)
        if verbose:
            print('nodes to connect:',
                  'friend - ',
                  str(friend_to_connect),
                  ', foafoaf - ',
                  str(foafoaf_to_connect))

        picked_try = {"eval":change,
                      "add_edge":{"from":friend_to_connect,"to":foafoaf_to_connect}}
        change_dict['attempt']['picked'] = picked_try

        change_dict['new_metric'] = change
        change_dict['performed'] = True
        return(change_dict)
    else:
        change_dict['attempt']['picked'] = 'no locally optimum change available'
        return(change_dict)

def friend_remove_edge_friend(node, original_graph, increase_metric, eval_function, greedy, verbose):
    original_metric = eval_function(original_graph, node)
    change_dict = {'new_metric':original_metric,
               'performed':False,
               'attempt':{'picked':{},'tries':{}}}
    if greedy:
        change = original_metric
    else:
        change = None
    friends_to_disconnect = [None, None]

    # checking if the node is already optimal
    if original_metric == 0 and not increase_metric:
        change_dict['attempt']['picked'] = \
            'no optimal change to make as evaluation is already optimal'
        return(change_dict)

    if not any(original_graph.neighbors(node)) and len(list(original_graph.neighbors(node))) < 2:
        change_dict['attempt']['picked'] = 'no moves possible'
        return(change_dict)

    try_ix = 0
    for friend1 in original_graph.neighbors(node):
        for friend2 in original_graph.neighbors(node):
            if friend1 != friend2 and \
                original_graph.get_edge_data(friend1, friend2) is not None:
                experimental_graph = original_graph.copy()
                experimental_graph.remove_edge(friend1, friend2)
                new_change = eval_function(experimental_graph, node)

                the_try = {"eval":new_change,"remove_edge":{"from":friend1,"to":friend2}}
                change_dict['attempt']['tries'][try_ix] = the_try
                try_ix += 1

                if increase_metric and (change is None or new_change > change):
                    change = new_change
                    friends_to_disconnect[0] = friend1
                    friends_to_disconnect[1] = friend2
                elif not increase_metric and (change is None or new_change < change):
                    change = new_change
                    friends_to_disconnect[0] = friend1
                    friends_to_disconnect[1] = friend2

    if friends_to_disconnect[0] is not None and friends_to_disconnect[1] is not None:
        original_graph.remove_edge(friends_to_disconnect[0], friends_to_disconnect[1])
        if verbose:
            print('nodes to disconnect:',
                  'friend1 - ',
                  str(friends_to_disconnect[0]),
                  ', friend2 - ',
                  str(friends_to_disconnect[1]))

        picked_try = {"eval":change,
                      "remove_edge":{"from":friends_to_disconnect[0],
                                     "to":friends_to_disconnect[1]}}
        change_dict['attempt']['picked'] = picked_try

        change_dict['new_metric'] = change
        change_dict['performed'] = True
        return(change_dict)
    else:
        change_dict['attempt']['picked'] = 'no locally optimum change available'
        return(change_dict)




def foaf_add_edge_foaf(node, original_graph, increase_metric, eval_function, greedy, verbose):
    original_metric = eval_function(original_graph, node)
    change_dict = {'new_metric':original_metric,
               'performed':False,
               'attempt':{'picked':{},'tries':{}}}
    if greedy:
        change = original_metric
    else:
        change = None
    foafs_to_connect = [None, None]

    # checking if the node is already optimal
    if original_metric == 0 and not increase_metric:
        change_dict['attempt']['picked'] = \
            'no optimal change to make as evaluation is already optimal'
        return(change_dict)

    if not any(original_graph.neighbors(node)):
        change_dict['attempt']['picked'] = 'no moves possible'
        return(change_dict)

    friend_set = set(original_graph.neighbors(node))
    foaf_set = set()

    # Finding foafs which are not friends or the node itself
    for friend in original_graph.neighbors(node):
        for foaf in original_graph.neighbors(friend):
            if foaf not in friend_set and foaf != node:
                foaf_set.add(foaf)

    try_ix = 0
    for viable_foaf1 in foaf_set:
        for viable_foaf2 in foaf_set:
            if viable_foaf1 != viable_foaf2 and \
                original_graph.get_edge_data(viable_foaf1, viable_foaf2) is None:
                experimental_graph = original_graph.copy()
                experimental_graph.add_edge(viable_foaf1, viable_foaf2)
                new_change = eval_function(experimental_graph, node)

                the_try = {"eval":new_change,
                           "add_edge":{"from":viable_foaf1,"to":viable_foaf2}}
                change_dict['attempt']['tries'][try_ix] = the_try
                try_ix += 1

                if increase_metric and (change is None or new_change > change):
                    change = new_change
                    foafs_to_connect[0] = viable_foaf1
                    foafs_to_connect[1] = viable_foaf2
                elif not increase_metric and (change is None or new_change < change):
                    change = new_change
                    foafs_to_connect[0] = viable_foaf1
                    foafs_to_connect[1] = viable_foaf2

    if foafs_to_connect[0] is not None and foafs_to_connect[1] is not None:
        original_graph.add_edge(foafs_to_connect[0], foafs_to_connect[1])
        if verbose:
            print('nodes to connect:', 'foaf1 - ',
                  str(foafs_to_connect[0]), ',foaf2 - ',
                  str(foafs_to_connect[1]))

        picked_try = {"eval":change,
                      "add_edge":{"from":foafs_to_connect[0],"to":foafs_to_connect[1]}}
        change_dict['attempt']['picked'] = picked_try

        change_dict['new_metric'] = change
        change_dict['performed'] = True
        return(change_dict)
    else:
        change_dict['attempt']['picked'] = 'no locally optimum change available'
        return(change_dict)


def foaf_add_edge_foafoaf(node, original_graph, increase_metric, eval_function, greedy, verbose):
    original_metric = eval_function(original_graph, node)
    change_dict = {'new_metric':original_metric,
               'performed':False,
               'attempt':{'picked':{},'tries':{}}}
    if greedy:
        change = original_metric
    else:
        change = None
    foaf_to_connect = None
    foafoaf_to_connect = None

    # checking if the node is already optimal
    if original_metric == 0 and not increase_metric:
        change_dict['attempt']['picked'] = \
            'no optimal change to make as evaluation is already optimal'
        return(change_dict)

    if not any(original_graph.neighbors(node)):
        change_dict['attempt']['picked'] = 'no moves possible'
        return(change_dict)


    self_node = set()
    self_node.add(node)

    all_friends = set(original_graph.neighbors(node))
    all_foafs = set()
    for friend in all_friends:
        for foaf in original_graph.neighbors(friend):
            all_foafs.add(foaf)

    all_foafs = (all_foafs - all_friends) - self_node
    all_foafoafs = set()
    for foaf in all_foafs:
        for foafoaf in original_graph.neighbors(foaf):
            all_foafoafs.add(foafoaf)

    all_foafoafs = ((all_foafoafs - all_foafs) - all_friends) - self_node

    try_ix = 0
    for foaf in all_foafs:
        for foafoaf in all_foafoafs:
            if foaf != foafoaf and \
                foaf not in all_friends and \
                original_graph.get_edge_data(foaf, foafoaf) is None:

                experimental_graph = original_graph.copy()
                experimental_graph.add_edge(foaf, foafoaf)
                new_change = eval_function(experimental_graph, node)

                the_try = {"eval":new_change,"add_edge":{"from":foaf,"to":foafoaf}}
                change_dict['attempt']['tries'][try_ix] = the_try
                try_ix += 1

                if increase_metric and (change is None or new_change > change):
                    change = new_change
                    foaf_to_connect = foaf
                    foafoaf_to_connect = foafoaf
                elif not increase_metric and (change is None or new_change < change):
                    change = new_change
                    foaf_to_connect = foaf
                    foafoaf_to_connect = foafoaf

    if (foaf_to_connect is not None and foafoaf_to_connect is not None):
        original_graph.add_edge(foaf_to_connect, foafoaf_to_connect)
        if verbose:
            print('nodes to connect:', 'foaf - ',
                  str(foaf_to_connect), ',foafoaf - ',
                  str(foafoaf_to_connect))

        picked_try = {"eval":change,
                      "add_edge":{"from":foaf_to_connect,"to":foafoaf_to_connect}}
        change_dict['attempt']['picked'] = picked_try

        change_dict['new_metric'] = change
        change_dict['performed'] = True
        return(change_dict)
    else:
        change_dict['attempt']['picked'] = 'no locally optimum change available'
        return(change_dict)


def foaf_remove_edge_foaf(node, original_graph, increase_metric, eval_function, greedy, verbose):
    original_metric = eval_function(original_graph, node)
    change_dict = {'new_metric':original_metric,
               'performed':False,
               'attempt':{'picked':{},'tries':{}}}
    if greedy:
        change = original_metric
    else:
        change = None
    foafs_to_disconnect = [None, None]

    # checking if the node is already optimal
    if original_metric == 0 and not increase_metric:
        change_dict['attempt']['picked'] = \
            'no optimal change to make as evaluation is already optimal'
        return(change_dict)

    if not any(original_graph.neighbors(node)):
        change_dict['attempt']['picked'] = 'no moves possible'
        return(change_dict)

    friend_set = set(original_graph.neighbors(node))
    foaf_set = set()

    # Finding foafs which are not friends or the node itself
    for friend in original_graph.neighbors(node):
        for foaf in original_graph.neighbors(friend):
            if foaf not in friend_set and foaf != node:
                foaf_set.add(foaf)

    try_ix = 0
    for viable_foaf1 in foaf_set:
        for viable_foaf2 in foaf_set:
            if viable_foaf1 != viable_foaf2 and \
                original_graph.get_edge_data(viable_foaf1, viable_foaf2) is not None:

                experimental_graph = original_graph.copy()
                experimental_graph.remove_edge(viable_foaf1, viable_foaf2)
                new_change = eval_function(experimental_graph, node)

                the_try = {"eval":new_change,
                           "remove_edge":{"from":viable_foaf1,"to":viable_foaf2}}
                change_dict['attempt']['tries'][try_ix] = the_try
                try_ix += 1

                if increase_metric and (change is None or new_change > change):
                    change = new_change
                    foafs_to_disconnect[0] = viable_foaf1
                    foafs_to_disconnect[1] = viable_foaf2
                elif not increase_metric and (change is None or new_change < change):
                    change = new_change
                    foafs_to_disconnect[0] = viable_foaf1
                    foafs_to_disconnect[1] = viable_foaf2

    if foafs_to_disconnect[0] is not None and foafs_to_disconnect[1] is not None:
        original_graph.remove_edge(foafs_to_disconnect[0], foafs_to_disconnect[1])
        if verbose:
            print('nodes to disconnect:', 'foaf1 - ',
                  str(foafs_to_disconnect[0]),
                  ', foaf2 - ',
                  str(foafs_to_disconnect[1]))

        picked_try = {"eval":change,
                      "remove_edge":{"from":foafs_to_disconnect[0],"to":foafs_to_disconnect[1]}}
        change_dict['attempt']['picked'] = picked_try

        change_dict['new_metric'] = change
        change_dict['performed'] = True
        return(change_dict)
    else:
        change_dict['attempt']['picked'] = 'no locally optimum change available'
        return(change_dict)


def friend_remove_edge_foaf(node, original_graph, increase_metric, eval_function, greedy, verbose):
    original_metric = eval_function(original_graph, node)
    change_dict = {'new_metric':original_metric,
               'performed':False,
               'attempt':{'picked':{},'tries':{}}}
    if greedy:
        change = original_metric
    else:
        change = None
    friend_disconnected_from_foaf = None
    foaf_disconnected_from_friend = None

    # checking if the node is already optimal
    if original_metric == 0 and not increase_metric:
        change_dict['attempt']['picked'] = \
            'no optimal change to make as evaluation is already optimal'
        return(change_dict)

    if not any(original_graph.neighbors(node)):
        change_dict['attempt']['picked'] = 'no moves possible'
        return(change_dict)

    friend_set = set(original_graph.neighbors(node))

    try_ix = 0
    for friend in original_graph.neighbors(node):
        for foaf in original_graph.neighbors(friend):
            if foaf != node and \
               foaf not in friend_set and \
               original_graph.get_edge_data(friend, foaf) is not None:

                experimental_graph = original_graph.copy()
                experimental_graph.remove_edge(friend, foaf)
                new_change = eval_function(experimental_graph, node)

                the_try = {"eval":new_change,
                           "remove_edge":{"from":friend,"to":foaf}}
                change_dict['attempt']['tries'][try_ix] = the_try
                try_ix += 1

                if increase_metric and (change is None or new_change > change):
                    change = new_change
                    friend_disconnected_from_foaf = friend
                    foaf_disconnected_from_friend = foaf
                elif not increase_metric and (change is None or new_change < change):
                    change = new_change
                    friend_disconnected_from_foaf = friend
                    foaf_disconnected_from_friend = foaf

    if friend_disconnected_from_foaf is not None and \
       foaf_disconnected_from_friend is not None:

        original_graph.remove_edge(friend_disconnected_from_foaf, foaf_disconnected_from_friend)
        if verbose:
            print('nodes to disconnect:', 'friend - ',
                  str(friend_disconnected_from_foaf),
                  ', foaf - ',
                  str(foaf_disconnected_from_friend))

        picked_try = {"eval":change,
                      "remove_edge":{"from":friend_disconnected_from_foaf,
                                     "to":foaf_disconnected_from_friend}}
        change_dict['attempt']['picked'] = picked_try

        change_dict['new_metric'] = change
        change_dict['performed'] = True
        return(change_dict)
    else:
        change_dict['attempt']['picked'] = 'no locally optimum change available'
        return(change_dict)


def self_add_edge_any(node, original_graph, increase_metric, eval_function, greedy, verbose):
    original_metric = eval_function(original_graph, node)
    change_dict = {'new_metric':original_metric,
               'performed':False,
               'attempt':{'picked':{},'tries':{}}}
    if greedy:
        change = original_metric
    else:
        change = None
    other_node_to_connect_self_to = None

    # checking if the node is already optimal
    if original_metric == 0 and not increase_metric:
        change_dict['attempt']['picked'] = \
            'no optimal change to make as evaluation is already optimal'
        return(change_dict)

    all_nodes_possible = set(original_graph.nodes())
    all_friends = set(original_graph.neighbors(node))
    all_foafs = set()
    for friend in all_friends:
        for foaf in original_graph.neighbors(friend):
            all_foafs.add(foaf)
    # removing all nodes not viable from all nodes possible
    viable_nodes = (all_nodes_possible - all_friends) - all_foafs

    try_ix = 0
    for viable_node in viable_nodes:
        experimental_graph = original_graph.copy()
        experimental_graph.add_edge(node, viable_node)
        new_change = eval_function(experimental_graph, node)

        the_try = {"eval":new_change,"add_edge":{"from":node,"to":viable_node}}
        change_dict['attempt']['tries'][try_ix] = the_try
        try_ix += 1

        if increase_metric and (change is None or new_change > change):
            change = new_change
            other_node_to_connect_self_to = viable_node
        elif not increase_metric and (change is None or new_change < change):
            change = new_change
            other_node_to_connect_self_to = viable_node

    if other_node_to_connect_self_to is not None:
        original_graph.add_edge(node, other_node_to_connect_self_to)
        if verbose:
            print('nodes to connect:', 'self - ',
                  str(node), ', other node -',
                  str(other_node_to_connect_self_to))

        picked_try = {"eval":change,
                      "add_edge":{"from":node,"to":other_node_to_connect_self_to}}
        change_dict['attempt']['picked'] = picked_try

        change_dict['new_metric'] = change
        change_dict['performed'] = True
        return(change_dict)
    else:
        change_dict['attempt']['picked'] = 'no locally optimum change available'
        return(change_dict)


def friend_add_edge_any(node, original_graph, increase_metric, eval_function, greedy, verbose):
    original_metric = eval_function(original_graph, node)
    change_dict = {'new_metric':original_metric,
               'performed':False,
               'attempt':{'picked':{},'tries':{}}}
    if greedy:
        change = original_metric
    else:
        change = None
    friend_selected = None
    other_node_to_connect_friend_to = None

    # checking if the node is already optimal
    if original_metric == 0 and not increase_metric:
        change_dict['attempt']['picked'] = \
            'no optimal change to make as evaluation is already optimal'
        return(change_dict)

    if not any(original_graph.neighbors(node)):
        change_dict['attempt']['picked'] = 'no moves possible'
        return(change_dict)

    try_ix = 0
    for friend in original_graph.neighbors(node):
        nodes_not_viable = set(original_graph.neighbors(friend))
        nodes_not_viable = nodes_not_viable.union(set(original_graph.neighbors(node)))
        for other_node in original_graph.nodes():
            if other_node not in nodes_not_viable and \
                original_graph.get_edge_data(friend, other_node) is None:
                experimental_graph = original_graph.copy()
                experimental_graph.add_edge(friend, other_node)
                new_change = eval_function(experimental_graph, node)

                the_try = {"eval":new_change,
                           "add_edge":{"from":friend,"to":other_node}}
                change_dict['attempt']['tries'][try_ix] = the_try
                try_ix += 1

                if increase_metric and (change is None or new_change > change):
                    change = new_change
                    friend_selected = friend
                    other_node_to_connect_friend_to = other_node
                elif not increase_metric and (change is None or new_change < change):
                    change = new_change
                    friend_selected = friend
                    other_node_to_connect_friend_to = other_node

    if friend_selected is not None and other_node_to_connect_friend_to is not None:
        original_graph.add_edge(friend_selected, other_node_to_connect_friend_to)
        if verbose:
            print('nodes to connect:',
                  'friend -',
                  str(friend_selected),
                  ', other node to connect friend to -',
                  str(other_node_to_connect_friend_to))

        picked_try = {"eval":change,
                      "add_edge":{"from":friend_selected,
                                  "to":other_node_to_connect_friend_to}}
        change_dict['attempt']['picked'] = picked_try

        change_dict['new_metric'] = change
        change_dict['performed'] = True
        return(change_dict)
    else:
        change_dict['attempt']['picked'] = 'no locally optimum change available'
        return(change_dict)

def foaf_add_edge_any(node, original_graph, increase_metric, eval_function, greedy, verbose):
    original_metric = eval_function(original_graph, node)
    change_dict = {'new_metric':original_metric,
               'performed':False,
               'attempt':{'picked':{},'tries':{}}}
    if greedy:
        change = original_metric
    else:
        change = None
    foaf_to_be_connected = None
    other_node_to_connect_foaf_to = None

    # checking if the node is already optimal
    if original_metric == 0 and not increase_metric:
        change_dict['attempt']['picked'] = \
            'no optimal change to make as evaluation is already optimal'
        return(change_dict)

    if not any(original_graph.neighbors(node)):
        change_dict['attempt']['picked'] = 'no moves possible'
        return(change_dict)

    all_nodes_possible = set(original_graph.nodes())
    self_node = set()
    self_node.add(node)
    all_friends = set(original_graph.neighbors(node))
    all_foafs = set()
    for friend in all_friends:
        for foaf in original_graph.neighbors(friend):
            all_foafs.add(foaf)

    all_foafs = (all_foafs - all_friends) - self_node

    viable_nodes = ((all_nodes_possible - all_friends) - all_foafs) - self_node

    try_ix = 0
    for foaf in all_foafs:
        for viable_node in viable_nodes:
            if original_graph.get_edge_data(foaf, viable_node) is None:
                experimental_graph = original_graph.copy()
                experimental_graph.add_edge(foaf, viable_node)
                new_change = eval_function(experimental_graph, node)

                the_try = {"eval":new_change,
                           "add_edge":{"from":foaf,"to":viable_node}}
                change_dict['attempt']['tries'][try_ix] = the_try
                try_ix += 1

                if increase_metric and (change is None or new_change > change):
                    change = new_change
                    foaf_to_be_connected = foaf
                    other_node_to_connect_foaf_to = viable_node
                elif not increase_metric and (change is None or new_change < change):
                    change = new_change
                    foaf_to_be_connected = foaf
                    other_node_to_connect_foaf_to = viable_node

    if foaf_to_be_connected is not None and other_node_to_connect_foaf_to is not None:
        original_graph.add_edge(foaf_to_be_connected, other_node_to_connect_foaf_to)
        if verbose:
            print('nodes to connect:',
                  'foaf - ',
                  str(foaf_to_be_connected),
                  ', other node -',
                  str(other_node_to_connect_foaf_to))

        picked_try = {"eval":change,
                      "add_edge":{"from":foaf_to_be_connected,
                                  "to":other_node_to_connect_foaf_to}}
        change_dict['attempt']['picked'] = picked_try

        change_dict['new_metric'] = change
        change_dict['performed'] = True
        return(change_dict)
    else:
        change_dict['attempt']['picked'] = 'no locally optimum change available'
        return(change_dict)


def foaf_remove_edge_foafoaf(node, original_graph, increase_metric, eval_function, greedy, verbose):
    original_metric = eval_function(original_graph, node)
    change_dict = {'new_metric':original_metric,
               'performed':False,
               'attempt':{'picked':{},'tries':{}}}
    if greedy:
        change = original_metric
    else:
        change = None
    foaf_to_disconnect = None
    foafoaf_to_disconnect = None

    # checking if the node is already optimal
    if original_metric == 0 and not increase_metric:
        change_dict['attempt']['picked'] = \
            'no optimal change to make as evaluation is already optimal'
        return(change_dict)

    if not any(original_graph.neighbors(node)):
        change_dict['attempt']['picked'] = 'no moves possible'
        return(change_dict)

    set_of_friends = set(original_graph.neighbors(node))

    all_nodes_possible = set(original_graph.nodes())
    self_node = set()
    self_node.add(node)
    all_friends = set(original_graph.neighbors(node))
    all_foafs = set()
    for friend in all_friends:
        for foaf in original_graph.neighbors(friend):
            all_foafs.add(foaf)

    all_foafs = (all_foafs - all_friends) - self_node
    all_foafoafs = set()
    for foaf in all_foafs:
        for foafoaf in original_graph.neighbors(foaf):
            all_foafoafs.add(foafoaf)

    all_foafoafs = ((all_foafoafs - all_foafs) - all_friends) - self_node

    try_ix = 0
    for foaf in all_foafs:
        for foafoaf in all_foafoafs:
            if foaf != foafoaf and original_graph.get_edge_data(foaf, foafoaf) is not None:
                experimental_graph = original_graph.copy()
                experimental_graph.remove_edge(foaf, foafoaf)
                new_change = eval_function(experimental_graph, node)

                the_try = {"eval":new_change,"remove_edge":{"from":foaf,"to":foafoaf}}
                change_dict['attempt']['tries'][try_ix] = the_try
                try_ix += 1

                if increase_metric and (change is None or new_change > change):
                    change = new_change
                    foaf_to_disconnect = foaf
                    foafoaf_to_disconnect = foafoaf
                elif not increase_metric and (change is None or new_change < change):
                    change = new_change
                    foaf_to_disconnect = foaf
                    foafoaf_to_disconnect = foafoaf

    if (foaf_to_disconnect is not None and foafoaf_to_disconnect is not None):
        original_graph.remove_edge(foaf_to_disconnect, foafoaf_to_disconnect)
        if verbose:
            print('nodes to disconnect:',
                  'foaf - ',
                  str(foaf_to_disconnect),
                  ', foafoaf - ',
                  str(foafoaf_to_disconnect))

        picked_try = {"eval":change,
                      "remove_edge":{"from":foaf_to_disconnect,
                                     "to":foafoaf_to_disconnect}}
        change_dict['attempt']['picked'] = picked_try

        change_dict['new_metric'] = change
        change_dict['performed'] = True
        return(change_dict)
    else:
        change_dict['attempt']['picked'] = 'no locally optimum change available'
        return(change_dict)

def any_add_edge_any(node, original_graph, increase_metric, eval_function, greedy, verbose):
    original_metric = eval_function(original_graph, node)
    change_dict = {'new_metric':original_metric,
               'performed':False,
               'attempt':{'picked':{},'tries':{}}}
    if greedy:
        change = original_metric
    else:
        change = None
    nodes_to_connect = [None, None]

    # checking if the node is already optimal
    if original_metric == 0 and not increase_metric:
        change_dict['attempt']['picked'] = \
            'no optimal change to make as evaluation is already optimal'
        return(change_dict)

    all_nodes_possible = set(original_graph.nodes())
    self_node = set()
    self_node.add(node)
    all_friends = set(original_graph.neighbors(node))
    all_foafs = set()
    for friend in all_friends:
        for foaf in original_graph.neighbors(friend):
            all_foafs.add(foaf)

    all_foafs = (all_foafs - all_friends) - self_node

    viable_nodes = ((all_nodes_possible - all_friends) - all_foafs) - self_node

    try_ix = 0
    for viable_node1 in viable_nodes:
        for viable_node2 in viable_nodes:
            if viable_node1 != viable_node2 and \
                original_graph.get_edge_data(viable_node1, viable_node2) is None:

                experimental_graph = original_graph.copy()
                experimental_graph.add_edge(viable_node1, viable_node2)
                new_change = eval_function(experimental_graph, node)

                the_try = {"eval":new_change,
                           "add_edge":{"from":viable_node1,"to":viable_node2}}
                change_dict['attempt']['tries'][try_ix] = the_try
                try_ix += 1

                if increase_metric and (change is None or new_change > change):
                    change = new_change
                    nodes_to_connect[0] = viable_node1
                    nodes_to_connect[1] = viable_node2
                elif not increase_metric and (change is None or new_change < change):
                    change = new_change
                    nodes_to_connect[0] = viable_node1
                    nodes_to_connect[1] = viable_node2

    if (nodes_to_connect[0] is not None and nodes_to_connect[1] is not None):
        original_graph.add_edge(nodes_to_connect[0], nodes_to_connect[1])
        if verbose:
            print('nodes to connect:',
                  'node1 - ',
                  str(nodes_to_connect[0]),
                  ', node2 - ',
                  str(nodes_to_connect[1]))

        picked_try = {"eval":change,
                      "add_edge":{"from":nodes_to_connect[0],"to":nodes_to_connect[1]}}
        change_dict['attempt']['picked'] = picked_try

        change_dict['new_metric'] = change
        change_dict['performed'] = True
        return(change_dict)
    else:
        change_dict['attempt']['picked'] = 'no locally optimum change available'
        return(change_dict)

def any_remove_edge_any(node, original_graph, increase_metric, eval_function, greedy, verbose):
    original_metric = eval_function(original_graph, node)
    change_dict = {'new_metric':original_metric,
               'performed':False,
               'attempt':{'picked':{},'tries':{}}}
    if greedy:
        change = original_metric
    else:
        change = None
    nodes_to_disconnect = [None, None]

    # checking if the node is already optimal
    if original_metric == 0 and not increase_metric:
        change_dict['attempt']['picked'] = \
            'no optimal change to make as evaluation is already optimal'
        return(change_dict)

    all_nodes_possible = set(original_graph.nodes())
    self_node = set()
    self_node.add(node)
    all_friends = set(original_graph.neighbors(node))
    all_foafs = set()
    for friend in all_friends:
        for foaf in original_graph.neighbors(friend):
            all_foafs.add(foaf)

    all_foafs = (all_foafs - all_friends) - self_node

    viable_nodes = ((all_nodes_possible - all_friends) - all_foafs) - self_node

    try_ix = 0
    for viable_node1 in viable_nodes:
        for viable_node2 in viable_nodes:
            if viable_node1 != viable_node2 and \
                original_graph.get_edge_data(viable_node1, viable_node2) is not None:

                experimental_graph = original_graph.copy()
                experimental_graph.remove_edge(viable_node1, viable_node2)
                new_change = eval_function(experimental_graph, node)

                the_try = {"eval":new_change,
                           "remove_edge":{"from":viable_node1,"to":viable_node2}}
                change_dict['attempt']['tries'][try_ix] = the_try
                try_ix += 1

                if increase_metric and (change is None or new_change > change):
                    change = new_change
                    nodes_to_disconnect[0] = viable_node1
                    nodes_to_disconnect[1] = viable_node2
                elif not increase_metric and (change is None or new_change < change):
                    change = new_change
                    nodes_to_disconnect[0] = viable_node1
                    nodes_to_disconnect[1] = viable_node2

    if nodes_to_disconnect[0] is not None and nodes_to_disconnect[1] is not None:
        original_graph.remove_edge(nodes_to_disconnect[0], nodes_to_disconnect[1])
        if verbose:
            print('nodes to disconnect:',
                  'node1 - ',
                  str(nodes_to_disconnect[0]),
                  ', node2 - ',
                  str(nodes_to_disconnect[1]))

        picked_try = {"eval":change,
                      "remove_edge":{"from":nodes_to_disconnect[0],
                                     "to":nodes_to_disconnect[1]}}
        change_dict['attempt']['picked'] = picked_try

        change_dict['new_metric'] = change
        change_dict['performed'] = True
        return(change_dict)
    else:
        change_dict['attempt']['picked'] = 'no locally optimum change available'
        return(change_dict)


# # NODES

# May not need this?
# def remove_node_self(node, graph):

def add_node_to_self(node, original_graph, increase_metric, eval_function, greedy, verbose):
    original_metric = eval_function(original_graph, node)
    change_dict = {'new_metric':original_metric,
               'performed':False,
               'attempt':{'picked':{},'tries':{}}}
    if greedy:
        change = original_metric
    else:
        change = None
    new_node_introduced_and_add_edge_to = None

    # checking if the node is already optimal
    if original_metric == 0 and not increase_metric:
        change_dict['attempt']['picked'] = \
            'no optimal change to make as evaluation is already optimal'
        return(change_dict)

    all_nodes = list(original_graph.nodes())
    new_node = all_nodes[-1] + 1

    experimental_graph = original_graph.copy()
    experimental_graph.add_node(new_node)
    experimental_graph.add_edge(node, new_node)
    new_change = eval_function(experimental_graph, node)

    the_try = {"eval":new_change,
               "add_node":new_node,
               "add_edge":{"from":node,"to":new_node}}
    change_dict['attempt']['tries'][0] = the_try

    if increase_metric and (change is None or new_change > change):
        change = new_change
        new_node_introduced_and_add_edge_to = new_node
    elif not increase_metric and (change is None or new_change < change):
        change = new_change
        new_node_introduced_and_add_edge_to = new_node

#     original_graph.add_node(new_node)
    if new_node_introduced_and_add_edge_to is not None:
        original_graph.add_node(new_node_introduced_and_add_edge_to)
        original_graph.add_edge(node, new_node_introduced_and_add_edge_to)
        if verbose:
            print('node created:', str(new_node_introduced_and_add_edge_to))
            print('nodes to connect:',
                  'self - ',
                  str(node),
                  ', node just created - ',
                  str(new_node_introduced_and_add_edge_to))

        picked_try = {"eval":change,
                      "add_node":new_node_introduced_and_add_edge_to,
                      "add_edge":{"from":node,"to":new_node_introduced_and_add_edge_to}}
        change_dict['attempt']['picked'] = picked_try

        change_dict['new_metric'] = change
        change_dict['performed'] = True
        return(change_dict)
    else:
        change_dict['attempt']['picked'] = 'no locally optimum change available'
        return(change_dict)

def remove_node_to_self(node, original_graph, increase_metric, eval_function, greedy, verbose):
    original_metric = eval_function(original_graph, node)
    change_dict = {'new_metric':original_metric,
               'performed':False,
               'attempt':{'picked':{},'tries':{}}}
    if greedy:
        change = original_metric
    else:
        change = None
    node_to_be_removed = None

    # checking if the node is already optimal
    if original_metric == 0 and not increase_metric:
        change_dict['attempt']['picked'] = \
            'no optimal change to make as evaluation is already optimal'
        return(change_dict)

    if not any(original_graph.neighbors(node)):
        change_dict['attempt']['picked'] = 'no moves possible'
        return(change_dict)

    try_ix = 0
    for friend in original_graph.neighbors(node):
        experimental_graph = original_graph.copy()
        experimental_graph.remove_node(friend)
        new_change = eval_function(experimental_graph, node)

        the_try = {"eval":new_change,"remove_node":friend}
        change_dict['attempt']['tries'][try_ix] = the_try
        try_ix += 1

        if increase_metric and (change is None or new_change > change):
            change = new_change
            node_to_be_removed = friend
        elif not increase_metric and (change is None or new_change < change):
            change = new_change
            node_to_be_removed = friend

    if node_to_be_removed is not None:
        original_graph.remove_node(node_to_be_removed)
        if verbose:
            print('node to remove:', 'friend - ', str(node_to_be_removed))

        picked_try = {"eval":change,"remove_node":node_to_be_removed}
        change_dict['attempt']['picked'] = picked_try

        change_dict['new_metric'] = change
        change_dict['performed'] = True
        return(change_dict)
    else:
        change_dict['attempt']['picked'] = 'no locally optimum change available'
        return(change_dict)


def add_node_to_friend(node, original_graph, increase_metric, eval_function, greedy, verbose):
    original_metric = eval_function(original_graph, node)
    change_dict = {'new_metric':original_metric,
               'performed':False,
               'attempt':{'picked':{},'tries':{}}}
    if greedy:
        change = original_metric
    else:
        change = None
    new_node_introduced_and_add_edge_to = None
    friend_to_add_connected_node_to = None

    # checking if the node is already optimal
    if original_metric == 0 and not increase_metric:
        change_dict['attempt']['picked'] = \
            'no optimal change to make as evaluation is already optimal'
        return(change_dict)

    if not any(original_graph.neighbors(node)):
        change_dict['attempt']['picked'] = 'no moves possible'
        return(change_dict)

    all_nodes = list(original_graph.nodes())
    new_node = all_nodes[-1] + 1

    try_ix = 0
    for friend in original_graph.neighbors(node):
        experimental_graph = original_graph.copy()
        experimental_graph.add_node(new_node)
        experimental_graph.add_edge(friend, new_node)
        new_change = eval_function(experimental_graph, node)

        the_try = {"eval":new_change,
                   "add_node":new_node,
                   "add_edge":{"from":friend,"to":new_node}}
        change_dict['attempt']['tries'][try_ix] = the_try
        try_ix += 1

        if increase_metric and (change is None or new_change > change):
            change = new_change
            new_node_introduced_and_add_edge_to = new_node
            friend_to_add_connected_node_to = friend
        elif not increase_metric and (change is None or new_change < change):
            change = new_change
            new_node_introduced_and_add_edge_to = new_node
            friend_to_add_connected_node_to = friend

#     original_graph.add_node(new_node)
    if new_node_introduced_and_add_edge_to is not None and \
       friend_to_add_connected_node_to is not None:

        original_graph.add_node(new_node_introduced_and_add_edge_to)
        original_graph.add_edge(friend_to_add_connected_node_to,
                                new_node_introduced_and_add_edge_to)
        if verbose:
            print('node created:', str(new_node_introduced_and_add_edge_to))
            print('nodes to connect:', 'friend - ',
                  str(friend_to_add_connected_node_to),
                  ', node just created - ',
                  str(new_node_introduced_and_add_edge_to))

        picked_try = {"eval":change,
                      "add_node":new_node_introduced_and_add_edge_to,
                      "add_edge":{"from":friend_to_add_connected_node_to,
                                  "to":new_node_introduced_and_add_edge_to}}
        change_dict['attempt']['picked'] = picked_try

        change_dict['new_metric'] = change
        change_dict['performed'] = True
        return(change_dict)
    else:
        change_dict['attempt']['picked'] = 'no locally optimum change available'
        return(change_dict)

def remove_node_to_friend(node, original_graph, increase_metric, eval_function, greedy, verbose):
    original_metric = eval_function(original_graph, node)
    change_dict = {'new_metric':original_metric,
               'performed':False,
               'attempt':{'picked':{},'tries':{}}}
    if greedy:
        change = original_metric
    else:
        change = None
#     friend_who_will_lose_friend = None
    node_to_be_removed = None

    # checking if the node is already optimal
    if original_metric == 0 and not increase_metric:
        change_dict['attempt']['picked'] = \
            'no optimal change to make as evaluation is already optimal'
        return(change_dict)

    if not any(original_graph.neighbors(node)):
        change_dict['attempt']['picked'] = 'no moves possible'
        return(change_dict)

    self_node = set()
    self_node.add(node)
    all_friends = set(original_graph.neighbors(node))
    all_foafs = set()
    for friend in all_friends:
        foafs = set(original_graph.neighbors(friend))
        all_foafs = all_foafs.union(foafs)

    all_foafs = (all_foafs - all_friends) - self_node

    try_ix = 0
    for foaf in all_foafs:
        experimental_graph = original_graph.copy()
        experimental_graph.remove_node(foaf)
        new_change = eval_function(experimental_graph, node)

        the_try = {"eval":new_change,"remove_node":foaf}
        change_dict['attempt']['tries'][try_ix] = the_try
        try_ix += 1

        if increase_metric and (change is None or new_change > change):
            change = new_change
#             friend_who_will_lose_friend = friend
            node_to_be_removed = foaf
        elif not increase_metric and (change is None or new_change < change):
            change = new_change
#             friend_who_will_lose_friend = friend
            node_to_be_removed = foaf

#     if friend_who_will_lose_friend is not None and node_to_be_removed is not None:
    if node_to_be_removed is not None:
        original_graph.remove_node(node_to_be_removed)
        if verbose:
            print('foaf to remove:', str(node_to_be_removed))

        picked_try = {"eval":change,"remove_node":node_to_be_removed}
        change_dict['attempt']['picked'] = picked_try

        change_dict['new_metric'] = change
        change_dict['performed'] = True
        return(change_dict)
    else:
        change_dict['attempt']['picked'] = 'no locally optimum change available'
        return(change_dict)

def add_node_to_foaf(node, original_graph, increase_metric, eval_function, greedy, verbose):
    original_metric = eval_function(original_graph, node)
    change_dict = {'new_metric':original_metric,
               'performed':False,
               'attempt':{'picked':{},'tries':{}}}
    if greedy:
        change = original_metric
    else:
        change = None
    new_node_introduced_and_add_edge_to = None
    foaf_to_add_connected_node_to = None

    # checking if the node is already optimal
    if original_metric == 0 and not increase_metric:
        change_dict['attempt']['picked'] = \
            'no optimal change to make as evaluation is already optimal'
        return(change_dict)

    if not any(original_graph.neighbors(node)):
        change_dict['attempt']['picked'] = 'no moves possible'
        return(change_dict)

    all_nodes = list(original_graph.nodes())
    new_node = all_nodes[-1] + 1

    all_friends = set(original_graph.neighbors(node))
    foaf_set = set()

    for friend in all_friends:
        foafs = set(original_graph.neighbors(friend))
        foaf_set = foaf_set.union(foafs)

    try_ix = 0
    for foaf in foaf_set:
        if foaf != node and foaf not in all_friends:
            experimental_graph = original_graph.copy()
            experimental_graph.add_node(new_node)
            experimental_graph.add_edge(foaf, new_node)
            new_change = eval_function(experimental_graph, node)

            the_try = {"eval":new_change,
                       "add_node":new_node,
                       "add_edge":{"from":foaf,"to":new_node}}
            change_dict['attempt']['tries'][try_ix] = the_try
            try_ix += 1

            if increase_metric and (change is None or new_change > change):
                change = new_change
                new_node_introduced_and_add_edge_to = new_node
                foaf_to_add_connected_node_to = foaf
            elif not increase_metric and (change is None or new_change < change):
                change = new_change
                new_node_introduced_and_add_edge_to = new_node
                foaf_to_add_connected_node_to = foaf

#     original_graph.add_node(new_node)
    if new_node_introduced_and_add_edge_to is not None \
        and foaf_to_add_connected_node_to is not None:

        original_graph.add_node(new_node_introduced_and_add_edge_to)
        original_graph.add_edge(foaf_to_add_connected_node_to, \
                                new_node_introduced_and_add_edge_to)
        if verbose:
            print('node created:', str(new_node_introduced_and_add_edge_to))
            print('nodes to connect:', 'foaf - ',
                  str(foaf_to_add_connected_node_to),
                  ', node just created - ',
                  str(new_node_introduced_and_add_edge_to))

        picked_try = {"eval":change,
                      "add_node":new_node_introduced_and_add_edge_to,
                      "add_edge":{"from":foaf_to_add_connected_node_to,
                                  "to":new_node_introduced_and_add_edge_to}}
        change_dict['attempt']['picked'] = picked_try

        change_dict['new_metric'] = change
        change_dict['performed'] = True
        return(change_dict)
    else:
        change_dict['attempt']['picked'] = 'no locally optimum change available'
        return(change_dict)

def remove_node_to_foaf(node, original_graph, increase_metric, eval_function, greedy, verbose):
    original_metric = eval_function(original_graph, node)
    change_dict = {'new_metric':original_metric,
               'performed':False,
               'attempt':{'picked':{},'tries':{}}}
    if greedy:
        change = original_metric
    else:
        change = None
#     foaf_who_will_lose_friend = None
    node_to_be_removed = None

    # checking if the node is already optimal
    if original_metric == 0 and not increase_metric:
        change_dict['attempt']['picked'] = \
            'no optimal change to make as evaluation is already optimal'
        return(change_dict)

    if not any(original_graph.neighbors(node)):
        change_dict['attempt']['picked'] = 'no moves possible'
        return(change_dict)

    try_ix = 0
    self_node = set()
    self_node.add(node)
    all_friends = set(original_graph.neighbors(node))
    all_foafs = set()
    for friend in all_friends:
        foafs = set(original_graph.neighbors(friend))
        all_foafs = all_foafs.union(foafs)

    all_foafs = (all_foafs - all_friends) - self_node

    all_foafoafs = set()
    for foaf in all_foafs:
        foafoaf = set(original_graph.neighbors(foaf))
        all_foafoafs = all_foafoafs.union(foafoaf)

    all_foafoafs = ((all_foafoafs - all_foafs) - all_friends) - self_node

    try_ix = 0
    for foafoaf in all_foafoafs:
        experimental_graph = original_graph.copy()
        experimental_graph.remove_node(foafoaf)
        new_change = eval_function(experimental_graph, node)

        the_try = {"eval":new_change,"remove_node":foafoaf}
        change_dict['attempt']['tries'][try_ix] = the_try
        try_ix += 1

        if increase_metric and (change is None or new_change > change):
            change = new_change
#             foaf_who_will_lose_friend = foaf
            node_to_be_removed = foafoaf
        elif not increase_metric and (change is None or new_change < change):
            change = new_change
#             foaf_who_will_lose_friend = foaf
            node_to_be_removed = foafoaf

#     if foaf_who_will_lose_friend is not None and node_to_be_removed is not None:
    if node_to_be_removed is not None:
        original_graph.remove_node(node_to_be_removed)
        if verbose:
            print('foafoaf to remove:', str(node_to_be_removed))

        picked_try = {"eval":change,"remove_node":node_to_be_removed}
        change_dict['attempt']['picked'] = picked_try

        change_dict['new_metric'] = change
        change_dict['performed'] = True
        return(change_dict)
    else:
        change_dict['attempt']['picked'] = 'no locally optimum change available'
        return(change_dict)


def add_node_anywhere(node, original_graph, increase_metric, eval_function, greedy, verbose):
    original_metric = eval_function(original_graph, node)
    change_dict = {'new_metric':original_metric,
               'performed':False,
               'attempt':{'picked':{},'tries':{}}}
    if greedy:
        change = original_metric
    else:
        change = None
    new_node_introduced_and_add_edge_to = None
    existing_node_connected_to_new_node = None

    # checking if the node is already optimal
    if original_metric == 0 and not increase_metric:
        change_dict['attempt']['picked'] = \
            'no optimal change to make as evaluation is already optimal'
        return(change_dict)

    all_nodes = list(original_graph.nodes())
    new_node = all_nodes[-1] + 1

    all_nodes_possible = set(original_graph.nodes())
    self_node = set()
    self_node.add(node)
    all_friends = set(original_graph.neighbors(node))
    all_foafs = set()
    for friend in all_friends:
        for foaf in original_graph.neighbors(friend):
            all_foafs.add(foaf)

    all_foafs = (all_foafs - all_friends) - self_node

    viable_nodes = ((all_nodes_possible - all_friends) - all_foafs) - self_node

    try_ix = 0
    for possible_tangent_node in viable_nodes:
        experimental_graph = original_graph.copy()
        experimental_graph.add_node(new_node)
        experimental_graph.add_edge(possible_tangent_node, new_node)
        new_change = eval_function(experimental_graph, node)

        the_try = {"eval":new_change,
                   "add_node":new_node,
                   "add_edge":{"from":possible_tangent_node,"to":new_node}}
        change_dict['attempt']['tries'][try_ix] = the_try
        try_ix += 1

        if increase_metric and (change is None or new_change > change):
            change = new_change
            new_node_introduced_and_add_edge_to = new_node
            existing_node_connected_to_new_node = possible_tangent_node
        elif not increase_metric and (change is None or new_change < change):
            change = new_change
            new_node_introduced_and_add_edge_to = new_node
            existing_node_connected_to_new_node = possible_tangent_node

    if new_node_introduced_and_add_edge_to is not None and \
       existing_node_connected_to_new_node is not None:

        original_graph.add_node(new_node_introduced_and_add_edge_to)
        original_graph.add_edge(existing_node_connected_to_new_node,
                                new_node_introduced_and_add_edge_to)
        if verbose:
            print('new node:', str(new_node_introduced_and_add_edge_to))
            print('node to add and a new node to:',
                  str(existing_node_connected_to_new_node))

        picked_try = {"eval":change,
                      "add_node":new_node_introduced_and_add_edge_to,
                      "add_edge":{"from":existing_node_connected_to_new_node,
                                  "to":new_node_introduced_and_add_edge_to}}
        change_dict['attempt']['picked'] = picked_try

        change_dict['new_metric'] = change
        change_dict['performed'] = True
        return(change_dict)
    else:
        change_dict['attempt']['picked'] = 'no locally optimum change available'
        return(change_dict)


def remove_node_anywhere(node, original_graph, increase_metric, eval_function, greedy, verbose):
    original_metric = eval_function(original_graph, node)
    change_dict = {'new_metric':original_metric,
               'performed':False,
               'attempt':{'picked':{},'tries':{}}}
    if greedy:
        change = original_metric
    else:
        change = None
    node_to_be_removed = None

    # checking if the node is already optimal
    if original_metric == 0 and not increase_metric:
        change_dict['attempt']['picked'] = \
            'no optimal change to make as evaluation is already optimal'
        return(change_dict)

    all_nodes = list(original_graph.nodes())
    new_node = all_nodes[-1] + 1

    all_nodes_possible = set(original_graph.nodes())
    self_node = set()
    self_node.add(node)
    all_friends = set(original_graph.neighbors(node))
    all_foafs = set()
    for friend in all_friends:
        for foaf in original_graph.neighbors(friend):
            all_foafs.add(foaf)

    all_foafs = (all_foafs - all_friends) - self_node

    all_foafoafs = set()
    for foaf in all_foafs:
        for foafoaf in original_graph.neighbors(foaf):
            all_foafoafs.add(foafoaf)

    all_foafoafs = ((all_foafoafs - all_foafs) - all_friends) - self_node

    viable_nodes = (((all_nodes_possible - all_foafoafs) - all_foafs) - all_friends) - self_node

    try_ix = 0
    for possible_node_to_remove in viable_nodes:
        experimental_graph = original_graph.copy()
        experimental_graph.remove_node(possible_node_to_remove)
        new_change = eval_function(experimental_graph, node)

        the_try = {"eval":new_change,"remove_node":possible_node_to_remove}
        change_dict['attempt']['tries'][try_ix] = the_try
        try_ix += 1

        if increase_metric and (change is None or new_change > change):
            change = new_change
            node_to_be_removed = possible_node_to_remove
        elif not increase_metric and (change is None or new_change < change):
            change = new_change
            node_to_be_removed = possible_node_to_remove

    if node_to_be_removed is not None:
        original_graph.remove_node(node_to_be_removed)
        if verbose:
            print('node to be removed:', str(node_to_be_removed))

        picked_try = {"eval":change,"remove_node":node_to_be_removed}
        change_dict['attempt']['picked'] = picked_try

        change_dict['new_metric'] = change
        change_dict['performed'] = True
        return(change_dict)
    else:
        change_dict['attempt']['picked'] = 'no locally optimum change available'
        return(change_dict)

def turn_text_sequences_into_functions(text_sequences):
    functional_sequences = []
    for text_sequence in text_sequences:
        functional_sequences.append(adv_function_mappings_text_to_func[text_sequence])
    return(functional_sequences)

# Cellular Network Generation
def cellular_network(cluster_size, clusters, p, seed, directed):
    def mapping(x):
        return x + cluster_size

    cellular_network_graph = \
        nx.gnp_random_graph(n = cluster_size, p = p, seed = seed, directed = directed)

    for cluster_ix in range(1, clusters):
        F = nx.relabel_nodes(cellular_network_graph, mapping)
        T = nx.compose(F, cellular_network_graph)
        cellular_network_graph = T

    for cluster_ix_outer in range(0, clusters):
        for cluster_ix_inner in range(0, clusters):
            if (cluster_ix_outer >= cluster_ix_inner):
                outer_node_selected = \
                    randint(cluster_ix_outer * cluster_size, 9 + cluster_ix_outer * cluster_size)
                inner_node_selected = \
                    randint(cluster_ix_inner * cluster_size, 9 + cluster_ix_inner * cluster_size)
                cellular_network_graph.add_edge(outer_node_selected, inner_node_selected)

    return(cellular_network_graph)



def empty_adv_seq():
    global all_adversarial_sequences
    global all_positional_sequences
    all_adversarial_sequences = []
    all_positional_sequences = set()

#ITERATIVE VERSION
# Finding sequences iterative
all_adversarial_sequences = []
all_positional_sequences = set()

def find_adversarial_possibility_sequences(spending_balance,
                                           total_budget,
                                           adversarial_sequence,
                                           adversarial_possibilities):

    # Function Variables
    max_adversarial_possibility_index = len(adversarial_possibilities) - 1
    max_seq_len = ceil(total_budget / adversarial_possibilities[1][0])
    add_next_seq_pos_max = [False for i in range(0, max_seq_len)]
    seq_pos = -1

    # Useful Inner Functions
    def find_spending_balance():
        spending_balance = 0
        for seq in adversarial_sequence:
            spending_balance += seq[1][0]
        return spending_balance

    def trueify_rightmost_sequence_inclusive_from_seqpos(seq_pos):
        for pos in range(seq_pos, max_seq_len):
            add_next_seq_pos_max[pos] = True

    # Pre Initialization, filling in the max adversarial position [max, max, etc...]
    while True:
        adversarial_sequence.append(adversarial_possibilities[max_adversarial_possibility_index])
        seq_pos += 1
        spending_balance = find_spending_balance()
        if spending_balance >= total_budget:
            break

    #INIT
    while True:
        #USEFUL PROPERTY
        if adversarial_sequence:
            last_adversarial_sequence = adversarial_sequence[len(adversarial_sequence)-1][0]
        else:
            print('Finished finding all possible adversarial sequences equal to the budget!')
            return(all_adversarial_sequences)

        #RECURSIVE CHECKS
        # Equal Budget, save sequence
        if spending_balance == total_budget:
            trueify_rightmost_sequence_inclusive_from_seqpos(seq_pos)
            adversarial_sequence_copy = adversarial_sequence.copy()
            all_adversarial_sequences.append(adversarial_sequence_copy)

            if last_adversarial_sequence >= 1:
                # Change last sequence to a lower cost function because there is one available
                adversarial_sequence[len(adversarial_sequence) - 1] = \
                    adversarial_possibilities[last_adversarial_sequence - 1]
                spending_balance = find_spending_balance()
                continue
            else:
                # No lower cost function available so we remove the position
                del adversarial_sequence[-1]
                seq_pos -= 1
                spending_balance = find_spending_balance()
                continue
        # Over Budget
        elif spending_balance > total_budget:
            trueify_rightmost_sequence_inclusive_from_seqpos(seq_pos)
            if last_adversarial_sequence >= 1:
                adversarial_sequence[len(adversarial_sequence) - 1] = \
                    adversarial_possibilities[last_adversarial_sequence - 1]
                spending_balance = find_spending_balance()
                continue
            else:
                del adversarial_sequence[-1]
                seq_pos -= 1
                spending_balance = find_spending_balance()
                continue
        # Under Budget
        elif spending_balance < total_budget:
            if add_next_seq_pos_max[seq_pos]:
                add_next_seq_pos_max[seq_pos] = False
                adversarial_sequence.append(adversarial_possibilities[max_adversarial_possibility_index])
                seq_pos += 1
                spending_balance = find_spending_balance()
                continue
            else:
                trueify_rightmost_sequence_inclusive_from_seqpos(seq_pos)
                if last_adversarial_sequence >= 1:
                    adversarial_sequence[len(adversarial_sequence) - 1] = \
                        adversarial_possibilities[last_adversarial_sequence - 1]
                    spending_balance = find_spending_balance()
                    continue
                else:
                    del adversarial_sequence[-1]
                    seq_pos -= 1
                    spending_balance = find_spending_balance()
                    continue

# name to function mappings
adv_function_mappings_text_to_func = {
    'self_remove_edge_friend': self_remove_edge_friend,
    'friend_add_edge_friend': friend_add_edge_friend,
    'self_add_edge_foaf': self_add_edge_foaf,
    'friend_add_edge_foaf': friend_add_edge_foaf,
    'foaf_add_edge_foaf': foaf_add_edge_foaf,
    'friend_remove_edge_friend': friend_remove_edge_friend,
    'self_add_edge_foafoaf': self_add_edge_foafoaf,
    'friend_add_edge_foafoaf': friend_add_edge_foafoaf,
    'friend_remove_edge_foaf': friend_remove_edge_foaf,
    'foaf_add_edge_foafoaf': foaf_add_edge_foafoaf,
    'foaf_remove_edge_foaf': foaf_remove_edge_foaf,
    'foaf_remove_edge_foafoaf': foaf_remove_edge_foafoaf,
    'self_add_edge_any': self_add_edge_any,
    'friend_add_edge_any': friend_add_edge_any,
    'foaf_add_edge_any': foaf_add_edge_any,
    'any_add_edge_any': any_add_edge_any,
    'any_remove_edge_any': any_remove_edge_any,
    'add_node_to_self': add_node_to_self,
    'add_node_to_friend': add_node_to_friend,
    'add_node_to_foaf': add_node_to_foaf,
    'remove_node_to_self': remove_node_to_self,
    'remove_node_to_friend': remove_node_to_friend,
    'remove_node_to_foaf': remove_node_to_foaf,
    'add_node_anywhere': add_node_anywhere,
    'remove_node_anywhere': remove_node_anywhere
}

def find_optimal_sequence_json(starting_eval, should_increase, all_sequences_dict):
    best_outcome = None
    optimal_sequence = None
    cost = None
    remainder = None
    seq_key = None

    # print('ALL SEQUENCES DICT')
    # print(all_sequences_dict)

    # finding an inital sequence which has case other than None
    initial_non_none_key = None
    for seq_num in list(all_sequences_dict.keys()):
        if all_sequences_dict[seq_num]['final_eval'] is not None:
            initial_non_none_key = seq_num
            best_outcome = all_sequences_dict[initial_non_none_key]['final_eval']
            optimal_sequence = all_sequences_dict[initial_non_none_key]['sequence']
            cost = all_sequences_dict[initial_non_none_key]['cost']
            remainder = all_sequences_dict[initial_non_none_key]['remainder']
            seq_key = initial_non_none_key

    if initial_non_none_key is None:
        # No sequences present so we return
        return_dict = {}
        return_dict["post_experiment_metrics"] = \
            'No sequences and therefore results present.'
        return(return_dict)

    for key in all_sequences_dict.keys():
        outcome = all_sequences_dict[key]['final_eval']
        if outcome == None:
            continue

        if should_increase and outcome >= best_outcome:
            best_outcome = outcome
            optimal_sequence = all_sequences_dict[key]['sequence']
            cost = all_sequences_dict[key]['cost']
            remainder = all_sequences_dict[key]['remainder']
            seq_key = key
        elif not should_increase and outcome <= best_outcome:
            best_outcome = outcome
            optimal_sequence = all_sequences_dict[key]['sequence']
            cost = all_sequences_dict[key]['cost']
            remainder = all_sequences_dict[key]['remainder']
            seq_key = key


    readable_optimal_sequence = []

    if optimal_sequence is not None:
        for step in optimal_sequence.values():
            readable_step = list(step.keys())[0]
            if step[readable_step]['picked'] == 'no moves possible' \
                or step[readable_step]['picked'] == 'no locally optimum change available' \
                or step[readable_step]['picked'] == 'no optimal change to make as evaluation is already optimal':
                continue
            readable_optimal_sequence.append(readable_step)

    # print('best outcome')
    # print(best_outcome)

    return_dict = {}
    return_dict['post_experiment_metrics'] = {}
    return_dict['post_experiment_metrics']['optimal_sequence'] = {}
    return_dict['post_experiment_metrics']['best_outcome'] = best_outcome
    return_dict['post_experiment_metrics']['optimal_sequence']['key'] = seq_key
    return_dict['post_experiment_metrics']['optimal_sequence']['details'] = readable_optimal_sequence
    return_dict['post_experiment_metrics']['cost'] = cost
    return_dict['post_experiment_metrics']['remainder'] = remainder
    return(return_dict)

def find_optimal_adversarial_attack(all_adversarial_sequences,
                                    node,
                                    original_graph,
                                    budget,
                                    increase_metric,
                                    eval_function,
                                    greedy,
                                    verbose):
    attack_traversal = {}

    # print('ALL ADV SEQ')
    # print(all_adversarial_sequences)

    with progressbar.ProgressBar(max_value=len(all_adversarial_sequences)) as bar:
        change = eval_function(original_graph, node)
        optimal_sequence = None

        for ix, sequence in enumerate(all_adversarial_sequences):
            attack_traversal[ix] = {}
            time.sleep(0.1)
            bar.update(ix)
            new_change = None

            graph_playground = original_graph.copy()

            sequence_cost = 0
            attack_traversal[ix]['sequence'] = {}

            for s_ix,step in enumerate(sequence):
                step_dict = {step[1][1][0]:{}}
                attack_traversal[ix]['sequence'][s_ix] = step_dict
                change_dict = step[1][1][1](node,
                                            graph_playground,
                                            increase_metric,
                                            eval_function,
                                            greedy,
                                            verbose)
                attack_traversal[ix]['sequence'][s_ix][step[1][1][0]] = \
                                                        change_dict['attempt']
                if change_dict['performed'] is False:
                    break
                new_change = change_dict['new_metric']
                sequence_cost += step[1][0]

            attack_traversal[ix]['cost'] = sequence_cost
            attack_traversal[ix]['remainder'] = budget - sequence_cost
            attack_traversal[ix]['final_eval'] = new_change

            if increase_metric:
                if new_change is not None and new_change > change:
                    change = new_change
                    optimal_sequence = sequence
            else:
                if new_change is not None and new_change < change:
                    change = new_change
                    optimal_sequence = sequence

        # print('THE ATTACK TRAVERSAL')
        # print(attack_traversal)

        return ({'new_evaluation_metric':change,
                 'optimal_sequence':optimal_sequence,
                 'attack_traversal':attack_traversal})

# MULTIPROCESSING
def foaa_multiprocessing(process_number, processing_slice,
                         all_adversarial_sequences,
                         node, original_graph, budget,
                         increase_metric,
                         eval_function,
                         greedy,
                         verbose,
                         processes_return_dict):

    all_adversarial_sequences_to_process = \
        all_adversarial_sequences[processing_slice[0]:processing_slice[1]]
    return_result = \
        find_optimal_adversarial_attack(all_adversarial_sequences_to_process,
                                        node,
                                        original_graph,
                                        budget,
                                        increase_metric,
                                        eval_function,
                                        greedy,
                                        verbose)
    processes_return_dict[process_number] = return_result
