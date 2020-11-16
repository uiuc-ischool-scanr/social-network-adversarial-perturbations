
# Author: Mihai Avram
# E-Mail: mihai.v.avram@gmail.com

# This is the main file which takes or generates input graph and configuration
# Then runs experiments and simulations and outputs the results

# Importing functions needed for adversarial learning on social graphs
from adfunc.adfunc import *

# Used for reading/writing configuration paths
import json

# Used for reading configuration paths
import os

if __name__ == "__main__":
	#store the list of specific config paths in paths_list variable
	paths_list = []

	CURRENT_DIRECTORY_OF_SCRIPT = os.path.dirname(os.path.realpath(__file__))
	INPUT_CONFIGS_FOLDER = os.path.join(CURRENT_DIRECTORY_OF_SCRIPT, "input-configs")

	for input_config in os.listdir(INPUT_CONFIGS_FOLDER):
	    if input_config.endswith(".json"):
	        paths_list.append(os.path.join(INPUT_CONFIGS_FOLDER, input_config))

	#for each file in paths_list
	for file_path in paths_list:
	    # INITIALIZING ALL NEEDED CONFIGS
		# Configuration data structure to be filled in later from the config file
	    all_configs = {
		    # Paths
		    "aggregate_config_paths": paths_list,
		    "specific_config_path": file_path,
	        "aggregate_graphs_path": None,
		    "input_path":None,
		    "output_path":None,
		    # Print verbose JSON of all moves or just picked move
		    "print_minimum_output":None,
		    # Graph
		    "graph":None,
		    "graph_name":None,
		    "graph_type":None,
		    # Sampling
		    # How many times to sample at the high-level
		    "bootstrap_sampling_times":None,
		    # How many sampling tiers to sample for e.g. 2 means bottom 50% and top 50%
		    "num_tiers":None,
		    # Which of the two tiers is selected, if num_tiers is 2, and this is 2,
		    # then this refers to the final tier top 50%, 1 refers to the bottom 50%
		    "specific_tier_selected":None,
		    # Threshold for selecting the top percentile nodes based on evaluation metric
		    "threshold_top_percentile":None,
		    # How many nodes to sample from each sampling tier
		    "tier_sampling_size":None,
		    # Sample with replacement or without replacement
		    "with_replacement":None,
		    # Sample all nodes from the tier
		    "sample_all_nodes":None,
		    # Evaluation
		    # What centrality metric or evaluation metric to use
		    "eval_function":None,
		    "eval_function_text":None,
		    # Increase or decrease
		    "eval_direction":None,
		    # Budget Costs and Functions
		    # Budgets to perform adversarial attacks with
		    "one_time_budget":None,
		    "min_budget":None,
		    "max_budget":None,
		    "sample_budget":None,
		    # Adversarial functions names and costs
		    "adversarial_functions_costs":None,
		    # Actual adversarial functions
		    "adversarial_functions":None,
		    # Misc
		    "verbose":None,
		    "greedy":None,
		    "num_of_processes":None,
		    "save_sequences_on_file":None,
		    "read_sequences_from_file":None,
		    "load_sequences_file_path":None
	    }


	    # Initializing All Configs
	    all_configs = initialize_configs(all_configs)

	    graph_number = 0
	    # Iterating all included input graphs if they have been provided
	    for graph_path in os.listdir(all_configs["aggregate_graphs_path"]):
	        if all_configs["aggregate_graphs_path"] != "./":
	            graph_number += 1
	            all_configs["input_path"] = \
	                all_configs["aggregate_graphs_path"] + "/" + graph_path
	            graph = nx.read_edgelist(all_configs["input_path"], nodetype=int)
	            all_configs["graph"] = graph.to_undirected()
	            all_configs["graph_type"] = "from_file"
	            all_configs["graph_name"] = all_configs["input_path"].split("/")[-1]

	        # ADV LEARNING SOCIAL GRAPH EXPERIMENTATION PROCESS STARTS BELOW:
	        for bootstrap_sample in range(all_configs["bootstrap_sampling_times"]):
	            # Per-sample configurations
	            node = None
	            budget = None
	            if all_configs["sample_budget"]:
	                budget = randint(1, all_configs["max_budget"])
	            else:
	                budget = all_configs["one_time_budget"]

	            # Setting top percentile threshold
	            all_configs["threshold_top_percentile"] = \
	                (all_configs["num_tiers"] - all_configs["specific_tier_selected"])/10

	            # Node Sampling
	            tiered_node_sample = pick_node_sample(all_configs["eval_function_text"],
	                                                  all_configs["graph"],
	                                                  all_configs["num_tiers"],
	                                                  all_configs["specific_tier_selected"],
	                                                  all_configs["tier_sampling_size"],
	                                                  all_configs["with_replacement"],
	                                                  all_configs["sample_all_nodes"])

	            if not tiered_node_sample:
	                sys.exit("No samples were selected, please check graph/sampling variables")

	            #RUNNING ADV LEARNING PROCESS FOR EACH SAMPLE
	            for tier_ix, tier in enumerate(tiered_node_sample):
	                if not tier:
	                    # no samples available in tier
	                    continue
	                for subsample_ix, subsample in enumerate(tier):

	                    # Initializing results json
	                    node = subsample
	                    starting_percentile = \
	                        compute_node_percentile(all_configs["graph"],
	                                                              node,
	                                                              all_configs["eval_function_text"])

	                    # Initial graph json version serialized
	                    initial_graph_edgelist = json_graph.node_link_data(all_configs["graph"])
	                    pre_current_metric = all_configs["eval_function"](all_configs["graph"], node)
	                    results_json_dict = {
	                                    "configurations":
	                                    {
	                                        "from_config_file":all_configs["specific_config_path"],
	                                        "num_tiers":all_configs["num_tiers"],
	                                        "graph_type":all_configs["graph_type"],
	                                        "graph_name":all_configs["graph_name"],
	                                        "threshold_top_percentile":all_configs["threshold_top_percentile"],
	                                        "bootstrap_sampling_times":all_configs["bootstrap_sampling_times"],
	                                        "specific_tier_selected":tier_ix,
	                                        "eval_metric":all_configs["eval_function_text"],
	                                        "direction":all_configs["eval_direction"],
	                                        "print_minimum_output":all_configs["print_minimum_output"],
	                                        "one_time_budget":all_configs["one_time_budget"],
	                                        "min_budget":all_configs["min_budget"],
	                                        "max_budget":all_configs["max_budget"],
	                                        "sample_budget":all_configs["sample_budget"],
	                                        "adversarial_functions_costs":all_configs["adversarial_functions_costs"],
	                                        "verbose":all_configs["verbose"],
	                                        "greedy":all_configs["greedy"],
	                                        "num_of_processes":all_configs["num_of_processes"],
	                                    },
	                                    "pre_experiment_metrics":
	                                    {
	                                        "self_node":node,
	                                        "budget_selected_for_experiment":budget,
	                                        "starting_eval":pre_current_metric,
	                                        "starting_percentile":starting_percentile,
	                                        "starting_graph":initial_graph_edgelist
	                                    },
	                                    "post_experiment_metrics":
	                                    {
	                                        "ending_eval":None,
	                                        "ending_percentile":None,
	                                        "outcome_difference":None,
	                                        "optimal_sequence":None,
	                                        "budget_remainder":None,
	                                        "ending_graph":None
	                                    }
	                                }

	                    # Initializing evaluation direction
	                    increase_metric = None
	                    if all_configs["eval_direction"] == "increase":
	                        increase_metric = True
	                    else:
	                        increase_metric = False

	                    print('THE BUDGET')
	                    print(budget)

	                    # Emptying adv. sequences before finding new ones
	                    empty_adv_seq()

	                    # Finding all possible sequences for adversarial learning
	                    print('\n Starting Adversarial Learning Process')
	                    all_adversarial_sequences = None
	                    if all_configs["read_sequences_from_file"]:
	                        print('Reading sequences from file.')
	                        all_adversarial_sequences = \
	                            read_sequences_from_file(all_configs["load_sequences_file_path"])
	                    else:
	                        all_adversarial_sequences = \
	                            find_adversarial_possibility_sequences(0,
	                                                                   budget,
	                                                                   [],
	                                                                   all_configs["adversarial_functions"])

	                    if all_configs["save_sequences_on_file"] == True:
	                        # Saving sequences to a file for later use to save compute time
	                        print('Saving sequences to file.')
	                        save_sequences_on_file(all_adversarial_sequences,
	                                               all_configs["output_path"])

	                    # shuffling all of the adversarial sequences
	                    #before multiprocessing to work towards an equitable workload
	                    shuffle(all_adversarial_sequences)

	                    # print('PRINTING SEQUENCES:')
	                    # for sequence in all_adversarial_sequences:
	                    #     print(sequence)
	                    #     print('\n')

	                    # Finding optimal sequence for adversarial learning
	                    print('\n Finding Optimal Adversarial Sequence')
	                    graph_copy = all_configs["graph"].copy()

	                    # Multiprocessing
	                    num_of_adv_sequences = len(all_adversarial_sequences)
	                    if num_of_adv_sequences <= 10:
	                        all_configs["num_of_processes"] = num_of_adv_sequences
	                    print('NUM OF ADV SEQUENCES')
	                    print(num_of_adv_sequences)

	                    slice_length = \
	                        ceil(num_of_adv_sequences / all_configs["num_of_processes"])
	                    processing_slices = \
	                        [[slice_length*x, slice_length*(x+1)] for x in range(0, all_configs["num_of_processes"])]

	                    print('PROCESSING SLICES')
	                    print(processing_slices)

	                    processes = []
	                    manager = Manager()
	                    processes_return_dict = manager.dict()
	                    for p in range(0, all_configs["num_of_processes"]):
	                        process_number = p
	                        process = Process(target=foaa_multiprocessing,
	                                          args= \
	                                            (process_number,
	                                             processing_slices[p],
	                                             all_adversarial_sequences,
	                                             node,
	                                             graph_copy,
	                                             budget,
	                                             increase_metric,
	                                             all_configs["eval_function"],
	                                             all_configs["greedy"],
	                                             all_configs["verbose"],
	                                             processes_return_dict))

	                        processes.append(process)
	                        process.start()
	                    for p in range(0, all_configs["num_of_processes"]):
	                        processes[p].join()
	                    print('All threads finished!')

	                    all_sequences_dict = {}
	                    creation_ix = 0

	                    for process_result in processes_return_dict.values():
	                        if bool(process_result['attack_traversal']):
	                            for sequence in process_result['attack_traversal'].values():
	                                all_sequences_dict[creation_ix] = sequence
	                                creation_ix += 1

	                    results_json_dict["experiment"] = {}
	                    results_json_dict["experiment"]["perturbations"] = all_sequences_dict

	                    optimal_sequence_results = \
	                        find_optimal_sequence_json(results_json_dict["pre_experiment_metrics"]["starting_eval"],
	                                                   increase_metric,
	                                                   all_sequences_dict)

	                    final_graph = None

	                    if (optimal_sequence_results["post_experiment_metrics"] != \
	                        'No sequences and therefore results present.'):

	                        if optimal_sequence_results["post_experiment_metrics"]['optimal_sequence'] is not None:
	                            print('OPTIMAL SEQUENCES FOUND')
	                            print(optimal_sequence_results["post_experiment_metrics"]['optimal_sequence'])
	                            results_json_dict["post_experiment_metrics"]["optimal_sequence"] = \
	                                optimal_sequence_results["post_experiment_metrics"]['optimal_sequence']

	                        if optimal_sequence_results["post_experiment_metrics"]['best_outcome'] is not None:
	                            results_json_dict["post_experiment_metrics"]["ending_eval"] = \
	                                optimal_sequence_results["post_experiment_metrics"]['best_outcome']
	                            results_json_dict["post_experiment_metrics"]["outcome_difference"] = \
	                                optimal_sequence_results["post_experiment_metrics"]['best_outcome'] - \
	                                    results_json_dict["pre_experiment_metrics"]["starting_eval"]

	                        if optimal_sequence_results["post_experiment_metrics"]['cost'] is not None:
	                            results_json_dict["post_experiment_metrics"]["budget_remainder"] = \
	                                budget - optimal_sequence_results["post_experiment_metrics"]['cost']

	                        # FINDING OPTIMAL SEQUENCE FROM ALL SEQUENCES FOUND FROM MULTITHREADING
	                        final_adversarial_sequences_to_compare = []
	                        for process_result in processes_return_dict.values():
	                            if process_result['optimal_sequence'] is not None:
	                                final_adversarial_sequences_to_compare.append(process_result['optimal_sequence'])

	                        optimal_functions = \
	                            turn_text_sequences_into_functions(optimal_sequence_results["post_experiment_metrics"]['optimal_sequence']['details'])

	                        #PERFORMING OPTIMAL ADVERSARIAL ATTACK
	                        print('PERFORMING OPTIMAL ADVERSARIAL ATTACK:')
	                        final_graph = all_configs["graph"].copy()
	                        verbose = True
	                        if optimal_functions is not None:
	                            for function_step in optimal_functions:
	                                function_step(node,
	                                              final_graph,
	                                              increase_metric,
	                                              all_configs["eval_function"],
	                                              all_configs["greedy"],
	                                              verbose)
	                        else:
	                            print('No optimal sequence found for the given criteria.')

	                        post_current_metric = all_configs["eval_function"](final_graph, node)
	                        ending_percentile = \
	                            compute_node_percentile(final_graph,
	                                                    node,
	                                                    all_configs["eval_function_text"])
	                        results_json_dict["post_experiment_metrics"]["ending_percentile"] = \
	                            ending_percentile

	                        # WRITING FINAL GRAPH AND RESULTS TO FILE
	                        ending_graph_edgelist = json_graph.node_link_data(final_graph)
	                        results_json_dict["post_experiment_metrics"]["ending_graph"] = \
	                            ending_graph_edgelist
	                    else:
	                        results_json_dict["post_experiment_metrics"]["ending_graph"] = \
	                            initial_graph_edgelist
	                        results_json_dict["post_experiment_metrics"]["ending_percentile"] = \
	                            starting_percentile

	                    output_file_name = \
	                        all_configs["output_path"] + 'adv_learning_results_' +\
	                        str(all_configs["eval_direction"]) + \
	                        '_' +\
	                        str(all_configs["eval_function_text"]) +\
	                        '_graphtype_' +\
	                        str(all_configs["graph_type"]) +\
	                        '_budget_' +\
	                        str(budget) +\
	                        '_bootstrapsample_' +\
	                        str(bootstrap_sample) +\
	                        '_tier_' + \
	                        str(tier_ix) + \
	                        '_subsample_' + \
	                        str(subsample_ix) + \
	                        '_imported-graph-number_' + \
	                        str(graph_number) + \
	                        '.json'

	                    # Printing minimum amount of content if needed
	                    if all_configs["print_minimum_output"]:
	                        try:
	                            final_results_sequence_key = \
	                                results_json_dict["post_experiment_metrics"]["optimal_sequence"]["key"]
	                            if final_results_sequence_key is None:
	                                raise
	                            final_sequence_contents = \
	                                results_json_dict["experiment"]["perturbations"][final_results_sequence_key]
	                            results_json_dict["experiment"]["perturbations"] = {}
	                            results_json_dict["experiment"]["perturbations"][final_results_sequence_key] = \
	                                final_sequence_contents
	                        except:
	                            results_json_dict["experiment"]["perturbations"] = {}
	                    with open(output_file_name, 'w') as dump_file:
	                        json.dump(results_json_dict, dump_file)

	        if all_configs["aggregate_graphs_path"] == "./":
	            # Aggregate path is empty so we only use one input graph
	            break
