# Adversarial Network Simulation and Analysis Framework

An open-source framework to assess the robustness of network structures and metrics to adversarial attacks and ascertain adversarial attack patterns.

##### Development Goal

Our ultimate goal is to create a framework that can support any graph configurations, simulations, and be the test-bed for finding changes and optimal results of various changes that can be modeled in the form of a network.

## More Background

Almost any information problem can be postulated as a network with nodes and relations. For instance, a simple food supply chain may involve the agricultural farms which create the goods, which are then transported to suppliers, followed by wholesalers, supermarkets, and finally, the curated food gets to consumers. In this example, each of these entities can be a node in the network, and their connection can be an edge. By the same token, a plethora of other real-world scenarios can be modeled as networks. Other examples that come to mind are hyperlink networks that are used to assess the raking of a given Google search or even a Tweet chain that can express how misinformation is spreading on social media.

Unfortunately, most of these networks can be attacked and undermined by bad actors (adversaries) which can aim to change the flow of information or perception of influence in those real-world situations (adversarial attacks) for their own benefit.

Our open-source Python framework provides a way to simulate such networks, adversaries, and adversarial attacks for any domain, and find patterns in how robust the networks are to these attacks, as well as the patterns that adversaries can employ to yield their goals.

We believe that by understanding the robustness of networks to adversarial attacks, as well as the patterns of those attacks, can better enable society to mitigate or even eliminate such attacks in the future.

## Prerequisites

- [Python](https://www.python.org/downloads/) installed on your machine.
- [Jupyter Notebook](https://jupyter.org/install) installed on your machine.

## Configuring the Framework/Experiments

#### Installation
1. Copy or clone this repository to your local machine e.g. `git clone https://github.com/uiuc-ischool-scanr/social-network-adversarial-perturbations`
1. Install Python (version 3 or greater) and pip on your Machine if you haven't already. To check that they are installed run `python --version` and `pip --version` in your terminal and ensure they return a proper version.
1. Install Jupyter Notebook.
1. Install all requirements and dependencies of this framework by running `pip install -r requirements.txt` from the root repository location (e.g. `adversarial-learning-social-graphs` folder).

#### Input-Graph Generation

This tool can either use external `.edgelist` type graphs to be imported in the configuration files, or we can generate synthetic graphs ourselves. In order to generate them, we do the following:

1. Check the `/graph-generator/generate_sample_graphs.py` script which you can pick what parameters you want to generate the graphs with, what types of graphs to generate, and how many graphs to generate. This script uses [NetworkX Graph Generators](https://networkx.github.io/documentation/networkx-2.0/reference/generators.html) which provides documentation for how these generators work.
1. Feel free to tweak the parameters of `generate_sample_graphs.py` for your type of experiments/graphs.
1. Execute `python generate_sample_graphs.py` to generate the graphs in the `created_graphs_output` folder.

#### Input-Config Generation

The framework uses configuration files to run the experiments/simulations. In order to generate such files (so that we don't have to do it manually), we can generate bulk/batch configuration files to run experiments in batch. For more information on the details of a configuration and how that fits in with the framework/experiment, check out `Anatomy of a Configuration File` below. We generate configurations in bulk by doing the following:

1. Investigate `experiment-config-creator/creator-main.py` for how the configurations can be generated in batch. Note that this script generates many configuration files and many folders in the `output-experiments` location. This is done with the idea that one configuration file is one "type" of experiment with name `exp1` (name of config file as well with `.json` at the end), which then should generate all output results into the `output-experiments/exp1` folder.
1. Feel free to tweak the parameters of `creator-main.py` for your type of experiments.
1. Execute `python creator-main.py` which should generate all configuration files in the `input-configs` folder, and all respective output experiment folders in the `output-experiment` folder.

## Running the Framework
- Simply execute `python advlearningsocgraph_main.py` from the main folder.
- After the script completes, the results should be present under `/output-experiments/<exp-folder(s)>/`.

## Visualizing Results
1. Browse to the `adversarial-learning-social-graphs/result-analyzers/` path.
1. Run the command `jupyter notebook`.
1. Go to `http://localhost:8888/` in your browser as directed by the command line if it did not direct you there already.
1. To evaluate a single experiment, launch the `InterpretSingleExperiment.ipynb` and change the `folder_path` variable to contain the folder where the output `.json` file lies. Then change the `file_name` variable to the name of the file you would like to inspect e.g. `file_name = 'adv_learning_results_increase_betweenness_centrality_graphtype_from_file_budget_1_bootstrapsample_0_tier_0_subsample_0.json'`. Run this jupyter notebook to visualize results for that specific experiment.
1. To evaluate a whole experiment set (where the results of an experiment set should all be included in the same folder), launch the `InterpretAggregateExperimentSet.ipynb` and change the `final_experiment_output_path` to the folder of the experiment present in `output-experiments`. After running the whole notebook it should generate a plot of all optimal moves for that experiment set.
1. To generate many experiment sets in bulk, launch the `InterpretManyAggregateExperimentSets.ipynb` and execute all scripts from the notebook in order. This should look into all experiments/folders under `output-experiments` and generate the plots for each experiment type in `output-plots`. One can then look into each plot in `output-plots` and look for patterns/results.
1. Execute these scripts according to the Jupyter Notebook [documentation](https://jupyter.org/documentation).

## How do all the Components Work Together (Pipeline)

Everything starts from the main `advlearnsocgraph_main.py` file which, when execuded, first grabs each configuration file from `input-configs` one by one, then for each config, configures the parameters of the simulation/experiment from that given config, including iterating through all the plots that config may point to. For instnace, if a config points to all X sampled graphs of type `random`, then each config will have X experiments for each particular graph. Then, the framework uses helpful functions/constants from `adfunc/adfunc.py` to configure the framework and run the experiments. During the experimentation process, the framework first finds all possible moves given the configurations, then it finds the optimal move given the defined configuration/criteria. Once one experiment is done, an output `.json` file is written to the presecribed `output-path` of the config at hand. This output file contains all configurations, pre-evaluation metrics/graphs and post-evaluation metrics/graphs as well as all tries for the experiment in detail in case further analysis needs to be done. Note that each experiment and therefore `.json` file is generated for each graph and each config. For example, if there are 10 configuration files in `input-configs` and each point to 10 graphs respectively, there should be 100 total output `.json` files in 10 respective folders where the folder name is the name of the config, all present in `output-experiments`.

## Citation
* Parulian, N. N., Lu, T., Mishra, S., Avram, M., & Diesner, J. (2020). Effectiveness of the Execution and Prevention of Metric-Based Adversarial Attacks on Social Network Data. Information, 11(6), 306.
* Avram, M. V., Mishra, S., Parulian, N. N., & Diesner, J. (2019, August). Adversarial perturbations to manipulate the perception of power and influence in networks. In 2019 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining (ASONAM) (pp. 986-994). IEEE.


## Other Important Details

### Anatomy of a Configuration File

See comments above each component decorated with `//`

```
{
	// Used if a single graph needs to be used (e.g. running experiment for a real-world graph). Path to this file in .edgelist format should be placed here
	"input_path": "",
	// Used for running multiple input graphs where in this case /cellular/ includes many cellular type graphs all in .edgelist format
	"aggregate_graphs_path":"./graph-generator/created_graphs_output/cellular/",
	// If one needs to name a graph, they can name the graph here
	"graph_name": "",
	// Where the output result(s) of thes experiment should be placed
	"output_path": "./output-experiments//GRAPH_cellular_CENTRALITY_betweenness_DIRECTION_decrease_TIER_10_REACHTYPE_1/",
	// If one wants to see detailed output of the experiment while the experiment is running (in the console), change this to True
	"print_minimum_output": "False",
	// When importing graphs externally, all these should be False. However, when generating graphs without importing them, one of these graphs should be true.
	"graph": {
		"scale_free": "False",
		"small_world": "False",
		"random": "False",
		"cellular": "False"
	},
	// Used for sampling nodes in the network (where nodes are the ego/self of the experiment) where we try to optimize some metric according to that node
	"sampling": {
		// How many nodes to sample for each graph, each node results in one experiment for that graph.
		"bootstrap_sampling_times": 1,
		// The following two configurations have to do with generating nodes according to percentiles. Think of num_tiers as the total, and the specific_tier_selected as the selected tier. Hence if selecting specific_tier_selected as 10 and num_tiers as 10, this should be 90-100%ile node selected according to the evaluation metric. While if specific_tier_selected is 1 and num_tiers is 10, then this should be 0-10%ile node. For instance, if we select eval_function (below) as betweenness_centrality, then it should be 0-10%ile node of betweenness_centrality where 90% of nodes have higher betweenness_centrality than that node.
		"num_tiers": 10,
		"specific_tier_selected": 10,
		// How many nodes should we sample for each tier selected, note that this increases the number of experiments done.
		"tier_sampling_size": 1,
		// Sample with replacement or not?
		"with_replacement": "True",
		// Sample all nodes in that tier (note that this increases the number of experiments run)
		"sample_all_nodes": "False"
	},
	// Evaluation criteria of the selected noe
	"evaluation": {
		// What centrality/valuation should we optimize for? Select True for the one selected, and False for the others.
		"eval_function": {
			"degree_centrality": "False",
			"betweenness_centrality": "True",
			"closeness_centrality": "False",
			"eigenvector_centrality": "False"
		},
		// Should this centrality/evaluation be increased or decreased through perturbations?
		"eval_direction": {
			"increase": "False",
			"decrease": "True"
		}
	},
	// Each of the moves may be costly, this is the total budget that the node/adversary has
	"budget": {
		// If not sampling, this is the budget the node/adversary will have
		"one_time_budget": 1,
		// If we sample a budget, select the min and max number of the sampling and select sample_budget to be true. E.g. if min_budget is 5 and max_budget is 10, the idea is that the budget will be sampled somewhere between 5 and 10 (inclusively?)
		"min_budget": 1,
		"max_budget": 1,
		"sample_budget": "False"
	},
	// All the moves and costs of each moves. If adding or removing moves we are changing the reach of the node (where maybe in some experiments/simulations, the node is not allowed to make such moves depending on the set-up) Also, see "Reach types and their explanations" below
	"adversarial_functions_costs": {
		"self_remove_edge_friend": 1,
		"self_add_edge_foaf": 1,
		"add_node_to_self": 1,
		"remove_node_to_self": 1
	},
	"verbose": "False",
	// Has to do with the selection of the move when looking through different moves of each of the adversarial_functions. E.g. if we are doing 'self_remove_edge_friend', then it looks for all the friends and tries each one, if it is greedy it picks the move that actually increases the centrality measure. If none of them do, none of them are picked. If it is not greedy, it picks the best move from the move-type regardless of whether or not it improve the centrality measure or not. Double-check adfunc.py for details of this.
	"greedy": "True",
	// Number of processes to run this experiment with. This will only optimize the (all moves finding) part of the algorithm which is one of the slowest parts.
	"num_of_processes": 2,
	// Should the (all moves finding) part of the experiment save the results somewhere so that they don't need to be computed again next time an experiment runs a few days/weeks/months later? Then save them to a file. Note that this may not be much of an optimization and unless this is improved, it should be used sparingly.
	"save_sequences_on_file": "False",
	"read_sequences_from_file": {
		"read_from_file": "False",
		"load_sequences_file_path": ""
	}
}
```

### Reach types and their explanations

```
1) "self_remove_edge_friend"
    - Removes an immediate edge to the self node

2) "friend_add_edge_friend"
    - Adds an edge from a friend of the self node to another friend

3) "self_add_edge_foaf"
    - Adds an edge from the self node to a friend of a friend of the self node

4) "friend_add_edge_foaf"
    - Adds an edge from a friend to a friend of a friend

5) "foaf_add_edge_foaf"
    - Adds an edge from a friend of a friend to another friend of a friend

6) "friend_remove_edge_friend"
    - Removes the connection between two friend nodes of the self

7) "self_add_edge_foafoaf"
    - Adds an edge from the self node to a friend of a friend of a friend

8) "friend_add_edge_foafoaf"
    - Adds an edge from a friend node to a friend of a friend of a friend

9) "friend_remove_edge_foaf"
    - Removes the edge between a friend and a friend of friend

10) "foaf_add_edge_foafoaf"
    - Adds an edge between a friend of a friend and a friend of a friend of a friend

11) "foaf_remove_edge_foaf"
    - Removes an edge between two nodes that are friend of a friend

12) "foaf_remove_edge_foafoaf"
    -Removes an edge between a friend of a friend and a friend of a friend of a friend

13) "self_add_edge_any"
    - Adds an edge between the self node and any node in the network

14) "friend_add_edge_any"
    - Adds an edge between a friend and any node in the network

15) "foaf_add_edge_any"
    - Adds an edge between a friend of a friend and any node in the network

16) "any_add_edge_any"
    - Adds an edge between any two nodes in the network

17) "any_remove_edge_any"
    - Removes an edge between any two nodes in the network

18) "add_node_to_self"
    - Creates a new node, and adds an edge between the self node and the newly created node

19) "add_node_to_friend"
    - Creates a new node, and adds an edge between a friend and the newly created node

20) "add_node_to_foaf"
    - Creates a new node, and adds an edge between a friend of a friend and the newly created node

21) "remove_node_to_self"
    - Removes a node which is a friend node to the self

22) "remove_node_to_friend"
    - Removes a node which is connected to a friend

23) "remove_node_to_foaf"
    - Removes a node which is connected to a friend of a friend

24) "add_node_anywhere"
    - Creates a new node, and connects it to any other node in the network

25) "remove_node_anywhere"
    - Removes any node in the network
```


