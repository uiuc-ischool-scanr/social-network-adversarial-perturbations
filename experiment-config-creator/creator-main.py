import json
import os

if __name__ == "__main__":
	graph_types = ["random", "cellular", "scale_free", "small_world"]
	eval_functions = ["degree",
		"betweenness",
		"closeness",
		"eigenvector"]

	one_time_budget=[1]
	specific_tier_selected=[10,1]
	adversarial_function_costs=[1,2,3]
	input_path= "./graph-generator/created_graphs_output/"
	config_output_path="../input-configs/"
	experiment_output_base_path = "./output-experiments/"
	experiment_output_base_path_relative_to_script = "../output-experiments/"

	for i in graph_types:
		for j in eval_functions:
			for k in one_time_budget:
				for l in specific_tier_selected:
					for m in adversarial_function_costs:
						if(l==1):
							direction= "increase"
						else:
							direction= "decrease"
						exp_name = "GRAPH_" \
									+ i \
									+ "_CENTRALITY_" \
									+ j \
									+ "_DIRECTION_" \
									+ direction \
									+ "_TIER_" \
									+ str(l) \
									+ "_REACHTYPE_" \
									+ str(m)
						output_config = {"input_path":"",
							"aggregate_graphs_path":(input_path + i + "/"),
							"graph_name":"",
							"output_path": experiment_output_base_path + "/" + exp_name + "/",
							"print_minimum_output":"False",
							"graph": {
								"scale_free":str(i=="scale_free"),
								"small_world":str(i=="small_world"),
								"random":str(i=="random"),
								"cellular":str(i=="cellular")
							},
							"sampling": {
								"bootstrap_sampling_times":1,
								"num_tiers":10,
								"specific_tier_selected":l,
								"tier_sampling_size":1,
								"with_replacement":"True",
								"sample_all_nodes":"False"
							},
							"evaluation":{
								"eval_function":{
									"degree_centrality":str(j=="degree"),
									"betweenness_centrality":str(j=="betweenness"),
									"closeness_centrality":str(j=="closeness"),
									"eigenvector_centrality":str(j=="eigenvector")
								},
								"eval_direction":{
									"increase":str(l==1),
									"decrease":str(l==10)
								}
							},
							"budget":{
								"one_time_budget":k,
								"min_budget":1,
								"max_budget":k,
								"sample_budget":"False"
							}
						}
						if(m == 1):
							output_config["adversarial_functions_costs"]={
								"self_remove_edge_friend":1,
								"self_add_edge_foaf":1,
								"add_node_to_self":1,
								"remove_node_to_self":1
							}
						if(m==2):
							output_config["adversarial_functions_costs"]={
								"self_remove_edge_friend":1,
								"add_node_to_self":1,
								"friend_add_edge_friend":1,
								"friend_add_edge_foaf":1,
								"remove_node_to_self":1,
								"foaf_add_edge_foaf":1,
								"friend_remove_edge_friend":1,
								"self_add_edge_foafoaf":1,
								"friend_add_edge_foafoaf":1,
								"friend_remove_edge_foaf":1,
								"foaf_add_edge_foafoaf":1,
								"foaf_remove_edge_foaf":1,
								"foaf_remove_edge_foafoaf":1,
								"add_node_to_friend":1,
								"add_node_to_foaf":1,
								"remove_node_to_friend":1,
								"remove_node_to_foaf":1
							}
						if(m==3):
							output_config["adversarial_functions_costs"]={
								"self_remove_edge_friend":1,
								"friend_add_edge_friend":1,
								"self_add_edge_foaf":1,
								"friend_add_edge_foaf":1,
								"foaf_add_edge_foaf":1,
								"friend_remove_edge_friend":1,
								"self_add_edge_foafoaf":1,
								"friend_add_edge_foafoaf":1,
								"friend_remove_edge_foaf":1,
								"foaf_add_edge_foafoaf":1,
								"foaf_remove_edge_foaf":1,
								"foaf_remove_edge_foafoaf":1,
								"self_add_edge_any":1,
								"friend_add_edge_any":1,
								"foaf_add_edge_any":1,
								"any_add_edge_any":1,
								"any_remove_edge_any":1,
								"add_node_to_self":1,
								"add_node_to_friend":1,
								"add_node_to_foaf":1,
								"remove_node_to_self":1,
								"remove_node_to_friend":1,
								"remove_node_to_foaf":1,
								"add_node_anywhere":1,
								"remove_node_anywhere":1
							}
						output_config["verbose"]="False"
						output_config["greedy"]="True"
						output_config["num_of_processes"]=8
						output_config["save_sequences_on_file"]="False"
						output_config["read_sequences_from_file"]={
							"read_from_file":"False",
							"load_sequences_file_path":""
						}
						content = json.dumps(output_config, indent=4)
						# Creating configuration file
						with open(config_output_path + "/" + exp_name + ".json", "w") as f:
							f.write(content)
						# Creating output location of experiments using this configuration file
						out_experiment_dir_path = experiment_output_base_path_relative_to_script + "/" + exp_name
						if not os.path.exists(out_experiment_dir_path):
							os.makedirs(out_experiment_dir_path)
