import argparse
import copy
import importlib
import pathlib
import random
import time

import dgl
import networkx as nx
import numpy as np
import torch
from dgl.data import DGLDataset, register_data_args
from heft.core import schedule
from sklearn import preprocessing

from edgnn.core.app import App
from edgnn.core.data.constants import GRAPH, N_CLASSES, N_ENTITIES, N_RELS
from edgnn.core.models.constants import (AIFB, GRAPH_CLASSIFICATION, MUTAG,
                                    MUTAGENICITY, NODE_CLASSIFICATION, PTC_FM,
                                    PTC_FR, PTC_MM, PTC_MR)
from edgnn.utils.inits import to_cuda
from edgnn.utils.io import create_default_path, print_graph_stats, read_params

## from TP HEFT
def face_recognition_task_graph():
    name_dict = {"Source":0,"Copy":1,"Tiler":2,"Detect1":3,"Detect2":4,"Detect3":5,"Feature merger":6,"Graph Spiltter":7,"Classify1":8,"Classify2":9
    ,"Reco. Merge":10,"Display":11}
    dic_task_graph = dict()
    dic_task_graph[name_dict["Source"]]=[name_dict["Copy"]]
    dic_task_graph[name_dict["Copy"]] = [ name_dict["Tiler"],name_dict["Feature merger"],name_dict["Display"]]
    dic_task_graph[name_dict["Tiler"]] = [ name_dict["Detect1"],name_dict["Detect2"],name_dict["Detect3"] ]
    dic_task_graph[name_dict["Detect1"]] = [name_dict["Feature merger"]]
    dic_task_graph[name_dict["Detect2"]] = [name_dict["Feature merger"]]
    dic_task_graph[name_dict["Detect3"]] = [name_dict["Feature merger"]]
    dic_task_graph[name_dict["Feature merger"]] = [name_dict["Graph Spiltter"]]
    dic_task_graph[name_dict["Graph Spiltter"]] = [ name_dict["Classify1"],name_dict["Classify2"],name_dict["Reco. Merge"] ]
    dic_task_graph[name_dict["Classify1"]] = [name_dict["Reco. Merge"]]
    dic_task_graph[name_dict["Classify2"]] = [name_dict["Reco. Merge"]]
    dic_task_graph[name_dict["Reco. Merge"]] =[name_dict["Display"]]

    #print("Task Graph Face Recognition\n",dic_task_graph)
    return dic_task_graph

def obj_and_pose_recognition_task_graph():
    name_dict = {"Source":0,
    "Copy":1,
    "Scaler":2,
    "Tiler":3,
    "SIFT1":4,
    "SIFT2":5,
    "SIFT3":6,
    "SIFT4":7,
    "SIFT5":8,
    "Feature merger":9,
    "Descaler":10,
    "Feature spiltter":11,
    "Model matcher1":12,
    "Model matcher2":13,
    "Model matcher3":14,
    "Match joiner":15,
    "Cluster spiltter":16,
    "Clustering1":17,
    "Clustering2":18,
    "Cluster joiner":19,
    "RANSAC":20,
    "Display":21}
    dic_task_graph = dict()
    dic_task_graph[name_dict["Source"]]=[name_dict["Copy"]]
    dic_task_graph[name_dict["Copy"]] = [ name_dict["Scaler"],name_dict["Display"]]
    dic_task_graph[name_dict["Scaler"]] = [ name_dict["Tiler"],name_dict["Descaler"]]
    dic_task_graph[name_dict["Tiler"]] = [ name_dict["SIFT1"],name_dict["SIFT2"],name_dict["SIFT3"],name_dict["SIFT4"],name_dict["SIFT5"],name_dict["Feature merger"] ]
    dic_task_graph[name_dict["SIFT1"]] = [name_dict["Feature merger"]]
    dic_task_graph[name_dict["SIFT2"]] = [name_dict["Feature merger"]]
    dic_task_graph[name_dict["SIFT3"]] = [name_dict["Feature merger"]]
    dic_task_graph[name_dict["SIFT4"]] = [name_dict["Feature merger"]]
    dic_task_graph[name_dict["SIFT5"]] = [name_dict["Feature merger"]]
    dic_task_graph[name_dict["Feature merger"]] = [name_dict["Descaler"]]
    dic_task_graph[name_dict["Descaler"]] = [name_dict["Feature spiltter"]]
    dic_task_graph[name_dict["Feature spiltter"]] = [name_dict["Model matcher1"],name_dict["Model matcher2"],name_dict["Model matcher3"]]
    dic_task_graph[name_dict["Model matcher1"]] = [name_dict["Match joiner"]]
    dic_task_graph[name_dict["Model matcher2"]] = [name_dict["Match joiner"]]
    dic_task_graph[name_dict["Model matcher3"]] = [name_dict["Match joiner"]]
    dic_task_graph[name_dict["Match joiner"]] = [name_dict["Cluster spiltter"]]
    dic_task_graph[name_dict["Cluster spiltter"]] = [name_dict["Clustering1"],name_dict["Clustering2"]]
    dic_task_graph[name_dict["Clustering1"]] = [name_dict["Cluster joiner"]]
    dic_task_graph[name_dict["Clustering2"]] = [name_dict["Cluster joiner"]]
    dic_task_graph[name_dict["Cluster joiner"]] = [name_dict["RANSAC"]]
    dic_task_graph[name_dict["RANSAC"]] =[name_dict["Display"]]

    #print("Task Graph Obj_and_Pose Recognition\n",dic_task_graph)
    return dic_task_graph
def gesture_recognition_task_graph():
    name_dict = {"Source":0,
    "Copy":1,
    "L_Pair generator":2,
    "L_Scaler":3,
    "L_Tiler":4,
    "L_motionSIFT1":5,
    "L_motionSIFT2":6,
    "L_motionSIFT3":7,
    "L_motionSIFT4":8,
    "L_motionSIFT5":9,
    "L_Feature merger":10,
    "L_Descaler":11,
    "L_Copy":12,
    "R_Scaler":13,
    "R_Tiler":14,
    "R_Face detect1":15,
    "R_Face detect2":16,
    "R_Face detect3":17,
    "R_Face detect4":18,
    "R_Face merger":19,
    "R_Descaler":20,
    "R_Copy":21,
    "Display":22}
    dic_task_graph = dict()
    dic_task_graph[name_dict["Source"]]=[name_dict["Copy"]]
    dic_task_graph[name_dict["Copy"]] = [ name_dict["L_Pair generator"],name_dict["Display"],name_dict["R_Scaler"]]
    dic_task_graph[name_dict["L_Pair generator"]] = [ name_dict["L_Scaler"]]
    dic_task_graph[name_dict["L_Scaler"]] = [ name_dict["L_Tiler"],name_dict["L_Descaler"]]
    dic_task_graph[name_dict["L_Tiler"]] = [ name_dict["L_motionSIFT1"],name_dict["L_motionSIFT2"],name_dict["L_motionSIFT3"],
    name_dict["L_motionSIFT4"],name_dict["L_motionSIFT5"],name_dict["L_Feature merger"] ]
    dic_task_graph[name_dict["L_motionSIFT1"]] = [name_dict["L_Feature merger"]]
    dic_task_graph[name_dict["L_motionSIFT2"]] = [name_dict["L_Feature merger"]]
    dic_task_graph[name_dict["L_motionSIFT3"]] = [name_dict["L_Feature merger"]]
    dic_task_graph[name_dict["L_motionSIFT4"]] = [name_dict["L_Feature merger"]]
    dic_task_graph[name_dict["L_motionSIFT5"]] = [name_dict["L_Feature merger"]]
    dic_task_graph[name_dict["L_Feature merger"]] = [name_dict["L_Descaler"]]
    dic_task_graph[name_dict["L_Descaler"]] = [name_dict["L_Copy"]]
    dic_task_graph[name_dict["L_Copy"]] = [name_dict["Display"]]
    
    dic_task_graph[name_dict["R_Scaler"]] = [ name_dict["R_Tiler"],name_dict["R_Descaler"]]
    dic_task_graph[name_dict["R_Tiler"]] = [ name_dict["R_Face detect1"],name_dict["R_Face detect2"],name_dict["R_Face detect3"],
    name_dict["R_Face detect4"],name_dict["R_Face merger"] ]
    dic_task_graph[name_dict["R_Face detect1"]] = [name_dict["R_Face merger"]]
    dic_task_graph[name_dict["R_Face detect2"]] = [name_dict["R_Face merger"]]
    dic_task_graph[name_dict["R_Face detect3"]] = [name_dict["R_Face merger"]]
    dic_task_graph[name_dict["R_Face detect4"]] = [name_dict["R_Face merger"]]
    dic_task_graph[name_dict["R_Face merger"]] = [name_dict["R_Descaler"]]
    dic_task_graph[name_dict["R_Descaler"]] = [name_dict["R_Copy"]]
    dic_task_graph[name_dict["R_Copy"]] = [name_dict["Display"]]
    #print("Task Graph Gesture Recognition\n",dic_task_graph)
    return dic_task_graph
##
def create_single_dict_from_list_dicts(DAGs):
    big_DAG = dict()
    for DAG in DAGs:
        for k,v in DAG.items():
            big_DAG[k]=v
    return big_DAG

    # big_DAG = defaultdict(list)
    # for DAG in DAGs:
    #     for k,v in DAG.items():
    #         big_DAG[k]=v
    # return big_DAG
def distinct_element_dict(d):
    # returns a set(int)
    # d is a dictionary(key:int,value:)
    output = set()
    for k,v in d.items():
        output.add(k)
        for value in v:
            if value is not None:
                output.add(value)
    #print("SET:::",output)
    return output




def make_DAG(num_tasks,prob_edge):
    """
    Create single source single destination DAG
    """
    G_making=nx.gnp_random_graph(num_tasks,prob_edge,directed=True)
    G_directed = nx.DiGraph([(u,v) for (u,v) in G_making.edges() if u<v])
    
    set_out_going_nodes,set_in_going_nodes = set(),set() 
    for (u,v) in G_directed.edges():
         set_out_going_nodes.add(u)
         set_in_going_nodes.add(v)
    min_node_idx,max_node_idx = min(G_directed.nodes()),max(G_directed.nodes())     
    sources = set_out_going_nodes.difference(set_in_going_nodes)     
    destinations = set_in_going_nodes.difference(set_out_going_nodes)
    # if there are more than ONE source
    if len(sources)>1:
        for n in sources:
            G_directed.add_edge(min_node_idx-1, n)
    # if there are more than ONE destination
    if len(destinations)>1:
        for n in destinations:
            G_directed.add_edge(n,max_node_idx+1)

    # if FINAL source id is <1 ===> shift it to start from 0
    G_out = nx.DiGraph()
    if len(sources)>1 and min_node_idx==0:
        for (u,v) in G_directed.edges():
            G_out.add_edge(u+1,v+1)
    else:
        G_out = G_directed    

    if not nx.is_directed_acyclic_graph(G_out):
        raise NameError('it is NOT DAG!!!')
    #if not nx.is_directed_acyclic_graph(G_out):
    #    raise NameError('it is NOT DAG!!!')


    dag_start_from_0 = dict()
    for (u,v) in sorted(G_out.edges(),key=lambda x:[x[0],x[1]]):
        if u not in dag_start_from_0.keys():
            dag_start_from_0[u] = [v]
        else:
            dag_start_from_0[u].append(v)    
    
    # add destination into the DICTIONARY
    #dag_start_from_0[max(G_out.nodes())]=None    
    return dag_start_from_0
    
    # #print(nx.is_directed_acyclic_graph(DAG))
    # A = nx.to_numpy_matrix(DAG_create_from_edge)
    # dag_grapg = dict()#defaultdict(list)
    # outgoing_nodes,ingoing_nodes = set(),set()
    # for i in range(A.shape[0]):
    #     for j in range(A.shape[1]):
    #         if A[i,j]==1.0:
    #             outgoing_nodes.add(i)
    #             ingoing_nodes.add(j)
    #             if i not in dag_grapg.keys():
    #                 dag_grapg[i]=[j]
    #             else:
    #                 dag_grapg[i].append(j)
    # sources = outgoing_nodes.difference(ingoing_nodes)
    # destinations = ingoing_nodes.difference(outgoing_nodes)
    # if len(sources)>1:
    #     dag_grapg[-1] = list(sources)
    # if len(destinations)>1:
    #     for dest in destinations:
    #         if dest not in dag_grapg.keys():
    #             dag_grapg[dest]=[num_tasks]
    #         else:    
    #             dag_grapg[dest].append(num_tasks) 
    # if min(dag_grapg.keys())==0:
    #     return dag_grapg
    # dag_start_from_0 = dict()
    # for k in sorted(dag_grapg.keys()):
    #     dag_start_from_0[k+1] = [v+1 for v in dag_grapg[k]]
    
    # return dag_start_from_0

def make_DAG_MODIFIED(num_tasks,min_deg,max_deg,min_width,max_width,depth,num_of_ahead_layers_considered_for_link):
    """
    num_of_ahead_layers_considered_for_link (int) >= 1
    """
    #max_width
    max_width = min_width = int((num_tasks-1)*1.0/depth)

    nodes_has_incoming_edge = set()
    nodes_has_outcoming_edge = set()
    dict_layer_nodes = {0:[0]}
    index_starting_node = 0
    for i in range(1,depth):
        num_nodes_in_this_layer = random.randrange(min_width,max_width+1)
        dict_layer_nodes[i] = [k for k in range(index_starting_node+1,index_starting_node+num_nodes_in_this_layer+1)]
        index_starting_node += num_nodes_in_this_layer
    
    if index_starting_node+1 <num_tasks:
        dict_layer_nodes[depth] = [k for k in range(index_starting_node+1,num_tasks)]
        depth += 1
    dict_layer_nodes[depth] = [num_tasks]
    source,destination = 0, num_tasks
    print("Layers\n",dict_layer_nodes[max([k for k in dict_layer_nodes.keys()])][-1]+1, dict_layer_nodes)    

    dic_task_graph = dict()
    dic_task_graph[0] = dict_layer_nodes[1] # connect source to all layer-1 nodes
    nodes_has_outcoming_edge.add(0)
    nodes_has_incoming_edge.update(dict_layer_nodes[1])
    
    # connect destination to all nodes in previous nodes
    for node in  dict_layer_nodes[depth-1]:
        if node not in dic_task_graph.keys():
            dic_task_graph[node] = [destination]
        else:
            dic_task_graph[node].append(destination)
        
        nodes_has_outcoming_edge.add(node)
    nodes_has_incoming_edge.add(destination)

    for k in range(1,max(dict_layer_nodes.keys())-num_of_ahead_layers_considered_for_link):
        
        for v in dict_layer_nodes[k]:
            num_child = random.randrange(min_deg,max_deg+1)
            candidate_nodes_to_set_link = [x for x in range(dict_layer_nodes[k+1][0],dict_layer_nodes[k+num_of_ahead_layers_considered_for_link][-1])]
            print("num_child ",num_child, " candidate_nodes_to_set_link ", candidate_nodes_to_set_link)
            dic_task_graph[v] = random.sample(candidate_nodes_to_set_link,min(num_child,len(candidate_nodes_to_set_link)))
            
            nodes_has_outcoming_edge.add(v)
            nodes_has_incoming_edge.update(dic_task_graph[v])
            print("node ",v," children ",dic_task_graph[v])
    
    # make a connection between any node that has no incoming edge (then add a connection just to previous layer)    
    for l in range(1,depth):
        for node in dict_layer_nodes[l]:
            if node not in nodes_has_incoming_edge:
                rand_node_prev_layer = random.randint(dict_layer_nodes[l-1][0],dict_layer_nodes[l-1][-1])
                if rand_node_prev_layer not in dic_task_graph.keys():
                    dic_task_graph[rand_node_prev_layer]=[node]
                else:    
                    dic_task_graph[rand_node_prev_layer].append(node)

                nodes_has_outcoming_edge.add(rand_node_prev_layer)  # update both node and the node from prev layer that a link established between them
                nodes_has_incoming_edge.add(node)

    # make a connection between any node that has no incoming edge (then add a connection just to previous layer)    
    for l in range(depth-1,0,-1):
        for node in dict_layer_nodes[l]:
            if node not in nodes_has_outcoming_edge:
                rand_node_next_layer = random.randint(dict_layer_nodes[l+1][0],dict_layer_nodes[l+1][-1])
                if node not in dic_task_graph.keys():
                    dic_task_graph[node]=[rand_node_next_layer]
                else:
                    dic_task_graph[node].append(rand_node_next_layer)

                nodes_has_incoming_edge.add(rand_node_next_layer)  # update both node and the node from prev layer that a link established between them
                nodes_has_outcoming_edge.add(node)
    if len(nodes_has_incoming_edge)!=len(nodes_has_outcoming_edge) or len(nodes_has_incoming_edge)!=num_tasks or len(nodes_has_outcoming_edge)!=num_tasks:
        raise NameError('it is NOT correct!!!') 
    # for k in sorted(dic_task_graph.keys()):
    #     print("=",k,dic_task_graph[k]) 
    return dic_task_graph


def adjust_dict(sub_DAG,offset):
    output = dict()
    for k in sorted(sub_DAG.keys()):
        output[k+offset] = [v+offset for v in sub_DAG[k]]
    return output
#def create_big_DAG(num_tasks,num_graphs,prob_edge):
def create_big_DAG(num_tasks,num_graphs,prob_edge,min_deg,max_deg,min_width,max_width,depth,num_of_ahead_layers_considered_for_link,given_graph):
    """
    given_graph: list [True,name_of_task_graph]
    """
    starting_index_for_new_DAG = 0
    DAGs = []
    for i in range(num_graphs):
        if i==0:
            #sub_DAG = make_DAG(num_tasks,prob_edge)
            if given_graph[0]:
                if given_graph[1]=="face_recognition":
                    sub_DAG = face_recognition_task_graph()
                elif given_graph[1]=="obj_and_pose_recognition":
                    sub_DAG = obj_and_pose_recognition_task_graph()
                elif given_graph[1]=="gesture_recognition":
                    sub_DAG = gesture_recognition_task_graph()
                else:
                    raise NameError("Name of known task graph is NOT correct!")

            else:
                sub_DAG = make_DAG_MODIFIED(num_tasks,min_deg,max_deg,min_width,max_width,depth,num_of_ahead_layers_considered_for_link)
                    
        sub_DAG = adjust_dict(sub_DAG,starting_index_for_new_DAG)
        starting_index_for_new_DAG = len(distinct_element_dict(sub_DAG))
        DAGs.append(sub_DAG)
    return DAGs

def get_labes_for_small_DAG(small_DAG,comp_amount,speed,average_com_across_nodes):
    """
    - small_DAG(dictionary-key:int,value:list(int))
    - comp_amount (tensor of ALL(in bigger DAGs) tasks)
    - speed (tensor)
    - average_com_across_nodes (int)
    """
    dict_processing_tasks = dict()
    for i in list(distinct_element_dict(small_DAG)):
        x = comp_amount[0,i]
        dict_processing_tasks[i]= x.item()
    def commcost(ni, nj, A, B):
        if(A==B):
            return 0
        # elif ni==-1 or ni==-2 or nj==-1 or nj==-2:
        #     return 0.0000000000000001
        else:
            #global average_com_across_nodes
            return average_com_across_nodes
    def compcost(job,agent):
        #global new_P,E
        nonlocal dict_processing_tasks,speed
        execution_sp = speed[0,ord(agent)-97]
        x = dict_processing_tasks[job]/ execution_sp.item()   
        return x
    string_naming_machines_with_alphabets = ''
    for i in range(speed.shape[1]):
        string_naming_machines_with_alphabets += chr(97+i)

    start_time_heft =  time.time()   
    orders, jobson = schedule(small_DAG, string_naming_machines_with_alphabets, compcost, commcost)
    time_heft = time.time() - start_time_heft

    categorical_labels = [v for k,v in jobson.items()]
    le = preprocessing.LabelEncoder()
    targets = le.fit_transform(categorical_labels)
    #node_labels = torch.as_tensor(targets)
    print("targets",type(targets),targets)
    return time_heft,targets,targets.tolist()


def calculate_makespan(dict_assignment,dag,computations,speed,comm_matrix):
    """
    - dict_assignment(dict): key is task id, value is machine id
    - computations (list): computations
    - speed(list): executation speed
    - comm_matrix(list(list))
    """
    dict_comp = dict()
    dict_comm = dict()
    dict_total_time = dict()
    for t,l in dict_assignment.items():
        dict_comp[t]= computations[t]*1.0/speed[l]
    
    # set_out_going = set(dag.keys())
    # temp_list_values = []
    # for t in dag.keys():
    #     temp_list_values.extend(dag[t])
    # set_in_going = set(temp_list_values)
    # source = (set_out_going.difference(set_in_going)).pop()
    # dest = (set_in_going.difference(set_out_going)).pop()
    #print("dict_assignment",dict_assignment)
    source = min(dict_assignment.keys()) 
    dest=max(dict_assignment.keys())
    stack = [source]
    dict_total_time[source]=dict_comp[source]#0.0
    seen_tasks = set()
    #print("(makespan cal) DAG",dag,"\nassignment",dict_assignment)
    while stack:
        poped_task = stack.pop()
        for t in dag[poped_task]:
            if t not in dict_total_time.keys():
                #print("F-->poped_task",poped_task," t",t)
                #dict_total_time[t]=dict_comp[poped_task]+comm_matrix[dict_assignment[poped_task]][dict_assignment[t]]
                dict_total_time[t]=dict_total_time[poped_task]+comm_matrix[dict_assignment[poped_task]][dict_assignment[t]]

            else:
                #print("ELSE-->poped_task",poped_task," t",t)
                dict_total_time[t]=max(dict_total_time[t], dict_comp[poped_task]+comm_matrix[dict_assignment[poped_task]][dict_assignment[t]])

            if t!=dest and t not in seen_tasks:# in set_out_going:
                #print("Append-->poped_task",poped_task," t",t)
                stack.append(t)
                seen_tasks.add(t)
    #print("source and dest",source,dest,dict_total_time[dest])
    return dict_total_time[dest]





"""
=========================================================================================================
=========================================================================================================
==================================== Functions for TP-HEFT ==============================================
=========================================================================================================
=========================================================================================================
"""
def evalaute_time(M,C,E,P,dic_task_graph,dest_task):
    #M= M[:-1,:]
    new_M = np.zeros_like(M)
    max_time = 0
    dic_comp={k:[] for k in range(P.shape[0])}
    dic_comm={k:[] for k in range(P.shape[0])}
    sum_time={k:[] for k in range(P.shape[0])}
    min_all_same_task_total_sum_time= {k:0 for k in range(P.shape[0])}
    comptation_time_machines = np.zeros((len(E),))
    for j in range(M.shape[1]):
        #print(j,np.sum(M[:,j]*P[:,0]))
        #print(M.shape, P.shape, len(E))
        all_computation_on_this_machine = (M[:,j].T).dot(P)
        #print(j,all_computation_on_this_machine)
        comptation_time_machines[j]=all_computation_on_this_machine/E[j]# np.sum(M[:,j]*P[:,0])/E[j,0]
    #print(comptation_time_machines)

    
    
    for task in range(P.shape[0]):
        #find where task is, on what machine 
        if task!=dest_task:
            for j in range(M.shape[1]):
                if M[task,j]==1.0:
                    dic_comp[task].append([j,comptation_time_machines[j]])
                    #break

                    max_comm = 0
                    for child_task in dic_task_graph[task]:
                        for jj in range(M.shape[1]):
                            if M[child_task,jj] ==1.0:
                                if max_comm < C[j,jj]:
                                    max_comm = C[j,jj] 
                    dic_comm[task].append([j,max_comm])
                    sum_time[task].append([j,max_comm+comptation_time_machines[j]])
        
            #min_all_same_task_total_sum_time[task] = min([x[1] for x in sum_time[task]]) 
            temp_find_min_total_time = sorted(sum_time[task],key=lambda x:x[1])
            new_M[task,temp_find_min_total_time[0][0]]=1.0
            min_all_same_task_total_sum_time[task] = temp_find_min_total_time[0][1]
    #print("sum_time\n",sum_time)
    #print("min_all_same_task_total_sum_time\n",min_all_same_task_total_sum_time)
    #print("M\n",M)
    #print("dictionary_all\n",min_all_same_task_total_sum_time)
    #print(dic_task_graph)
    #print("E\n",E)
    #print("P\n",P)
    #print("M\n",M)
    #print("C\n",C)
    #print("comm\n",dic_comm)
    #print("comp\n",dic_comp)
    #print(sum_time)
    bottleneck_time = max([value for k,value in min_all_same_task_total_sum_time.items()])
    #bottleneck_time = max([value for k,value in sum_time.items()])
    #print(bottleneck_time)

    return new_M,bottleneck_time

def create_matrix_M_from_dict(mapping_TP,N,K):
    M = np.zeros((N,K))
    for k,v in mapping_TP.items():
        M[k,v]=1.0
    return M
def change_a_value_of_dict(input_dict,key_to_changed,new_value):
    output_dict = dict()
    for k in input_dict.keys():
        if k!=key_to_changed:
            output_dict[k]=copy.deepcopy(input_dict[k])
        else:
            output_dict[key_to_changed]=new_value
    return output_dict
def throughput_HEFT(E,P,C,dic_task_graph):
    # initialize all on the fastest machine
    #current_mapping_TP = np.zeors(())
    #np.random.randint(0,K-1,1)[0]
    E = np.array(E)
    P = np.array(P)
    C = np.array(C)
    N,K = P.shape[0],E.shape[0]
    print("N:",N," K:",K,"\nP",P)

    current_mapping_TP = {k:np.argmax(E) for k in dic_task_graph.keys()}
    # add destination task for initialization of assignments
    dest_task = list(distinct_element_dict(dic_task_graph).difference(set([k for k in dic_task_graph.keys()])))[0]
    current_mapping_TP[dest_task] = np.argmax(E)

    dest_task = list(distinct_element_dict(dic_task_graph).difference(set([k for k in dic_task_graph.keys()])))[0]


    print("INIT-->current_mapping_TP",current_mapping_TP)
    indices_E_TP = np.argsort(E)
    #M_TP = create_matrix_M_from_dict(current_mapping_TP,N,K)
    
    #_,current_val_TP = evalaute_time(M_TP,C,E,P,dic_task_graph,dest_task)
    print("P.tolist()",P.tolist())
    print("C.tolist()",C.tolist())
    current_val_TP = calculate_makespan(current_mapping_TP,dic_task_graph,P.tolist(),E.tolist(),C.tolist())
    
    continue_param = True
    while continue_param:
        best_val_TP = np.infty
        for k in dic_task_graph.keys():
            
            for ind_machine in range(K):
                if current_mapping_TP[k]!=ind_machine:
#                   print("machine",ind_machine,"tasks",dic_task_graph[k])
                    for ind_child_task in dic_task_graph[k]:
                    
                        #before_M_TP = create_matrix_M_from_dict(current_mapping_TP,N,K)
                        mapping_TP = change_a_value_of_dict(current_mapping_TP,ind_child_task,ind_machine)
                        #M_TP = create_matrix_M_from_dict(current_mapping_TP,N,K)
                        #_,new_value_TP = evalaute_time(M_TP,C,E,P,dic_task_graph,dest_task)
                        new_value_TP = calculate_makespan(mapping_TP,dic_task_graph,P.tolist(),E.tolist(),C.tolist())
                        if new_value_TP<best_val_TP:
                            best_val_TP,best_mapping_TP = new_value_TP,mapping_TP

                            current_mapping_TP=mapping_TP
                            #print("best move\n",create_matrix_M_from_dict(best_mapping_TP,len(dag.keys()),N,K))
        if best_val_TP > current_val_TP:
            current_val_TP,current_mapping_TP = best_val_TP,best_mapping_TP
            continue_param = True
        else:
            continue_param = False

    #M_current_mapping_TP =  create_matrix_M_from_dict(current_mapping_TP,N,K)

    output = []
    for k in sorted(current_mapping_TP.keys()):
        output.append(current_mapping_TP[k])
    print("MAPPING TP===>",output)
    return output#current_mapping_TP#M_current_mapping_TP




    # dict_comp = dict()
    # dict_comm = dict()
    # dict_total_time = dict()
    # for t,l in dict_assignment.items():
    #     dict_comp[t]= computations[t]*1.0/speed[l]
    
    # set_out_going = set(dag.keys())
    # temp_list_values = []
    # for t in dag.keys():
    #     temp_list_values.extend(dag[t])
    # set_in_going = set(temp_list_values)
    # source = (set_out_going.difference(set_in_going)).pop()
    # dest = (set_in_going.difference(set_out_going)).pop()
    
    # stack = [source]
    # dict_total_time[source]=0.0
    # while stack:
    #     poped_task = stack.pop()
    #     for t in dag[poped_task]:
    #         if t not in dict_total_time.keys():
    #             dict_total_time[t]=0.0
    #         else:
    #             dict_total_time[t]=max(dict_total_time[t], dict_comp[poped_task]+comm_matrix[dict_assignment[poped_task]][dict_assignment[t]])

    #         if t in set_out_going:
    #             stack.append(t)
    # #print("source and dest",source,dest,dict_total_time[dest])
    # return dict_total_time[dest]
def random_assingment(list_task,num_machines):
    # list_task(list int)
    # num_machines(int) 
    dict_random_assignment = dict()
    for t in list_task:
        random_machine = torch.randint(num_machines,(1,))
        dict_random_assignment[t] = random_machine.item()
    return dict_random_assignment



class KarateClubDataset(DGLDataset):
    def __init__(self,num_tasks,num_machines,num_graphs,prob_edge,min_deg,max_deg,min_width,max_width,depth,
        num_of_ahead_layers_considered_for_link,generate_LABEL=True,scheme_name_learned_from="heft",given_speed=None,given_comm=None,given_task=None):
        """
        prob_edge:      (float) prob of making an edge
        generate_LABEL: (bool)  should create LABELS for both nodes and edges
                                note: for large one-piece graph, make this False
        """
        self.num_tasks = num_tasks#10
        self.num_machines = num_machines#5 
        self.num_graphs = num_graphs#500
        self.prob_edge = prob_edge
        self.generate_LABEL = generate_LABEL
        self.scheme_name_learned_from = scheme_name_learned_from

        self.min_deg,self.max_deg,self.min_width,self.max_width,self.depth,self.num_of_ahead_layers_considered_for_link=min_deg,max_deg,min_width,max_width,depth,num_of_ahead_layers_considered_for_link
        self.given_speed,self.given_comm = given_speed,given_comm
        self.given_task = given_task
        super().__init__(name='karate_club')
        self.generate_LABEL = generate_LABEL

        self.actual_num_tasks = None
    def process(self):
        
        #DAGs = create_big_DAG(self.num_tasks,self.num_graphs,self.prob_edge)
        DAGs = create_big_DAG(self.num_tasks,self.num_graphs,self.prob_edge,self.min_deg,self.max_deg,self.min_width,self.max_width,self.depth,
            self.num_of_ahead_layers_considered_for_link,self.given_task)
        self.DAGs = DAGs
        DAG = create_single_dict_from_list_dicts(DAGs)
        #print("DA=======",len(distinct_element_dict(DAG)))
        comp_amount = .31*torch.rand(1,len(distinct_element_dict(DAG)))
        #print("comp_amount.shape",comp_amount.shape)
        if self.given_comm[0]:
            comm =self.given_comm[1]            
        else:
            comm = torch.rand(self.num_machines, self.num_machines)#,dtype=torch.float64)*1.0
            average_com_across_nodes = torch.mean(comm)
            for i in range(self.num_machines):
                comm[i,i] = 0.0
        if self.given_speed[0]:
            speed = self.given_speed[1]
        else:    
            speed = torch.rand(1,self.num_machines)#E = np.random.rand(num_machines,1) # E: execution speed of machines and is K by 1
        self.speed = speed
        self.comm = comm
        self.comp_amount = comp_amount
        

        node_features = torch.zeros([comp_amount.shape[1], self.num_machines])#, dtype=torch.float64)
        for i in range(comp_amount.shape[1]):
            node_features[i,:] = torch.div(comp_amount[0,i]*torch.ones(1, self.num_machines),speed)

        # categorical_labels = [v for k,v in jobson.items()]
        # le = preprocessing.LabelEncoder()
        # targets = le.fit_transform(categorical_labels)
        # node_labels = torch.as_tensor(targets)
        #print("DAGGG22",DAGs)
        if self.generate_LABEL:
            list_small_node_labels = []
            list_small_node_labels_LIST = []

            if self.scheme_name_learned_from == "heft":
                total_time = 0.0
                for small_DAG in DAGs:
                    #print("small_DAG",small_DAG)

                    this_time_heft, small_node_labels, small_node_labels_LIST= get_labes_for_small_DAG(small_DAG,comp_amount,speed,average_com_across_nodes)
                    total_time += this_time_heft

                    list_small_node_labels.extend(small_node_labels)
                    list_small_node_labels_LIST.append(small_node_labels_LIST)
                print("list_small_node_labels",list_small_node_labels,"\nlist_small_node_labels_LIST",list_small_node_labels_LIST)
                    #node_labels=torch.cat(node_labels,small_node_labels,1)
                    #print("Labeling for  is ",list_small_node_labels)
            else:
                total_time = 0.0
                print("Before TP HEFT ->self.comp_amount.shape",self.comp_amount)
                for small_DAG in DAGs:
                    print("====>>>>>>")
                    # list_t1=list(distinct_element_dict(small_DAG))
                    # indices_selected = torch.LongTensor(list_t1)#torch.tensor(list_t1)
                    # print("indices_selected",indices_selected,"comp size",comp_amount.shape)
                    
                    # selected_ind_comp_amount = torch.index_select(comp_amount, 1, indices_selected)
                    
                    # set_out_going_nodes_TP = set([x for x in small_DAG.keys()])
                    # set_all_nodes_TP  = distinct_element_dict(small_DAG)
                    # dest_task = list(set_all_nodes_TP.difference(set_out_going_nodes_TP))[0]
                    # print("dest_task",dest_task)
                    


                    start_time_TP_HEFT = time.time()
                    print("DAGs",small_DAG)
                    #print("speed.tolist()[0]",speed.tolist())
                    #print(comm.tolist())
                    small_node_labels = throughput_HEFT(speed.tolist()[0],comp_amount.tolist()[0],comm.tolist(),small_DAG)
                    total_time += (time.time() - start_time_TP_HEFT)
                    print("list_small_node_labels",small_node_labels)

                    list_small_node_labels.extend(small_node_labels)
                    list_small_node_labels_LIST.append(small_node_labels)
            #node_labels=torch.cat(list_small_node_labels,1)
            
            self.list_small_node_labels_LIST =list_small_node_labels_LIST
            self.total_time = total_time
            #print("hoooy",self.list_small_node_labels_LIST)
            node_labels = torch.as_tensor(list_small_node_labels)
        
        
        
        total_num_edges = sum([len(v) for k,v in DAG.items()])
        edge_features = torch.zeros(total_num_edges,self.num_machines**2)#,dtype=torch.float64)
        ###
        if self.generate_LABEL:
            edge_label = torch.zeros(total_num_edges,dtype=torch.int64)
        ###
        edges_src = torch.zeros(total_num_edges,dtype=torch.int64)
        edges_dst = torch.zeros(total_num_edges,dtype=torch.int64)
        
        ##########
        #to_be_edge_comm = torch.ones(1,num_machines*num_machines,dtype=torch.float64)
        ##########
        comm_in_one_row = torch.reshape(comm,(1,self.num_machines**2)) #comm_in_one_row.repeat(4, 1) #
        bandwidth = torch.zeros(1,self.num_machines**2)#torch.reshape(comm,(1,self.num_machines**2))
        for i in range(comm_in_one_row.shape[1]):
            x = comm_in_one_row[0,i]
            if x.item()==0:
                bandwidth[0,i] = 10**9
            else:
                bandwidth[0,i] =1/x.item()
        # print(">>>>>>>>>",self.comm)
        #print(bandwidth)
        #print("comm",comm,'\n',"comm_in_one_row",comm_in_one_row,'\n',"edge_features",edge_features)
        index = 0
        for k in sorted(DAG.keys()):
            for v in DAG[k]:
                #print(comm_in_one_row)
                edge_features.index_copy_(0,torch.tensor([index]),comm_in_one_row)
                #print('after',edge_features[index,:])
                edges_src[index] = k
                edges_dst[index] = v
                if self.generate_LABEL:
                    #print("index ",index,"-k ",k," node_labels.size",node_labels)
                    edge_label[index] = node_labels[k]
                index +=1
        
        # index = 0
        # for small_DAG_labels in self.list_small_node_labels_LIST:
        #     for label in small_DAG_labels:
        #         print()
        #         edge_label[index] = label
    
        #self.graph.edata['hel'] = edge_label
        # edge_features = torch.from_numpy(edges_data['Weight'].to_numpy())
        # edges_src = torch.from_numpy(edges_data['Src'].to_numpy())
        # edges_dst = torch.from_numpy(edges_data['Dst'].to_numpy())
        
        # print("==>>>>>>   comp_amount.shape[1]",comp_amount.shape[1],"distinct_element_dict",distinct_element_dict(DAG))
        # Suppose_to_be_empty = set(DAG.keys()).difference(set([x for x in range(len(DAG.keys()))]))

        # print("DIFF", set(DAG.keys()).difference(set([k for k in DAG])))
        # print("KEYS DAG",DAG.keys())
        # print("LEN KEYS",len(DAG.keys()),len(distinct_element_dict(DAG)))
        # print("range",[x for x in range(len(DAG.keys()))])
        # print("dag--->",DAG)
        # if len(Suppose_to_be_empty)>0:
        #     #print()
        #     raise NameError(len(Suppose_to_be_empty))
        self.graph = dgl.graph((edges_src, edges_dst), num_nodes=comp_amount.shape[1])
        #print("self.graph\n",self.graph)
        self.graph.ndata['hn_in'] = node_features
        self.graph.edata['he'] = edge_features
        #print("self.node_features\n", self.graph.ndata)
        if self.generate_LABEL:
             
            self.graph.ndata['hnl'] = node_labels
            self.graph.edata['hel'] = edge_label#torch.range(0, total_num_edges-1,dtype=torch.int64)
            self.node_labels = node_labels
            print("YES (IF) - > SHAPES",self.graph.ndata['hnl'].shape,self.graph.edata['hel'].shape,self.generate_LABEL)
        else:
            """
            Assigning something as labels (bcause FORWARD func needs some labels!!!)
            """
            self.graph.ndata['hnl'] = torch.randint(self.num_machines,(self.comp_amount.shape[1],))
            #torch.as_tensor([x for x in range(self.graph.ndata['hn_in'].shape[0])])
            self.graph.edata['hel'] = torch.randint(self.num_machines,(total_num_edges,))
            print("No (IF) - > SHAPES",self.graph.ndata['hnl'].shape,self.graph.edata['hel'].shape,self.generate_LABEL)
        # If your dataset is a node classification dataset, you will need to assign
        # masks indicating whether a node belongs to training, validation, and test set.
        self.total_num_edges=total_num_edges
        # n_nodes = comp_amount.shape[1]
        # n_train = int(n_nodes * 0.6)
        # n_val = int(n_nodes * 0.2)
        # train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        # val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        # test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        # train_mask[:n_train] = True
        # val_mask[n_train:n_train + n_val] = True
        # test_mask[n_train + n_val:] = True
        # self.graph.ndata['train_mask'] = train_mask
        # self.graph.ndata['val_mask'] = val_mask
        # self.graph.ndata['test_mask'] = test_mask
        
        #print("Train ---- Test----Val",train_mask,test_mask,val_mask)
    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1

    def get_labels(self):
        return self.node_labels

    def get_list_small_node_labels_LIST(self):
        return self.list_small_node_labels_LIST
    def get_num_machines(self):
        #print("hoooy",self.num_machines)
        return self.num_machines
    def get_total_num_tasks(self):
        return self.comp_amount.shape[1]
    def get_total_num_edges(self):
        self.total_num_edges
    def get_DAGs(self):
        return self.DAGs
    def get_generate_LABEL(self):
        return self.generate_LABEL
    def get_num_graphs(self):
        return self.num_graphs
    def get_total_time(self):
        return self.total_time
    def get_scheme_name_learned_from(self):
        return self.scheme_name_learned_from
    def is_scheme_HEFT(self):
        if self.scheme_name_learned_from == 'heft':
            return True
        return False    
    def get_setting(self):
        #print("SETTING",self.speed,self.comm,self.comp_amount)
        return self.speed,self.comm,self.comp_amount

MODULE = 'edgnn.core.data.{}'
AVAILABLE_DATASETS = {
    'dglrgcn',
    'dortmund'
}



def get_makespan_comparison_result(input_dataset,predicted_labels_test,ratio_num_graph_starting_point):
    print("predicted_labels_test",predicted_labels_test)
    rho = ratio_num_graph_starting_point
    DAGs_dict = input_dataset.get_DAGs()
    # create list of list task in each graph as well as list of list predicted labels
    whole_list_task = [list(distinct_element_dict(dag)) for dag in DAGs_dict[int(len(DAGs_dict)*rho):]]
    if input_dataset.get_generate_LABEL():
        #if input_dataset.is_scheme_HEFT():
        all_labels_HEFT = input_dataset.get_list_small_node_labels_LIST()
        desired_labels_HEFT = [list_labels_HEFT for list_labels_HEFT in all_labels_HEFT[int(len(all_labels_HEFT)*rho):]]
        #else:

    whole_pred_labels = predicted_labels_test.tolist()
    list_list_pred_labels = []#predicted_labels_test.tolist()
    #print("list_all_test_tasks",whole_list_task)
    ind_where_to_select = 0

    all_makespan = []
    speed,comm_matrix,computations=input_dataset.get_setting()
    #print("desired_labels_HEFT",desired_labels_HEFT)
    ave_makespan,ave_makespan_rand,ave_makespan_heft,ave_makespan_TP_HEFT = 0.0,0.0,0.0,0.0
    for ind_list,list_tasks in enumerate(whole_list_task):
        dict_assignment = dict()
        dict_assignment_HEFT_OR_TP_HEFT = dict()

        list_pred_labels = whole_pred_labels[ind_where_to_select:ind_where_to_select+len(list_tasks)]
        list_list_pred_labels.append(list_pred_labels)
        ind_where_to_select += len(list_pred_labels)
        for ind_task,task in enumerate(list_tasks):
            print("task",task," ind_task",ind_task, "list_pred_labels",list_pred_labels)
            dict_assignment[task]= list_pred_labels[ind_task]
            #print("desired_labels_HEFT[ind_list][ind_task]",desired_labels_HEFT[ind_list][ind_task])
            if input_dataset.get_generate_LABEL():
                dict_assignment_HEFT_OR_TP_HEFT[task]=desired_labels_HEFT[ind_list][ind_task]
        # calculatte makespan
        print("calculate_makespan FOR ----> Rand")
        dict_assignment_RAND = random_assingment(list_tasks,input_dataset.get_num_machines())
        makespan_random = calculate_makespan(dict_assignment_RAND,DAGs_dict[int(len(DAGs_dict)*rho):][ind_list],computations.tolist()[0],speed.tolist()[0],comm_matrix.tolist())
        
        print("calculate_makespan FOR ----> GCN")
        makespan = calculate_makespan(dict_assignment,DAGs_dict[int(len(DAGs_dict)*rho):][ind_list],computations.tolist()[0],speed.tolist()[0],comm_matrix.tolist())
        if input_dataset.get_generate_LABEL():
            print("calculate_makespan FOR ----> ", input_dataset.get_scheme_name_learned_from())
            makespan_HEFT = calculate_makespan(dict_assignment_HEFT_OR_TP_HEFT,DAGs_dict[int(len(DAGs_dict)*rho):][ind_list],computations.tolist()[0],speed.tolist()[0],comm_matrix.tolist())
        
        ave_makespan +=makespan
        ave_makespan_rand += makespan_random
        if input_dataset.get_generate_LABEL():
            ave_makespan_heft += makespan_HEFT
            print("MAKESPAN----",makespan_random,makespan,makespan_HEFT)
        else:
            print("MAKESPAN----",makespan_random,makespan)
        all_makespan.append(makespan)

    ave_makespan /=len(whole_list_task)
    ave_makespan_rand /=len(whole_list_task)
    if input_dataset.get_generate_LABEL():
        ave_makespan_heft /=len(whole_list_task)
        print("AVE MAKESPAN:",ave_makespan_rand,ave_makespan,ave_makespan_heft)   
    else:
        ave_makespan_heft = None
        print("AVE MAKESPAN:",ave_makespan_rand,ave_makespan)

    return ave_makespan_rand,ave_makespan,ave_makespan_heft
    # print("whole\n",whole_pred_labels)    
    # print("list_pred_labels_all_test_tasks\n",list_list_pred_labels)
    # print("Comm",comm_matrix)

def run_model(arg_config_fpath: str,
              arg_data_path: str,
              arg_dataset: str,
              arg_gpu: int = -1,
              arg_lr: float = 1e-3,
              arg_n_epochs: int = 200,
              arg_weight_decay: float = 5e-4,
              arg_batch_size: int = 16) -> None:
    Large_scale = True
    #ref_scheme = 'TP'
    num_of_all_machines = 6  
    min_deg,max_deg,min_width,max_width,depth,num_of_ahead_layers_considered_for_link = 3,4,5,5,9,2#3,5,3,3,6,1

    #dataset = KarateClubDataset(20,num_of_all_machines,40,0.25,True,'heft')
    #40,num_of_all_machines,10,0.4
    dataset = KarateClubDataset(50,num_of_all_machines,10,0.4,min_deg,max_deg,min_width,max_width,depth,
        num_of_ahead_layers_considered_for_link,True,'TP',[False,None],[False,None],[True,"face_recognition"])
    #dataset = KarateClubDataset(20,num_of_all_machines,10,0.4,True,'TP')
    speed_used_for_training,comm_used_for_training,_ = dataset.get_setting()
    graph = dataset[0]

    print(graph)
    #return 0

    if arg_gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(arg_gpu)

    default_path = create_default_path()
    print('\n*** Set default saving/loading path to:', default_path)

    if arg_dataset == AIFB or arg_dataset == MUTAG:
        module = importlib.import_module(MODULE.format('dglrgcn'))
        data = module.load_dglrgcn(arg_data_path)
        print("Injaaaaa",data)
        
        data = to_cuda(data) if cuda else data
        print("---------========== before =============-------------")
        config_params = read_params(arg_config_fpath, verbose=True)
        print("*********g",type(data[GRAPH].edata),data[GRAPH].edata,'\n',"*********g",data[GRAPH],'\n',
          "*********config_params",config_params[0],'\n',
          "*********n_classes",data[N_CLASSES],'\n'
          ,"*********n_rels",data[N_RELS],'\n',
          "*********n_entities",data[N_ENTITIES] if N_ENTITIES in data else None
          )
        #data = dataset
        data = dict()
        data[GRAPH] = graph
        # del data[GRAPH].ndata['feat']
        # del data[GRAPH].ndata['label']
        # del data[GRAPH].ndata['train_mask']
        # del data[GRAPH].ndata['val_mask']
        # del data[GRAPH].ndata['test_mask']
        data[N_RELS] = graph.num_edges()
        data[N_ENTITIES] = graph.num_nodes()
        data[N_CLASSES] = dataset.get_num_machines()
        data['labels'] = dataset.get_labels()


        n_nodes = graph.num_nodes()
        all_DAGs = dataset.get_DAGs()
        print(all_DAGs,len(all_DAGs))

        if len(all_DAGs)>1:
            n_train = sum([len(distinct_element_dict(dag)) for dag in all_DAGs[:int(len(all_DAGs)*.6)]])#    int(n_nodes * 0.6)
            n_val = sum([len(distinct_element_dict(dag)) for dag in all_DAGs[int(len(all_DAGs)*.6):int(len(all_DAGs)*.8)]])#      int(n_nodes * 0.2)
        else:
            n_train = int(n_nodes*.6)
            n_val = int(n_nodes*.2)
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[:n_train] = True
        val_mask[n_train:n_train + n_val] = True
        test_mask[n_train + n_val:] = True
        data['train_mask'] = train_mask
        data['val_mask'] = val_mask
        data['test_mask'] = test_mask


        #data['train_mask']=graph.ndata['train_mask']
        #data['val_mask']= graph.ndata['val_mask']
        #data['test_mask'] = graph.ndata['test_mask']
        #data[GRAPH].edata['hel'] = graph.edata['weight']
        #data[GRAPH].ndata['hnl'] = graph.ndata['feat']
        #print(data[GRAPH].edata['hel'])
        #return 0
        print("--------- after -------------")
        print("*********g",data[GRAPH],'\n',
          "*********config_params",config_params[0],'\n',
          "*********n_classes",data[N_CLASSES],'\n'
          ,"*********n_rels",data[N_RELS],'\n',
          "*********n_entities",data[N_ENTITIES])
        
        mode = NODE_CLASSIFICATION
    elif arg_dataset == MUTAGENICITY or arg_dataset == PTC_MR or arg_dataset == PTC_MM or arg_dataset == PTC_FR or arg_dataset == PTC_FM:
        module = importlib.import_module(MODULE.format('dortmund'))
        data = module.load_dortmund(arg_data_path)
        data = to_cuda(data) if cuda else data
        mode = GRAPH_CLASSIFICATION
    else:
        raise ValueError('Unable to load dataset', arg_dataset)

    print_graph_stats(data[GRAPH])

    config_params = read_params(arg_config_fpath, verbose=True)

    new_dict_config = config_params[0]
    new_dict_config['edge_dim'] = (dataset.get_num_machines())**2#25
    new_dict_config['node_dim'] = (dataset.get_num_machines())#5
    print("dim---",new_dict_config['edge_dim'])
    # new_dict_config['edge_dim'] = True
    # new_dict_config['node_dim'] = True
    # del new_dict_config['edge_one_hot']
    # del new_dict_config['node_one_hot']
    config_params[0] = new_dict_config
    print(config_params[0])

    #create GNN model
    
    # model = Model(g=data[GRAPH],
    #               config_params=config_params[0],
    #               n_classes=data[N_CLASSES],
    #               n_rels=data[N_RELS] if N_RELS in data else None, # dim edge
    #               n_entities=data[N_ENTITIES] if N_ENTITIES in data else None,
    #               is_cuda=cuda,
    #               mode=mode)

    if cuda:
        model.cuda()

    # 1. Training
    app = App()
    learning_config = {'lr': arg_lr, 'n_epochs': arg_n_epochs, 'weight_decay': arg_weight_decay, 'batch_size': arg_batch_size, 'cuda': cuda}
    print('\n*** Start training ***\n',arg_n_epochs)

    #arg_n_epochs  = 
    print("injaa->",data)
    app.train(data, config_params[0], learning_config, default_path, mode=mode)

    # 2. Testing
    print('\n*** Start testing ***\n')
    acc_test,predicted_labels_test,golden_labels_test = app.test(data, default_path, mode=mode)
    print("===== GOLDEN --- and --- predicted")
    print(golden_labels_test,'\n',predicted_labels_test)

    ave_makespan_rand,ave_makespan,ave_makespan_heft = get_makespan_comparison_result(dataset,predicted_labels_test,0.8)
    # return 0

    # test for TP HEFT
    # dataset_TP = KarateClubDataset(100,num_of_all_machines,10,.4,True,'TP')
    # print("graph",dataset_TP[0])
    # h_embd_TP = app.model.JUST_forward(dataset_TP[0])
    # _, indices_TP = torch.max(h_embd_TP, dim=1)
    # print('\n*** Start testing ***\n')
    # ave_makespan_rand,ave_makespan,ave_makespan_heft = get_makespan_comparison_result(dataset_TP,indices_TP,0)
    
    # =============================
    # for FACE and OBJ and POSE task graphs
    num_node = 30#,300,400]#, 110]
    prob = 0.3
    num_graphs_inference = 1
    min_deg,max_deg,min_width,max_width,depth = 3,4,5,5,9

    # Save mode
    torch.save(app.model.state_dict(), pathlib.Path.cwd().joinpath("model"))

    print("-------------------------- FACE Recognition -----------------------")
    dataset_SMALL = KarateClubDataset(num_node,num_of_all_machines,num_graphs_inference,prob,min_deg,max_deg,min_width,max_width,depth,
        num_of_ahead_layers_considered_for_link,True,'TP',[True,speed_used_for_training],[True,comm_used_for_training],[True,"face_recognition"])
    graph_SMALL = dataset_SMALL[0]
    start_time_Inference = time.time() # get Inference time
    h_embd = app.model.JUST_forward(graph_SMALL)
    _, indices = torch.max(h_embd, dim=1)
    end_time_Inference = time.time()
    result_inference_time = (end_time_Inference-start_time_Inference)/(1.0*num_graphs_inference)
    result_time_TP = dataset_SMALL.get_total_time()
    result_makespan_RAND,result_makespan,result_makespan_TP = get_makespan_comparison_result(dataset_SMALL,indices,0)
    print("result_makespan_RAND",result_makespan_RAND)
    print("result_makespan_TP",result_makespan_TP)
    print("result_makespan",result_makespan)
    print("Inference Time",result_inference_time)
    print("TP Time",result_time_TP)

    print("-------------------------- POSE Recognition -----------------------")
    dataset_SMALL = KarateClubDataset(num_node,num_of_all_machines,num_graphs_inference,prob,min_deg,max_deg,min_width,max_width,depth,
        num_of_ahead_layers_considered_for_link,True,'TP',[True,speed_used_for_training],[True,comm_used_for_training],[True,"obj_and_pose_recognition"])
    graph_SMALL = dataset_SMALL[0]
    start_time_Inference = time.time() # get Inference time
    h_embd = app.model.JUST_forward(graph_SMALL)
    _, indices = torch.max(h_embd, dim=1)
    end_time_Inference = time.time()
    result_inference_time = (end_time_Inference-start_time_Inference)/(1.0*num_graphs_inference)
    result_time_TP = dataset_SMALL.get_total_time()
    result_makespan_RAND,result_makespan,result_makespan_TP = get_makespan_comparison_result(dataset_SMALL,indices,0)
    print("result_makespan_RAND",result_makespan_RAND)
    print("result_makespan_TP",result_makespan_TP)
    print("result_makespan",result_makespan)
    print("Inference Time",result_inference_time)
    print("TP Time",result_time_TP)

    print("-------------------------- Gesture Recognition -----------------------")
    dataset_SMALL = KarateClubDataset(num_node,num_of_all_machines,num_graphs_inference,prob,min_deg,max_deg,min_width,max_width,depth,
        num_of_ahead_layers_considered_for_link,True,'TP',[True,speed_used_for_training],[True,comm_used_for_training],[True,"gesture_recognition"])
    graph_SMALL = dataset_SMALL[0]
    start_time_Inference = time.time() # get Inference time
    h_embd = app.model.JUST_forward(graph_SMALL)
    _, indices = torch.max(h_embd, dim=1)
    end_time_Inference = time.time()
    result_inference_time = (end_time_Inference-start_time_Inference)/(1.0*num_graphs_inference)
    result_time_TP = dataset_SMALL.get_total_time()
    result_makespan_RAND,result_makespan,result_makespan_TP = get_makespan_comparison_result(dataset_SMALL,indices,0)
    print("result_makespan_RAND",result_makespan_RAND)
    print("result_makespan_TP",result_makespan_TP)
    print("result_makespan",result_makespan)
    print("Inference Time",result_inference_time)
    print("TP Time",result_time_TP)
    return 0

def main():
    parser = argparse.ArgumentParser(description='Run graph neural networks.')
    register_data_args(parser)
    parser.add_argument("--config_fpath", type=str, required=True, 
                        help="Path to JSON configuration file.")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path from where to load the data (assuming they were preprocessed beforehand).")
    parser.add_argument("--gpu", type=int, default=-1, help="gpu")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--batch-size", type=int, default=16, help="batch size (only for graph classification)")

    args = parser.parse_args()

    run_model(
        arg_config_fpath=args.config_fpath,
        arg_data_path=args.data_path,
        arg_gpu=args.gpu,
        arg_lr=args.lr,
        arg_n_epochs=args.n_epochs,
        arg_weight_decay=args.weight_decay,
        arg_batch_size=args.batch_size,
    )


if __name__ == '__main__':
    main()