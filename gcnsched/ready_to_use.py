#!/usr/bin/env python3
"""
Run model script.
"""
from typing import Dict, List
import torch
import importlib

from edgnn.core.models.constants import NODE_CLASSIFICATION
from edgnn.core.app import App
from edgnn.utils.io import read_params

import dgl
from dgl.data import DGLDataset
import torch
import random
import networkx as nx

import time
import pathlib
import networkx as nx


thisdir = pathlib.Path(__file__).resolve().parent

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
    
def create_big_DAG(num_tasks,num_graphs,prob_edge,min_deg,max_deg,min_width,max_width,
    depth,num_of_ahead_layers_considered_for_link,given_graph,type_of_created_DAG):
    """
    given_graph: list [True,name_of_task_graph]
    """
    starting_index_for_new_DAG = 0
    DAGs = []
    for i in range(num_graphs):
        if i==0:
            #sub_DAG = make_DAG(num_tasks,prob_edge)
            print("---->>",given_graph)
            if given_graph[0]:
                if given_graph[1]=="face_recognition":
                    sub_DAG = face_recognition_task_graph()
                elif given_graph[1]=="obj_and_pose_recognition":
                    sub_DAG = obj_and_pose_recognition_task_graph()
                elif given_graph[1]=="gesture_recognition":
                    sub_DAG = gesture_recognition_task_graph()
                elif given_graph[1]=="designed":
                    sub_DAG = given_graph[2]
                else:
                    raise NameError("Name of known task graph is NOT correct!")

            else:
                if type_of_created_DAG == "depth-wise":
                    sub_DAG = make_DAG_MODIFIED(num_tasks,min_deg,max_deg,min_width,max_width,depth,num_of_ahead_layers_considered_for_link)
                elif type_of_created_DAG == "EP":
                    sub_DAG = make_DAG(num_tasks,prob_edge)
                else:
                    raise NameError("Name of type_of_created_DAG is NOT correct!")
                    
        sub_DAG = adjust_dict(sub_DAG,starting_index_for_new_DAG)
        starting_index_for_new_DAG = len(distinct_element_dict(sub_DAG))
        DAGs.append(sub_DAG)
    return DAGs


class KarateClubDataset(DGLDataset):
    def __init__(self,num_tasks,num_machines,num_graphs,prob_edge,min_deg,max_deg,min_width,max_width,depth,
        num_of_ahead_layers_considered_for_link,generate_LABEL=True,scheme_name_learned_from="heft",
        given_speed=None,given_comm=None,given_comp=None,given_task=None,type_of_created_DAG = "depth-wise"):
        
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
        self.given_speed,self.given_comm,self.given_comp = given_speed,given_comm,given_comp
        self.given_task = given_task
        self.type_of_created_DAG = type_of_created_DAG
        super().__init__(name='karate_club')
        self.generate_LABEL = generate_LABEL

        self.actual_num_tasks = None
    def process(self):
        
        #DAGs = create_big_DAG(self.num_tasks,self.num_graphs,self.prob_edge)
        DAGs = create_big_DAG(self.num_tasks,self.num_graphs,self.prob_edge,self.min_deg,self.max_deg,self.min_width,self.max_width,self.depth,
            self.num_of_ahead_layers_considered_for_link,self.given_task,self.type_of_created_DAG)
        self.DAGs = DAGs
        DAG = create_single_dict_from_list_dicts(DAGs)
        #print("DA=======",len(distinct_element_dict(DAG)))
        if self.given_comp[0]:
            comp_amount = self.given_comp[1] #.31*torch.rand(1,len(distinct_element_dict(DAG)))
        else:
            comp_amount = .31*torch.rand(1,len(distinct_element_dict(DAG)))
 
        #comp_amount = .31*torch.rand(1,len(distinct_element_dict(DAG)))
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

                    this_time_heft, small_node_labels, small_node_labels_LIST= get_labes_for_small_DAG(small_DAG,comp_amount,speed,comm)
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
            node_labels =torch.as_tensor(list_small_node_labels)
            #node_labels = torch.ones()
        
        
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

def find_schedule(num_of_all_machines: int,
                  num_node: int,
                  input_given_speed: torch.Tensor,
                  input_given_comm: torch.Tensor,
                  input_given_comp: torch.Tensor,
                  input_given_task_graph: Dict[int, List[int]]) -> torch.Tensor:
      
    prob = 0.25
    #min_deg,max_deg,min_width,max_width,depth = 2,2,10,10,10#2,3,5,5,7


    min_deg,max_deg,min_width,max_width,depth,num_of_ahead_layers_considered_for_link = 1,2,10,10,40,1#3,5,3,3,6,1
    num_graphs_inference = 1
    dataset_SMALL =KarateClubDataset(num_node,num_of_all_machines,num_graphs_inference,prob,min_deg,
                    max_deg,min_width,max_width,depth,num_of_ahead_layers_considered_for_link,False,'heft',
                    [True,input_given_speed],[True,input_given_comm],[True,input_given_comp],[True,"designed",input_given_task_graph])

    module = importlib.import_module(MODULE.format('dglrgcn'))
    mode = NODE_CLASSIFICATION 
    config_params = read_params(str(thisdir.joinpath("config_files/config_edGNN_node_class.json")), verbose=True)
    new_dict_config = config_params[0]
    new_dict_config['edge_dim'] = (dataset_SMALL.get_num_machines())**2#25
    new_dict_config['node_dim'] = (dataset_SMALL.get_num_machines())#5
    print("dim---",new_dict_config['edge_dim'])
    # new_dict_config['edge_dim'] = True
    # new_dict_config['node_dim'] = True
    # del new_dict_config['edge_one_hot']
    # del new_dict_config['node_one_hot']
    config_params[0] = new_dict_config
    
    learning_config = {'lr': 0.005, 'n_epochs': 80, 'weight_decay': 0, 'batch_size': 16, 'cuda': 0}
 

    graph_SMALL = dataset_SMALL[0]
    new_app = App()
                
    input_num_rels = graph_SMALL.num_edges()
    input_num_entities = graph_SMALL.num_nodes()
    input_num_classes = dataset_SMALL.get_num_machines()
    new_app.data_and_model_transfer(
        graph_SMALL,
        input_num_classes,
        input_num_entities,
        input_num_rels,
        config_params[0], 
        learning_config, 
        str(thisdir.joinpath("saved_model")),
        mode=mode
    )
                
    h_embd = new_app.model.JUST_forward(graph_SMALL)
                
    _, indices = torch.max(h_embd, dim=1)
    print(indices)
    return indices 


if __name__ == "__main__":
   num_of_all_machines = 6 
   num_node = 40
   dict_task_graph = face_recognition_task_graph()
   input_given_speed = torch.rand(1,num_of_all_machines)
   input_given_comm = torch.rand(num_of_all_machines, num_of_all_machines)
   input_given_comp = .31*torch.rand(1,len(distinct_element_dict(dict_task_graph))) 
   find_schedule(num_of_all_machines ,num_node ,input_given_speed,input_given_comm,input_given_comp,dict_task_graph)
    
