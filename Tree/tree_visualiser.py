#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 23:05:22 2017

@author: mjb
"""

# Generates visual representations of game trees.
# @classmod tree_visualiser
import torch
import Settings.game_settings as game_settings
import Game.card_to_string as card_to_string
import Settings.constants as constants
from Game.card_tools import card_tools
card_tools = card_tools()

import Settings.arguments as arguments


from graphviz import Source
#TODO: README
#dot tree_2.dot -Tpng -O

class TreeVisualiser:
    # Constructor
    def __init__(self):
        self.node_to_graphviz_counter = 0
        self.edge_to_graphviz_counter = 0
    
    # Generates a string representation of a tensor.
    # @param tensor a tensor
    # @param[opt] name a name for the tensor
    # @param[opt] format a format string to use with @{string.format} for each
    # element of the tensor
    # @param[opt] labels a list of labels for the elements of the tensor
    # @return a string representation of the tensor
    # @local
    def add_tensor(self, tensor, name, formatstr = '{:.2f}',labels = None):
      
      out = ''
      if name != "":
        out = '| ' + name + ': '
    
      for i in range(tensor.size(0)):
        for j in range(tensor.size(1)):
            if labels:
                out = out + labels[j] + ":"
        
            out = out + formatstr.format(tensor[i][j]) + ", "
        out = out + "| : "
      
      return out
      
    def add_list(self, input_list, name, formatstr = '{:.2f}'):
      
      out = ''
      if name != "":
        out = '| ' + name + ': '
    
      for i in range(len(input_list)):
        for j in range(len(input_list[i])):
#        if labels:
#          out = out + labels[i] + ":"
        
            out = out + formatstr.format(input_list[i][j]) + ", "
        out = out + "| : "
      
      return out
      
    
    # Generates a string representation of any range or value fields that are set
    # for the given tree node.
    # @param node the node
    # @return a string containing concatenated representations of any tensors
    # stored in the `ranges_absolute`, `cf_values`, or `cf_values_br` fields of
    # the node.
    # @local
#    def add_range_info(self, node):
#      out = ""
#      
##      if(node.ranges_absolute.dim() != 0): 
#      if(node.ranges_absolute.dim() != 0): 
#        out = out + self.add_tensor(node.ranges_absolute, 'abs_range')
#    
#      if(node.cf_values.dim() != 0):
#        #cf values computed by real tree dfs
#        out = out + self.add_tensor(node.cf_values, 'cf_values')
#
#      if(node.cf_values_br.dim() != 0):
#        #cf values that br has in real tree
#        out = out + self.add_tensor(node.cf_values_br, 'cf_values_br')
#
#      
#      return out
    
    # Generates data for a graphical representation of a node in a public tree.
    # @param node the node to generate data for
    # @return a table containing `name`, `label`, and `shape` fields for graphviz
    # @local
    def node_to_graphviz(self, node):  
      out = {}
      
      #1.0 label
      out['label'] = '"<f0>' + str(node.current_player) + '|<f1>' + str(node.node_id) + '*'
      
      if node.terminal:
        if node.type == constants.node_types.terminal_fold:
          out['label'] = out['label'] + '| TERMINAL FOLD'
        elif node.type == constants.node_types.terminal_call:
          out['label'] = out['label'] + '| TERMINAL CALL'
        else:
          assert(True)#'unknown terminal node type')
      else:
        out['label'] = out['label'] + '| bet0: ' + str(node.bets[0].item()) +\
                                      '| bet1: ' + str(node.bets[1].item()) 
#                                      '| bet2: ' + str(node.bets[2]) + ' ' + str(node.active[2]) 
#                                      '| bet3: ' + str(node.bets[3]) + ' ' + str(node.active[3]) +\
#                                      '| bet4: ' + str(node.bets[4]) + ' ' + str(node.active[4]) +\
#                                      '| bet5: ' + str(node.bets[5]) + ' ' + str(node.active[5])
#        
        if node.street != None:
          out['label'] = out['label'] + '| street: ' + str(node.street)
          out['label'] = out['label'] + '| board: ' + card_to_string.cards_to_string(node.board)
          out['label'] = out['label'] + '| depth: ' + str(node.depth)
      
        if node.table.dim() != 0:
          out['label'] = out['label'] + '| sl: ' + self.add_list(node.table,"")
          
#        if node.rl.dim() != 0:
#          out['label'] = out['label'] + '| rl: ' + self.add_tensor(node.rl,"")
#          
          
#      if node.margin != None:
#        out['label'] = out['label'] +  '| margin: ' + node.margin
    
#      out['label'] = out['label'] + self.add_range_info(node)  
#      
#      if node.cfv_infset.dim() != 0:
#        out['label'] = out['label'] +  '| cfv1: ' + str(node.cfv_infset[0])
#        out['label'] = out['label'] +  '| cfv2: ' + str(node.cfv_infset[1])
#        out['label'] = out['label'] +  '| cfv_br1: ' + str(node.cfv_br_infset[0])
#        out['label'] = out['label'] +  '| cfv_br2: ' + str(node.cfv_br_infset[1])
#        out['label'] = out['label'] +  '| epsilon1: ' + str(node.epsilon[0])
#        out['label'] = out['label'] +  '| epsilon2: ' + str(node.epsilon[1])  
#      
#      if node.has_key('lookahead_coordinates'):
#        out['label'] = out['label'] +  '| COORDINATES '
#        out['label'] = out['label'] +  '| action_id: ' + node.lookahead_coordinates[1]
#        out['label'] = out['label'] +  '| parent_action_id: ' + node.lookahead_coordinates[2]
#        out['label'] = out['label'] +  '| gp_id: ' + node.lookahead_coordinates[3]
      
      out['label'] = out['label'] + '"'
      
      #2.0 name
      out['name'] = '"node' + str(self.node_to_graphviz_counter) + '"'
      
      #3.0 shape
      out['shape'] = '"record"' 
        
      self.node_to_graphviz_counter = self.node_to_graphviz_counter + 1
      return out
    
    # Generates data for graphical representation of a public tree action as an
    # edge in a tree.
    # @param from the graphical node the edge comes from
    # @param to the graphical node the edge goes to
    # @param node the public tree node before at which the action is taken
    # @param child_node the public tree node that results from taking the action
    # @return a table containing fields `id_from`, `id_to`, `id` for graphviz and
    # a `strategy` field to use as a label for the edge
    # @local
    def nodes_to_graphviz_edge(self, fro, to, node, child_node):
      out = {}
      
      out['id_from'] = fro['name']
      out['id_to'] = to['name']
      out['id'] = self.edge_to_graphviz_counter
      
      #get the child id of the child node
      child_id = -1
      for i in range(len(node.children)):
          if node.children[i] is child_node:
              child_id = i
      
      assert(child_id != -1)
      #TODO:strategy
#      out['strategy'] = self.add_tensor(node.strategy[child_id].view(1,-1), "", labels=card_to_string.card_to_string_table)
      
      self.edge_to_graphviz_counter = self.edge_to_graphviz_counter + 1
      return out
    
    # Recursively generates graphviz data from a public tree.
    # @param node the current node in the public tree
    # @param nodes a table of graphical nodes generated so far
    # @param edges a table of graphical edges generated so far
    # @local
    def graphviz_dfs(self, node, nodes, edges):
    
      gv_node = self.node_to_graphviz(node)
      nodes.append(gv_node)
      
      for child_node in node.children:
        gv_node_child = self.graphviz_dfs(child_node, nodes, edges)
        gv_edge = self.nodes_to_graphviz_edge(gv_node, gv_node_child, node, child_node)
        edges.append(gv_edge)
    
      return gv_node
    
    # Generates `.dot` and `.svg` image files which graphically represent 
    # a game's public tree.
    # 
    # Each node in the image lists the acting player, the number of chips
    # committed by each player, the current betting round, public cards,
    # and the depth of the subtree after the node, as well as any probabilities
    # or values stored in the `ranges_absolute`, `cf_values`, or `cf_values_br`
    # fields of the node.
    # 
    # Each edge in the image lists the probability of the action being taken
    # with each private card.
    #
    # @param root the root of the game's public tree
    # @param filename a name used for the output files
    def graphviz(self, root, filename):
      filename = filename or 'tree_2.dot'
      
      out = 'digraph g {  graph [ rankdir = "LR"];node [fontsize = "16" shape = "ellipse"]; edge [];'
        
      nodes = []
      edges = []
      self.graphviz_dfs(root, nodes, edges)
        
      for i in range(len(nodes)):
        node = nodes[i]
        node_text = node['name'] + '[' + 'label=' + node['label'] + ' shape = ' + node['shape'] + '];'
          
        out = out + node_text
          
      for i in range(len(edges)):
        edge = edges[i]
#        edge_text = str(edge['id_from']) + ':f0 -> ' + str(edge['id_to']) + ':f0 [ id = ' + str(edge['id']) + ' label = "' + edge['strategy'] + '"];'
        edge_text = str(edge['id_from']) + ':f0 -> ' + str(edge['id_to']) + ':f0 [ id = ' + str(edge['id']) + ' label = "' + 'strategy' + '"];'
        out = out + edge_text
      out = out + '}'
        
      #write into:t file
#      with open(arguments.data_directory + 'Dot/' + filename, 'wb') as fout:
#          fout.write(str(out))
  
      
      #mjb 
      src = Source(out)
#      src.view()
      #run graphviz program to generate image
      src.render('dot ' + arguments.data_directory  + filename , view=True)
     
