#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 23:29:11 2017

@author: mjb
"""
import torch
import Setting.arguments as arguments
import Game.Evaluation.evaluator as evaluator
import Game.card_tools as card_toos

class RangeGenerator:
    def __init__(self):
        self.possible_hands_count
        self.possivle_hands_mask
        self.reverse_order
        self.reordered_range
        self.sorted_range
    
    
#    Recursively samples a section of the range vector.
#    @param cards an NxJ section of the range tensor, where N is the batch size
#    and J is the length of the range sub-vector
#    @param mass a vector of remaining probability mass for each batch member
#    @see generate_range
#   TODO what is mass?
    def _generate_recursion(self, cards, mass):
        batch_size = cards.size(0)
        assert mass.size(0) == batch_size
#       we terminate recursion at size of 1
        card_count = cards.size(1)
        if card_count == 1:
            cards.copy_(mass) 
        else:
            rand = torch.rand(batch_size)
        if arguments.gpu: 
            rand = rand.cuda()
        mass1 = mass.clone().cmul(rand)
        mass2 = mass - mass1
        halfSize = card_count / 2
#       if the tensor contains an odd number of cards, randomize which way the
#       middle card goes
        if halfSize % 1 != 0:
            halfSize = halfSize - 0.5
            halfSize = halfSize + torch.random(0,1)
        self._generate_recursion(cards[:,0:halfSize,:], mass1)
        self._generate_recursion(cards[:,halfSize + 1, -1,:], mass2)
  
#   Samples a batch of ranges with hands sorted by strength on the board.
#   @param range a NxK tensor in which to store the sampled ranges, where N is
#   the number of ranges to sample and K is the range size
#   @see generate_range
#   @local
    def _generate_sorted_range(self, range):
        batch_size = range.size(0)
        self._generate_recursion(range, arguments.Tensor(batch_size).fill(1))
        
#   Sets the (possibly empty) board cards to sample ranges with.
#   The sampled ranges will assign 0 probability to any private hands that
#   share any cards with the board.
#   @param board a possibly empty vector of board cards
    def set_board(self, board):
        hand_strengths = evaluator.batch_eval(board)    
#       mjb a rank*suit vector which represent all the vaild private card with 0 and 1
        possible_hand_indexes = card_toos.get_possible_hand_indexes(board)
        self.possible_hands_count = possible_hand_indexes.sum(1)[0]  
        self.possible_hands_mask = possible_hand_indexes.view(1, -1)
        if not arguments.gpu: 
            self.possible_hands_mask = self.possible_hands_mask.byte()
        non_coliding_strengths = arguments.Tensor(self.possible_hands_count) 
#       mjb choose the possible hand strengths use the mask( 0 1 0 1 1 1 etc)  
        non_coliding_strengths.maskedSelect(hand_strengths, self.possible_hands_mask)
        _, order = non_coliding_strengths.sort()
        _, self.reverse_order = order.sort() 
        self.reverse_order = self.reverse_order.view(1, -1).long()
        self.reordered_range = arguments.Tensor()
        self.sorted_range = arguments.Tensor()

#   Samples a batch of random range vectors.
#
#-- Each vector is sampled indepently by randomly splitting the probability
#-- mass between the bottom half and the top half of the range, and then
#-- recursing on the two halfs.
#-- 
#-- @{set_board} must be called first.
#--
#-- @param range a NxK tensor in which to store the sampled ranges, where N is
#-- the number of ranges to sample and K is the range size
    def generate_range(self, range):  
        batch_size = range.size(0)
        self.sorted_range.resize_(batch_size, self.possible_hands_count)
        self._generate_sorted_range(self.sorted_range, self.possible_hands_count)
#   --we have to reorder the the range back to undo the sort by strength
        index = self.reverse_order.expand_as(self.sorted_range)
        if arguments.gpu:
            index = index.cuda()
        self.reordered_range = self.sorted_range.gather(1, index)
   
        range.zero_()
        range.masked_copy_(self.possible_hands_mask.expandAs(range), self.reordered_range)
 
