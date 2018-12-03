#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 00:54:04 2017

@author: mjb
"""

from ACPC.network_communication import ACPCNetworkCommunication
from ACPC.msg_to_state import MsgToState
import Settings.arguments as arguments
import Settings.constants as constants
import re
import subprocess

class SixACPCGame:
    #if you want to fake what messages the acpc dealer sends, put them in the following list and uncomment it.
    debug_msg = None#{"MATCHSTATE:0:99::Kh|/", "MATCHSTATE:0:99:cr200:Kh |/", "MATCHSTATE:0:99:cr200:Kh|/Ks"}
    
    # Constructor
    def __init__(self, msg):
        self.debug_msg = msg
    
    
    # Connects to a specified ACPC server which acts as the dealer.
    # 
    # @param server the server that sends states to DeepStack, which responds
    # with actions
    # @param port the port to connect on
    # @see network_communication.connect
    def connect(self, server, port):
      if not self.debug_msg:
        self.network_communication = ACPCNetworkCommunication()
        if arguments.C_PLAYER:
            self.run_agent(server, port)
        self.network_communication.connect(server, port)

    # Receives and parses the next poker situation where DeepStack must act.
    # 
    # Blocks until the server sends a situation where DeepStack acts.
    # @return the parsed state representation of the poker situation (see
    # @{protocol_to_node.parse_state})
    # @return a public tree node for the state (see
    # @{protocol_to_node.parsed_state_to_node})
    def get_next_situation(self):
    
        while True:
            
            if self.debug_msg == []:
                return
            
            msg = None
    
            #1.0 get the message from the dealer
            if not self.debug_msg:
                msg = self.network_communication.get_line()
            else:
                msg = self.debug_msg.pop()
        
            print("Received acpc dealer message:")
            print(msg)
            
            #mjb if it is a show down or fold message, skip the first msg
            if re.search("(\w{2}\|\w{2})", msg) != None:
                continue
        
            #2.0 parse the string to our state representation
            msg2state = MsgToState(msg.strip('\n'))
            parsed_state = msg2state.state
            
            #3.0 figure out if we should act
            
            #current player to act is us
            if parsed_state.current_player == msg2state.viewing_player and not parsed_state.terminal:
                #we should not act since this is an allin situations
                print("Our turn")
        
                self.last_msg = msg
        
                return parsed_state
            #current player to act is the opponent
            else:
              print("Not our turn")
              
    # Generates a message to send to the ACPC protocol server, given DeepStack's
    # chosen action.
    # @param last_message the last state message sent by the server
    # @param adviced_action the action that DeepStack chooses to take, with fields
    # 
    # * `atype`: an element of @{constants.actions}
    # 
    # * `amount`: the number of chips to rraise (if `action` is rraise)
    # @return a string messsage in ACPC format to send to the server
    def action_to_message(self, last_message, adviced_action):
  
        out = last_message.replace('\n','')
  
        if adviced_action.atype == constants.actions.ccall:
            protocol_action = 'c'
        elif adviced_action.atype == constants.actions.fold:
            protocol_action = 'f'
        elif adviced_action.atype == constants.actions.rraise:
            protocol_action = 'r' + str(adviced_action.amount)
  
        out = out + ":" + protocol_action
  
        return out 
              
             
    
    # Informs the server that DeepStack is playing a specified action.
    # @param adviced_action a table specifying the action chosen by Deepstack,
    # with the fields:
    # 
    # * `action`: an element of @{constants.acpc_actions}
    # 
    # * `raise_amount`: the number of chips raised (if `action` is raise)
    def play_action(self, adviced_action):
        message = self.action_to_message(self.last_msg, adviced_action)
        print("Sending a message to the acpc dealer:")
        print(message)
    
        if not self.debug_msg:
            self.network_communication.send_line(message)

    # FIXME for education
    def run_agent(self, host, port):
        command = 'cd ../ABS && ./start.sh {} {}'.format(host, port)
        sp = subprocess.Popen(command, shell = True)
        sp.wait()