#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 01:50:32 2017

@author: mjb
"""

# Handles network communication for DeepStack.
# 
# Requires [luasocket](http://w3.impa.br/~diego/software/luasocket/)
# (can be installed with `luarocks install luasocket`).
# @classmod network_communication
import Settings.arguments as arguments
from socket import *
import sys

class ACPCNetworkCommunication:
    # Constructor
    def __init__(self):
        pass
    
    # Connects over a network socket.
    # 
    # @param server the server that sends states to DeepStack, and to which
    # DeepStack sends actions
    # @param port the port to connect on
    def connect(self, server, port):
      server = server if server != None else arguments.acpc_server
      port = port if port != None else arguments.acpc_server_port
    
      self.connection = socket(AF_INET, SOCK_STREAM)
      self.connection.connect((server, port))
      # mjb to read one line once 
      self.soc_file = self.connection.makefile()
      self._handshake()
    
    # Sends a handshake message to initialize network communication.
    # @local
    def _handshake(self):
        self.send_line("VERSION:2.0.0")
    
    # Sends a message to the server.
    # @param line a string to send to the server
    def send_line(self, line):
      self.connection.send((line + '\r\n').encode()) 
    
    # Waits for a text message from the server. Blocks until a message is
    # received.
    # @return the message received
    def get_line(self):  
      data = self.soc_file.readline()
      if not data:
          assert(False)
      return data
    
    # Ends the network communication.
    def close(self):  
      self.connection.close()
