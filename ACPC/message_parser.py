from ACPC.card_tool import CardTool

import Settings.game_settings as game_settings
import Settings.arguments as arguments
import Settings.constants as constants


class MessageParser(object):

    def __init__(self, message):
        self.message = message

        _, position, hand_number, betting_string, board_string = message.split(":")
        self.position = int(position)
        self.hand_number = int(hand_number)
        self.betting_string = betting_string.split("/")
        card_string = board_string.split("/")
        # hole string is the whole string represents all players hole, like "|JdTc||||"
        self.hole_string = card_string[0]
        # board string one-dimension array like "2d3cTh5c"
        # board string could be [], is the whole string represents boards of all rounds
        self.board_string = "".join(card_string[1:])
        # handle hole
        self.hole = self.parse_hole()
        self.hole_card = arguments.IntTensor(game_settings.player_count,game_settings.private_count).fill_(-1)
        for i in range(game_settings.player_count):
            if self.hole[i] != []:
                for j in range (len(self.hole[i])):
                    self.hole_card[i][j] = CardTool.string_to_card(self.hole[i][j])
            
        
        # handle board
        self.board = self.parse_board()
        self.board_card = arguments.IntTensor(game_settings.board_card_count).fill_(-1)
        for i in range(len(self.board)):
            self.board_card[i] = CardTool.string_to_card(self.board[i])

    def parse_hole(self):
        card_list = self.hole_string.split("|")
        for i in range(game_settings.player_count):
            string = card_list[i]
            card_list[i] = [string[x:x+2] for x in range(0, len(string), 2)]
        return card_list

    def parse_board(self):
        all_board_string = "".join(self.board_string)
        card_list = [all_board_string[x:x+2] for x in range(0, len(all_board_string), 2)]
        return card_list

    def get_position(self):
        return self.position

    def get_hand_number(self):
        return self.hand_number

    # return betting string of round, last round by default
    def get_betting_string(self, rd=None):
        if rd is not None:
            return self.betting_string[rd]
        return self.betting_string

    def get_hole_card(self, position=None):
        if position is not None:
            return self.hole_card[position]
        return self.hole_card

    def get_board_card(self, rd=None):
        if rd is not None:
            raise Exception
        return self.board_card

    # if round is not given, then return all board cards
    def get_board_string(self, rd=None):
        if rd is not None:
            # not implemented yet
            raise Exception
        return self.board_string


str1 = 'MATCHSTATE:1:31:r300r900r3000ccccc/r9000ffffc/cc/cc:|JdTc||||/2c2d2h/3c/3d'
#mp = MessageParser(str1)
#print(mp.hole, mp.board)
## print(mp.get_hole_card())
## print(mp.get_board_card())
## print(mp.hole)
## print(mp.board)
#print(mp.get_betting_string())
#print(mp.get_board_string())
#print(mp.get_board_card())
#import torch
#print(mp.get_hole_card(1))

