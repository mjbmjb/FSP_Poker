from random import randint
import Settings.game_settings as game_settings

class CardTool(object):

    rank = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    suit = ['c', 'd', 'h', 's']

    # rank = [ 'T', 'J', 'Q', 'K', 'A']
    # suit = ['h', 's']

    # rank = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
    # suit = ['h', 's', 'c', 'd']

    # construct a dict key='ranksuit' value in 0-51
    card_dict ={}
    for i in range(game_settings.rank_count):
        for j in range(game_settings.suit_count):
            str_temp = rank[i + (13 - game_settings.rank_count)] + suit[j + (4 - game_settings.suit_count)]
            card_dict[str_temp] = i * game_settings.suit_count + j

    # a reverse dict of card_dict
    string_dict={}
    for k, v in card_dict.items():
        string_dict[v] = k

    # return int number represent card
    @classmethod
    def string_to_card(cls, string):
        return cls.card_dict[string]

    # return string represent card
    @classmethod
    def card_to_string(cls, value):
        return cls.string_dict[value]

    @classmethod
    def rank_of_card(cls, card):
        return card // game_settings.suit_count

    @classmethod
    def suit_of_card(cls, card):
        return card % game_settings.suit_count
    @classmethod
    def deal_all_cards(cls, rd):
        dealed_cards = []
        card_number = 0
        if rd == 0:
            card_number = 12
        elif rd == 1:
            card_number = 15
        elif rd == 2:
            card_number = 16
        elif rd == 3:
            card_number = 17
        cards = list(range(52))

        for i in range(card_number):
            card_index = randint(0, 51-i)
            dealed_cards.append(cards[card_index])
            cards[card_index], cards[-1-i] = cards[-1-i], cards[card_index]
        holes = [([0] * 2) for i in range(6)]
        boards = []
        k = 0
        for i in range(6):
            for j in range(2):
                holes[i][j] = dealed_cards[k]
                k += 1
        boards = dealed_cards[k:]

        return holes, boards

    @classmethod
    def display(cls):
        print("card [0-51]")
        print('2c ' + str(cls.string_2_card('2c')))
        print('2d ' + str(cls.string_2_card('2d')))
        print('...')

# x = CardTool.deal_all_cards(rd=Round.RIVER)
# print(x[0], x[1])