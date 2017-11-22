from random import randint

class CardTool(object):

    rank = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    suit = ['c', 'd', 'h', 's']

    # construct a dict key='ranksuit' value in 0-51
    card_dict ={}
    for i in range(len(rank)):
        for j in range(len(suit)):
            str_temp = rank[i] + suit[j]
            card_dict[str_temp] = i * len(suit) + j

    # a reverse dict of card_dict
    string_dict={}
    for k, v in card_dict.items():
        string_dict[v] = k

    # return int number represent card
    @classmethod
    def string_2_card(cls, string):
        return cls.card_dict[string]

    # return string represent card
    @classmethod
    def card_to_string(cls, value):
        return cls.string_dict[value]

    @classmethod
    def rank_of_card(cls, card):
        return card // 4

    @classmethod
    def suit_of_card(cls, card):
        return card % 4

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