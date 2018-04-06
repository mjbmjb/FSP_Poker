extern "C"{
	#include "evalHandTables.h"
}
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <unordered_map>
#include <time.h>
#include <set>
#include <string.h>
using namespace std;


void evalShowdown(int board[],int board_size, int hole[][2], int player_number, int hs[]){
	Cardset public_cs = emptyCardset();
	for(int i=0; i<board_size; i++){
		addCardToCardset(&public_cs, board[i]%4, board[i]/4);
	}

	for(int i=0; i<player_number; i++) {
		Cardset cs = public_cs;
		addCardToCardset(&cs, hole[i][0]%4, hole[i][0]/4);
		addCardToCardset(&cs, hole[i][1]%4, hole[i][1]/4);
		hs[i] = rankCardset(cs);
	}
}
void evalShowdown_1(int board[], int board_size, int hole[], int player_number, int hs[]){
	Cardset public_cs = emptyCardset();
	for(int i=0; i<board_size; i++){
		addCardToCardset(&public_cs, board[i]%4, board[i]/4);
	}

	for(int i=0; i<player_number; i++) {
		Cardset cs = public_cs;
		addCardToCardset(&cs, hole[i]%4, hole[i]/4);
		hs[i] = rankCardset(cs);
	}

}
int main() {
    int hs[3];
//  int board[] = {6,30,31,39,43};
//	int hole[][2] = {{40,41},{50,51},{4,5},{8,9},{44,45},{28,29}};
    int board[] = {9};
    int hole[] = {6,5,4};
	evalShowdown_1(board, 1, hole, 3, hs);
	for (int i = 0;i < 3;i ++){
        cout<<hs[i]<<" ";
	}
}
