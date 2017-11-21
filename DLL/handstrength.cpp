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


void evalShowdown(int board[], int hole[][2], int player_number, int hs[]){
	Cardset public_cs = emptyCardset();
	for(int i=0; i<5; i++){
		addCardToCardset(&public_cs, board[i]%4, board[i]/4);
	}
	
	for(int i=0; i<player_number; i++) {
		Cardset cs = public_cs;
		addCardToCardset(&cs, hole[i][0]%4, hole[i][0]/4);
		addCardToCardset(&cs, hole[i][1]%4, hole[i][1]/4);
		hs[i] = rankCardset(cs);
	}

	// for (int i=0; i<player_number; i++){
	// 	printf("%d\n", hs[i]);
	// }
}
