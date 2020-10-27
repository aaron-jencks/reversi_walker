/*
 * ScoreTurnArea.cpp
 *
 *  Created on: 26.01.2015
 *      Author: alexey slovesnov
 */

#include "ScoreTurnArea.h"
#include "ReversiArea.h"

ScoreTurnArea::ScoreTurnArea() : BaseArea( (m_cellSize+1)*3+1,m_cellSize+2 ) {
}

ScoreTurnArea::~ScoreTurnArea() {
}

void ScoreTurnArea::draw() {
	int i,n;
	int turn;
	int cell;
	char c[32];
	const char* text[]={"turn","draw","win"};
	ReversiArea& area=getReversiArea();

	//turn or end game text
	if(area.isGameOver()){
		if( (i=area.count(BLACK)-area.count(WHITE)) ==0){
			n=1;
			turn=WHITE;//to draw text with black color
			cell=EMPTY;
		}
		else{
			turn = ( (isMinimize() && i<0) || (!isMinimize() && i>0) ) ? BLACK : WHITE;
			cell=turn;
			n=2;
		}
	}
	else{
		turn=area.m_turn;
		cell=turn;
		n=0;
	}

	for(i=0;i<3;i++){
		copy(chip(i==2 ? cell : i ),m_cr,i*(m_cellSize+1)+1,1);
	}

	for(i=0;i<2;i++){
		sprintf(c,"%d", getReversiArea().count(i==0 ? BLACK : WHITE) );
		drawText(m_cr,c,i*(m_cellSize+1)+1,0, m_cellSize,m_cellSize
				,true,true,i==0 ? whiteColor : blackColor,m_fontSize,true );
	}

	drawText(m_cr,text[n],2*(m_cellSize+1)+1,0, m_cellSize,m_cellSize,true,true
			,turn==BLACK ? whiteColor : blackColor,m_turnFontSize,true );

	drawNet(0,0,3,1);

	gtk_widget_queue_draw (m_area);

}
