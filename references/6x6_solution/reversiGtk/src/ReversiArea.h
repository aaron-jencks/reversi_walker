/*
 * ReversiArea.h
 *
 *  Created on: 25.01.2015
 *      Author: alexey slovesnov
 */

#ifndef REVERSIAREA_H_
#define REVERSIAREA_H_

#include "BaseArea.h"
#include "Position.h"

typedef Reversi<6,true,true> Reversi6;
typedef Reversi<5,true,true> Reversi5;
typedef Reversi<4,false,true> Reversi4;
typedef Reversi<3,false,true> Reversi3;

typedef Reversi<6,true,false> Reversi6Minimize;
typedef Reversi<5,true,false> Reversi5Minimize;
typedef Reversi<4,false,false> Reversi4Minimize;
typedef Reversi<3,false,false> Reversi3Minimize;

enum THREAD_ENUM{
	THREAD_INNER,
	THREAD_ALL,

	THREAD_SIZE
};

class ReversiArea : public BaseArea{
	static const int ESTIMATE_UNKNOWN=255;
	static const int ESTIMATE_CLEAR=ESTIMATE_UNKNOWN-1;
	static const int LAST_CELL_INVALID=-1;
	int m_lastCell;
	int m_tableSize;
	bool m_minimize;

	bool m_functionWait[THREAD_SIZE];
  GMutex m_mutex[THREAD_SIZE];
  GCond m_condition[THREAD_SIZE];

	ReversiBase*m_reversi;
	int m_animationStep;
	int m_animationTimer;

	std::vector<int> m_animationFlips;

public:
	int m_turn;
	std::vector<Position> m_turnList;
	unsigned m_turnListPosition;
  static const int m_size=512;

	ReversiArea();
	virtual ~ReversiArea();

	inline bool isThreadFinished(THREAD_ENUM e){
		return !isThreadRunning(e);
	}

	inline bool isThreadRunning(THREAD_ENUM e){
		return m_functionWait[e];
	}

	PositionMap& getEstimateMap(){
		assert(m_turnListPosition>=0 && m_turnListPosition<m_turnList.size());
		return m_turnList[m_turnListPosition].map;
	}


	void animationStep();
	void updateTurnList();

	virtual void draw();
	void drawCell(int index);
	inline void drawCell(int x,int y){
		drawCell(index(x,y));
	}
	int getStartPoint()const;

	void undoRedo(int index);
	void load();
	void save();
	bool fileChooser(bool open,std::string& filepath);
	void mouseLeftButtonDownThread(int i);
	void setEstimate(int index,int value);
	void showTypeEstimatesOn();

	inline bool isGameOver()const{
		return m_turn==EMPTY;
	}

	void newGame();
	void newGame(int moves,bool search);
	void updateAll();
	void setNewGameType();

	inline int count(char c){
		return m_reversi->count(c);
	}

	inline int index(int x,int y){
		return m_reversi->getIndex(x,y);
	}

	inline void getXY(int index,int&x,int&y){
		m_reversi->getXY(index,x,y);
	}

	inline bool possible(int x,int y,int white){
		return possible(index(x,y),white);
	}

	inline bool possible(int index,int white){
		//if game over the m_turn=EMPTY so before possible(x,y,m_turn) check isGameOver()
		if(isGameOver()){
			return false;
		}
		assert(white==BLACK || white==WHITE);
		return m_reversi->possibleMove(index,white);
	}

	inline char get(int x,int y){
		return get(index(x,y));
	}

	inline char get(int index){
		return m_reversi->get(index);
	}

	inline std::string toString(int index){
		return m_reversi->toString(index);
	}

	inline std::vector<int> flipList(int index,int white){
		if(isGameOver()){//in case of game over white=EMPTY so m_reversi->flipList gives wrong result
			std::vector<int> v;
			return v;
		}
		else{
			assert(white==BLACK || white==WHITE);
			return m_reversi->flipList(index,white);
		}
	}

	void makeMoveThread(int index,int white);

  void startWaitFunction(THREAD_ENUM e,GSourceFunc function,gpointer data);
  void finishWaitFunction(THREAD_ENUM e);
  void waitForFunction(THREAD_ENUM e);

	inline int estimateTurn(int index,int white,int alpha,int beta){
		assert(white==BLACK || white==WHITE);
		return m_reversi->estimateTurn(index,white,alpha,beta);
	}

	inline std::vector<int> getSymmetryCells(int index){
		return m_reversi->getSymmetryCells(index);
	}

	inline bool possible(int white){
		assert(white==BLACK || white==WHITE);
		return m_reversi->possibleMove(white);
	}

	void switchTurn(){
		if(possible(!m_turn)){
			m_turn=!m_turn;
		}
		else if(!possible(m_turn)){
			m_turn=EMPTY;
		}
	}

	void makeNeededComputerMovesThread();
	gpointer getCode(int x,int y,int e){
		return getCode(index(x,y),e);
	}

	gpointer getCode(int index,int e){
		return gpointer(index | (e<<8));
	}

	void fillEstimateListThread();

	void updateUndoRedo();

	inline int getCellIndex(double x,double y){
		int s=(m_cellSize+1)*getTableSize();
		int d=(m_size-s)/2;
		int e=d+s;
		if(x<d || y<d || x>=e || y>=e ){
			return LAST_CELL_INVALID;
		}
		else{
			return index( (x-d)/(m_cellSize+1) , (y-d)/(m_cellSize+1) );
		}
	}

	void mouseLeftButtonDown(GdkEventButton* event);
	void mouseMove(double x,double y);
	void mouseLeave(GdkEventCrossing* event);

};

#endif /* REVERSIAREA_H_ */
