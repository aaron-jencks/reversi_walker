/*
 * Position.h
 *
 *  Created on: 01.02.2015
 *      Author: alexey slovesnov
 */

#ifndef POSITION_H_
#define POSITION_H_

#include <map>
typedef std::map<int,int> PositionMap;

class Position {
public:
	int x,y;
	char color;
	PositionMap map;
	Position(int _x,int _y, char _color);
	Position();
	//have to use copy constructor because it's not possible to do shallow map copy
	Position(Position const &p);
	virtual ~Position();
};

#endif /* POSITION_H_ */
