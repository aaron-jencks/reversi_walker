/*
 * Position.cpp
 *
 *  Created on: 01.02.2015
 *      Author: alexey slovesnov
 */

#include "Base.h"
#include "Position.h"

Position::Position() {
	x=y=color=0;
}

Position::Position(int _x, int _y, char _color) {
	x=_x;
	y=_y;
	color=_color;
}

Position::Position(const Position& p) {
	x=p.x;
	y=p.y;
	color=p.color;
	map=p.map;
}

Position::~Position() {
}

