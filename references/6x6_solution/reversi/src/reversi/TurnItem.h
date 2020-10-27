/*
 * TurnItem.h
 *
 *  Created on: 24.01.2015
 *      Author: alexey slovesnov
 */

#ifndef TURNITEM_H_
#define TURNITEM_H_

struct TurnItem{
	int turn;
	TurnItem*next,*prev;
};

#endif /* TURNITEM_H_ */
