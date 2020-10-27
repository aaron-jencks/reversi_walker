/*
 * ScoreTurnArea.h
 *
 *  Created on: 26.01.2015
 *      Author: alexey slovesnov
 */

#ifndef SCORETURNAREA_H_
#define SCORETURNAREA_H_

#include "BaseArea.h"

class ScoreTurnArea: public BaseArea {
public:
	ScoreTurnArea();
	virtual ~ScoreTurnArea();
	virtual void draw();
};

#endif /* SCORETURNAREA_H_ */
