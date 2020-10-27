/*
 * BaseArea.h
 *
 *  Created on: 26.01.2015
 *      Author: alexey slovesnov
 */

#ifndef BASEAREA_H_
#define BASEAREA_H_

#include "Base.h"

class ReversiFrame;
class ReversiArea;
class ScoreTurnArea;

class BaseArea : public Base {
protected:
	cairo_t* m_cr;
	cairo_surface_t *m_surface;
	GtkWidget*m_area;

public:
	BaseArea(int width,int height);
	virtual ~BaseArea();
  virtual void draw()=0;

  GtkWidget* getArea(){
  	return m_area;
  }
  static ReversiFrame*m_reversiFrame;

  void copySurface(cairo_t* cr){
  	cairo_set_source_surface (cr, m_surface, 0, 0);
  	cairo_paint(cr);
  }

  ReversiArea& getReversiArea();
  ScoreTurnArea& getScoreTurnArea();
	int getTableSize()const;
	int getShowType()const;
	bool isMinimize()const;
	bool isAnimation()const;
	int getStartPosition()const;
	int getComputerPlayer()const;
	void enableWidgets(bool enable);


	void drawNet(int startX,int startY,int cellsX,int cellsY);

	void invalidateRect(gint x,gint y,gint width,gint height){
		gtk_widget_queue_draw_area(m_area,x,y,width,height);
	}

	/**
	 * centerx - center horizontally, centery - center vertically
	 * if center=true then draw centered text in rectangle r
	 * if center=false then draw text in point r.left,r.top. Parameters r.right, r.bottom ignore
	 */
	inline void drawText(cairo_t* ct,std::string text,int x,int y,int width,int height,bool centerx,bool centery){
		const GdkRGBA rgba= {0.,0.,0.,1.};
		drawText(ct,text,x,y,width,height,centerx,centery,rgba,m_fontSize,false);
	}

	void drawText(cairo_t* ct,std::string text,int x,int y,int width,int height,bool centerx,bool centery,const GdkRGBA rgba,int fontHeight,bool bold);

	PangoLayout* createPangoLayout(std::string text,int fontHeight,bool bold);

};

#endif /* BASEAREA_H_ */
