/*
 * BaseArea.cpp
 *
 *  Created on: 26.01.2015
 *      Author: alexey slovesnov
 */

#include "BaseArea.h"
#include "ReversiFrame.h"

ReversiFrame* BaseArea::m_reversiFrame;

static void on_draw_event(GtkWidget *widget, cairo_t *cr,BaseArea* area){
	area->copySurface(cr);
}

BaseArea::BaseArea(int width,int height) {
	m_area=gtk_drawing_area_new();
	m_surface=NULL;
	m_cr=NULL;

  createNew(m_surface, cairo_image_surface_create(CAIRO_FORMAT_ARGB32, width,height));
	createNew(m_cr , cairo_create(m_surface));

	gtk_widget_set_size_request(m_area,width,height);

	g_signal_connect(m_area, "draw",G_CALLBACK(on_draw_event), this);

}

BaseArea::~BaseArea() {
	destroy(m_surface);
	destroy(m_cr);
}

void BaseArea::drawText(cairo_t* ct, std::string text, int x, int y, int width,
		int height, bool centerx, bool centery,const GdkRGBA rgba,int fontHeight,bool bold) {

	gdk_cairo_set_source_rgba(ct, &rgba );
	PangoLayout *layout=createPangoLayout(text,fontHeight,bold);

	int w,h;
	pango_layout_get_pixel_size (layout,&w,&h);
	double px = x;
	double py = y;
	if(centerx){
		px+=(width-w)/2;
	}
	if(centery){
		py+=(height-h)/2;
	}

	cairo_move_to (ct, px, py);
	pango_cairo_update_layout(ct, layout);
	pango_cairo_show_layout(ct, layout);

	g_object_unref(layout);

}

ReversiArea& BaseArea::getReversiArea() {
	return m_reversiFrame->m_area;
}

ScoreTurnArea& BaseArea::getScoreTurnArea() {
	return m_reversiFrame->m_scoreTurnArea;
}

int BaseArea::getShowType() const {
	return m_reversiFrame->getShowType();
}

bool BaseArea::isMinimize() const {
	return m_reversiFrame->isMinimize();
}

int BaseArea::getStartPosition()const{
	return m_reversiFrame->getStartPosition();
}

int BaseArea::getComputerPlayer()const{
	return m_reversiFrame->getComputerPlayer();
}

int BaseArea::getTableSize() const {
	return m_reversiFrame->getTableSize();
}

bool BaseArea::isAnimation() const {
	return m_reversiFrame->isAnimation();
}

void BaseArea::enableWidgets(bool enable) {
	m_reversiFrame->enableWidgets(enable);
}

PangoLayout* BaseArea::createPangoLayout(std::string text,int fontHeight,bool bold) {
	PangoLayout *layout;
	std::string o;
	char c[32];
	layout = pango_cairo_create_layout(m_cr);
	sprintf(c,"Cambria, %d",fontHeight);
	//sprintf(c,"Georgia, %d",fontHeight);
	PangoFontDescription*desc=pango_font_description_from_string(c);
	pango_layout_set_font_description(layout, desc );
	if(bold){
		o="<b>"+text+"</b>";
	}
	else{
		o=text;
	}
	pango_layout_set_markup(layout, o.c_str(), -1);
	pango_font_description_free(desc);
	return layout;
}

void BaseArea::drawNet(int startX, int startY, int cellsX, int cellsY) {
	int i;
	double d;
	double sx=startX+0.5;
	double sy=startY+0.5;

	gdk_cairo_set_source_rgba(m_cr, &blackColor );
	cairo_set_line_width(m_cr,1.);
	//+0.5 to make line width equals 1

	for(i=0;i<=cellsX;i++){
		d=i*(m_cellSize+1);

		//horizontal
		cairo_move_to(m_cr,sx,sy+d);
		cairo_line_to(m_cr,sx+cellsX*(m_cellSize+1),sy+d);

		//vertical
		cairo_move_to(m_cr,sx+d,sy);
		cairo_line_to(m_cr,sx+d,sy+cellsX*(m_cellSize+1));
	}
	cairo_stroke(m_cr);

}
