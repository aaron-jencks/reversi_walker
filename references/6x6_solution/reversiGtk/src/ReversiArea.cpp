/*
 * ReversiArea.cpp
 *
 *  Created on: 25.01.2015
 *      Author: alexey slovesnov
 */

#include "ReversiArea.h"
#include "ReversiFrame.h"
#include <math.h>
#include <limits.h>
#include <set>

const int ANIMATION_STEP_TIME=100;//milliseconds
const int ANIMATION_STEPS=8;
static ReversiArea*area;

static gboolean mouse_press_event(GtkWidget *widget,GdkEventButton  *event,ReversiArea* area){
	if(event->button==1){
		area->mouseLeftButtonDown(event);
	}
	return TRUE;
}

static gboolean mouse_move_event(GtkWidget *widget,GdkEventButton  *event,ReversiArea* area){
	area->mouseMove(event->x,event->y);
	return TRUE;
}

//Mouse out signal. Note second parameter GdkEventCrossing not GdkEventButton
static gboolean mouse_leave_event(GtkWidget *widget,GdkEventCrossing  *event,ReversiArea* area){
	area->mouseLeave(event);
	return TRUE;
}

static gboolean mouse_enter_event(GtkWidget *widget,GdkEventCrossing  *event,ReversiArea* area){
	area->mouseMove(event->x,event->y);//after switch to another application and back to out window again
	return TRUE;
}

static gboolean timer_animation_handler(gpointer){
	area->animationStep();
	return TRUE;
}

static gpointer mouse_left_button_down_thread(gpointer data){
	area->mouseLeftButtonDownThread((int)data);
	return NULL;
}

static gpointer make_needed_computer_moves_thread(gpointer){
	area->makeNeededComputerMovesThread();
	return NULL;
}

static gboolean update_all(gpointer){
	area->updateAll();
	return G_SOURCE_REMOVE;
}

static gboolean update_turns_list_score_turn(gpointer){
	area->updateTurnList();
	area->getScoreTurnArea().draw();
	area->finishWaitFunction(THREAD_INNER);
	return G_SOURCE_REMOVE;
}

static gboolean create_new_turn(gpointer data){
	int i=(int)data;
	int index=i&0xff;
	int x,y;
	area->getXY(index,x,y);
	area->m_turnList.push_back(Position(x,y,i>>8));
	area->m_turnListPosition++;
	area->finishWaitFunction(THREAD_INNER);
	return G_SOURCE_REMOVE;
}

static gboolean draw_table(gpointer){
	area->draw();
	return G_SOURCE_REMOVE;
}

static gboolean enable_widgets(gpointer){
	area->enableWidgets(true);
	return G_SOURCE_REMOVE;
}

static gpointer fill_estimate_list_thread(gpointer){
	area->fillEstimateListThread();
	gdk_threads_add_idle(enable_widgets,0);
	gdk_threads_add_idle(update_all,0);
	return NULL;
}

static gboolean set_estimate(gpointer data){
	int i=(int)data;
	area->setEstimate( i&0xff, i>>8);
	return G_SOURCE_REMOVE;
}

ReversiArea::ReversiArea() : BaseArea(m_size,m_size){
	m_reversi=NULL;
	m_tableSize=0;
	m_lastCell=LAST_CELL_INVALID;
	area=this;
	int i;

	for(i=0;i<SIZE(m_mutex);i++){
		g_mutex_init(&m_mutex[i]);
	  g_cond_init(&m_condition[i]);
	  m_functionWait[i]=false;
	}

	//enable mouse down & up & motion
	gtk_widget_add_events(m_area, GDK_BUTTON_PRESS_MASK|GDK_POINTER_MOTION_MASK|GDK_LEAVE_NOTIFY_MASK|GDK_ENTER_NOTIFY_MASK);
 	g_signal_connect(m_area, "button_press_event",G_CALLBACK(mouse_press_event), this);
 	g_signal_connect(m_area, "motion-notify-event",G_CALLBACK(mouse_move_event), this);
 	g_signal_connect(m_area, "leave-notify-event",G_CALLBACK(mouse_leave_event), this);
 	g_signal_connect(m_area, "enter-notify-event",G_CALLBACK(mouse_enter_event), this);

}

ReversiArea::~ReversiArea() {
	m_reversi->free();
	int i;

	assert(SIZE(m_mutex)==SIZE(m_condition));
	for(i=0;i<SIZE(m_mutex);i++){
	  g_mutex_clear(&m_mutex[i]);
	  g_cond_clear(&m_condition[i]);
	}
}

int ReversiArea::getStartPoint()const{
	const int f=getTableSize()*(m_cellSize+1)+1;
	return (m_size-f)/2;
}

void ReversiArea::draw() {
	int x,y;
	char c[2];
	const int s=24;

	copy(m_background,m_cr,0,0);

	const int startx=getStartPoint();
	const int starty=getStartPoint();

	for(x=0;x<getTableSize();x++){
		for(y=0;y<getTableSize();y++){
			drawCell(x,y);
		}
	}
//	println("%d %d",getEstimateMap().size(),m_turnListPosition );

	const int cs=m_cellSize+1;

	c[1]=0;
	c[0]='A';
	for(x=0;x<getTableSize();x++,c[0]++){
		drawText(m_cr,c,startx+x*cs,starty-s, cs,s,true,true);
		drawText(m_cr,c,startx+x*cs,starty+getTableSize()*cs, cs,s,true,true);
	}

	c[0]='1';
	for(y=0;y<getTableSize();y++,c[0]++){
		drawText(m_cr,c,startx-s,starty+y*cs,s, cs,true,true);
		drawText(m_cr,c,startx+getTableSize()*cs,starty+y*cs,s, cs,true,true);
	}

	drawNet( startx,starty, getTableSize(),getTableSize() );
	gtk_widget_queue_draw (m_area);
}

void ReversiArea::drawCell(int index){
	int i,x,y;
	char c[8];
	GdkPixbuf *pixbuf;
	std::vector<int> flips;
	int size,color;
	getXY(index,x,y);
	const int startx=getStartPoint()+x*(m_cellSize+1)+1;
	const int starty=getStartPoint()+y*(m_cellSize+1)+1;

	if(m_lastCell!=LAST_CELL_INVALID){
		flips=flipList(m_lastCell,m_turn);
		if(std::find(flips.begin(),flips.end(),index)!=flips.end()){
			pixbuf=gdk_pixbuf_copy( chip(get(index)) );
			gdk_pixbuf_saturate_and_pixelate (pixbuf,pixbuf,0.2f,false);//desaturate image
			copy(pixbuf,m_cr,startx,starty);
			g_object_unref(pixbuf);
			return;
		}
	}
	if(std::find(m_animationFlips.begin(),m_animationFlips.end(),index)!=m_animationFlips.end()){
		copy(chip(EMPTY),m_cr,startx,starty	);
		color=get(index);
		size=abs(m_cellSize*cos(M_PI*m_animationStep/ANIMATION_STEPS));
		if(size!=0){
			pixbuf=gdk_pixbuf_copy( chip(m_animationStep<ANIMATION_STEPS/2 ? !color:color) );
			pixbuf=gdk_pixbuf_scale_simple( pixbuf ,m_cellSize,size,GDK_INTERP_BILINEAR);
			copy(pixbuf,m_cr,startx,starty+(m_cellSize-size)/2);
			g_object_unref(pixbuf);
		}
		return;
	}

	i=get(index);
	copy(chip(i),m_cr,startx,starty);

	if( isThreadFinished(THREAD_ALL) && getShowType()==SHOW_TYPE_POSSIBLE_MOVES && possible(index,m_turn)  ){
		size=gdk_pixbuf_get_height(m_possiblePixbuf);
		copy(m_possiblePixbuf,m_cr,startx+(m_cellSize-size)/2,starty+(m_cellSize-size)/2);
		return;
	}

	if(getShowType()==SHOW_TYPE_ESTIMATES){
		PositionMap::const_iterator it;
		PositionMap& estimate=getEstimateMap();
		for(it=estimate.begin();it!=estimate.end();it++){
			if(it->first==index){
				break;
			}
		}
		if(it!=estimate.end()){
			x=it->second;
			if(x==ESTIMATE_UNKNOWN){
				sprintf(c,"?");
			}
			else{
				sprintf(c,"%d",x);
			}

			for(it=estimate.begin();it!=estimate.end();it++){
				if(it->second==ESTIMATE_UNKNOWN){
					break;
				}
			}
			bool hasUnknown= it!=estimate.end();

			if(!hasUnknown){
				i = isMinimize() ? INT_MAX : INT_MIN;
				for(it=estimate.begin();it!=estimate.end();it++){
					if( (isMinimize() && it->second<i) || (!isMinimize() && it->second>i) ){
						i=it->second;
					}
				}
			}

			drawText(m_cr,c,startx,starty, m_cellSize,m_cellSize,true,true
					,hasUnknown || x!=i ? blackColor : bestColor,m_fontSize,true);
		}
	}

}

void ReversiArea::newGame(int moves,bool search){
	m_reversi->set(m_reversiFrame->getStartPosition());

	//switchTurn is change m_turn on BLACK, but if BLACK couldn't do a move (for some 3x3 initial positions) then m_turn will be white
	m_turn=WHITE;
	switchTurn();

	int i;
	for(i=1;i<=moves;i++){
		m_reversi->makeMove(index(m_turnList[i].x,m_turnList[i].y), m_turn);//no need animation here
		switchTurn();
	}

	if(search){
		m_functionWait[THREAD_ALL]=true;
	}
	updateAll();//anyway have to update because computer can thing on first move
	if(search){
		enableWidgets(false);
		g_thread_new("",make_needed_computer_moves_thread,gpointer(this));
	}
}

void ReversiArea::newGame() {
	setNewGameType();
	m_turnList.clear();
	m_turnList.push_back(Position());
	m_turnListPosition=0;
	newGame(0,true);
}

void ReversiArea::setNewGameType(){
	if(m_tableSize!=getTableSize() || m_minimize!=isMinimize() ){
		m_tableSize=getTableSize();
		m_minimize=isMinimize();
		if(m_reversi){
			m_reversi->free();
			delete m_reversi;
		}

#define M(size) else if(m_tableSize==size){if(m_minimize)m_reversi=new Reversi##size##Minimize();else m_reversi=new Reversi##size();}
		if(0){}
		M(6)
		M(5)
		M(4)
		M(3)
#undef M
		else{
			assert(0);
		}
	}
	m_reversi->prepare();
}

void ReversiArea::fillEstimateListThread() {
	int i,j,e,x,y;
	std::set<int> set;
	const int t=getTableSize();
	if(getShowType()==SHOW_TYPE_ESTIMATES){
		for(i=0;i<2;i++){
			for(y=0;y<t;y++){
				for(x=0;x<t;x++){
					if(possible(x,y,m_turn)){
						if(i==0){
							e = ESTIMATE_UNKNOWN;
							gdk_threads_add_idle(set_estimate,getCode(x,y,e) );
						}
						else{
							j=index(x,y);
							if(set.find(j)==set.end()){
								e = estimateTurn(j,m_turn,-t*t,t*t);
								std::vector<int> v=getSymmetryCells(j);
								std::vector<int>::const_iterator it;
								for(it=v.begin();it!=v.end();it++){
									set.insert(*it);
									gdk_threads_add_idle(set_estimate,getCode(*it,e) );
								}
							}
						}
					}
				}
			}
		}
		//best estimate should change last estimate color, so should redraw all estimates
		gdk_threads_add_idle(draw_table,0);
	}

}

void ReversiArea::mouseLeftButtonDown(GdkEventButton* event) {
	if(isThreadRunning(THREAD_ALL) || isGameOver() ){
		return;
	}
	int i=getCellIndex(event->x,event->y);
	if(i!=LAST_CELL_INVALID && possible(i,m_turn) ){
		enableWidgets(false);
		g_thread_new("",mouse_left_button_down_thread,gpointer(i));
	}
}

void ReversiArea::mouseMove(double x,double y) {
	if(isThreadRunning(THREAD_ALL)){
		return;
	}
	int i=getCellIndex(x,y);
	if(i!=m_lastCell){
		m_lastCell=i;
		draw();
	}
}

void ReversiArea::setEstimate(int index,int value) {
	if(value==ESTIMATE_CLEAR){
		getEstimateMap().erase(index);
	}
	else{
		getEstimateMap()[index]=value;
	}
	drawCell(index);
	int x,y;
	getXY(index,x,y);
	const int startx=getStartPoint()+x*(m_cellSize+1)+1;
	const int starty=getStartPoint()+y*(m_cellSize+1)+1;
	invalidateRect(startx,starty,m_cellSize,m_cellSize);
}

void ReversiArea::mouseLeave(GdkEventCrossing* event) {
	m_lastCell=LAST_CELL_INVALID;
	draw();
}

void ReversiArea::updateTurnList() {
	unsigned i;
	std::vector<Position>::const_iterator it;
	std::string s;
	char c[32];
	for(it=m_turnList.begin()+1,i=0 ; i<m_turnListPosition ; it++,i++){
		assert(it!=m_turnList.end());
		sprintf(c,"%s%2d.%s %c%d",i==0?"":"\n",i+1, it->color ==BLACK ? "black":"white" , 'a'+it->x,it->y+1);
		s+=c;
	}
	m_reversiFrame->setTurnList(s);
}

void ReversiArea::undoRedo(int index) {
	if(index==TOOLBAR_UNDOALL){
		m_turnListPosition=0;
	}
	else if(index==TOOLBAR_UNDO){
		m_turnListPosition--;
	}
	else if(index==TOOLBAR_REDO){
		m_turnListPosition++;
	}
	else if(index==TOOLBAR_REDOALL){
		m_turnListPosition=m_turnList.size()-1;
	}
	newGame(m_turnListPosition,false);

}

void ReversiArea::updateUndoRedo() {
	m_reversiFrame->updateButton(TOOLBAR_UNDOALL,m_turnListPosition>0);
	m_reversiFrame->updateButton(TOOLBAR_UNDO,m_turnListPosition>0);

	m_reversiFrame->updateButton(TOOLBAR_REDOALL,m_turnListPosition<m_turnList.size()-1);
	m_reversiFrame->updateButton(TOOLBAR_REDO,m_turnListPosition<m_turnList.size()-1);
}

void ReversiArea::updateAll() {
	updateTurnList();
	updateUndoRedo();
	draw();
	getScoreTurnArea().draw();
}

void ReversiArea::load() {
	std::string path;
	const char ESTIMATES[]="estimates";
	const char NONE[]="none";
	const char TURN[]="turn";

	if(fileChooser(true,path)){
		FILE*f=fopen(utf8ToLocale(path),"r");
		assert(f!=NULL);
		int i,x,y;
		char*p;
		int v[4];
		const int BUFFER_SIZE=128;
		char c[BUFFER_SIZE];
		fgets(c,BUFFER_SIZE,f);
		for(p=c,i=0;i<SIZE(v);i++){
			p=strchr(p,'=');
			assert(p!=NULL);
			v[i]=atoi(p+1);
			p+=2;
		}
		m_reversiFrame->setParameters(v);
		//set new table size, need before m_turnList.push_back because we use index() function
		setNewGameType();
		m_turnListPosition=v[3];

		m_turnList.clear();
		m_turn=WHITE;

		while(fgets(c,BUFFER_SIZE,f)){
			p=c;

			switchTurn();

			if(strncmp(p,TURN,strlen(TURN))==0){
				p+=strlen(TURN)+1;
				x=p[0]-'a';
				y=p[1]-'1';
			}
			else{
				x=y=0;
			}
			m_turnList.push_back(Position(x,y,m_turn));

			p=strstr(c,ESTIMATES);
			assert(p!=NULL);
			p+=strlen(ESTIMATES);
			if(strncmp(p+1,NONE,strlen(NONE))==0){
				continue;
			}

			p--;

			do{
				p+=2;
				i=index(p[0]-'a',p[1]-'1');//strtol change p, so use it now
				m_turnList.back().map[i]=strtol(p+2,&p,10);
			}while(*p==',');
		}

		fclose(f);
		newGame(m_turnListPosition,true);
	}
}

void ReversiArea::save() {
	std::string path;
	std::vector<Position>::const_iterator it;
	PositionMap::const_iterator itm;
	if(fileChooser(false,path)){
		if(path.find('.')==path.npos || path.rfind('.')<path.rfind('\\') ){//add default extension
			path+=".rvs";
		}
		FILE*f=fopen(utf8ToLocale(path),"w+");
		assert(f!=NULL);
		fprintf(f,"tableSize=%d minimize=%d startPosition=%d currentTurn=%d\n",getTableSize(),isMinimize()?1:0
				,getStartPosition(),m_turnListPosition );
		for(it=m_turnList.begin() ; it!=m_turnList.end() ; it++ ){
			if(it!=m_turnList.begin()){
				fprintf(f,"turn %c%d ", 'a'+it->x, it->y+1);
			}

			fprintf(f,"estimates");
			if(it->map.empty()){
				fprintf(f," none");
			}
			else{
				for(itm=it->map.begin() ; itm!=it->map.end() ; itm++ ){
					fprintf(f,"%s%s %d",itm==it->map.begin()?" ":", ", toString(itm->first).c_str() , itm->second );
				}
			}
			fprintf(f,"\n");
		}

		fclose(f);
	}
}

bool ReversiArea::fileChooser(bool open,std::string& filepath) {
	//always show images for buttons
  g_object_set( gtk_settings_get_default(), "gtk-button-images", TRUE, NULL);

  GtkWidget *button;

	GtkWidget *dialog = gtk_file_chooser_dialog_new (open ? "open":"save",
		GTK_WINDOW(m_reversiFrame->m_window),
		open ? GTK_FILE_CHOOSER_ACTION_OPEN : GTK_FILE_CHOOSER_ACTION_SAVE,

		open ? "_Open" : "_Save", GTK_RESPONSE_ACCEPT,
		"_Cancel", GTK_RESPONSE_CANCEL,
		NULL);

	//w=gtk_image_new_from_icon_name (open ? "folder-open" : "media-floppy",  GTK_ICON_SIZE_LARGE_TOOLBAR);
	button= gtk_dialog_get_widget_for_response (GTK_DIALOG(dialog),GTK_RESPONSE_ACCEPT);
	gtk_button_set_image(GTK_BUTTON(button),image(open?"open32.png":"save32.png"));

	//w=gtk_image_new_from_icon_name ("emblem-unreadable",  GTK_ICON_SIZE_LARGE_TOOLBAR);
	button= gtk_dialog_get_widget_for_response (GTK_DIALOG(dialog),GTK_RESPONSE_CANCEL);
	gtk_button_set_image(GTK_BUTTON(button),image("cancel32.png"));

	GtkFileFilter *gtkFilter;
	const char* ext[]={"rvs","*"};
	const char* name[]={"reversi files","all files"};
	std::string s;
	assert(SIZE(ext)==SIZE(name));
	int i;
	for(i=0;i<SIZE(ext);i++){
		gtkFilter= gtk_file_filter_new();
		s="*.";
		s+=ext[i];
		gtk_file_filter_add_pattern(gtkFilter, s.c_str() );
		s="("+s+") "+name[i];
		gtk_file_filter_set_name(gtkFilter, s.c_str() );
		gtk_file_chooser_add_filter(GTK_FILE_CHOOSER(dialog), gtkFilter);
	}

	bool ok=gtk_dialog_run (GTK_DIALOG (dialog)) == GTK_RESPONSE_ACCEPT;
	if (ok){
		char *file = gtk_file_chooser_get_filename (GTK_FILE_CHOOSER (dialog));
		filepath=file;//utf8 locale
		g_free (file);
	}
	gtk_widget_destroy (dialog);
	return ok;

}

void ReversiArea::makeNeededComputerMovesThread() {
	m_functionWait[THREAD_ALL]=true;
	int x,y;
	int t=getTableSize();
	while( (m_turn==BLACK && getComputerPlayer()==COMPUTER_BLACK) || (m_turn==WHITE && getComputerPlayer()==COMPUTER_WHITE) ){
		for(x=0;x<getTableSize();x++){
			for(y=0;y<getTableSize();y++){
				if(possible(x,y,m_turn)){
					gdk_threads_add_idle(set_estimate,getCode(x,y,ESTIMATE_UNKNOWN ) );
				}
			}
		}

		std::vector<int> v=m_reversi->getOptimalMoves(m_turn,-t*t,t*t);
		srand(time(NULL));//need to call srand inside thread
		int index=v[rand()%v.size()];

		for(x=0;x<getTableSize();x++){
			for(y=0;y<getTableSize();y++){
				if(possible(x,y,m_turn)){
					gdk_threads_add_idle(set_estimate,getCode(x,y,ESTIMATE_CLEAR ) );
				}
			}
		}
		makeMoveThread(index,m_turn);


	}
	fillEstimateListThread();
	//since it's end of any thread function signal about all threads finished
	finishWaitFunction(THREAD_ALL);
	gdk_threads_add_idle(enable_widgets,0);
	gdk_threads_add_idle(update_all,0);
}


void ReversiArea::showTypeEstimatesOn() {
	enableWidgets(false);
	g_thread_new("",fill_estimate_list_thread,gpointer(this));
}

void ReversiArea::makeMoveThread(int index, int white) {
	if(isAnimation()){
		m_animationFlips=flipList(index,white);//get flips before make move
	}

	m_reversi->makeMove(index,white);

	//intermediate update turn list, score, turn
	int x,y;
	getXY(index,x,y);
	startWaitFunction(THREAD_INNER,create_new_turn,getCode(x,y,white));
	waitForFunction(THREAD_INNER);

	if(isAnimation()){
		m_animationStep = 0;
		m_functionWait[THREAD_INNER]=true;
		m_animationTimer=g_timeout_add(ANIMATION_STEP_TIME, timer_animation_handler, (gpointer) this);
		waitForFunction(THREAD_INNER);
	}
	else{
		gdk_threads_add_idle(draw_table,0);//draw while computer think on next turn
	}

	switchTurn();
	startWaitFunction(THREAD_INNER,update_turns_list_score_turn,NULL);
	waitForFunction(THREAD_INNER);

}

void ReversiArea::animationStep() {
	//0<=m_animationStep<ANIMATION_STEPS
	if(m_animationStep++==ANIMATION_STEPS-1){
		m_animationFlips.clear();
		draw();
		g_source_remove(m_animationTimer);
		finishWaitFunction(THREAD_INNER);
	}
	else{
		draw();
	}
}

void ReversiArea::mouseLeftButtonDownThread(int i) {
	m_functionWait[THREAD_ALL]=true;
	m_turnList.resize(m_turnListPosition+1);//clear end of turn list, now go to another path
	makeMoveThread(i,m_turn);
	makeNeededComputerMovesThread();
}

void ReversiArea::startWaitFunction(THREAD_ENUM e,GSourceFunc function, gpointer data) {
	m_functionWait[e]=true;
	gdk_threads_add_idle(function,data);
}

void ReversiArea::finishWaitFunction(THREAD_ENUM e) {
  g_mutex_lock (m_mutex+e);
  g_cond_signal (m_condition+e);
  m_functionWait[e]=false;
  g_mutex_unlock (m_mutex+e);
}

void ReversiArea::waitForFunction(THREAD_ENUM e) {
	g_mutex_lock (m_mutex+e);
  while(m_functionWait[e]){
  	g_cond_wait (m_condition+e, m_mutex+e);
  }
	g_mutex_unlock (m_mutex+e);
}

