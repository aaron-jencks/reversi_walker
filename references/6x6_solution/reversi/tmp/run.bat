@echo off
REM use "run.bat N_THREADS"
REM or "run.bat"  run 8 threads

set count=%1
IF "%1"=="" set count=8
set /a count-=1
for /l %%c in (0,1,%count%) DO  (
	start /LOW /B reversi.exe i.txt 6 %%c 1
)
