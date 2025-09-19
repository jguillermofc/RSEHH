@echo off

echo Program to execute tests
pause

set DEC=10 25 50 100

for %%A in (%DEC%) do (
    python hyperheuristic.py 10 %%A 100 10000 3 RSE 100 10000 MMD 1 Mean
)

pause
