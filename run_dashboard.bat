@echo off
rem Launches the BIA405 dashboard with the working Python 3.13 install.
rem (Anaconda's numpy 1.26 + scipy 1.17 pair is broken, so the bare
rem  "streamlit" command crashes - this bypasses PATH entirely.)
"C:\Users\ASUS\AppData\Local\Programs\Python\Python313\python.exe" -m streamlit run "%~dp0dashboard.py"
pause
