# Slotter
A simple tool to assign workers to timeslots.  
Written with way to much AI, terrible code hygiene and very inefficient.  
Maybe I will clean up in the future, maybe I won't.

## Dependencies
Best create a venv (choose a name as you see fit):  
```
python -m venv my_venv
```
Then install the requirements using pip:  
```
pip install -r requirements.txt
```

## Usage
Call via CLI, provide options and a preference matrix:
```
python slotter.py -m 1 -M 2 preferences.csv
```

Preferences could look like this:
```
Name,Slot1,Slot2,Slot3
Alice,1,2,3
Bob,3,2,1
Carol,2,3,1
```

Available options:
```
-m      minimal number of slots assigned per person
-M      maximum number of slots assigned per person
-l      log level [INFO | DEBUG | WARNING | ERROR | CRITICAL]
```


