# Gaming Decision Challenge:
- CSE5522 Challenge
- Shahriar Hooshmand
- Ohio State University, Columbus, OH

## Project Description:
------------------------------------
My family likes to play games, but considers a wide variety of factors when considering which game to play. I’ve created a dataset of instances on deciding whether to play “Apples To Apples” or “Settlers of Catan” based on the following attributes:

- dayOfWeek: Weekday, Saturday, Sunday • timeOfDay: morning, afternoon, evening • timeToPlay: < 30, 30 − 60, > 60
- mood: silly, happy, tired
- friendsVisiting: no, yes • kidsPlaying: no, yes
- atHome: no, yes
- snacks: no, yes
- game (predicted value): SettersOfCatan, ApplesToApples

The dataset comprises the following files:
- `game_attributes.txt`: The list of attributes and their possible values
- `game_attrdata train.dat`: A comma-separated list of values for each attribute (in the same order as in ) for each training instance, with one instance per line
- `game_attrdata_test.dat`: The test instances in the same format

Here we implement **averaged perceptron algorithm** to perform the decision. The detailed [project description](data/project_desc.pdf) and [report of analyses](proj_report.pdf) are also attached. 

## To use this software:
------------------------------------
1. Install the libraries in [requirements.txt](requirements.txt) to be able to run the scripts. This can be done by: 
    ```bash
    pip install -r requirements.txt 
    ```  
2. Run the following command to perform the analysis:

    ```bash
	python3 perceptron.py <train_data> <test_data> 
    ```  

* For example:

    ```bash
	python3 perceptron.py "data/game_attrdata_train.dat" "data/game_attrdata_test.dat"
    ```  

Outputs are logged in `output.log`. Results of analysis are plotted in [`Training.pdf`](Training.pdf). 


## Feedback, bugs, questions 
-------------------------------
Please reach out to me by email to shahriarhoushmand@gmail.com for any inquiry. Comments and feedbacks are greatly appreciated. 
