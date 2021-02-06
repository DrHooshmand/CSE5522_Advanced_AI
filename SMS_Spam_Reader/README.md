# SMS Spam classifier: Naive Bayes Model
- CSE5522 Challenge
- Shahriar Hooshmand
- Ohio State University, Columbus, OH

## Project Description:
------------------------------------
In this project, Naive Bayes model is employed to construct a spam classifier for SMSs, where the two document categories are spam and ham (i.e., not spam). The entire exercise is meant to be done programmatically. The detailed [project description](project_desc.pdf) and [report of results](proj_report.pdf) can be found attached. 



## To use this software:
------------------------------------
1. Install the libraries in [requirements.txt](requirements.txt) to be able to run the scripts. This can be done by: 

    ```bash
    pip install -r requirements.txt 
    ```  

2. Use python 3.6.1 to run the script as follows: 

3. For method 1 implementation (separation of words based on independence assumption in Naive bayes, see the explanation in the [report](proj_report.pdf): 

    ```bash
	python3.6 spamreader_Method1.py
    ```  

4. For method 2 implementation (scanning over the dictionary for calculating the condition probs, see the explanation in the [report](proj_report.pdf) ):

    ```bash
    python3.6 spamreader_Method2.py
    ```  

5. Also you can find the output of both methods along with the applied correction in the following 4 text files:
    ```bash
    Method1_normal
    Method1_bonus
    Method2_normal
    Method2_bonus
    ```  

## Feedback, bugs, questions 
-------------------------------
Please reach out to me by email to shahriarhoushmand@gmail.com for any inquiry. Comments and feedbacks are greatly appreciated. 
