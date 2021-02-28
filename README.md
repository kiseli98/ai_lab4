# AI Lab4 - ML Model Deployment (CI/CD)
This porject contains implementation of laboratory work for Artificial Intelligence course (UTM 2021).

 
The task was to publicly deploy ML model developed in lab 3 (https://github.com/kiseli98/ai_lab3/). The lab was implemented with the usage of Azure ML and Azure WebService tools.

*Note: This repo contains static model and the code for web service back-end/front-end. Dynamic model that adopts CI/CD process can be found here: https://github.com/kiseli98/azure-ml/  (Azure ML + GitHub Actions)*

# Approach

In order to make the model available for public it was decided to deploy it as a webservice. 

1. Firstly, we need to export the model as a serialized object (*.pkl file*)
2. To predict the median complex value, we need to collect the data from new input values provided in the form and then use our Linear Regression model to predict the output and display the result in the form. Hence, we are creating a simple HTML form for that purpose (*templates/index.html*).
3. We are using Flask to create web app (*application.py*)
4. Configuration and dependencies are listed in *Procfile* and *requirements.txt*
5. A free azure account is required https://azure.microsoft.com/en-in/free/
6. Once logged it, navigate to *App Services*, follow the instructions, link GitHub repo and then deploy the app 
7. CI/CD processes and be added via GitHub actions



