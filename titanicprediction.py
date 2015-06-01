import pprint
import time
from apiclient import discovery
from apiclient import sample_tools
from oauth2client import client
import csv as csv
import numpy as np
import json

def main():
# Authenticate to the access  
  service, flags = sample_tools.init(
      [], 'prediction', 'v1.6', __doc__, 'titanic.json', parents=[],
      scope='https://www.googleapis.com/auth/prediction')
# Test data_test
  test_data = csv.reader(open('test_data.csv','rb'))
  test_data.next()
  data1 = []
  data = []
  for row in test_data:
	data.append(row)
  data1 = np.array(data)
  correct_answer = 0
  for row in range(0,100):
  	try:
# Get access to the Prediction API.
    		prediction = service.trainedmodels()
# Make a prediction from the model built previously using the API
		Pclass = data[row][1]
		sex = data[row][2]
  		Age = data[row][3]
		SibSp = data[row][4]
		Parch = data[row][5]
		Fare = data[row][6]
		Embarked = data[row][7]
    		result = prediction.predict(project='256748303744', id='titanic5', body={ "input": {  "csvInstance": 		            [Pclass,sex,Age,SibSp,Parch,Fare,Embarked] }}).execute()
#print 'Prediction results...'
#Compare the result with test labels	
		if (result[u'outputLabel'] == data[row][0]):
			correct_answer = correct_answer + 1


    		print(str(row)+" - prediction: " + result[u'outputLabel']+ ", label: " + data[row][0])

  	except client.AccessTokenRefreshError:
   	 	print ("The credentials have been revoked or expired, please re-run"
      		"the application to re-authorize")

  print(str(correct_answer)+'%'+' correct')


if __name__ == '__main__':
  main()
