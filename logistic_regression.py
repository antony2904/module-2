from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score
from sklearn import metrics 
from sklearn.datasets import load_iris 

iris = load_iris() 
x = iris.data 
y = iris.target
class_names = iris.target_names  # Get the actual class names

from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = 4)
logreg = LogisticRegression() 
logreg.fit(x_train, y_train)  
y_predict = logreg.predict(x_test) 
print(metrics.accuracy_score(y_test, y_predict))

# Now, let's make predictions on a new set of data
new_data = [[5.1, 3.5, 1.4, 0.2],  # Example data point 1
[6.2, 3.4, 5.4, 2.3],  # Example data point 2
[7.7, 3.0, 6.1, 2.3]]  # Example data point 3

# Use the trained model to predict the target class for the new data

predicted_classes = logreg.predict(new_data)

# Convert predicted values to class labels (rounded to the nearest integer)

predicted_classes = [round(prediction) for prediction in predicted_classes]

# Map predicted class indices to class names

predicted_class_names = [class_names[predicted_class] for predicted_class in predicted_classes]

# Print the predicted class names for the new data

print('Predicted Class Names for New Data:')
for i, predicted_class_name in enumerate(predicted_class_names):
    print(f'Data Point {i+1}: {predicted_class_name}')