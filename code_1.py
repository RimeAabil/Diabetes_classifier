# Importing libraries
#import sys
#sys.path.append(r'C:\Users\hp04\AppData\Local\Programs\Python\Python311\Lib\site-packages') 
#import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import time 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

# Importing tabulate for better table formatting
from tabulate import tabulate

# Importing termcolor to print text with colors 
import termcolor
from termcolor import colored

# Set Seaborn context and theme
sns.set_context("notebook")  # Valid contexts are 'paper', 'notebook', 'talk', 'poster'
sns.set_theme(style='whitegrid')  # Set the style to 'whitegrid'

print(colored('----------------------          Data collecting ...             --------------------------------','blue'))

time.sleep(3)
# Data collecting and cleaning   

# Importing the dataset
'''' It is a dataset available on kaggle : Diabetes Dataset For Beginners '''

# Correcting file path with raw string literal
path = r'C:\Users\hp04\Desktop\Perso\Diabetes_classifier\model\diabetes.csv'

# Reading the dataset into a pandas dataframe
data = pd.read_csv(path)

# Information about dataset attributes
'''
Pregnancies: To express the Number of pregnancies
Glucose: To express the Glucose level in blood
BloodPressure: To express the Blood pressure measurement
SkinThickness: To express the thickness of the skin
Insulin: To express the Insulin level in blood
BMI: To express the Body mass index
DiabetesPedigreeFunction: To express the Diabetes percentage. It is the probability of diabetes based on the person's family history  
DPF=∑(Weight of Relative Type × Age of Onset)
Age: To express the age
Outcome: To express the final result: 1 is Yes and 0 is No
'''

print(colored("Here's a sample of our dataset:",'blue'))
# Displaying the first 5 rows in a table format
print(data.head())
time.sleep(5)

print(colored('--------------------------------       Some statistical measures     -----------------------------','blue'))

# Statistical summary of the dataset using describe 
print(data.describe())
print(colored('\nDataset infos :','blue'))
print(data.info()) 

time.sleep(3)
# Null values existence test :
print(colored("\nThe number of null values in each collumn :",'blue'))
print(data.isnull().sum()) 

time.sleep(3)
# Required data types for each collumns :
print(colored("\nRequired data types for each collumns :",'blue'))
print(data.dtypes)
time.sleep(3)

print(colored('----------------------          Data visualization (using various plots)             --------------------------------','blue'))

# Set Seaborn style and context
sns.set_theme(style="whitegrid")

# Get the numerical columns for histograms, KDE, and box plots
numerical_cols = data.select_dtypes(include=['float64','int64']).columns

# Create a figure with subplots (2 rows and 3 columns to fit all 6 columns)
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(20, 12))

# Flatten the axes array for easy access
axes = axes.flatten()

# Plot histograms and KDE for numerical columns
for i, column in enumerate(numerical_cols):
    sns.histplot(data[column], bins=50, kde=True, ax=axes[i], color='blue')
    axes[i].set_title(f'{column} Distribution (Histogram + KDE)')
    axes[i].set_xlabel(column)
    axes[i].set_ylabel('Frequency')

# Adjust layout to avoid overlap
plt.tight_layout()
plt.show()
time.sleep(7)




print(colored('''------------------------             Correlation matrix            --------------------------------------'''))
'''Correlation measures the strength and direction of the linear relationship between two variables.'''

# Compute the correlation matrix
corr_matrix = data.corr()

# Sort the correlation values for 'Outcome' in descending order
corr_with_outcome = corr_matrix['Outcome'].sort_values(ascending=False)

# Display the sorted correlation values
print(colored('Correlation matrix:','blue'))
print(corr_with_outcome)



# Create a scatter plot: Outcome vs Glucose
plt.figure(figsize=(8, 6))  # Set the size of the plot
plt.scatter(data['Outcome'], data['Glucose'], s=100, alpha=0.08, color='blue')

# Add titles and labels
plt.title('Scatter Plot: Outcome vs Glucose')
plt.xlabel('Outcome (Diabetes: 1=Yes, 0=No)')
plt.ylabel('Glucose Level')

# Show the plot
plt.show()
# Create the message with colored parts
message = colored('\nCorrelation: ', 'blue') + "The correlation between Glucose and Outcome is high. It indicates a positive relationship: as glucose levels increase, the probability of the Outcome being 1 (diabetes positive) increases. This suggests that glucose levels are a key indicator of whether someone is likely to be diagnosed with diabetes."

# Print the message
print(message)
time.sleep(5)

'''---------------------------    Spotting and handling outliers       --------------------------'''

'''
Outliers can be detected using various methods, such as:
* IQR (Interquartile Range) Method: Values below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR are considered outliers.
'''
    
# Calculating the IQR (Interquartile Range) for BMI and Glucose
'''Q1_bmi = data['BMI'].quantile(0.25)
Q3_bmi = data['BMI'].quantile(0.75)
IQR_bmi = Q3_bmi - Q1_bmi

Q1_glucose = data['Glucose'].quantile(0.25)
Q3_glucose = data['Glucose'].quantile(0.75)
IQR_glucose = Q3_glucose - Q1_glucose

# Defining lower and upper bounds for outliers
lower_bound_bmi = Q1_bmi - 1.5 * IQR_bmi
upper_bound_bmi = Q3_bmi + 1.5 * IQR_bmi

lower_bound_glucose = Q1_glucose - 1.5 * IQR_glucose
upper_bound_glucose = Q3_glucose + 1.5 * IQR_glucose

# Identifying outliers in BMI and Glucose
outliers_bmi = data[(data['BMI'] < lower_bound_bmi) | (data['BMI'] > upper_bound_bmi)]
outliers_glucose = data[(data['Glucose'] < lower_bound_glucose) | (data['Glucose'] > upper_bound_glucose)]

print(f"Outliers in BMI:\n{outliers_bmi}")
print(f"Outliers in Glucose:\n{outliers_glucose}")

# Replacing outliers in BMI with median
median_bmi = data['BMI'].median()
median_glucose = data['Glucose'].median()

data['BMI'] = np.where((data['BMI'] < lower_bound_bmi) | (data['BMI'] > upper_bound_bmi), median_bmi, data['BMI'])
data['Glucose'] = np.where((data['Glucose'] < lower_bound_glucose) | (data['Glucose'] > upper_bound_glucose), median_glucose, data['Glucose'])
'''

'''---------------------             Seperating the data into training and testing data    -------------------------'''

x=data.drop("Outcome",axis=1)  #Dropping the last column in the dataset
#x contains all the independant variables of the dataset 
y=data["Outcome"]


#Training and testing data 
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=50)
'''
Here, 20% of the data will be used to test the model
Fixing the random seed to 50 for example ensures that the data is split the same way every time I run the code, to get consistant results
'''
# Initialize a list to store the cross-validation scores for each k
neighbors_range = range(1, 21)  # Try values of n_neighbors from 1 to 20
cv_scores = []  # List to store mean cross-validation scores for each k

# Loop over different values of k (number of neighbors)
for k in neighbors_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    # Perform cross-validation on the training data and calculate accuracy
    scores = cross_val_score(knn, X_train, Y_train, cv=5, scoring='accuracy')  # 5-fold cross-validation
    cv_scores.append(np.mean(scores))  # Get the mean accuracy score

# Find the value of k that gives the best score
best_k = neighbors_range[np.argmax(cv_scores)]
print(f"Best number of neighbors: {best_k}")

# Train the model with the best k
knn_best =KNeighborsClassifier(n_neighbors=best_k, weights='distance',algorithm='auto')
knn.fit(X_train, Y_train)
knn_best.fit(X_train, Y_train)

# Make predictions on the test data
Y_pred = knn_best.predict(X_test)
'''----------------------              Model evaluation : Accuracy test                  --------------------------------------'''

'''
The predict() method takes the feature matrix X_test as input.
For each instance in X_test, the KNN algorithm:
1-Computes the distance between the test instance and all training instances.
2-Selects the n_neighbors nearest training instances.
3-Predicts the class (or value, for regression) based on the majority class of the nearest neighbors 
'''
# Model's Accuracy (assessing how well the model performed)
# Model evaluation
print(colored("After training the model with the best number of neighbors...", 'red'))
print("Accuracy:", accuracy_score(Y_test, Y_pred))

#print the classification report and confusion matrix
print(classification_report(Y_test, Y_pred))
print(confusion_matrix(Y_test, Y_pred))

'''--------------------------                Test on user's data           ---------------------------  '''
print(colored("Test time :)",'green'))

import pandas as pd
import numpy as np

# Function to get user input and make prediction
def get_user_input_and_predict():
    
    # Print the disclaimer and accuracy
    print("Disclaimer: This model is just a prediction tool and cannot replace medical advice.")
    print("Model accuracy: 74% (based on the training dataset). Please consult a healthcare professional for a proper diagnosis.\n")
    
    try:
        # Collecting user input for each feature (with units specified)
        pregnancies = int(input("Enter number of pregnancies (unit: count): "))
        glucose = float(input("Enter glucose level (unit: mg/dL): "))
        blood_pressure = float(input("Enter blood pressure (unit: mmHg): "))
        skin_thickness = float(input("Enter skin thickness (unit: mm): "))
        insulin = float(input("Enter insulin level (unit: µU/mL): "))
        bmi = float(input("Enter BMI (Body Mass Index) (unit: kg/m²): "))
        diabetes_pedigree_function = float(input("Enter diabetes pedigree function (unit: scale from 0 to 1): "))
        age = int(input("Enter age (unit: years): "))
        
        # Creating a DataFrame for the user input, ensuring the same columns as the training data
        user_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin,
                                   bmi, diabetes_pedigree_function, age]],
                                  columns=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                                           "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"])

        # Make the prediction using the trained model
        prediction = knn_best.predict(user_data)
        
        # Show the result to the user
        if prediction[0] == 1:
            print("\nThe prediction is: Positive for diabetes (1).")
        else:
            print("\nThe prediction is: Negative for diabetes (0).")
    
    except ValueError:
        print("Invalid input. Please enter numeric values for all fields.")

get_user_input_and_predict()