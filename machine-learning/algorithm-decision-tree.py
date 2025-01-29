import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import graphviz

# Set display options for pandas DataFrames
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Load the seeds dataset
df = pd.read_csv('seeds.csv')

# Separate variables into the measurements (inputs) and class (ouputs)
inputs = df.drop(columns=['class'])
outputs = df['class']

# Split the data into training and testing sets
inputs_train, inputs_test, outputs_train, outputs_test = train_test_split(inputs, outputs, test_size=0.2, random_state=17)

# The code for applying decision trees has been generated using ChatGPT
# Initialize the Decision Tree model
dtree = DecisionTreeClassifier(random_state=17)

# Train the model
dtree.fit(inputs_train, outputs_train)

# Visualize the decision tree
dot_data = export_graphviz(dtree, out_file=None,
                           feature_names=inputs.columns,
                           class_names=outputs.unique().astype(str),  # Convert class names to string
                           filled=True, rounded=True,
                           special_characters=True)

graph = graphviz.Source(dot_data)
graph.render("decision_tree_seeds")
graph.view()

# Make predictions on the test set
outputs_pred = dtree.predict(inputs_test)

# Show actual vs predicted class for testing dataset
comparison_values = pd.DataFrame({
    'Actual Class': outputs_test,
    'Predicted Class': outputs_pred
})
print(comparison_values)

# Evaluate the model

# Classification report
print("\nClassification Report:")
print(classification_report(outputs_test, outputs_pred, digits=4))

# Confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(outputs_test, outputs_pred, labels=outputs.unique())
print(cm)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=outputs.unique(), yticklabels=outputs.unique())
plt.title('Confusion Matrix - Decision Tree')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()
