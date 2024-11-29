import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import precision_recall_curve, auc
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize

# Set display options for pandas DataFrames
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Load the dataset
df = pd.read_csv('penguins.csv')

# Split the data into 80% training and 20% testing
training_dataset, testing_dataset = train_test_split(df, test_size=0.2, random_state=17)

# Classification using chaining rules (forward chaining)
def classify_penguin(row):
    # Rules based on island
    if row['island'] == 'Torgersen':
        return 'Adelie'
    # Rules based on bill depth and bill length
    elif row['bill_depth_mm'] > 16 and row['bill_length_mm'] < 45:
        return 'Adelie'
    elif row['bill_depth_mm'] > 16 and row['bill_length_mm'] > 45:
        return 'Chinstrap'
    elif row['bill_depth_mm'] < 16:
        return 'Gentoo'
    # Rules based on flipper length and bill length
    elif row['flipper_length_mm'] < 205 and row['bill_length_mm'] < 45:
        return 'Adelie'
    elif row['flipper_length_mm'] < 205 and row['bill_length_mm'] > 45:
        return 'Chinstrap'
    elif row['flipper_length_mm'] > 205:
        return 'Gentoo'
    # Rules based on bill length and body mass
    elif row['bill_length_mm'] < 43:
        return 'Adelie'
    elif row['bill_length_mm'] > 43 and row['body_mass_g'] < 4250:
        return 'Chinstrap'
    elif row['bill_length_mm'] > 43 and row['body_mass_g'] > 4250:
        return 'Gentoo'
    # Rules based on flipper length and bill depth
    elif row['flipper_length_mm'] > 200 and row['bill_depth_mm'] < 18:
        return 'Gentoo'
    # Rules based on body mass and bill depth
    elif row['body_mass_g'] > 4000 and row['bill_depth_mm'] < 17:
        return 'Gentoo'
    # Rules based on flipper length and body mass
    elif row['flipper_length_mm'] > 205:
        return 'Gentoo'
    # If no rules apply
    return 'Unknown'

# Apply classification rules to the testing dataset
testing_dataset['predicted_species'] = testing_dataset.apply(classify_penguin, axis=1)

# Show predicted vs actual species for testing dataset
print(testing_dataset[['species', 'predicted_species']])

# The code for classification metrics for evaluation has been generated using ChatGPT
# Actual labels and predicted labels
y_true = testing_dataset['species']
y_pred = testing_dataset['predicted_species']

# Accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Precision, Recall, F1 Score (for multi-class classification)
precision = precision_score(y_true, y_pred, average='macro', labels=y_pred.unique())
recall = recall_score(y_true, y_pred, average='macro', labels=y_pred.unique())
f1 = f1_score(y_true, y_pred, average='macro', labels=y_pred.unique())
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Define the order of the classes explicitly
class_order = ['Adelie', 'Chinstrap', 'Gentoo']

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred, labels=class_order)
print(f'Confusion Matrix:\n{cm}')

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm.T, annot=True, fmt='g', cmap='Blues', xticklabels=class_order, yticklabels=class_order)
plt.title('Figure 8 - Confusion Matrix - Forward Chaining')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()

