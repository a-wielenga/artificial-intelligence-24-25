import pandas as pd
from sklearn.model_selection import train_test_split

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

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

testing_dataset['predicted_species'] = testing_dataset.apply(classify_penguin, axis=1)
print(testing_dataset[['species', 'predicted_species']])
