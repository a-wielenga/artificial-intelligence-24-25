import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

df = pd.read_csv('penguins.csv')

# Split the data into 80% training and 20% testing
training_dataset, testing_dataset = train_test_split(df, test_size=0.2, random_state=17)

# The code for the following charts & graphs for inital data analysis has been generated using ChatGPT, with edits to generate a graph for each relationship between variables to be explored

# Fig 1. Bar Chart - Island vs Species
species_island_count = training_dataset.groupby(['species', 'island']).size().reset_index(name='count')

plt.figure(figsize=(10, 6))
sns.barplot(x='island', y='count', hue='species', data=species_island_count)

plt.xlabel('Island')
plt.ylabel('Number of Penguins')
plt.title('Figure 1 - Number of Penguins of Each Species on Different Islands')
plt.legend(title='Species')

# Fig 2. Scatter Graph - Bill Length vs Bill Depth
plt.figure(figsize=(8, 6))
species = training_dataset['species'].unique()

for species_name in species:
    species_data = training_dataset[training_dataset['species'] == species_name]
    plt.scatter(species_data['bill_length_mm'], species_data['bill_depth_mm'], label=species_name, alpha=0.6)

plt.xlabel('Bill Length (mm)')
plt.ylabel('Bill Depth (mm)')
plt.title('Figure 2 - Bill Length vs Bill Depth by Species')
plt.legend(title='Species')

# Fig 3. Scatter Graph - Bill Length vs Flipper Length
plt.figure(figsize=(8, 6))

for species_name in species:
    species_data = training_dataset[training_dataset['species'] == species_name]
    plt.scatter(species_data['bill_length_mm'], species_data['flipper_length_mm'], label=species_name, alpha=0.6)

plt.xlabel('Bill Length (mm)')
plt.ylabel('Flipper Length (mm)')
plt.title('Figure 3 - Bill Length vs Flipper Length by Species')
plt.legend(title='Species')

# Fig 4. Scatter Graph - Bill Length vs Body Mass
plt.figure(figsize=(8, 6))

for species_name in species:
    species_data = training_dataset[training_dataset['species'] == species_name]
    plt.scatter(species_data['bill_length_mm'], species_data['body_mass_g'], label=species_name, alpha=0.6)

plt.xlabel('Bill Length (mm)')
plt.ylabel('Body Mass (g)')
plt.title('Figure 4 - Bill Length vs Body Mass by Species')
plt.legend(title='Species')

# Fig 5. Scatter Graph - Bill Depth vs Flipper Length
plt.figure(figsize=(8, 6))

for species_name in species:
    species_data = training_dataset[training_dataset['species'] == species_name]
    plt.scatter(species_data['bill_depth_mm'], species_data['flipper_length_mm'], label=species_name, alpha=0.6)

plt.xlabel('Bill Depth (mm)')
plt.ylabel('Flipper Length (mm)')
plt.title('Figure 5 - Bill Depth vs Flipper Length by Species')
plt.legend(title='Species')

# Fig 6. Scatter Graph - Bill Depth vs Body Mass
plt.figure(figsize=(8, 6))

for species_name in species:
    species_data = training_dataset[training_dataset['species'] == species_name]
    plt.scatter(species_data['bill_depth_mm'], species_data['body_mass_g'], label=species_name, alpha=0.6)

plt.xlabel('Bill Depth (mm)')
plt.ylabel('Body Mass (g)')
plt.title('Figure 6 - Bill Depth vs Body Mass by Species')
plt.legend(title='Species')

# Fig 7. Scatter Graph - Flipper Length vs Body Mass
plt.figure(figsize=(8, 6))

for species_name in species:
    species_data = training_dataset[training_dataset['species'] == species_name]
    plt.scatter(species_data['flipper_length_mm'], species_data['body_mass_g'], label=species_name, alpha=0.6)

plt.xlabel('Flipper Length (mm)')
plt.ylabel('Body Mass (g)')
plt.title('Figure 7 - Flipper Length vs Body Mass by Species')
plt.legend(title='Species')

# Plots all the graphs all in one go
plt.show()
