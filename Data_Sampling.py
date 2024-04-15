
# importing the required libraries
import pandas as pd
import numpy as np

# loading the reddit_opinion_ru_ua dataset
data = pd.read_csv(r'C:\Users\acer\Dropbox\PC\Downloads\reddit_data.csv')

num_zeros = (data['controversiality'] == 1).sum()

print("Number of zeros in the column:", num_zeros)

# calculating the total number of comments for each post
total_comments = data.groupby('post_id').size().reset_index(name='total_comments')

# merging total_comments to the original DataFrame
data = pd.merge(data, total_comments, on='post_id')

data.head()

# normalizing the total_comments column to obtain weights
data['weights'] = data['total_comments'] / (data['total_comments'].min()*10)

# rounding down the normalized comments
data['weights'] = data['weights'].apply(np.floor)

# generating the sampled dataset
# performing weighted random sampling
sampled_df = pd.DataFrame(columns=data.columns)
for post_id, group in data.groupby('post_id'):
    num_comments = int(group['weights'].iloc[0])
    sampled_comments = group.sample(n=num_comments, random_state=42)
    sampled_df = pd.concat([sampled_df, sampled_comments], ignore_index=True)

print("Sampled Dataset:")
sampled_df

# dropping the 'weights' column
sampled_df = sampled_df.drop(columns=['weights'])

sampled_df.info()

# checking for duplicates in the sampled dataframe
duplicate_values = sampled_df[sampled_df.duplicated(subset='post_id')]

# counting the number of duplicate rows found
num_duplicates = duplicate_values.shape[0]

# printing the number of duplicate rows found
print(num_duplicates)  

# counting the occurrences of 0s and 1s in the 'controversially' column
controversy_counts = sampled_df['controversiality'].value_counts()

# printing the counts
print("Controversiality Counts:")
for label, count in controversy_counts.items():
    print(f"{label}: {count}")

# counting the number of unique post_ids falling under controversiality 0 and 1
controversy_counts = sampled_df.groupby('controversiality')['post_id'].nunique()

# printing the results
print("Controversiality Counts:")
print("Controversiality 0:", controversy_counts.get(0, 0))
print("Controversiality 1:", controversy_counts.get(1, 0))

# selecting all the rows having controversiality 1
controversial_rows = sampled_df[sampled_df['controversiality'] == 1]

# sampling all controversial rows
sampled_controversial = controversial_rows

# selecting non-controversial rows
non_controversial_rows = sampled_df[sampled_df['controversiality'] == 0]

# grouping the DataFrame by 'post_id'
grouped_by_post_id = non_controversial_rows.groupby('post_id')

# Get the unique post_ids and reset the index to turn post_id into a column
unique_post_ids = grouped_by_post_id.size().reset_index(name='count')

# Randomly select 'post_ids'
# Specify the number of post_ids to select
num_post_ids_to_select = 4000
selected_post_ids = unique_post_ids.sample(n=num_post_ids_to_select, random_state=42)['post_id'].unique()

# Initialize an empty list to store sampled rows
sampled_rows = []

# Iterate over selected post_ids and retrieve all rows associated with each
for post_id in selected_post_ids:
    sampled_rows.extend(non_controversial_rows[non_controversial_rows['post_id'] == post_id].values.tolist())

# Create a DataFrame from the sampled rows
sampled_non_controversial = pd.DataFrame(sampled_rows, columns=non_controversial_rows.columns)

# Concatenate controversial and non-controversial samples
sampled_df = pd.concat([sampled_controversial, sampled_non_controversial])

# Shuffle the dataset
sampled_df = sampled_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Display the sampled dataset
sampled_df.info()

# Count the occurrences of 0s and 1s in the 'controversially' column
controversy_counts = sampled_df['controversiality'].value_counts()

# Print the counts
print("Controversiality Counts:")
for label, count in controversy_counts.items():
    print(f"{label}: {count}")

# Count the number of unique post_ids falling under controversiality 0 and 1
controversy_counts = sampled_df.groupby('controversiality')['post_id'].nunique()

# Print the counts
print("Controversiality Counts:")
print("Controversiality 0:", controversy_counts.get(0, 0))
print("Controversiality 1:", controversy_counts.get(1, 0))

# downloading the resulting dataset
# sampled_df.to_csv('balanced_dataset.csv', index=False)




