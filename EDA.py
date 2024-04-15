#!/usr/bin/env python
# coding: utf-8

# # INSTALLING WORDCLOUD

# In[ ]:





# # LOADING DATASET

# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# In[3]:


df = pd.read_csv(r"C:\Users\HP\Downloads\sampled_dataset.csv")


# In[26]:


display(df)


# # HISTOGRAM OF CONTROVERSIALITY

# In[4]:


# Set seaborn style
sns.set_style("whitegrid")


# In[5]:


# 1. Histogram of Controversiality
plt.figure(figsize=(8, 6))
sns.histplot(df['controversiality'], bins=20, kde=True, color='red')
plt.title('Histogram of Controversiality')
plt.xlabel('Controversiality')
plt.ylabel('Frequency')
plt.show()


# # INSIGHTS:
# 
# The histogram of Controversiality shows a highly skewed distribution, with the majority of the data concentrated at the lower end of the scale near 0. This indicates that a large proportion of the observations have low levels of controversiality. However, there is a long tail towards the higher end of the scale, suggesting that there are some posts that are highly controversial.

# # BAR CHART OF SUBREDDITS

# In[8]:


# 2. Bar Chart of Subreddits
plt.figure(figsize=(10, 6))
sns.countplot(y='subreddit', data=df, order=df['subreddit'].value_counts().index[:10], color='red')
plt.title('Top 10 Subreddits by Frequency')
plt.xlabel('Frequency')
plt.ylabel('Subreddit')
plt.show()


# # INSIGHTS:
# 
# The subreddit "UkraineWarVideoReport" has the highest frequency, indicating that it is the most prevalent subreddit among the top 10. This suggests a significant level of engagement or activity related to videos and reports concerning the Ukraine war on this platform. Following closely are "UkraineRussiaReport" and "worldnews," which also exhibit substantial frequencies, indicating their prominence in discussions within the dataset. Other subreddits such as "UkrainianConflict," "ukraine," and "europe" also feature prominently, highlighting the prevalence of discussions related to Ukrainian affairs and European geopolitics. 

# # LINE PLOT OF COMMENT SCORE OVER TIME

# In[28]:


# 3. Line Plot of Comment Score over Time
df['created_time'] = pd.to_datetime(df['created_time'])
plt.figure(figsize=(10, 6))
sns.lineplot(x='created_time', y='score', data=df)
plt.title('Comment Score over Time')
plt.xlabel('Time')
plt.ylabel('Comment Score')
plt.show()


# # INSIGHTS:
# 
# The line plot of comment score over time showcases the fluctuation in comment scores throughout the years. It reveals a generally stable trend until around 2022, where a significant spike in comment scores occurs, indicating a surge in engagement or controversial discussions. This abrupt rise suggests a notable event or change in the platform dynamics that led to increased user interaction. The sustained high comment scores in the latter part of the timeline could signify continued interest or participation in discussions

# # SCATTER PLOT OF COMMENT SCORE VS UPS

# In[10]:


# 4. Scatter Plot of Comment Score vs. Ups
plt.figure(figsize=(8, 6))
sns.scatterplot(x='score', y='ups', data=df)
plt.title('Comment Score vs. Ups')
plt.xlabel('Comment Score')
plt.ylabel('Ups')
plt.show()


# # INSIGHTS:
# 
# The scatter plot of comment score versus ups reveals a strong positive correlation between these two variables. As the comment score increases, there is a notable trend of higher ups, indicating that highly-scored comments tend to receive more upvotes. This suggests that users are more likely to upvote comments that have already garnered significant attention or engagement from other users. However, it's important to note that while a majority of comments with low scores also have lower ups, there are some outliers where comments with relatively low scores still manage to receive a considerable number of upvotes. This could be attributed to various factors such as the relevance of the comment to the discussion, its humor, or its insightful nature, indicating that upvoting behavior is influenced by more than just the comment score alone.

# # BOX PLOT OF CONTROVERSIALITY BY SUBREDDIT

# In[29]:


# 5. Box Plot of Controversiality by Subreddit
plt.figure(figsize=(12, 8))
sns.boxplot(x='subreddit', y='controversiality', data=df)
plt.title('Controversiality by Subreddit')
plt.xlabel('Subreddit')
plt.ylabel('Controversiality')
plt.xticks(rotation=90)
plt.show()


# # INSIGHTS:
# 
# The box plot of controversy by subreddit showcases the distribution of controversy scores across different subreddits. However, the plot appears to be skewed towards one extreme, with most subreddits concentrated at either the minimum or maximum controversy score. This indicates that the controversy scores are not evenly distributed across subreddits but are rather heavily skewed towards either highly controversial or non-controversial topics within the dataset. It suggests that certain subreddits tend to generate significantly more controversial discussions compared to others, which could be attributed to various factors such as the nature of the topics discussed, the demographics of the subreddit's user base, or the moderation policies enforced within the subreddit. 

# # PIE CHART OF CONTROVERSIALITY DISTRIBUTION

# In[12]:


# 7. Pie Chart of Controversiality Distribution
plt.figure(figsize=(6, 6))
df['controversiality_label'] = df['controversiality'].apply(lambda x: 'Controversial' if x == 1 else 'Non-Controversial')
df['controversiality_label'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Controversiality Distribution')
plt.ylabel('')
plt.show()


# # INSIGHTS:
# 
# 
# The pie chart of controversy distribution illustrates the proportion of controversial and non-controversial instances within the dataset. It reveals that the vast majority, approximately 93.7%, of instances are classified as non-controversial, while only a small fraction, approximately 6.3%, are deemed controversial. This suggests that the dataset predominantly consists of content that is considered non-controversial, indicating a prevalence of discussions or topics that do not elicit significant disagreement or debate. However, the existence of a minority of controversial instances highlights the presence of contentious topics or discussions within the dataset, albeit to a lesser extent compared to non-controversial content.

# # HEATMAP OF CORRELATION MATRIX

# In[13]:


# Select only numeric columns
numeric_df = df.select_dtypes(include=['int64', 'float64'])

# 8. Heatmap of Correlation Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


# # INSIGHTS:
# 
# Positive correlations are depicted in shades of red, while negative correlations are shown in shades of blue. Stronger correlations are indicated by darker colors.
# 
# From the heatmap, we can observe several interesting patterns. For instance, there is a strong positive correlation between features such as post_score and total_comments, suggesting that posts with higher scores tend to attract more comments. Conversely, features like user_awardee_karma and user_total_karma exhibit weaker correlations with other variables, indicating less direct influence on other aspects of the dataset. Additionally, the diagonal line of dark red indicates perfect correlation between a feature and itself, as expected.

# # CREATING WORD CLOUD FOR CONTROVERSIAL COMMENTS

# In[14]:


from wordcloud import WordCloud

# Extract controversial and non-controversial comments
controversial_comments = df[df['controversiality'] == 1]['self_text'].str.cat(sep=' ')
non_controversial_comments = df[df['controversiality'] == 0]['self_text'].str.cat(sep=' ')


# In[15]:


# Create WordCloud for controversial comments
wordcloud_controversial = WordCloud(width=800, height=400, background_color='white').generate(controversial_comments)

# Plot WordCloud for controversial comments
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_controversial, interpolation='bilinear')
plt.title('Word Cloud for Controversial Comments')
plt.axis('off')
plt.show()


# # INSIGHTS:
# 
# 
# The word cloud generated from controversial comments provides valuable insights into the prominent topics and themes discussed within these comments. Larger and bolder words indicate higher frequency occurrences in the text. From the word cloud, we can observe that terms like "war," "Ukraine," and "Russia" appear prominently, suggesting that discussions revolve around geopolitical tensions and conflicts. Additionally, words like "NATO," "conflict," and "Putin" are also prevalent, indicating specific entities and topics of interest within the controversial comments

# # WORD CLOUD FOR NON - CONTROVERSIAL COMMENTS

# In[16]:


# Create WordCloud for non-controversial comments
wordcloud_non_controversial = WordCloud(width=800, height=400, background_color='white').generate(non_controversial_comments)


# Plot WordCloud for non-controversial comments
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_non_controversial, interpolation='bilinear')
plt.title('Word Cloud for Non-Controversial Comments')
plt.axis('off')
plt.show()


# # INSIGHTS:
# 
# The word cloud generated from non-controversial comments offers insights into the prevalent themes and topics discussed within these comments. Similar to the word cloud from controversial comments, larger and bolder words represent higher frequency occurrences in the text. In this word cloud, terms like "Ukraine," "Russia," and "war" still appear prominently, indicating that these topics are commonly discussed even in non-controversial contexts. Additionally, words like "Putin," "European," and "people" also stand out, suggesting discussions on political figures, regional dynamics, and general sentiments.

# # COMPARISON OF CONTROVERSIAL AND NON - CONTROVERSIAL COMMENTS BY SUBREDDIT 

# In[19]:


# plt.figure(figsize=(12, 6))
sns.countplot(x='subreddit', hue='controversiality', data=df)
plt.title('Controversiality by Subreddit')
plt.xlabel('Subreddit')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.legend(title='Controversiality', labels=['Non-Controversial', 'Controversial'])
plt.show()


# # INSIGHTS:
# 
# It provides insights into the disparity between controversial and non-controversial discussions within different subreddit communities. The graph indicates that certain subreddits have a higher frequency of controversial comments compared to others. This visualization helps in understanding the engagement levels and the nature of discussions within different subreddits, highlighting topics that evoke strong opinions or disagreements among users

# # COMMENT SCORE BY CONTROVERSIALITY

# In[20]:


plt.figure(figsize=(8, 6))
sns.boxplot(x='controversiality', y='score', data=df)
plt.title('Comment Score by Controversiality')
plt.xlabel('Controversiality')
plt.ylabel('Comment Score')
plt.xticks(ticks=[0, 1], labels=['Non-Controversial', 'Controversial'])
plt.show()


# # INSIGHTS:
# 
# The graph titled "Comment Score by Controversiality" illustrates the distribution of comment scores categorized by their controversiality. It reveals significant differences in comment scores between controversial and non-controversial comments. Controversial comments tend to have a wider range of scores, indicating a more polarized reception from the audience, while non-controversial comments generally cluster around lower scores with less variance. This suggests that controversial comments provoke diverse reactions, leading to both highly upvoted and heavily downvoted responses, whereas non-controversial comments typically receive more moderate scores.

# # TOTAL COMMENTS BY CONTROVERSIALITY

# In[22]:


plt.figure(figsize=(8, 6))
sns.violinplot(x='controversiality', y='total_comments', data=df)
plt.title('Total Comments by Controversiality')
plt.xlabel('Controversiality')
plt.ylabel('Total Comments')
plt.xticks(ticks=[0, 1], labels=['Non-Controversial', 'Controversial'])
plt.show()


# # INSIGHTS:
# 
# 
# The plot titled "Total Comments by Controversiality" showcases the distribution of total comments categorized by their controversiality. It highlights a notable disparity in the distribution of comments between controversial and non-controversial categories. Non-controversial comments tend to have a more symmetric distribution, indicating a relatively consistent level of engagement across different posts. Conversely, controversial comments exhibit a skewed distribution, with a greater concentration of comments towards higher counts. This suggests that controversial topics or discussions tend to attract a higher volume of comments compared to non-controversial ones

# # CONTROVERSIALITY DISTRIBUTION OVER TIME

# In[23]:


plt.figure(figsize=(12, 6))
df['created_time'] = pd.to_datetime(df['created_time'])
df['year_month'] = df['created_time'].dt.to_period('M')
sns.countplot(x='year_month', hue='controversiality', data=df)
plt.title('Controversiality Distribution over Time')
plt.xlabel('Year-Month')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Controversiality', labels=['Non-Controversial', 'Controversial'])
plt.show()


# # INSIGHTS:
# 
# The plot titled "Controversiality Distribution over Time" illustrates the distribution of controversial and non-controversial comments over different time periods. It reveals fluctuations in the volume of comments across months and years, with certain periods showing spikes in activity. Notably, there are peaks in controversial comments during specific months, indicating times of heightened controversy or engagement with contentious topics. Additionally, the visualization suggests that controversial discussions may occur in bursts, with intermittent periods of high and low activity. 

# # DISTRIBUTION OF COMMENT SCORES

# In[24]:


plt.figure(figsize=(8, 6))
sns.histplot(df['score'], bins=30, kde=True)
plt.title('Distribution of Comment Scores')
plt.xlabel('Comment Score')
plt.ylabel('Frequency')
plt.show()


# # INSIGHTS:
# 
# The plot titled "Distribution of Comment Scores" depicts the frequency distribution of comment scores within the dataset. It reveals that the majority of comments have low scores, clustered around zero, indicating a large number of comments with minimal engagement or interaction. Additionally, there are a few outliers with extremely high scores, representing comments that have received significant attention or positive feedback. The distribution suggests that while most comments may go unnoticed or receive little recognition, there are notable exceptions that garner substantial engagement.

# # PAIR PLOT

# In[27]:


numerical_vars = ['score', 'user_awardee_karma', 'post_score', 'total_comments']

# Create pair plot
sns.pairplot(df[numerical_vars])
plt.show()


# # INSIGHTS:
# 
# The pair plots visualize the relationships between different numerical variables within the dataset. From the plots, it can be observed that there are varying degrees of correlation between the variables. For example, the 'score' variable exhibits a positive correlation with 'user_awardee_karma' and 'post_score', indicating that higher scores tend to be associated with higher user karma and post scores. Similarly, the 'total_comments' variable shows a positive correlation with 'user_awardee_karma' and 'post_score', suggesting that more comments are linked with higher user karma and post scores.

# # CONCLUSION

# 
# The analysis conducted on the dataset reveals several insightful findings. Firstly, when comparing controversial and non-controversial comments across different subreddits, it becomes evident that certain subreddits generate more controversy than others. Additionally, examining the comment score by controversiality indicates that controversial comments tend to have higher scores, indicating greater engagement or interest from users. Moreover, the distribution of total comments by controversiality demonstrates that controversial comments are less frequent but can generate a substantial amount of discussion compared to non-controversial ones. Furthermore, exploring controversiality distribution over time unveils fluctuations, highlighting periods of heightened controversy. The distribution of comment scores showcases a skewed distribution, with a majority of comments having lower scores, but a small proportion receiving exceptionally high scores. Finally, the pair plots offer insights into the relationships between various numerical variables, revealing correlations that can aid in understanding user behavior and engagement patterns on the platform. Overall, these analyses provide a comprehensive understanding of comment dynamics, user engagement, and trends within the dataset.

# In[ ]:




