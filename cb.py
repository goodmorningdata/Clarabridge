import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from string import punctuation
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist

# Use Seaborn formatting for plots and set color palette.
sns.set()
sns.set_palette('Dark2')

# FUNCTION = column_insights
# Print number of NaN values, number of unique values, and unique
# values with count, mean, and median for a dataframe column.
def column_insights(df, cname):
    print('\n------------ {} ------------'.format(cname))
    print('Type: {}'.format(df[cname].dtype))
    print('Number of NaN values: {}'.format(df[cname].isna().sum()))
    num_unique = df[cname].nunique()
    print('Number of unique values: {}'.format(num_unique))
    if num_unique < 20:
        print(df[cname].value_counts())
        df_col = (df.groupby(cname)['RATING']
                .agg(['count', 'mean', 'median'])
                .reset_index())
        df_col = df_col.sort_values(by='mean')
        print(df_col)

# FUNCTION = rating_pie_chart
# Rating pie chart for all reviews.
def rating_pie_chart(df, title):
    df_pie = df['RATING'].value_counts()

    fig, ax = plt.subplots()
    ax.pie(df_pie, labels=df_pie.index)
    ax.set_aspect('equal')
    plt.title(title)
    plt.show()

# FUNCTION = segment_grid
# Plot average review rating per category value for all categories of
# interest as separate bar plots for each category arranged in a grid.
def segment_grid(df, cols, col_names):

    overall_mean = df['RATING'].mean()

    fig = plt.figure(figsize=(10,5))

    for i, col in enumerate(sorted(cols), start=1):
        plt.subplot(2,3,i)
        df_attr = (df.groupby(col)['RATING']
                     .agg(['mean', 'count'])
                     .reset_index())
        df_attr = df_attr.sort_values(by='mean')
        ax = plt.barh(df_attr[col], df_attr['mean'], color='blue', alpha=0.5, align='center')
        for j, p in enumerate(ax.patches):
            width = p.get_width()
            plt.text(0.25+p.get_width(),
                     p.get_y() + p.get_height()/2 - 0.1,
                     df_attr.iloc[j]['count'],
                     size=7, color='black')

        plt.axvline(x=overall_mean, color='orange', alpha=0.75)
        plt.xticks(range(0,6), size=7)
        plt.title(sorted(col_names)[i-1])

    plt.tight_layout()
    plt.show()

# FUNCTION = histogram_grid
# Plot histograms for dataframe data grouped by col parameter.
# Histograms are arranged in two rows with the number of columns
# dependent of the number of unique values in parameter column.
def histogram_grid(df, col):
    overall_mean = df['RATING'].mean()
    grid_cols = math.ceil(df[col].nunique()/2)
    vals = np.sort(df[col].unique())

    fig = plt.figure(figsize=(10,5))
    for i, val in enumerate(vals, start=1):
        plt.subplot(2,grid_cols,i)
        df_val = df.loc[df[col] == val]
        bins = np.arange(1,7) - 0.5
        plt.hist(df_val['RATING'], bins=bins, color='blue', alpha=0.5)
        plt.axvline(df_val['RATING'].mean(), color='k', linestyle='dashed',
                    linewidth=1, alpha=0.5)
        plt.axvline(x=overall_mean, color='orange', alpha=0.75)
        plt.xticks(range(1,6), size=7)
        plt.yticks(size=7)
        plt.xlim([0, 6])
        plt.title(val)

    plt.tight_layout()
    plt.show()

# FUNCTION = brand_insights
# Plots to provide insight into the categorical variable, BRAND_DESC.
def brand_insights(df):
    histogram_grid(df, 'BRAND_DESC')

# FUNCTION = loyalty_insights
# Plots to provide insight into the categorical variable, LOYALTY_DESC.
def loyalty_insights(df):
    histogram_grid(df, 'LOYALTY_DESC')

# FUNCTION = age_group_insights
# Plots to provide insight into the categorical variable, AGE_GROUP_DESC.
def age_group_insights(df):
    histogram_grid(df, 'AGE_GROUP_DESC')

# FUNCTION = find_high_freq_words
# Find top 30 occuring words in review text.
def find_high_freq_words(df):
    # Join text in Verbatim column for all rows
    text = []
    for index, row in df.iterrows():
        text.append(row.Verbatim)
    text = ' '.join(text)

    # Tokenize Verbatim column into words
    word_tokens = word_tokenize(text)

    # Remove stop words and punctuation
    stop_words = set(stopwords.words("english"))
    more_stop_words = ["n't", "'s", "...", "us", "got", "get", "one", "would", "could", "2", "3", "4", "nyc", "york", "hyatt", "hilton", "go", "le", "square"]
    stop_words.update(more_stop_words)
    words = [w for w in word_tokens if w.lower() not in stop_words]
    words = [w for w in words if w not in punctuation]

    #Create frequency distribution dataframe
    fdist = FreqDist(words)
    df = pd.DataFrame.from_dict(fdist, orient='index').reset_index()
    df.columns = ['word', 'count']
    df = df.sort_values(by=['count'], ascending=False)

    return df

# FUNCTION = plot_word_frequency
# Bar plot of occurence count of high frequency words.
def plot_word_frequency(df, title):
    fig, ax = plt.subplots()
    ax.barh(df.head(30)['word'], df.head(30)['count'], color='blue', alpha=0.5)
    ax.invert_yaxis()
    plt.title(title)
    plt.tight_layout()
    plt.show()

def get_sentences(df):
    # Join text in Verbatim column for all rows
    text = []
    for index, row in df.iterrows():
        text.append(row.Verbatim)
    text = ' '.join(text)

    return sent_tokenize(text)

def find_pain_points(df):
    # Filter dataframe for RATING = 1 or 2 and Sentence Sentiment =
    # Negative or Strongly negative.
    df_low = df.loc[df['RATING'].isin([1,2]) & df['Sentence Sentiment'].isin(['Negative', 'Strongly negative'])]

    # Find high frequency words for low reviews
    df_low_words = find_high_freq_words(df_low)
    plot_word_frequency(df_low_words,
        "High frequency words in negative reviews")

    pain_point_set = ["dirty", "rooms", "bed", "bathroom", "beds", "small", "manager", "booked", "staff", "desk", "service"]

    sent_low = get_sentences(df_low)

    df_low_sents = pd.DataFrame(columns=['word', 'sentence'])
    for word in pain_point_set:
        for s in sent_low:
            if word in s:
                df_low_sents = df_low_sents.append({'word': word, 'sentence':s}, ignore_index=True)

    print(df_low_sents)
    df_low_sents.to_excel('pain_points.xlsx')

def find_highlights(df):
    # Filter dataframe for RATING = 4 or 5 and Sentence Sentiment =
    # Positive or Strongly positive.
    df_high = df.loc[df['RATING'].isin([4,5]) & df['Sentence Sentiment'].isin(['Positive', 'Strongly positive'])]

    # Find high frequency words for high reviews
    df_high_words = find_high_freq_words(df_high)
    plot_word_frequency(df_high_words,
        "High frequency words in positive reviews")

    pain_point_set = [""]
    sent_high = get_sentences(df_high)

    df_high['Verbatim'].to_excel('highlights.xlsx')

def main():
    filename = 'EM_Onsite_Presentation_Jan_2019_v2.xlsx'
    df = pd.read_excel(filename, header=0)
    df['BRAND_DESC'].replace({"consolidated hotels intl":"con hotels intl"}, regex=True, inplace=True)

    # cols = ['NaturalId', 'Source', 'AGE', 'AGE_GROUP_DESC', 'AUTHOR',
    #         'BRAND_DESC', 'CITY', 'Document Date', 'GENDER', 'HOTEL',
    #         'Language', 'LOYALTY_DESC', 'Natural Id', 'NPS_LEVEL',
    #         'NPS_POSNEG_FLAG', 'Parent Natural ID', 'RATING', 'REGION_DESC',
    #         'RESPONSE_E', 'Time of Day(UTC)', 'TRIP_PURPOSE_DESC',
    #         'VerbatimType', 'Verbatim', 'Sentence Sentiment']
    #
    # include_cols = ['Source', 'BRAND_DESC', 'REGION_DESC', 'CITY',
    #                 'AGE_GROUP_DESC', 'GENDER',  'LOYALTY_DESC',
    #                 'NPS_LEVEL', 'TRIP_PURPOSE_DESC', 'RATING', 'Verbatim',
    #                 'Sentence Sentiment']
    #
    # report_cols = ['Source', 'BRAND_DESC', 'REGION_DESC', 'CITY',
    #                'AGE_GROUP_DESC', 'GENDER',  'LOYALTY_DESC',
    #                'NPS_LEVEL', 'TRIP_PURPOSE_DESC']
    #
    # exclude_cols = ['NaturalId', 'AGE', 'AUTHOR', 'Document Date', 'HOTEL',
    #                 'Language', 'Natural Id', 'NPS_POSNEG_FLAG',
    #                 'Parent Natural ID', 'RESPONSE_E', 'Time of Day(UTC)',
    #                 'VerbatimType']
    #
    insight_cols = ['BRAND_DESC', 'REGION_DESC', 'CITY', 'AGE_GROUP_DESC',
                    'LOYALTY_DESC', 'TRIP_PURPOSE_DESC']
    insight_cols_name = ['Brand', 'Region', 'City', 'Age group',
                         'Loyalty category', 'Trip purpose']

    # Information about each column in the dataframe
    # for i, c in enumerate(df.columns):
    #     column_insights(df, c)

    # Rating pie chart for all reviews
    # rating_pie_chart(df, "Rating Distribution")

    # Grid of rating by segment data
    # segment_grid(df, insight_cols, insight_cols_name)

    # Drill down into brand data
    # brand_insights(df)

    # Drill down into loyalty data
    # loyalty_insights(df)

    # Drill down into age group data
    # age_group_insights(df)

    find_pain_points(df)
    find_highlights(df)
    # words_df = process_text(df)
    # words_bar_chart(words_df, "All reviews")
    #
    # low_score_df = df[df['RATING'] < 3]
    # words_df = process_text(low_score_df)
    # words_bar_chart(words_df, "All reviews - low scores")
    #
    # high_score_df = df[df['RATING'] > 3]
    # words_df = process_text(high_score_df)
    # words_bar_chart(words_df, "All reviews - high scores")

    # Check that NPS_LEVEL is calculated from RATING and only use one.
    # detractor = Rating 1 or 2, neutral = Rating 3, promoter = Rating 4 or 5
    # df_nps = (df.groupby(['NPS_LEVEL', 'RATING'])
    #                .agg(['count', 'mean', 'median'])
    #                .reset_index())
    # print(df_nps)

if __name__ == '__main__':
    main()

# Rating Bar Plot
# for x in df['TRIP_PURPOSE_DESC'].unique():
#     print(x)
#     df_filter = df[df['TRIP_PURPOSE_DESC'] == x]
#     labels, counts = np.unique(df_filter["RATING"], return_counts=True)
#     print(labels)
#     print(counts)
#     plt.bar(labels, counts, align='center')
#
# plt.xlabel('Rating')
# plt.ylabel('Rating count')
# plt.title('Rating Distribution')
# plt.show()

# Bar plot - count of ratings vs. trip purpose

    # Bigrams
    # bgs = nltk.bigrams(word_tokens)
    # fdist = nltk.FreqDist(bgs)
    # for obj in fdist.most_common():
    #     if obj[1] > 2: print(obj)

    # Collocations
    #bigram_measures = nltk.collocations.BigramAssocMeasures()
    # #trigram_measures = nltk.collocations.TrigramAssocMeasures()
    #finder = BigramCollocationFinder.from_words(word_tokens, window_size=3)
    # finder.apply_freq_filter(2)
    # #print(finder.score_ngrams(bigram_measures.pmi))
    #print(finder.nbest(bigram_measures.pmi,30))
    # for k,v in finder.ngram_fd.items():
    #     print(k,v)
    #print(finder.nbest(bigram_measures.pmi, 10))


# NPS = Net Promoter Score = an index ranging from -100 to 100 that measures the willingness of customers to recommend a company's products or services to others. It is used as a proxy for gauging the customer's overall satisfaction with a company's product or service and the customer's loyalty to the brand.
