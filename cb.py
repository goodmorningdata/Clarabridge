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
from wordcloud import WordCloud

# Use Seaborn formatting for plots and set color palette.
sns.set()
sns.set_palette('Dark2')

# FUNCTION = column_insights
# Print number of NaN values, number of unique values, and unique
# values with count, mean, and median for a dataframe column.
def column_insights(df, cname):
    print("\n------------ {} ------------".format(cname))
    print("Type: {}".format(df[cname].dtype))
    print("Number of NaN values: {}".format(df[cname].isna().sum()))
    num_unique = df[cname].nunique()
    print("Number of unique values: {}".format(num_unique))
    if num_unique < 20:
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

    overall_mean = df['RATING'].mean()
    print("\nRating mean = {:.2f}\n".format(overall_mean))

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
        ax = plt.barh(df_attr[col], df_attr['mean'], color='blue',
                      alpha=0.5, align='center')
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

    fig.savefig('segment_grid')

# FUNCTION = histogram_grid
# Plot histograms for dataframe data grouped by col parameter.
# Histograms are arranged in two rows with the number of columns
# dependent of the number of unique values in parameter column.
def histogram_grid(df, col):
    overall_mean = df['RATING'].mean()
    grid_cols = math.ceil(df[col].nunique()/2)
    vals = np.sort(df[col].unique())

    if (col == 'LOYALTY_DESC'):
        vals = ['silver', 'gold', 'platinum', 'standard', 'non-member']

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

    fig.savefig(col.lower())

# FUNCTION = brand_insights
# Plots to provide insight into the categorical variable, BRAND_DESC.
def brand_insights(df):
    histogram_grid(df, 'BRAND_DESC')

# FUNCTION = loyalty_insights
# Plots to provide insight into the categorical variable, LOYALTY_DESC.
def loyalty_insights(df):
    histogram_grid(df, 'LOYALTY_DESC')

# FUNCTION = city_insights
# Plots to provide insight into the categorical variable, CITY.
def city_insights(df):
    histogram_grid(df, 'CITY')

# FUNCTION = age_group_insights
# Plots to provide insight into the categorical variable, AGE_GROUP_DESC.
def age_group_insights(df):
    histogram_grid(df, 'AGE_GROUP_DESC')

# FUNCTION = join_verbatim
# Join all text in the Verbatim column into one string.
def join_verbatim(df):
    text = []
    for index, row in df.iterrows():
        text.append(row.Verbatim)

    return ' '.join(text)

# FUNCTION = find_high_freq_words
# Find top 30 occuring words in review text.
def find_high_freq_words(df):

    # Join text in Verbatim column for all rows
    text = join_verbatim(df)

    # Tokenize Verbatim column into words
    word_tokens = word_tokenize(text)

    # Remove stop words and punctuation
    stop_words = set(stopwords.words("english"))
    more_stop_words = ["n't", "'s", "...", "us", "got", "get", "one", "would",
                       "could", "two", "2", "3", "4", "nyc", "york", "hyatt",
                       "hilton", "go", "le", "square", "--", "new", "also"]
    stop_words.update(more_stop_words)
    words = [w for w in word_tokens if w.lower() not in stop_words]
    words = [w for w in words if w not in punctuation]

    # Create frequency distribution dataframe
    fdist = FreqDist(words)
    df = pd.DataFrame.from_dict(fdist, orient='index').reset_index()
    df.columns = ['word', 'count']
    df = df.sort_values(by=['count'], ascending=False)

    return df

# FUNCTION = plot_word_frequency
# Bar plot of occurence count of high frequency words.
def plot_word_frequency(df, title, filename):
    fig, ax = plt.subplots()
    ax.barh(df.head(30)['word'], df.head(30)['count'], color='blue', alpha=0.5)
    ax.invert_yaxis()
    plt.title(title)
    plt.tight_layout()
    plt.show()

    fig.savefig(filename)

# FUNCTION = get_sentences
# Split review text into sentences.
def get_sentences(df):

    # Join text in Verbatim column for all rows
    text = join_verbatim(df)

    return sent_tokenize(text)

# FUNCTION = show_wordcloud
# Create word cloud of text in Verbatim column.
def show_wordcloud(df, title = None):

    # Join text in Verbatim column for all rows
    text = join_verbatim(df)

    wordcloud = WordCloud(
        background_color = 'white',
        max_words = 200,
        #max_font_size = 50,
        scale = 3,
        width=600,
        height=400,
        colormap="Paired",
        random_state = 42
    ).generate(text)

    fig = plt.figure(figsize = (6, 6))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.show()
    plt.savefig('wordcloud.png', bbox_inches='tight')

# FUNCTION = find_pain_points
# Find high frequency words in negative reviews.
def find_pain_points(df):

    # Find high frequency words for all reviews
    df_all_words = find_high_freq_words(df)
    plot_word_frequency(df_all_words,
        "High frequency words in all reviews",
        "high_freq_all")

    # Filter dataframe for RATING = 1 or 2 and Sentence Sentiment =
    # Negative or Strongly negative.
    df_low = df.loc[df['RATING'].isin([1,2]) &
        df['Sentence Sentiment'].isin(['Negative', 'Strongly negative'])]

    print("\nThere are {} low reviews, a percentage of {:.0f} out of {} total "
          "reviews.".format(df_low.shape[0], df_low.shape[0]/df.shape[0]*100,
          df.shape[0]))

    # Find high frequency words for low reviews
    df_low_words = find_high_freq_words(df_low)
    plot_word_frequency(df_low_words,
        "High frequency words in negative reviews",
        "high_freq_neg")

    pain_point_set = ["dirty", "rooms", "bed", "bathroom", "beds", "small",
                      "manager", "booked", "staff", "desk", "service", "lobby"]

    # Find percent of low reviews that contain pain point set word.
    print("\nPercent of low reviews that pain point set words are found in:")
    total_low_reviews = df_low.shape[0]
    for w in pain_point_set:
        count = df_low['Verbatim'].str.contains(w).sum()
        print(" - {} is found in {:.2f} percent of low reviews."
              .format(w, count/total_low_reviews*100))

    # Export all sentences that contain pain point words.
    sent_low = get_sentences(df_low)
    df_low_sents = pd.DataFrame(columns=['word', 'sentence'])
    for word in pain_point_set:
        for s in sent_low:
            if word in s:
                df_low_sents = df_low_sents.append(
                    {'word': word, 'sentence':s}, ignore_index=True)
    df_low_sents.to_excel('pain_points.xlsx')

    show_wordcloud(df_low)

# FUNCTION = find_highlights
# Find high frequency words in positive reviews.
def find_highlights(df):
    # Filter dataframe for RATING = 4 or 5 and Sentence Sentiment =
    # Positive or Strongly positive.
    df_high = df.loc[df['RATING'].isin([4,5]) &
        df['Sentence Sentiment'].isin(['Positive', 'Strongly positive'])]

    print("\nThere are {} high reviews, a percentage of {:.0f} out of {} total "
          "reviews.".format(df_high.shape[0], df_high.shape[0]/df.shape[0]*100,
          df.shape[0]))

    # Find high frequency words for high reviews
    df_high_words = find_high_freq_words(df_high)
    plot_word_frequency(df_high_words,
        "High frequency words in positive reviews",
        "high_freq_pos")

    highlight_set = ["room", "rooms", "good", "nice", "floor", "great",
                     "staff", "clean", "friendly", "helpful", "service",
                     "comfortable", "bathroom"]

    # Find percent of high reviews that contain highlight set word.
    print("\nPercent of high reviews that highlight set words are found in:")
    total_high_reviews = df_high.shape[0]
    for w in highlight_set:
        count = df_high['Verbatim'].str.contains(w).sum()
        print(" - {} is found in {:.2f} percent of high reviews.".format(w, count/total_high_reviews*100))

    # Export all sentences that contain highlight words.
    sent_high = get_sentences(df_high)
    df_high_sents = pd.DataFrame(columns=['word', 'sentence'])
    for word in highlight_set:
        for s in sent_high:
            if word in s:
                df_high_sents = df_high_sents.append(
                    {'word': word, 'sentence':s}, ignore_index=True)
    df_high_sents.to_excel('highlights.xlsx')

def main():
    filename = 'EM_Onsite_Presentation_Jan_2019_v2.xlsx'
    df = pd.read_excel(filename, header=0)
    df['BRAND_DESC'].replace({"consolidated hotels intl":"con hotels intl"}, regex=True, inplace=True)

    insight_cols = ['BRAND_DESC', 'REGION_DESC', 'CITY', 'AGE_GROUP_DESC',
                    'LOYALTY_DESC', 'TRIP_PURPOSE_DESC']
    insight_cols_name = ['Brand', 'Region', 'City', 'Age group',
                         'Loyalty category', 'Trip purpose']

    # Information about each column in the dataframe
    for i, c in enumerate(df.columns):
        column_insights(df, c)

    # Rating pie chart for all reviews
    rating_pie_chart(df, "Rating Distribution")

    # Grid of rating by segment data
    segment_grid(df, insight_cols, insight_cols_name)

    # Drill down into brand data
    brand_insights(df)

    # Drill down into loyalty data
    loyalty_insights(df)

    # Drill down into city data
    city_insights(df)

    # Drill down into age group data
    # age_group_insights(df)

    find_pain_points(df)
    find_highlights(df)

if __name__ == '__main__':
    main()

# NPS = Net Promoter Score = an index ranging from -100 to 100 that measures the willingness of customers to recommend a company's products or services to others. It is used as a proxy for gauging the customer's overall satisfaction with a company's product or service and the customer's loyalty to the brand.
