from sklearn.svm import LinearSVC
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.feature_selection import  chi2
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import classification_report
import string
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from spellchecker import SpellChecker

def simplify_text(df):
    l = WordNetLemmatizer()
    spell_check = SpellChecker()
    texts = df['text'].tolist()

    for i in range(len(texts)):
        if texts[i] == "None":
            continue
        # removes numbers and puctuations
        text = re.sub(r'\d+', '', texts[i])
        text = "".join([char.lower() for char in text if char not in string.punctuation])
        
        # greatly simplifys tweet and removes all capitalizations
        tweet_tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True, preserve_case=False) 
        words = tweet_tokenizer.tokenize(text)
        text = ''
        # removes user from words
        for j in range(len(words)):
            if 'user' in words[j]:
                continue
            if j == len(words):
                text += words[j]
                break
            text += words[j] + " "
        # simplify and spellcheck
        text = l.lemmatize(text)
        # no spell check it takes waaaay to long
        # text = spell_check.correction(text)
        texts[i] = text
    # sends all user data to lowercase
    users = df['user'].tolist()
    for i in range(len(users)):
        users[i] = users[i].lower()
    df['user'] = users
    df['text'] = texts
    return df

def create_data(train, test, df, col):
    y = [train[col], test[col]]

    # Most common words by region
    word_list = []
    tfidf_feature = TfidfVectorizer(lowercase=False,analyzer='word', sublinear_tf=True, norm='l2', min_df=2, max_df= 0.5)
    features = tfidf_feature.fit_transform(df['text'])

    # Create a mapping from region to ids
    # labels and label dictionary to use for the tfidf
    labels = df[col].factorize()[0]
    df['region_ids'] = labels
    label2id = dict(df[['region', 'region_ids']].drop_duplicates().sort_values('region_ids').values)
    # finding most common words by region
    for label, num in sorted(label2id.items()):
        index = np.argsort(chi2(features, labels == num)[0])
        words = np.array(tfidf_feature.get_feature_names_out())[index]
        words = [word for word in words]

        # top 50 words
        word_list.extend(words[-50:])
        # to print out most common words by region:
        # print(words[-50:])

    # Word tfidf
    # Uses a list of words to vectorize the words from the text of each tweet.
    tfidf_word = TfidfVectorizer(lowercase=False,analyzer='word', sublinear_tf=True, norm='l2', min_df=2, max_df= 0.5, vocabulary=set(word_list))
    tfidf = tfidf_word.fit_transform(train['text'])
    tfidf_test = tfidf_word.transform(test['text'])

    # User tfidf bc user is also important to region location
    # Uses the same idea as before but with users.
    tfidf_user = TfidfVectorizer(lowercase=False, analyzer='word', sublinear_tf=True, norm='l2', min_df=2, max_df= 0.5,vocabulary=set(df['user'].unique()))
    user = tfidf_user.fit_transform(train['user'])
    user2 = tfidf_user.transform(test['user'])
    
    # Final word/user vector matrix for learning
    x_train = hstack((tfidf, user))
    x_test = hstack((tfidf_test, user2))

    x = [x_train,x_test]

    # Also creating the data for the individual states using the same vectorizer for words and users
    pacific = ["Alaska","Hawaii","Washington","Oregon", 'California']
    mountain = ["Montana","Idaho","Wyoming","Nevada", 'Utah', 'Colorado', 'Arizona', 'New Mexico']
    nw_central = ["North Dakota","South Dakota","Minnesota","Nebraska", 'Iowa', 'Kansas', 'Missouri']
    ne_central = ["Wisconsin","Michigan","Illinois","Indiana", 'Ohio']
    sw_central = ["Oklahoma","Texas","Arkansas","Louisiana"]
    se_central = ["Kentucky","Tennessee","Mississippi","Alabama"]
    s_atlantic = ["Florida","Georgia","South Carolina","North Carolina",'Virginia','West Verginia','District of Columbia','Maryland','Delaware']
    m_atlantic = ["New York","Pennsylvania","New Jersey"]
    new_england = ["Maine","New Hampshire","Vermont","Massachusetts",'Connecticut','Rhode Island']
    data_segments = [pacific,mountain,nw_central,ne_central,sw_central,se_central,s_atlantic,m_atlantic,new_england]

    state_data = []
    for i in range(9):
        train1 = train[train['state'].isin(data_segments[i])]
        test1 = test[test['state'].isin(data_segments[i])]
        tfidf = tfidf_word.fit_transform(train1['text'])
        tfidf_test = tfidf_word.fit_transform(test1['text'])
        user = tfidf_user.fit_transform(train1['user'])
        user2 = tfidf_user.fit_transform(test1['user'])
                        # train x                 test x                       train y         test y
        state_data.append([hstack((tfidf, user)), hstack((tfidf_test, user2)), train1['state'], test1['state']])
    return x, y, state_data

# Overall regonal model that will predict the regions
region_model = LinearSVC(random_state=0, class_weight='balanced')

# 9 models for 9 regions
# These will predict the states
states_models = [
    LinearSVC(random_state=0, class_weight='balanced'),
    LinearSVC(random_state=1, class_weight='balanced'),
    LinearSVC(random_state=2, class_weight='balanced'),
    LinearSVC(random_state=3, class_weight='balanced'),
    LinearSVC(random_state=4, class_weight='balanced'),
    LinearSVC(random_state=5, class_weight='balanced'),
    LinearSVC(random_state=6, class_weight='balanced'),
    LinearSVC(random_state=7, class_weight='balanced'),
    LinearSVC(random_state=8, class_weight='balanced')
]

# read data
df = pd.read_csv('region_data.csv', encoding ='latin1', sep = ',')
df = df.fillna("None")

# Simplifys the text by premoving puctuation, numbers, retweets, and capitals.
# Also uses WordNetLemmatizer which changes words back to their base form. EX. better -> good, walked -> walk, went -> go etc...
df = simplify_text(df)

# Used for separating regions and states
regions = ["Pacific", "Mountain", "NW Central", "NE Central", "SW Central", "SE Central","S Atlantic", "M Atlantic", "New England"]

# Separates df data for just US states, removes any non us tweets.
df = df[df['region'].isin(regions)]

# Splits data into train and test
region_df_train, region_df_test = train_test_split(df, test_size=0.2)

# Puts the df data into a useable form for learning.
print("Creating Data")
x, y, state_data = create_data(region_df_train, region_df_test, df, 'region')

# Used later for testing accuracy
pred_x = x[1]
pred_y = y[1]

# train and eval regonal model
print("Training Region Model")
model = region_model.fit(x[0], y[0])
pred = model.predict(x[1])
real = y[1].to_list()
print(classification_report(real, pred))

# Saves region model
print("Saving Region Model")
pickle.dump(model, open("models/region_model.sav", 'wb'))

# Trains, evals, and saves state models
models = ["pacific", "mountain", "nwcentral", "necentral", "swcentral", "secentral","satlantic", "matlantic", "newengland"]
for i in range(9):
    print(models[i])
    print("-----------------------------------------------------")

    state_train = [state_data[i][0], state_data[i][2]]
    state_test = [state_data[i][1],state_data[i][3]]

    print(state_train[0].shape)

    model = states_models[i].fit(state_train[0], state_train[1])
    pred = model.predict(state_test[0])
    real = state_test[1].to_list()
    print(classification_report(real, pred))

    pickle.dump(model, open("models/"+models[i]+"_model.sav", 'wb'))


# PREDICTING LOCATION FROM TEXT
# This takes the text of the tweet ONLY and first predicts the region of the tweet and then predicts the state of the tweet.
predicted_states = [states_models[0].predict(pred_x),states_models[1].predict(pred_x),states_models[2].predict(pred_x),states_models[3].predict(pred_x),states_models[4].predict(pred_x),states_models[5].predict(pred_x),states_models[6].predict(pred_x),states_models[7].predict(pred_x),states_models[8].predict(pred_x)]
predicted = region_model.predict(pred_x).tolist()
pred_states = []
for i in range(len(predicted)):
    if predicted[i] == "Pacific":
        pred_states.append(predicted_states[0][i])
    elif predicted[i] == "Mountain":
        pred_states.append(predicted_states[1][i])
    elif predicted[i] == "NW Central":
        pred_states.append(predicted_states[2][i])
    elif predicted[i] == "NE Central":
        pred_states.append(predicted_states[3][i])
    elif predicted[i] == "SW Central":
        pred_states.append(predicted_states[4][i])
    elif predicted[i] == "SE Central":
        pred_states.append(predicted_states[5][i])
    elif predicted[i] == "S Atlantic":
        pred_states.append(predicted_states[6][i])
    elif predicted[i] == "M Atlantic":
        pred_states.append(predicted_states[7][i])
    elif predicted[i] == "New England":
        pred_states.append(predicted_states[8][i])

# Final output is the precision, recall, f1 score, and accuracy of both region and state model combind.
real = region_df_test['state'].to_list()
print(classification_report(real, pred_states))
