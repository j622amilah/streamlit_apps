# cd /home/oem2/Documents/PROGRAMMING/Github_analysis_PROJECTS/Text_analysis/text_prediction_app
# streamlit run my_script.py
# strg + w

!pip install tensorflow

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import io
import csv
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.feature_extraction.text import CountVectorizer

import streamlit as st



class predict_text:
    
    def __init__(self, file_loc, delimiter):
        self.file_loc = file_loc
        self.delimiter = delimiter
        self.NUM_WORDS = 1000
        self.EMBEDDING_DIM = 16
        self.MAXLEN = 120
        self.PADDING = 'post'
        self.OOV_TOKEN = "<OOV>"
        self.TRAINING_SPLIT = .8

    def make_a_properlist(self, vec):

        out = []
        for i in range(len(vec)):
            out = out + [np.ravel(vec[i])]

        if self.is_empty(out) == False:
            vecout = np.concatenate(out).ravel().tolist()
        else:
            vecout = list(np.ravel(out))

        return vecout

    def is_empty(self, vec):
        vec = np.array(vec, dtype=object)
        if vec.shape[0] == 0:
            out = True
        else:
            out = False

        return out
        
    def get_word_count_uniquewords(self, word_tokens, list_to_remove):

        # -------------------------------------
        # Process word tokens
        # -------------------------------------
        vectorizer = CountVectorizer()

        # -------------------------------------
        # 1. Count word tokens and get a unique list of words : count how many times a word appears
        # Get the document-term frequency array: you have to do this first because it links cvtext to vectorizer
        X = vectorizer.fit_transform(word_tokens)
        word_count0 = np.ravel(np.sum(X, axis=0)) # sum vertically

        # Get document-term frequency list : returns unique words in the document that are mentioned at least once
        unique_words0 = np.ravel(vectorizer.get_feature_names())
        # -------------------------------------
        # 3. Remove undesireable words AGAIN and adjust the unique_words and word_count vectors

        # first let's do a marker method
        marker_vec = np.zeros((len(unique_words0), 1))

        # search for the remove tokens in tok, an put a 1 in the marker_vec
        for i in range(len(unique_words0)):
            for j in range(len(list_to_remove)):
                if unique_words0[i] == list_to_remove[j]:
                    marker_vec[i] = 1

        unique_words = []
        word_count = []
        for i in range(len(marker_vec)):
            if (marker_vec[i] == 0) & (len(unique_words0[i]) > 4):
                unique_words.append(unique_words0[i])
                word_count.append(word_count0[i])

        m = len(np.ravel(word_count))
        # -------------------------------------

        # Matrix of unique words and how many times they appear
        mat = np.concatenate([np.reshape(np.ravel(word_count), (m,1)), np.reshape(unique_words, (m,1))], axis=1)

        # print('There are ' + str(len(word_tokens)) + ' word tokens, but ' + str(len(unique_words)) + ' words are unique.')

        # 2. (Option) sort the unique_words by the word_count such that most frequent words are 1st
        # Gives the index of unique_word_count sorted from min to max
        sort_index = np.argsort(word_count)

        # Convert from matrix to array, so we can manipulate the entries
        # Puts the response vector in an proper array vector
        A = np.array(sort_index.T)

        # But we want the index of unique_word_count sorted max to min
        Ainvert = A[::-1]

        # Convert the array to a list : this is a list where each entry is a list
        Ainv_list = []
        for i in range(len(Ainvert)):
            Ainv_list.append(Ainvert[i])

        # Top num_of_words counted words in document : cvkeywords
        keywords = []
        wc = []
        p = np.ravel(word_count)

        #print('Ainv_list' + str(Ainv_list))

        top_words = len(Ainv_list)  # 20
        for i in range(top_words):
            keywords.append(unique_words[Ainv_list[i]])
            wc.append(p[Ainv_list[i]])

        # Matrix of unique words and how many times they appear
        mat_sort = np.concatenate([np.reshape(np.ravel(wc), (top_words,1)), np.reshape(np.ravel(keywords), (top_words,1))], axis=1)
        # print(mat_sort)
        # -------------------------------------

        return wc, keywords, mat_sort
        
        
    def remove_stopwords(self, sentence):
        """
        Removes a list of stopwords

        Args:
            sentence (string): sentence to remove the stopwords from

        Returns:
            sentence (string): lowercase sentence without the stopwords
        """
        # List of stopwords
        stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", 
                     "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", 
                     "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", 
                     "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", 
                     "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", 
                     "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", 
                     "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", 
                     "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", 
                     "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", 
                     "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", 
                     "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", 
                     "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", 
                     "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", 
                     "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", 
                     "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]

        # Sentence converted to lowercase-only
        sentence = sentence.lower()

        words = sentence.split()
        no_words = [w for w in words if w not in stopwords]
        sentence = " ".join(no_words)
        
        return sentence

    
    def parse_data_from_file(self):
        sentences = []
        labels = []

        with open(self.file_loc, encoding='utf8', errors="surrogateescape") as cvinfo:
            reader = csv.reader(cvinfo, delimiter=self.delimiter)
            next(reader)
            for row in reader:
                labels.append(row[0])
                sentence = row[1]
                sentence = self.remove_stopwords(sentence)
                sentences.append(sentence)
        return sentences, labels
    
    
    def train_val_split(self, sentences, labels, training_split):
        # Compute the number of sentences that will be used for training (should be an integer)
        train_size = int(training_split*len(labels))

        # Split the sentences and labels into train/validation splits
        # tr_ind = np.random.permutation(len(labels))[0:train_size]
        # OR
        # They used the first train_size number of sentences
        all_indexes = np.arange(len(labels))
        tr_ind = all_indexes[0:train_size]
        train_sentences = [sentences[i] for i in tr_ind]
        train_labels = [labels[i] for i in tr_ind]

        val_ind = np.setdiff1d(np.arange(len(labels)), tr_ind)
        validation_sentences = [sentences[i] for i in val_ind]
        validation_labels = [labels[i] for i in val_ind]
    
        return train_sentences, validation_sentences, train_labels, validation_labels
    
    
    def fit_tokenizer(self, train_sentences):
        # Instantiate the Tokenizer class, passing in the correct values for num_words and oov_token
        tokenizer = Tokenizer(num_words=self.NUM_WORDS, oov_token=self.OOV_TOKEN)

        # Fit the tokenizer to the training sentences
        tokenizer.fit_on_texts(train_sentences)
        return tokenizer
    
    
    def seq_and_pad(self, sentences, tokenizer):
        """
        Generates an array of token sequences and pads them to the same length

        Args:
            sentences (list of string): list of sentences to tokenize and pad
            tokenizer (object): Tokenizer instance containing the word-index dictionary
            padding (string): type of padding to use
            maxlen (int): maximum length of the token sequence

        Returns:
            padded_sequences (array of int): tokenized sentences padded to the same length
        """    
        ### START CODE HERE

        # Convert sentences to sequences
        sequences = tokenizer.texts_to_sequences(sentences)

        # Pad the sequences using the correct padding and maxlen
        padded_sequences = pad_sequences(sequences, maxlen=self.MAXLEN, truncating=self.PADDING)

        return padded_sequences
    
    
    def tokenize_labels(self, all_labels, split_labels):
        """
        Tokenizes the labels

        Args:
            all_labels (list of string): labels to generate the word-index from
            split_labels (list of string): labels to tokenize

        Returns:
            label_seq_np (array of int): tokenized labels
        """
        ### START CODE HERE

        # Instantiate the Tokenizer (no additional arguments needed)
        label_tokenizer = Tokenizer(num_words=len(all_labels))

        # Fit the tokenizer on all the labels
        label_tokenizer.fit_on_texts(all_labels)
        
        # Convert labels to sequences
        label_seq = label_tokenizer.texts_to_sequences(split_labels)

        # Convert sequences to a numpy array. Don't forget to substact 1 from every entry in the array!
        label_seq_np = np.array([i-1 for i in np.array(label_seq)])

        ### END CODE HERE

        return label_seq_np
    
    
    def create_model(self):
        """
        Creates a text classifier model

        Args:
            num_words (int): size of the vocabulary for the Embedding layer input
            embedding_dim (int): dimensionality of the Embedding layer output
            maxlen (int): length of the input sequences

        Returns:
            model (tf.keras Model): the text classifier model
        """

        tf.random.set_seed(123)

        model = tf.keras.Sequential([ 
            tf.keras.layers.Embedding(self.NUM_WORDS, self.EMBEDDING_DIM, input_length=self.MAXLEN),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(5, activation='softmax')
        ])

        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        return model
    
    
    def predict_text_input(self, model, tokenizer, txt_input):
        # Prepare seed_text
        token_list = tokenizer.texts_to_sequences([txt_input])[0]

        # Pad the sequence
        token_list = pad_sequences([token_list], maxlen=self.MAXLEN, padding='pre')
        probabilities = model.predict(token_list, verbose=0)
        selected_cat = np.argmax(probabilities)
        print('selected_cat: ', selected_cat)

        return selected_cat
    
    
    # def generate_vis_files():
        
    
    
    def main(self):
        sentences, labels = self.parse_data_from_file()
        
        # --------------------------------------
        
        print('There are ', len(sentences), 'sentences provided.')
        unq_labels = list(set(labels))
        num_diag = len(unq_labels)
        print('There are ', num_diag, 'prediction categories.')
        
        # --------------------------------------
        
        # Give some information about the text
        num = np.random.randint(100, size=100)[0]
        print('Random sentence: ', sentences[num])
        print('Corresponding label: ', labels[num])
        # --------------------------------------
        
        # Give some information about the labels
        dicc = {}
        for i in list(set(labels)):
            temp = []
            for ind, j in enumerate(labels):
                if j == i:
                    temp.append(ind)
            dicc[i] = temp

        list_to_remove = ['b', "their", "based", "which", 'would', 'https']
        k = list(dicc.keys())
        sendic = {}
        for i in k:
            vec = [sentences[j] for j in dicc[i]]
            wc, keywords, mat_sort = self.get_word_count_uniquewords(vec, list_to_remove)
            sendic[i] = [list(mat_sort[0]), list(mat_sort[1]), list(mat_sort[2]), list(mat_sort[3])]

        print('Your labels represent the following : ', sendic)
        
        # --------------------------------------

        train_split = 0.7
        X_train, X_val, Y_train, Y_val = self.train_val_split(sentences, labels, train_split)
        X_train = np.array(X_train)
        X_val = np.array(X_val)
        Y_train = np.array(Y_train)
        Y_val = np.array(Y_val)

        # print('X_train.shape: ', X_train.shape)
        # print('Y_train.shape: ', Y_train.shape)
        # print('X_val.shape: ', X_val.shape)
        # print('Y_val.shape: ', Y_val.shape)

        # --------------------------------------

        tokenizer = self.fit_tokenizer(X_train)
        word_index = tokenizer.word_index

        # --------------------------------------

        train_padded_seq = self.seq_and_pad(X_train, tokenizer)
        val_padded_seq = self.seq_and_pad(X_val, tokenizer)
        # print('train_padded_seq.shape: ', train_padded_seq.shape)
        # print('val_padded_seq.shape: ', val_padded_seq.shape)

        # --------------------------------------

        train_label_seq = self.tokenize_labels(labels, Y_train)
        tls = self.make_a_properlist(train_label_seq)
        label_meaning = dict(zip(tls, labels))
        val_label_seq = self.tokenize_labels(labels, Y_val)

        # --------------------------------------

        model = self.create_model()

        history = model.fit(train_padded_seq, train_label_seq, epochs=30, validation_data=(val_padded_seq, val_label_seq))

        # --------------------------------------
        
        return label_meaning, tokenizer, model
    
    
def head():

    st.title('Welcome to this test application!!')
    # OR
    st.markdown("""
        <h1 style='text-align: center; margin-bottom: -35px;'>
        Test Streamlit: Predict text
        </h1>
    """, unsafe_allow_html=True
    )
    
    st.caption("""
        <p style='text-align: center'>
        by <a href='https://github.com/j622amilah/streamlit_apps'>Github_repo</a>
        </p>
    """, unsafe_allow_html=True
    )
    
    st.write(
        "This is a silly application to learn how to use streamlit.",
        "Working with applications on the Internet seems difficult.",
        "Click the button \U0001F642."
    )
    
    file_loc = st.text_input('Enter the URL location of your text to analyse: ')
    # /home/oem2/Documents/ONLINE_CLASSES/DeepLearning_AI_TensorFlow_Developer/3_Natural_Language_Processing_Tensorflow/Semaine2/3_sentence_classification/bbc-text.csv
    
    # df = read_data('filename.csv')
    # choice = df.sample(1)
    
    
    delimiter = st.text_input('Enter the delimiter that separates the label from the text, for each row: ')
    # ,
    
    txt_input = st.text_input('Enter some text to predict: ')
    # tennis game tomorrow
    
    return file_loc, delimiter, txt_input


def body(file_loc, delimiter, txt_input):
    pt = predict_text(file_loc, delimiter=delimiter)
    label_meaning, tokenizer, model = pt.main()
    st.info('Building the prediction model was a success!', icon='\U0001F916')
    
    
    st.info('Let us test the model!!', icon='\U0001F916')
    
    selected_cat = pt.predict_text_input(model, tokenizer, txt_input)
    st.write(label_meaning[selected_cat])

    st.stop()

file_loc, delimiter, txt_input = head()

if st.button('Build the Predictive Model'):
    st.write('file_loc : ', file_loc)
    st.write('delimiter : ', delimiter)
    st.write('txt_input : ', txt_input)
    
    body(file_loc, delimiter, txt_input)





