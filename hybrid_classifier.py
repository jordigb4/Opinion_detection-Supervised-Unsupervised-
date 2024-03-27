import text_normalizer as tn
from nltk.wsd import lesk
import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from collections import Counter
import nltk
import numpy as np
from collections import defaultdict
class HybridClassifier():
    def __init__(self):
        self.clf_lr = joblib.load('models/log_reg_model.pkl')
        self.clf_svm = joblib.load('models/svm_model.pkl')
        self.bow = joblib.load('models/bow.pkl')
        nltk.download('sentiwordnet')
        nltk.download('movie_reviews')
        nltk.download('wordnet')
        
    def predict(self, text,ponderations = [1/3,1/3,1/3],test = False, test_data = None):
 
        X_test_supervised = self.__preprocessing(text)
        X_test_unsupervised = tn.normalize_corpus(text, split_phrases = True, stopword_removal = False)
        pred_lr = self.__predict_lr(X_test_supervised); pred_svm = self.__predict_svm(X_test_supervised); pred_unsuper = self.__predict_unsuper(X_test_unsupervised)
        predictions = [pred_lr, pred_svm, pred_unsuper]
        if test:
            result = []
            for weights in test_data:
                answers = np.array(self.__combine_predictions(predictions,weights))
                result.append((answers,weights))
            return result
        answers = np.array(self.__combine_predictions(predictions,ponderations))
        return answers

    def __predict_lr(self, text):
        return self.clf_lr.predict(text)
    
    def __predict_svm(self, text):
        return self.clf_svm.predict(text)
    
    def __predict_unsuper(self, text):
        
        return self.__sentiwn_negation(text)
    
    def __Negation(self, sentence):	
        '''
        Input: String representing a sentence
        Output: Tokenized sentence with negation handled (List of words)
        '''
        temp = int(0)
        tokens = nltk.word_tokenize(sentence)
        tokens = [token.strip() for token in tokens]
        for i in range(len(tokens)):  
            if tokens[i-1] in ['not', 'no']:
                antonyms = []
                syn = lesk(word_tokenize(sentence), tokens[i], pos = 'a')
                if syn:
                    w1 = syn.name()
                else:
                    continue
                temp = 0
                for l in syn.lemmas():
                    if l.antonyms():
                        antonyms.append(l.antonyms()[0].name())
                        max_dissimilarity = 0
                        for ant in antonyms:
                            syns = wn.synsets(ant)
                            w2 = syns[0].name()
                            syns = wn.synsets(tokens[i])
                            w1 = syns[0].name()
                            word1 = wn.synset(w1)
                            word2 = wn.synset(w2)
                            if isinstance(word1.wup_similarity(word2), float) or isinstance(word1.wup_similarity(word2), int):
                                temp = 1 - word1.wup_similarity(word2)
                            if temp>max_dissimilarity:
                                max_dissimilarity = temp
                                antonym_max = ant
                                tokens[i] = antonym_max
                                tokens[i-1] = ''
        while '' in tokens:
            tokens.remove('')
        return tokens

    def __sentiwn_negation(self, corpora):
        stopword_list = nltk.corpus.stopwords.words('english')
        stopword_list.remove('no') ; stopword_list.remove('not')

        coef = {'n': 1, 'v': 0.6, 'r': 0.8, 'a': 2, 's': 2}
        pred = list()
        for review in corpora:
            final_score = token_count = 0
            phrases = review.split('linebreak')
            for phrase in phrases:
                words = self.__Negation(phrase)
                for word in words:
                    if word not in stopword_list:
                        synset = lesk(word_tokenize(phrase), word)
                        if synset:
                            sentiSynset = swn.senti_synset(synset.name())
                            final_score += coef[synset.pos()]*(sentiSynset.pos_score() - sentiSynset.neg_score())
                            token_count += 1

            norm_final_score = round(float(final_score) / token_count, 3)
            final_sentiment = 1 if norm_final_score >= 0 else 0
            pred.append(final_sentiment)
        return pred
    def __combine_predictions(self,vector_predictions,ponderations):
        ensemble_predictions = []

        for preds in zip(*vector_predictions):
            vote_counts = defaultdict(float)

            for pred, weight in zip(preds, ponderations):
                vote_counts[pred] += weight


            chosen_pred = max(vote_counts, key=vote_counts.get)
            ensemble_predictions.append(chosen_pred)

        return ensemble_predictions
    
    def __preprocessing(self,text):
        normalized = tn.normalize_corpus(text)
        full_data = self.__feature_extraction(normalized)
        return full_data

    def __feature_extraction(self,text):
        full_data = self.bow.transform(text)
        return full_data
    
    