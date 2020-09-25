#!/usr/bin/env python
# -*- coding: utf-8 -*-

# katestable
'''
:author:  madkote
:contact: madkote(at)bluewin.ch

K-testable languages with DFA
-----------------------------
:TODO add here some description

Usage
-----
>>> from katestable import KTestable
>>> kt = KTestable.build(3, [...])
>>> kt.detect('abba')
'''

import numpy as np
from itertools import product
import string

VERSION = (0, 2, 0)

__all__ = ['KTestable']
__author__ = 'madkote <madkote(at)bluewin.ch>'
__version__ = '.'.join(str(x) for x in VERSION)


# ============================================================================
# INNER
# ============================================================================
class DFA(object):
    '''
    (Simple) Deterministic Finite Automata.
    '''
    def __init__(self, alphabet, accepts, start, states, trans):
        '''
        The constructor.
        :param alphabet: Alphabet of DFA
        :param accepts: Final states
        :param start: Start state
        :param states: States
        :param trans: Transitions (delta)
        '''
        self.alphabet = alphabet
        self.accepts = accepts
        self.current_state = start
        self.start = start
        self.states = states
        self.trans = trans
        self.count_failure = 0

    def reset(self):
        '''
        Reset DFA: set the current state to the start state.
        '''
        self.current_state = self.start

    def status(self):
        '''
        Get the status of the DFA.
        :return: True if the current state is in the set of the final states,
            otherwise False.
        '''
        return self.current_state in self.accepts


class KDFA(DFA):
    '''
    DFA built from K-Testable-Machine
    >>> dfa = DFA.build(ktmachine)
    >>> dfa.detect('00011100')
    '''
    STATE_FAILURE = 'FAILURE'

    def _detect_char(self, c):
        '''
        Detect the character: find the transition for the given character.
        :param c: Character to be detected
        '''
        if c in self.alphabet:
            # new state     
            state = self.current_state
            self.current_state = self.trans[self.current_state][c]
            if self.current_state == 'FAILURE':
                self.current_state = state
                self.count_failure += 1
        else:
            # unknown character
            self.current_state = self.STATE_FAILURE

    def _detect_string(self, s):
        '''
        Detect the string: the final state is crucial.
        :param s: String to be detected.
        '''
        for c in s:
            self._detect_char(c)

    def detect(self, s):
        '''
        Detect the string (string should belong to the language).
        :param s: String to be analyzed
        :return: True if the string is detected and belongs to the language,
            otherwise False.
        '''
        self.reset()    
        self.count_failure = 0    
        self._detect_string(s)
        return self.count_failure, self.status()

    @staticmethod
    def build(ktmachine):
        '''
        Build the DFA from the K-Testable-Machine.
        -# merge prefixes and short strings :: I&C
        -# create states as prefixes from I&C
        -# create states from valid strings T as suff(k-1) and pref(k-1)
        -# initialize transitions
        -# transitions for I&C
        -# transitions for set of valid strings T
        -# add state from suffixes to accepted states
        -# add state from short string to accepted states
        :param ktmachine: K-Testable-Machine
        :return: DFA from K-Testable-Machine
        '''
        accepts = []
        start = ''
        failure = KDFA.STATE_FAILURE
        states = [start, failure]
        trans = {}
        # merge I&C and create states as prefixes from I&C
        for x in set(ktmachine.prefixes) | set(ktmachine.shortstr):
            for i in range(1, len(x)+1):
                state = x[:i]
                if state not in states:
                    states.append(state)
        # create states from valid strings T as suff(k-1) and pref(k-1)
        for x in ktmachine.validstr:
            if len(x) > 1:
                tpref = x[:len(x)-1]
                tsuff = x[1:]
                if tpref not in states:
                    states.append(tpref)
                if tsuff not in states:
                    states.append(tsuff)
        # initialize transitions
        for x in states:
            trans[x] = {}
            for y in ktmachine.alphabet:
                trans[x][y] = failure
        # transitions for set of I and C
        for x in set(ktmachine.prefixes) | set(ktmachine.shortstr):
            if not x or x == failure:
                continue
            for i in range(0, len(x)):
                char = x[i]
                if i == 0:
                    source = ''
                else:
                    source = x[:i]
                dest = x[:i+1]
                trans[source][char] = dest
        # transitions for set of valid strings T
        for x in ktmachine.validstr:
            if len(x) < 2:
                # suffix and prefix are required
                continue
            source = x[:len(x)-1]
            dest = x[1:]
            char = x[-1]
            trans[source][char] = dest
        # add state from suffixes to accepted states
        for x in ktmachine.suffixes:
            if x not in accepts:
                accepts.append(x)
        # add state from short string to accepted states
        for x in ktmachine.shortstr:
            if x not in accepts:
                accepts.append(x)
        # DFA
        return KDFA(ktmachine.alphabet, accepts, start, states, trans)


class KTMachine(object):
    '''
    K-Testable-Machine.
    The machine can be built from the language and the K-value.
    Usage:
    >>> KTMachine.build(k, language)
    '''
    def __init__(self, k, alphabet, prefixes, shortstr, suffixes, validstr):
        '''
        The constructor of the machine
        for K, M=(alphabet, prefixes, shortstr, suffixes, validstr).
        :param k: K value
        :param alphabet: Alphabet
        :param prefixes: Prefixes (k-1)
        :param shortstr: short strings (<k)
        :param suffixes: suffixes (k-1)
        :param validstr: allowed/valid strings (k)
        '''
        self.k = k
        self.alphabet = alphabet
        self.prefixes = prefixes
        self.shortstr = shortstr
        self.suffixes = suffixes
        self.validstr = validstr

    @staticmethod
    def build(k, language):
        '''
        Build the machine from a language.
        -# get all short strings (size < k)
        -# get all prefixes and suffixes (k-1)
        -# extract allowed/valid strings (size == k)
        -# build alphabet
        :param k: K value
        :param language: The language as a list of language's words
        :return: The K-Testable-Machine
        '''
        alphabet = []
        prefixes = []
        shortstr = []
        suffixes = []
        validstr = []
        # build
        for word in language:
            # get all short strings (size < k)
            if len(word) < k:
                if word not in shortstr:
                    shortstr.append(word)
            # get all prefixes and suffixes (k-1)
            if len(word) >= (k-1):
                p = word[:k-1]
                s = word[len(word)-k+1:]
                if p not in prefixes:
                    prefixes.append(p)
                if s not in suffixes:
                    suffixes.append(s)
            # extract allowed strings (size == k)
            if len(word) >= k:
                for i in range(0, len(word)-k+1):
                    tword = word[i:i+k]
                    if tword not in validstr:
                        validstr.append(tword)
            # build alphabet by each character
            for c in word:
                if c not in alphabet:
                    alphabet.append(c)
        # done
        return KTMachine(k, alphabet, prefixes, shortstr, suffixes, validstr)


# ============================================================================
# PUBLIC
# ============================================================================
class KTestable(object):
    '''
    K-Testable public facade.
    '''
    def __init__(self, kdfa):
        '''
        The constructor with DFA
        :param kdfa: K-testable DFA
        '''
        self.kdfa = kdfa

    def detect(self, s):
        '''
        Detect the string (string should belong to the language).
        :param s: String to be analyzed
        :return: True if the string is detected and belongs to the language,
            otherwise False.
        '''
        return self.kdfa.detect(s)

    @staticmethod
    def build(k, language):
        '''
        Build the machine from a language.
        :param k: K value
        :param language: The language as a list of language's words
        :return: The K-Testable
        '''
        return KTestable(KDFA.build(KTMachine.build(k, language)))




class KTestableSyntactic:
   
    def __init__(self, x_train, y_train, x_test, y_test, alphabet_size):
        self.ktestable  = 2
        self.x_train = x_train 
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.inference = []
        self.alphabet_size = alphabet_size
        #print('******>',self.x_test)
        #print('******>',self.y_test)   
       
    def __define_class(self,values):
        output = []
        seen = set()
        for value in values:
            if value not in seen:
                output.append(value)
                seen.add(value)
        return output

    '''
    def __define_class(self,values):
        output = []
        seen = set()
        for value in values:
            if value not in seen:
                output.append(value)
                seen.add(value)
        return output
    '''

    def __group_values_class(self,x_t, y_t):
        indexs = []
        y_t_len = len(y_t)
        for i in range(y_t_len):
            if (i != (y_t_len-1)):
                if y_t[i] != y_t[i+1]:
                    indexs.append(i+1)
        return np.split(x_t, indexs)
    
    def __create_sequences_to_classes(self, grupo_of_class):

        ascii_uppercase = list(string.ascii_uppercase)
        array = []
        for i in product('ABCDEFGHIJKLMNOPQRSTUVWXYZ', '0123456789'):
            array.append(i[0]+i[1])
        k = self.alphabet_size
        j = 0
        for i in range(len(ascii_uppercase) + 1, k + 1):
            ascii_uppercase.append(array[j])
            j = j + 1
          
        sequence_temp = []
        for sequence in grupo_of_class:
            new_sequence = []
            for c in sequence:
                new_sequence.append(ascii_uppercase[int(c)])
            sequence_temp.append(new_sequence)
        return sequence_temp
    
    '''
    def __create_sequences_to_classes(self, grupo_of_class):
        
        sequence_temp = []
        for j in grupo_of_class:
            temp = '-'.join(str(x) for x in j) 
            #temp = ''.join(str(x) for x in j)      
            sequence_temp.append(temp)

        return sequence_temp
    '''
    
    
    def __define_languages(self, x_t, y_t):
        array_values_classes = self.__group_values_class(x_t, y_t)    
        languages = []
        
        for i in array_values_classes:    
            languages.append(self.__create_sequences_to_classes(i))   
        
        return languages
    
    
    def __recognize(self, test_sentence):
        """ 
        This method recognizes a single image 
        It can be utilized individually as well.
        param test_img: representation of an image using visual words. Example: [ 40.  16.  16.  13.  13.].
        """ 
        preditions = []
        for language in self.inference:
            for cl, kt in language.items():
                errors, detect = kt.detect(test_sentence) # retorna os erros e se pertence a linguagem ou não. Ex: 1, True
                preditions.append([cl, [errors, detect]])               
        
        pred_true = []
        pred_false = []
        for pred in preditions:
            if pred[1][1]:
                pred_true.append([pred[0], pred[1][0]])
            else:
                pred_false.append([pred[0], pred[1][0]])


        class_prediction = None
        if len(pred_true) > 0:
            erros = pred_true[0][1]
            class_prediction = pred_true[0][0]
            for p in pred_true:
                if erros > p[1]:
                    erros = p[1]
                    class_prediction = p[0]
        else:      
            erros = pred_false[0][1]
            class_prediction = pred_false[0][0]
            for p in pred_false:
                if erros > p[1]:
                    erros = p[1]
                    class_prediction = p[0]           
       
        return class_prediction             
        
      
    def trainModel(self):
        """
        Uses k-testable inference classifier 
        """        
        languages = (self.__define_languages(self.x_train, self.y_train))
        classes = self.__define_class(self.y_train)
       
        for lang, cl in zip(languages, classes):
            language = []
            for word in lang:
                language.append(''.join(word))
         
            self.inference.append({cl:KTestable.build(self.ktestable, language)})
    
           
    def testModel(self):

        languages = (self.__define_languages(self.x_test, self.y_test))
        classes = self.__define_class(self.y_test)
        predictions = []
        cls = []
     
        for sentences, c in zip(languages, classes):
            for sentence in sentences:
                word = ''.join(sentence)
                cl = self.__recognize(word)
                predictions.append(cl)
                cls.append(c)            
        
        return (np.asarray([int(i) for i in predictions]), np.asarray([int(i) for i in cls]))







# ============================================================================
# DEMO
# ============================================================================
def demo():
    '''
    Simple demo
    '''
    k = 2
    language1 = ['a', 'aa', 'abba', 'abbba', 'aaabbbbba']
    kt = KTestable.build(k, language1)
    print(kt.detect('abba'))
    print(kt.detect('bbabba'))
    print(kt.detect('ab'))
    #x = kt.detect('abab')
    

if __name__ == '__main__':
    demo()
