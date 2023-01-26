'''
from https://github.com/amirveyseh/MadDog under CC BY-NC-SA 4.0
'''

import string

if __name__ != "__main__":
    import spacy

    nlp = spacy.load("en_core_web_sm")

with open('../input/stopWords.txt') as file:
    stop_words = [l.strip() for l in file.readlines()]


class Extractor:
    def __init__(self):
        pass

    def short_extract(self, sentence, threshold, starting_lower_case, ignore_dot=False):
        shorts = []
        for i, t in enumerate(sentence):
            if ignore_dot:
                t = t.replace('.', '')
                # t = t.replace('-','')
            if len(t) == 0:
                continue
            # FIXED [issue: of an enhanced Node B ( eNB ) ]
            if not starting_lower_case:
                if t[0].isupper() and len([c for c in t if c.isupper()]) / len(t) > threshold and 2 <= len(t) <= 10:
                    shorts.append(i)
            else:
                if len([c for c in t if c.isupper()]) / len(t) > threshold and 2 <= len(t) <= 10:
                    shorts.append(i)
        return shorts

    def extract_cand_long(self, sentence, token, ind, ignore_punc=False, add_punc=False, small_window=False):
        '''
        extract candidate long form of the form "long form (short form)" or "short form (long form)"

        :param sentence: tokenized sentence
        :param token: acronym
        :param ind: position of the acronym
        :return: candidate long form, candidate is on left or right of the short form
        '''
        if not small_window:
            long_cand_length = min([len(token) + 10, len(token) * 3])
        else:
            long_cand_length = min([len(token) + 5, len(token) * 2])
        cand_long = []
        cand_long_index = []
        left = True
        right_ind = 1
        left_ind = 1
        # FIXED [issue: ]
        if add_punc:
            excluded_puncs = ['=', ':']
        else:
            excluded_puncs = []
        # FIXED [issue: such as Latent Semantic Analysis ( LSA ; )]
        if ignore_punc:
            while ind + right_ind < len(sentence) and sentence[ind + right_ind] in [p for p in string.punctuation if
                                                                                    p != '(' and p != ')' and p not in excluded_puncs]:
                right_ind += 1
            while ind - left_ind > 0 and sentence[ind - left_ind] in [p for p in string.punctuation if
                                                                      p != '(' and p != ')' and p not in excluded_puncs]:
                left_ind -= 1
        ####
        if ind < len(sentence) - 2 - right_ind and (
                sentence[ind + right_ind] == '(' or sentence[ind + right_ind] == '=' or sentence[
            ind + right_ind] in excluded_puncs):
            left = False
            for j in range(ind + right_ind + 1, min([ind + right_ind + 1 + long_cand_length, len(sentence)])):
                if sentence[j] != ')':
                    cand_long.append(sentence[j])
                    cand_long_index.append(j)
                else:
                    break
        elif 1 < ind - (left_ind - 1) and ind + right_ind < len(sentence) and (
                (sentence[ind - left_ind] == '(' and sentence[ind + right_ind] == ')') or sentence[
            ind - left_ind] in excluded_puncs):
            for k in range(0, long_cand_length):
                j = ind - left_ind - 1 - k
                if j > -1:
                    cand_long.insert(0, sentence[j])
                    cand_long_index.insert(0, j)
        return cand_long, cand_long_index, left

    # FIXED [issue: The Stopping Trained in America PhDs from Leaving the Economy Act ( or STAPLE Act ) has bee introduced]
    def extract_high_recall_cand_long(self, sentence, token, ind, small_window=False, left=False):
        '''
        Find the candidate long form for a give acronym for high recall extraction
        example: The Stopping Trained in America PhDs from Leaving the Economy Act ( or STAPLE Act ) has bee introduced

        :param sentence:
        :param token:
        :param ind:
        :param small_window:
        :return:
        '''
        long_cand_length = min([len(token) + 10, len(token) * 3])
        cand_long = []
        cand_long_index = []
        if not left:
            for j in range(ind + 1, min([ind + long_cand_length, len(sentence)])):
                cand_long.append(sentence[j])
                cand_long_index.append(j)
        else:
            for k in range(0, long_cand_length):
                j = ind - 1 - k
                if j > -1:
                    cand_long.insert(0, sentence[j])
                    cand_long_index.insert(0, j)
        return cand_long, cand_long_index, left

    def create_diction(self, sentence, labels, all_acronyms=True, tag='', map_chars=False, diction={}):
        '''
        convert sequential labels into {short-form: long-form} dictionary

        :param sentence: tokenized sentence
        :param labels: labels of form B-short, B-long, I-short, I-long, O
        :return: dictionary
        '''
        shorts = []
        longs = []
        isShort = True
        phr = []
        for i in range(len(sentence)):
            if labels[i] == 'O' or (isShort and 'long' in labels[i]) or (not isShort and 'short' in labels[i]) or (
            labels[i].startswith('B')):
                if len(phr):
                    if isShort:
                        shorts.append((phr[0], phr[-1]))
                    else:
                        longs.append((phr[0], phr[-1]))
                    phr = []
            if 'short' in labels[i]:
                isShort = True
                phr.append(i)
            if 'long' in labels[i]:
                isShort = False
                phr.append(i)
        if len(phr):
            if isShort:
                shorts.append((phr[0], phr[-1]))
            else:
                longs.append((phr[0], phr[-1]))
        acr_long = {}
        for long in longs:
            best_short = []
            ## check if the long form is already mapped in given diction
            if long in diction and diction[long] in shorts:
                best_short = diction[long]
            best_dist = float('inf')
            #### FIXED [issue: long form incorrectly mapped to the closest acronym in the sentence]
            #### FIXED [issue: multiple short forms could be character matched with the long form]
            if not best_short:
                best_short_cands = []
                for short in shorts:
                    long_form = self.character_match(sentence[short[0]], sentence[long[0]:long[1] + 1],
                                                     list(range(long[1] + 1 - long[0])), output_string=True,
                                                     is_candidate=False)
                    if long_form:
                        best_short_cands.append(short)
                if len(best_short_cands) == 1:
                    best_short = best_short_cands[0]
            #####
            #### FIXED [QALD-6 (the workshop of question answering over linked-data 6) at ESWIC 2016]
            if not best_short and map_chars:
                best_short_cands = []
                for short in shorts:
                    long_form = self.map_chars(sentence[short[0]], sentence[long[0]:long[1] + 1])
                    if long_form:
                        best_short_cands.append(short)
                if len(best_short_cands) == 1:
                    best_short = best_short_cands[0]
            ####
            #### FIXED [issue: US Securities and Exchange Commission EDGAR ( SEC ) database]
            if not best_short:
                best_short_cands = []
                for short in shorts:
                    is_mapped = self.map_chars_with_capitals(sentence[short[0]], sentence[long[0]:long[1] + 1])
                    if is_mapped:
                        best_short_cands.append(short)
                if len(best_short_cands) == 1:
                    best_short = best_short_cands[0]
            ####
            # FIXED [issue: RNNs , Long Short - Term Memory ( LSTM ) architecture]
            if not best_short and long[1] < len(sentence) - 2 and sentence[long[1] + 1] == '(' and 'short' in labels[
                long[1] + 2]:
                for short in shorts:
                    if short[0] == long[1] + 2:
                        best_short = short
                        break
            if not best_short and long[0] > 1 and sentence[long[0] - 1] == '(' and 'short' in labels[long[0] - 2]:
                for short in shorts:
                    if short[1] == long[0] - 2:
                        best_short = short
                        break
            ####
            if not best_short:
                for short in shorts:
                    if short[0] > long[1]:
                        dist = short[0] - long[1]
                    else:
                        dist = long[0] - short[1]
                    if dist < best_dist:
                        best_dist = dist
                        best_short = short
            if best_short:
                short_form_info = ' '.join(sentence[best_short[0]:best_short[1] + 1])
                long_form_info = [' '.join(sentence[long[0]:long[1] + 1]), best_short, [long[0], long[1]], tag, 1]
                if short_form_info in acr_long:
                    long_form_info[4] += 1
                acr_long[short_form_info] = long_form_info
        if all_acronyms:
            for short in shorts:
                acr = ' '.join(sentence[short[0]:short[1] + 1])
                if acr not in acr_long:
                    acr_long[acr] = ['', short, [], tag, 1]
        return acr_long

    #### FIXED [QALD-6 (the workshop of question answering over linked-data 6) at ESWIC 2016]
    def map_chars(self, acronym, long):
        '''
        This function evaluate the long for based on number of initials overlapping with the acronym and if it is above a threshold it assigns the long form the the acronym

        :param acronym:
        :param long:
        :return:
        '''
        capitals = []
        for c in acronym:
            if c.isupper():
                capitals.append(c.lower())
        initials = [w[0].lower() for w in long]
        ratio = len([c for c in initials if c in capitals]) / len(initials)
        if ratio >= 0.6:
            return long
        else:
            return None

    #### FIXED [issue: US Securities and Exchange Commission EDGAR ( SEC ) database]
    def map_chars_with_capitals(self, acronym, long):
        '''
        This function maps the acronym to the long-form which has the same initial capitals as the acronym

        :param acronym:
        :param long:
        :return:
        '''
        capitals = []
        for c in acronym:
            if c.isupper():
                capitals.append(c.lower())
        long_capital_initials = []
        for w in long:
            if w[0].isupper():
                long_capital_initials.append(w[0].lower())
        if len(capitals) == len(long_capital_initials) and all(
                capitals[i] == long_capital_initials[i] for i in range(len(capitals))):
            return True
        else:
            return False

    def schwartz_extract(self, sentence, shorts, remove_parentheses, ignore_hyphen=False, ignore_punc=False,
                         add_punc=False, small_window=False, no_stop_words=False, ignore_righthand=False,
                         map_chars=False,default_diction=False):
        labels = ['O'] * len(sentence)
        diction = {}
        for i, t in enumerate(sentence):
            if i in shorts:
                labels[i] = 'B-short'
                # FIXED [issue: We show that stochastic gradient Markov chain Monte Carlo ( SG - MCMC ) - a class of ]
                if ignore_hyphen:
                    t = t.replace('-', '')
                # FIXED [issue: such as Latent Semantic Analysis ( LSA ; )]
                cand_long, cand_long_index, left = self.extract_cand_long(sentence, t, i, ignore_punc=ignore_punc,
                                                                          add_punc=add_punc, small_window=small_window)
                cand_long = ' '.join(cand_long)
                long_form = ""
                ## findBestLongForm
                if len(cand_long) > 0:
                    if left:
                        sIndex = len(t) - 1
                        lIndex = len(cand_long) - 1
                        while sIndex >= 0:
                            curChar = t[sIndex].lower()
                            if curChar.isdigit() or curChar.isalpha():
                                while (lIndex >= 0 and cand_long[lIndex].lower() != curChar) or (
                                        sIndex == 0 and lIndex > 0 and (
                                        cand_long[lIndex - 1].isdigit() or cand_long[lIndex - 1].isalpha())):
                                    lIndex -= 1
                                if lIndex < 0:
                                    break
                                lIndex -= 1
                            sIndex -= 1
                        if lIndex >= -1:
                            try:
                                lIndex = cand_long.rindex(" ", 0, lIndex + 1) + 1
                            except:
                                lIndex = 0
                            if cand_long:
                                cand_long = cand_long[lIndex:]
                                long_form = cand_long
                    else:
                        sIndex = 0
                        lIndex = 0
                        if t[0].lower() == cand_long[0].lower() or ignore_righthand:
                            while sIndex < len(t):
                                curChar = t[sIndex].lower()
                                if curChar.isdigit() or curChar.isalpha():
                                    while (lIndex < len(cand_long) and cand_long[lIndex].lower() != curChar) or (
                                            ignore_righthand and (sIndex == 0 and lIndex > 0 and (
                                            cand_long[lIndex - 1].isdigit() or cand_long[lIndex - 1].isalpha()))) or (
                                            lIndex != 0 and cand_long[lIndex - 1] != ' ' and ' ' in cand_long[
                                                                                                    lIndex:] and
                                            cand_long[cand_long[lIndex:].index(' ') + lIndex + 1].lower() == curChar):
                                        lIndex += 1
                                        if lIndex >= len(cand_long):
                                            break
                                    if lIndex >= len(cand_long):
                                        break
                                    lIndex += 1
                                sIndex += 1
                            if lIndex < len(cand_long):
                                try:
                                    lIndex = cand_long[lIndex:].index(" ") + lIndex + 1
                                except:
                                    lIndex = len(cand_long)
                                if cand_long:
                                    cand_long = cand_long[:lIndex]
                                    long_form = cand_long
                    # FIXED [issue : 'good results on the product review ( CR ) and on the question - type ( TREC ) tasks']
                    if remove_parentheses:
                        if '(' in long_form or ')' in long_form:
                            long_form = ''
                    # FIXED [issue: TN: The Number of ]
                    long_form = long_form.split()
                    if no_stop_words and long_form:
                        if long_form[0].lower() in stop_words:
                            long_form = []
                    if long_form:
                        if left:
                            long_form_index = cand_long_index[-len(long_form):]
                        else:
                            long_form_index = cand_long_index[:len(long_form)]
                        first = True
                        for j in range(len(sentence)):
                            if j in long_form_index:
                                if first:
                                    labels[j] = 'B-long'
                                    first = False
                                else:
                                    labels[j] = 'I-long'
                        if default_diction:
                            diction[(long_form_index[0], long_form_index[-1])] = (i, i)
        return self.create_diction(sentence, labels, tag='Schwartz', map_chars=map_chars, diction=diction)

    def bounded_schwartz_extract(self, sentence, shorts, remove_parentheses, ignore_hyphen=False, ignore_punc=False,
                                 add_punc=False, small_window=False, no_stop_words=False, ignore_righthand=False,
                                 map_chars=False, high_recall=False, high_recall_left=False, tag='Bounded Schwartz',default_diction=False):
        '''
        This function uses the same rule as schwartz but for the format "long form (short form)" will select long forms that the last word in the long form is selected to form the acronym
        example: User - guided Social Media Crawling method ( USMC ) that

        :param remove_parentheses:
        :param sentence:
        :param shorts:
        :return:
        '''
        labels = ['O'] * len(sentence)
        diction = {}
        for i, t in enumerate(sentence):
            if i in shorts:
                labels[i] = 'B-short'
                # FIXED [issue: We show that stochastic gradient Markov chain Monte Carlo ( SG - MCMC ) - a class of ]
                if ignore_hyphen:
                    t = t.replace('-', '')
                # FIXED [issue: The Stopping Trained in America PhDs from Leaving the Economy Act ( or STAPLE Act ) has bee introduced]
                if high_recall:
                    cand_long, cand_long_index, left = self.extract_high_recall_cand_long(sentence, t, i,
                                                                                          small_window=small_window,
                                                                                          left=high_recall_left)
                else:
                    # FIXED [issue: such as Latent Semantic Analysis ( LSA ; )]
                    cand_long, cand_long_index, left = self.extract_cand_long(sentence, t, i, ignore_punc=ignore_punc,
                                                                              add_punc=add_punc,
                                                                              small_window=small_window)
                cand_long = ' '.join(cand_long)
                long_form = ""
                ## findBestLongForm
                if len(cand_long) > 0:
                    if left:
                        sIndex = len(t) - 1
                        lIndex = len(cand_long) - 1
                        first_ind = len(cand_long)
                        while sIndex >= 0:
                            curChar = t[sIndex].lower()
                            if curChar.isdigit() or curChar.isalpha():
                                while (lIndex >= 0 and cand_long[lIndex].lower() != curChar) or (
                                        sIndex == 0 and lIndex > 0 and (
                                        cand_long[lIndex - 1].isdigit() or cand_long[lIndex - 1].isalpha())):
                                    lIndex -= 1
                                if first_ind == len(cand_long):
                                    first_ind = lIndex
                                if lIndex < 0:
                                    break
                                lIndex -= 1
                            sIndex -= 1
                        if lIndex >= 0 or lIndex == -1 and cand_long[0].lower() == t[0].lower():
                            try:
                                lIndex = cand_long.rindex(" ", 0, lIndex + 1) + 1
                                try:
                                    rIndex = cand_long[first_ind:].index(" ") + first_ind
                                except:
                                    rIndex = len(cand_long)
                            except:
                                lIndex = 0
                                try:
                                    rIndex = cand_long[first_ind:].index(" ") + first_ind
                                except:
                                    rIndex = len(cand_long)
                            if cand_long:
                                index_map = {}
                                word_ind = 0
                                for ind, c in enumerate(cand_long):
                                    if c == ' ':
                                        word_ind += 1
                                    index_map[ind] = word_ind
                                last_word_index = index_map[rIndex - 1]
                                cand_long = cand_long[lIndex:rIndex]
                                long_form = cand_long
                    else:
                        sIndex = 0
                        lIndex = 0
                        first_ind = -1
                        if t[0].lower() == cand_long[0].lower() or ignore_righthand:
                            while sIndex < len(t):
                                curChar = t[sIndex].lower()
                                if curChar.isdigit() or curChar.isalpha():
                                    while (lIndex < len(cand_long) and cand_long[lIndex].lower() != curChar) or (
                                            ignore_righthand and (sIndex == 0 and lIndex > 0 and (
                                            cand_long[lIndex - 1].isdigit() or cand_long[lIndex - 1].isalpha()))) or (
                                            lIndex != 0 and cand_long[lIndex - 1] != ' ' and ' ' in cand_long[
                                                                                                    lIndex:] and
                                            cand_long[cand_long[lIndex:].index(' ') + lIndex + 1].lower() == curChar):
                                        lIndex += 1
                                        if lIndex >= len(cand_long):
                                            break
                                    if first_ind == -1:
                                        first_ind = lIndex
                                    if lIndex >= len(cand_long):
                                        break
                                    lIndex += 1
                                sIndex += 1
                            if lIndex < len(cand_long) or (
                                    first_ind < len(cand_long) and lIndex == len(cand_long) and cand_long[-1] == t[-1]):
                                try:
                                    lIndex = cand_long[lIndex:].index(" ") + lIndex + 1
                                except:
                                    lIndex = len(cand_long)
                                if cand_long:
                                    if not ignore_righthand:
                                        first_ind = 0
                                    index_map = {}
                                    word_ind = 0
                                    for ind, c in enumerate(cand_long):
                                        if c == ' ':
                                            word_ind += 1
                                        index_map[ind] = word_ind
                                    first_word_index = index_map[first_ind]
                                    cand_long = cand_long[first_ind:lIndex]
                                    long_form = cand_long
                    # FIXED [issue : 'good results on the product review ( CR ) and on the question - type ( TREC ) tasks']
                    if remove_parentheses:
                        if '(' in long_form or ')' in long_form:
                            long_form = ''
                    # FIXED [issue: TN: The Number of ]
                    long_form = long_form.split()
                    if no_stop_words and long_form:
                        if long_form[0].lower() in stop_words:
                            long_form = []
                    if long_form:
                        if left:
                            long_form_index = cand_long_index[last_word_index - len(long_form) + 1:last_word_index + 1]
                        else:
                            long_form_index = cand_long_index[first_word_index:first_word_index + len(long_form)]
                        first = True
                        for j in range(len(sentence)):
                            if j in long_form_index:
                                if first:
                                    labels[j] = 'B-long'
                                    first = False
                                else:
                                    labels[j] = 'I-long'
                        if default_diction:
                            diction[(long_form_index[0],long_form_index[-1])] = (i,i)
        return self.create_diction(sentence, labels, tag=tag, map_chars=map_chars,diction=diction)

    # FIXED [issue: The Stopping Trained in America PhDs from Leaving the Economy Act ( or STAPLE Act ) has bee introduced]
    def high_recall_schwartz(self, sentence, shorts, remove_parentheses, ignore_hyphen=False, ignore_punc=False,
                             add_punc=False, small_window=False, no_stop_words=False, ignore_righthand=False,
                             map_chars=False):
        '''
        This function use bounded schwartz rules for acronyms which are not necessarily in parentheses
        example: The Stopping Trained in America PhDs from Leaving the Economy Act ( or STAPLE Act ) has bee introduced

        :param sentence:
        :param shorts:
        :param remove_parentheses:
        :param ignore_hyphen:
        :param ignore_punc:
        :param add_punc:
        :param small_window:
        :param no_stop_words:
        :param ignore_righthand:
        :param map_chars:
        :return:
        '''
        pairs_left = self.bounded_schwartz_extract(sentence, shorts, remove_parentheses, ignore_hyphen=True,
                                                   ignore_punc=ignore_punc, add_punc=add_punc,
                                                   small_window=small_window, no_stop_words=no_stop_words,
                                                   ignore_righthand=ignore_righthand, map_chars=True, high_recall=True,
                                                   high_recall_left=True, tag='High Recall Schwartz')
        pairs_right = self.bounded_schwartz_extract(sentence, shorts, remove_parentheses, ignore_hyphen=True,
                                                    ignore_punc=ignore_punc, add_punc=add_punc,
                                                    small_window=small_window, no_stop_words=no_stop_words,
                                                    ignore_righthand=ignore_righthand, map_chars=True, high_recall=True,
                                                    high_recall_left=False, tag='High Recall Schwartz')
        for acr, lf in pairs_right.items():
            if len(lf[0]) > 0 and (acr not in pairs_left or len(pairs_left[acr][0]) == 0):
                pairs_left[acr] = lf
        res = {}
        for acr, lf in pairs_left.items():
            if acr == ''.join([w[0] for w in lf[0].split() if w[0].isupper()]) or acr.lower() == ''.join(
                    w[0] for w in lf[0].split() if w not in string.punctuation and w not in stop_words).lower():
                res[acr] = lf
        return res

    def character_match(self, acronym, long, long_index, left=False, output_string=False, is_candidate=True):
        capitals = []
        long_form = []
        for c in acronym:
            if c.isupper():
                capitals.append(c)
        # FIXED [issue: different modern GAN architectures : Deep Convolutional ( DC ) GAN , Spectral Normalization ( SN ) GAN , and Spectral Normalization GAN with Gradient Penalty ( SNGP ) .]
        if not is_candidate:
            long_capital_initials = []
            for w in long:
                if w[0].isupper():
                    long_capital_initials.append(w[0])
        ####
        if left:
            capitals = capitals[::-1]
            long = long[::-1]
            long_index = long_index[::-1]
        for j, c in enumerate(capitals):
            if j >= len(long):
                long_form = []
                break
            else:
                if long[j][0].lower() == c.lower():
                    long_form.append(long_index[j])
                else:
                    long_form = []
                    break
        # FIXED [issue: different modern GAN architectures : Deep Convolutional ( DC ) GAN , Spectral Normalization ( SN ) GAN , and Spectral Normalization GAN with Gradient Penalty ( SNGP ) .]
        if not is_candidate:
            if len(long_capital_initials) != len(long_form) and len(long_capital_initials) > 0:
                long_form = []
        ####
        long_form.sort()
        if output_string:
            if long_form:
                return long[long_form[0]:long_form[-1] + 1]
            else:
                return ""
        else:
            return long_form

    # FIXED [issue: annotation software application , Text Annotation Graphs , or TAG , that provides a rich set of]
    def high_recall_character_match(self, sentence, shorts, all_acronyms, ignore_hyphen=False, map_chars=False,default_diction=False):
        '''
        This function finds the long form of the acronyms that are not surrounded by parentheses in the text using scritc rule of character matching (the initial of the sequence of the words in the candidate long form should form the acronym)
        example: annotation software application , Text Annotation Graphs , or TAG , that provides a rich set of ...

        :param sentence:
        :param shorts:
        :param all_acronyms:
        :return:
        '''
        labels = ['O'] * len(sentence)
        diction = {}
        for i, t in enumerate(sentence):
            if i in shorts:
                labels[i] = 'B-short'
                # FIXED [issue: We show that stochastic gradient Markov chain Monte Carlo ( SG - MCMC ) - a class of ]
                if ignore_hyphen:
                    t = t.replace('-', '')
                capitals = []
                for c in t:
                    if c.isupper():
                        capitals.append(c)
                cand_long = sentence[max(i - len(capitals) - 10, 0):i]
                long_form = ''
                long_form_index = []
                for j in range(max(len(cand_long) - len(capitals), 0)):
                    if ''.join(w[0] for w in cand_long[j:j + len(capitals)]) == t:
                        long_form = ' '.join(cand_long[j:j + len(capitals)])
                        long_form_index = list(range(max(max(i - len(capitals) - 10, 0) + j, 0),
                                                     max(max(i - len(capitals) - 10, 0) + j, 0) + len(capitals)))
                        break
                if not long_form:
                    cand_long = sentence[i + 1:len(capitals) + i + 10]
                    for j in range(max(len(cand_long) - len(capitals), 0)):
                        if ''.join(w[0] for w in cand_long[j:j + len(capitals)]) == t:
                            long_form = ' '.join(cand_long[j:j + len(capitals)])
                            long_form_index = list(range(i + 1 + j, i + j + len(capitals) + 1))
                            break
                long_form = long_form.split()
                if long_form:
                    if long_form[0] in stop_words or long_form[-1] in stop_words:
                        long_form = []
                    if any(lf in string.punctuation for lf in long_form):
                        long_form = []
                    if __name__ != "__main__":
                        NPs = [np.text for np in nlp(' '.join(sentence)).noun_chunks]
                        long_form_str = ' '.join(long_form)
                        if all(long_form_str not in np for np in NPs):
                            long_form = []
                if long_form:
                    for j in long_form_index:
                        labels[j] = 'I-long'
                    labels[long_form_index[0]] = 'B-long'
                    if default_diction:
                        diction[(long_form_index[0], long_form_index[-1])] = (i, i)
        return self.create_diction(sentence, labels, all_acronyms=all_acronyms, tag='high recall character match',
                                   map_chars=map_chars,diction=diction)

    def character_match_extract(self, sentence, shorts, all_acronyms, check_all_capitals=False, ignore_hyphen=False,
                                ignore_punc=False, map_chars=False,default_diction=False):
        labels = ['O'] * len(sentence)
        diction = {}
        for i, t in enumerate(sentence):
            if i in shorts:
                labels[i] = 'B-short'
                # FIXED [issue: We show that stochastic gradient Markov chain Monte Carlo ( SG - MCMC ) - a class of ]
                if ignore_hyphen:
                    t = t.replace('-', '')
                # FIXED [issue: acronyms with lowercase letters, example:  of an enhanced Node B ( eNB )  ]
                if check_all_capitals:
                    if len(t) != len([c for c in t if c.isupper()]):
                        continue
                # FIXED [issue: such as Latent Semantic Analysis ( LSA ; )]
                cand_long, cand_long_index, left = self.extract_cand_long(sentence, t, i, ignore_punc=ignore_punc)
                long_form = []
                if cand_long:
                    long_form = self.character_match(t, cand_long, cand_long_index, left, is_candidate=True)
                if long_form:
                    labels[long_form[0]] = 'B-long'
                    for l in long_form[1:]:
                        labels[l] = 'I-long'
                    if default_diction:
                        diction[(long_form[0], long_form[-1])] = (i, i)
        return self.create_diction(sentence, labels, all_acronyms=all_acronyms, tag='character match',
                                   map_chars=map_chars, diction=diction)

    # FIXED [issue: roman numbers]
    def filterout_roman_numbers(self, diction):
        '''
        This function removes roman numbers from the list of extracted acronyms. It removes only numbers from 1 to 20.
        :param diction:
        :return:
        '''
        acronyms = set(diction.keys())
        for acr in acronyms:
            # instead of all roman acronyms we remove only 1 to 20:
            # if bool(re.search(r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$", acr)):
            if acr in ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'XIII', 'XIV', 'XV',
                       'XVI', 'XVII', 'XVIII', 'XIX', 'XX']:
                del diction[acr]
        return diction

    # FIXED [issue: 'In International Semantic Web Conference , ( ISWC ) ,']
    def remove_punctuations(self, diction):
        '''
        Remove head+tailing punctuations

        :param diction:
        :return:
        '''

        for acr, info in diction.items():
            if len(info[0]) > 0:
                if info[0][0] in string.punctuation:
                    info[0] = info[0][2:]
                    info[2][0] = info[2][0] + 1
                    info[3] = 'remove punctuation'
            if len(info[0]) > 0:
                if info[0][-1] in string.punctuation:
                    info[0] = info[0][:-2]
                    info[2][1] = info[2][1] - 1
                    info[3] = 'remove punctuation'

        return diction

    # FIXED [issue: and Cantab Capital Institute for Mathematics of Information ( CCIMI )]
    def initial_capitals_extract(self, sentence, shorts, all_acronyms, ignore_hyphen=False, map_chars=False,default_diction=False):
        '''
        This function captures long form which their initials is capital and could form the acronym in the format "long form (acronym)" or "(acronym) long form"
        example:

        :param sentence:
        :param shorts:
        :param all_acronyms:
        :return:
        '''
        labels = ['O'] * len(sentence)
        diction = {}
        for i, t in enumerate(sentence):
            if i in shorts:
                labels[i] = 'B-short'
                # FIXED [issue: We show that stochastic gradient Markov chain Monte Carlo ( SG - MCMC ) - a class of ]
                if ignore_hyphen:
                    t = t.replace('-', '')
                capitals = []
                for c in t:
                    if c.isupper():
                        capitals.append(c)
                cand_long, cand_long_index, left = self.extract_cand_long(sentence, t, i)
                capital_initials = []
                capital_initials_index = []
                for j, w in enumerate(cand_long):
                    lll = labels[i + j - len(cand_long) - 1]
                    if w[0].isupper() and labels[i + j - len(cand_long) - 1] == 'O':
                        capital_initials.append(w[0])
                        capital_initials_index.append(j)
                if ''.join(capital_initials) == t:
                    long_form = cand_long[capital_initials_index[0]:capital_initials_index[-1] + 1]
                    long_form_index = cand_long_index[capital_initials_index[0]:capital_initials_index[-1] + 1]
                    for lfi in long_form_index:
                        labels[lfi] = 'I-long'
                    labels[long_form_index[0]] = 'B-long'
                    if default_diction:
                        diction[(long_form_index[0], long_form_index[-1])] = (i, i)
        return self.create_diction(sentence, labels, all_acronyms=all_acronyms, tag='Capital Initials',
                                   map_chars=map_chars,diction=diction)

    # FIXED [issue: for C - GAN indicates ]
    def hyphen_in_acronym(self, sentence, shorts):
        '''
        This function merge two acronyms if there is a hyphen between them
        example: for C - GAN indicates

        :param sentence:
        :param shorts:
        :return:
        '''

        new_shorts = []
        for short in shorts:
            i = short + 1
            next_hyphen = False
            while i < len(sentence) and sentence[i] == '-':
                next_hyphen = True
                i += 1
            j = short - 1
            before_hyphen = False
            while j > 0 and sentence[j] == '-':
                before_hyphen = True
                j -= 1
            # FIXED [check length of the new acronym. issue: SPG - GCN)In Table]
            # if i < len(sentence) and sentence[i].isupper() and len(sentence[i]) <= 2:
            if i < len(sentence) and sentence[i].isupper() and next_hyphen:
                for ind in range(short + 1, i + 1):
                    new_shorts += [ind]
            # FIXED [check length of the new acronym. issue: SPG - GCN)In Table]
            # if j > -1 and sentence[j].isupper() and len(sentence[j]) <= 2:
            if j > -1 and sentence[j].isupper() and before_hyphen:
                for ind in range(j, short):
                    new_shorts += [ind]

        shorts.extend(new_shorts)
        return shorts

    # FIXED [issue: We show that stochastic gradient Markov chain Monte Carlo ( SG - MCMC ) - a class of ]
    def merge_hyphened_acronyms(self, sentence, labels=[]):
        '''
        This function merge hyphened acronyms
        example: We show that stochastic gradient Markov chain Monte Carlo ( SG - MCMC ) - a class of

        :param sentence:
        :return:
        '''
        new_sentence = []
        new_labels = []
        merge = False
        shorts = self.short_extract(sentence, 0.6, True)
        shorts += self.hyphen_in_acronym(sentence, shorts)

        for i, t in enumerate(sentence):
            if i in shorts and i - 1 in shorts and i + 1 in shorts and t == '-':
                merge = True
                if len(new_sentence) > 0:
                    new_sentence[-1] += '-'
                else:
                    new_sentence += ['-']
                continue
            if merge:
                if len(new_sentence) > 0:
                    new_sentence[-1] += t
                else:
                    new_sentence += [t]
            else:
                new_sentence.append(t)
                if labels:
                    new_labels.append(labels[i])
            merge = False

        return new_sentence, new_labels

    # FIXED [issue: we use encoder RNN ( ER )]
    def add_embedded_acronym(self, diction, shorts, sentence):
        '''
        This function will add the embeded acronyms into the dictionary
        example: we use encoder RNN ( ER )

        :param diction:
        :param shorts:
        :return:
        '''
        short_captured = []
        long_captured = []
        for acr, info in diction.items():
            short_captured.append(info[1][0])
            if info[2]:
                long_captured.extend(list(range(info[2][0], info[2][1])))
        for short in shorts:
            if short not in short_captured and short in long_captured and sentence[short] not in diction:
                diction[sentence[short]] = ['', (short, short), [], 'embedded acronym']
        return diction

    # FIXED [issue: acronym stands for template]
    def extract_templates(self, sentence, shorts, map_chars=False):
        '''
        Extract acronym and long forms based on templates
        example: PM stands for Product Manager

        :param sentence:
        :param shorts:
        :return:
        '''
        labels = ['O'] * len(sentence)
        for i, t in enumerate(sentence):
            if i in shorts:
                labels[i] = 'B-short'
                capitals = []
                for c in t:
                    if c.isupper():
                        capitals.append(c)
                if i < len(sentence) - len(capitals) - 2:
                    if sentence[i + 1] == 'stands' and sentence[i + 2] == 'for':
                        if ''.join(w[0] for w in sentence[i + 3:i + 3 + len(capitals)]) == ''.join(capitals):
                            labels[i + 3:i + 3 + len(capitals)] = ['I-long'] * len(capitals)
                            labels[i + 3] = 'B-long'
        return self.create_diction(sentence, labels, all_acronyms=False, tag='Template', map_chars=map_chars)

    # FIXED [issue: preserve number of meanins extracted from other method]
    def update_pair(self, old_pair, new_pair):
        for acr, info in new_pair.items():
            if acr not in old_pair:
                old_pair[acr] = info
            else:
                info[4] = max(info[4],old_pair[acr][4])
                old_pair[acr] = info
        return old_pair

    def extract(self, sentence, active_rules):
        # FIXED [issue: of an enhanced Node B ( eNB ) ]
        shorts = self.short_extract(sentence, 0.6, active_rules['starting_lower_case'],
                                    ignore_dot=active_rules['ignore_dot'])
        # FIXED [issue: acronyms like StESs]
        if active_rules['low_short_threshold']:
            shorts += self.short_extract(sentence, 0.50, active_rules['starting_lower_case'],
                                         ignore_dot=active_rules['ignore_dot'])
        ####
        # FIXED [issue: for C - GAN indicates ]
        if active_rules['hyphen_in_acronym']:
            shorts += self.hyphen_in_acronym(sentence, shorts)
        ####
        pairs = {}
        if active_rules['schwartz']:
            # FIXED [issue: such as Latent Semantic Analysis ( LSA ; )]
            pairs = self.schwartz_extract(sentence, shorts, active_rules['no_parentheses'],
                                          ignore_punc=active_rules['ignore_punc_in_parentheses'],
                                          add_punc=active_rules['extend_punc'],
                                          small_window=active_rules['small_window'],
                                          no_stop_words=active_rules['no_beginning_stop_word'],
                                          ignore_righthand=active_rules['ignore_right_hand'],
                                          map_chars=active_rules['map_chars'],
                                          default_diction=active_rules['default_diction'])
        # FIXED [issue: 'User - guided Social Media Crawling method ( USMC ) that']
        if active_rules['bounded_schwartz']:
            # FIXED [issue: such as Latent Semantic Analysis ( LSA ; )]
            bounded_pairs = self.bounded_schwartz_extract(sentence, shorts, active_rules['no_parentheses'],
                                                          ignore_punc=active_rules['ignore_punc_in_parentheses'],
                                                          add_punc=active_rules['extend_punc'],
                                                          small_window=active_rules['small_window'],
                                                          no_stop_words=active_rules['no_beginning_stop_word'],
                                                          ignore_righthand=active_rules['ignore_right_hand'],
                                                          map_chars=active_rules['map_chars'],
                                                          default_diction=active_rules['default_diction'])
            # pairs.update(bounded_pairs)
            pairs = self.update_pair(pairs, bounded_pairs)
        # FIXED [issue: The Stopping Trained in America PhDs from Leaving the Economy Act ( or STAPLE Act ) has bee introduced]
        if active_rules['high_recall_schwartz']:
            hr_paris = self.high_recall_schwartz(sentence, shorts, active_rules['no_parentheses'],
                                                 ignore_punc=active_rules['ignore_punc_in_parentheses'],
                                                 add_punc=active_rules['extend_punc'],
                                                 small_window=active_rules['small_window'],
                                                 no_stop_words=active_rules['no_beginning_stop_word'],
                                                 ignore_righthand=active_rules['ignore_right_hand'],
                                                 map_chars=active_rules['map_chars'],
                                                 default_diction=active_rules['default_diction'])
            # pairs.update(hr_paris)
            pairs = self.update_pair(pairs,hr_paris)
        if active_rules['character']:
            # FIXED [issue: acronyms with lowercase letters, example: of an enhanced Node B ( eNB )  ]
            # FIXED [issue: such as Latent Semantic Analysis ( LSA ; )]
            character_pairs = self.character_match_extract(sentence, shorts, not active_rules['schwartz'],
                                                           check_all_capitals=active_rules['check_all_capitals'],
                                                           ignore_punc=active_rules['ignore_punc_in_parentheses'],
                                                           map_chars=active_rules['map_chars'],
                                                           default_diction=active_rules['default_diction'])
            # pairs.update(character_pairs)
            pairs = self.update_pair(pairs, character_pairs)
        # FIXED [issue: annotation software application , Text Annotation Graphs , or TAG , that provides a rich set of]
        if active_rules['high_recall_character_match']:
            character_pairs = self.high_recall_character_match(sentence, shorts, not active_rules['schwartz'],
                                                               map_chars=active_rules['map_chars'],default_diction=active_rules['default_diction'])
            acronyms = character_pairs.keys()
            for acr in acronyms:
                if acr not in pairs or len(pairs[acr][0]) == 0:
                    pairs[acr] = character_pairs[acr]
        # FIXED [issue: and Cantab Capital Institute for Mathematics of Information ( CCIMI )]
        if active_rules['initial_capitals']:
            character_pairs = self.initial_capitals_extract(sentence, shorts, not active_rules['schwartz'],
                                                            map_chars=active_rules['map_chars'],default_diction=active_rules['default_diction'])
            # pairs.update(character_pairs)
            pairs = self.update_pair(pairs,character_pairs)
        # FIXED [issue: acronym stands for long form]
        if active_rules['template']:
            template_pairs = self.extract_templates(sentence, shorts, map_chars=active_rules['map_chars'])
            # pairs.update(template_pairs)
            pairs = self.update_pair(pairs,template_pairs)
        # FIXED [issue: we use encoder RNN ( ER )]
        if active_rules['capture_embedded_acronym']:
            pairs = self.add_embedded_acronym(pairs, shorts, sentence)
        # FIXED [issue: roman numbers]
        if active_rules['roman']:
            pairs = self.filterout_roman_numbers(pairs)
        # FIXED [issue: 'In International Semantic Web Conference , ( ISWC ) ,']
        if active_rules['remove_punctuation']:
            pairs = self.remove_punctuations(pairs)
        return pairs

        failures = []
        sucess = []
        for i in range(len(gold_label)):
            gold_diction = self.create_diction(dataset[i]['token'], gold_label[i], tag='gold')
            pred_diction = pred_dictions[i]
            if gold_diction.keys() != pred_diction.keys() or set(v[0] for v in gold_diction.values()) != set(
                    v[0] for v in pred_diction.values()):
                failures.append([gold_diction, pred_diction, dataset[i]['token'], dataset[i]['id']])
            else:
                sucess.append([gold_diction, pred_diction, dataset[i]['token'], dataset[i]['id']])
        failure_ratio = 'Failures: {:.2%}'.format(len(failures) / len(dataset)) + '\n'
        print(failure_ratio)
        results += failure_ratio
        return failures, sucess, results