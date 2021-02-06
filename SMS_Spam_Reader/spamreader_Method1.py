
# data from https://archive.ics.uci.edu/ml/datasets/sms+spam+collection

import string

def tokenize(s):
    return s.translate(punct_stripper).split()


if __name__ == "__main__":
    items = []

    print('reading SMSSpamCollection')
    with open('SMSSpamCollection') as datafile: #, encoding='latin-1') as datafile:
        for line in datafile:
            row = line.rstrip().split('\t')
            # store first two fields as (label, message)
            items.append((row[0], row[1]))
    print('read', len(items), 'items')
    print()

    print('first five items:')
    for item in items[:5]:
        print(item)
    print()

    # very simple tokenizer that first strips punctuation
    punct_stripper = str.maketrans(dict.fromkeys(string.punctuation))


    print('tokenizing')
    items = [(item[0], tokenize(item[1])) for item in items]
    print()

    print('first five tokenized items:')
    for item in items[:5]:
        print(item)
    print()

    print('making 80/20 train/test split')
    train_size = int(0.8 * len(items))
    train_items, test_items = items[:train_size], items[train_size:]
    print('train set size:', len(train_items))
    print('test set size:', len(test_items))
    print()

    acc_words =[]
    w_spam = []
    w_ham = []
    n_spam = 0
    n_ham = 0
    for (tag, elem) in train_items:
        if tag == "ham":
            n_ham +=1
        elif tag == "spam":
            n_spam += 1

        for word in elem:
            if word in acc_words:
                if tag == "ham":
                    w_ham[acc_words.index(word)] += 1
                elif tag == "spam":
                    w_spam[acc_words.index(word)] += 1
            else:
                acc_words.append(word)
                if tag == "ham":
                    w_ham.append(1)
                    w_spam.append(0)
                elif tag == "spam":
                    w_spam.append(1)
                    w_ham.append(0)

# # Normal
#     cond_ham = [ (item) / sum(w_ham) for item in w_ham]
#     cond_spam = [ (item) / sum(w_spam) for item in w_spam]

##Laplace implementation (BONUS!)
    cond_ham = [ (item+1) / (sum(w_ham) + len(acc_words)) for item in w_ham]
    cond_spam = [ (item+1) / (sum(w_spam) +len(acc_words)) for item in w_spam]


    p_ham = n_ham / len(train_items)
    p_spam = n_spam / len(train_items)


    tag_test = []
    tag_eval = []
    counter = 0
    for (tag,elem) in test_items:
        p_E_giv_spam = 1
        p_E_giv_ham = 1

        if tag == "ham":
            tag_test.append(0)
        elif tag == "spam":
            tag_test.append(1)


        p_w_mult = 1
        for word in elem:
            if word in acc_words:
                p_E_giv_spam = p_E_giv_spam * cond_spam[acc_words.index(word)]
                p_E_giv_ham = p_E_giv_ham * cond_ham[acc_words.index(word)]

                p_w_mult = p_w_mult * (w_ham[acc_words.index(word)] + w_spam[acc_words.index(word)]) / len(acc_words)
                # p_w_mult = 1


        P_ham_giv_E = p_E_giv_ham * p_ham / (p_w_mult)
        P_spam_giv_E = p_E_giv_spam * p_spam / (p_w_mult)



        if P_spam_giv_E >= P_ham_giv_E:
            tag_eval.append(1)
            eval = "spam"
            #print("This case probably is spam")
        else:
            tag_eval.append(0)
            eval = "ham"
        counter +=1
        # if counter %22 == 0 and counter >=51:
        #     print(counter/len(test_items) *100)
        if counter <= 50:

            print("Test set: %s P_ham= %s P_spam= %s Predicted tag: %s" % (counter, P_ham_giv_E, P_spam_giv_E, eval))




    count = 0
    for i in range(len(tag_eval)):
        if tag_eval[i] == tag_test[i]:
            count+=1
    accuracy = count/ len(tag_eval) *100
    print("Accuracy is : %s " % accuracy + "percent")






print("done")

