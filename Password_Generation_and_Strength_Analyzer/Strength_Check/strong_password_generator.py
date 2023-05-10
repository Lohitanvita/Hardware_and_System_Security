from random_words import RandomWords
import random
import string


def generate_random_words(words_length, fullname, petName, emailId):
    password_words = []
    word_source = RandomWords()
    if words_length > 7:
        word_length1 = words_length//2
        word1 = word_source.random_word(min_letter_count=word_length1)
        if word1 != fullname and word1 != petName and word1 != emailId:
            password_words.append(word1)
        else:
            word1 = word_source.random_word(min_letter_count=word_length1)
            password_words.append(word1)

        word_length2 = words_length - word_length1
        word2 = word_source.random_word(min_letter_count=word_length2)
        if word2 != fullname and word2 != petName and word2 != emailId:
            password_words.append(word2)
        else:
            word2 = word_source.random_word(min_letter_count=word_length1)
            password_words.append(word2)
    else:
        get_words = word_source.random_word(min_letter_count=words_length)
        password_words.append(get_words)
    return password_words


def generate_random_numbers(num_count):
    random_number = []
    for digit in range(0, num_count):
        random_number.append(random.choice(string.digits))
    return random_number


def generate_special_characters(special_count):
    password_punc = []
    punc = "!()-[]{}\,<>./?@#$%^&*_~"
    min_count = 0
    while min_count < special_count:
        punctuation = random.choice(punc)
        password_punc.append(punctuation)
        min_count += 1
    return password_punc


def generate_password(words_length, num_count, special_count, fullname, emailId, petName, DOB):
    conv_dict = {'o': '0', 'i': '1', 'e': '3', 'b': '6', 'B': '8', 'g': '9', 'I': '1', 'r': '2'}

    new_word_generator = generate_random_words(words_length, fullname, petName, emailId)
    rand_index = random.randint(0, len(new_word_generator)-1)
    new_word = new_word_generator[rand_index]
    new_word = ''.join(map(random.choice, zip(new_word.lower(), new_word.upper())))
    for ch in new_word:
        if ch in conv_dict:
            new_word = new_word.replace(ch, conv_dict[ch], 1)
            break

    new_word_generator[rand_index] = new_word
    new_num_generator = generate_random_numbers(num_count)
    new_splchar_generator = generate_special_characters(special_count)
    new_password = new_word_generator + new_num_generator + new_splchar_generator

    DOB_split = (DOB.replace('-', " ")).split()
    print("DOB_split", DOB_split)
    for i in range(0, len(DOB_split)):
        if DOB_split[i] in new_password:
            generate_random_numbers(num_count)

    pre_shuffle = new_password[1:]
    random.SystemRandom().shuffle(pre_shuffle)
    post_shuffle = list(new_password[0]) + pre_shuffle
    password = ''.join(post_shuffle)
    return password


def main(fullname, emailId, petName, DOB, pswd_length):
    print(fullname, emailId, petName, DOB, pswd_length)
    password_length = int(pswd_length)

    if password_length > 0:
        len_word = round(0.6 * password_length)
        len_nums = round(0.2 * password_length)
        len_splchars = password_length - (len_nums + len_word)
        result = generate_password(len_word, len_nums, len_splchars, fullname, emailId, petName, DOB)
        return result

