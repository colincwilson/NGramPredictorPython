import sys
import kenlm
from pathlib import Path

lm2_filename = Path.home() / \
        f'Languages/00English/Wiki-Text-103/wiki_2gram.binary'
model2 = kenlm.Model(str(lm2_filename))

lm5_filename = Path.home() / \
        f'Languages/00English/Wiki-Text-103/wiki_5gram.binary'
model5 = kenlm.Model(str(lm5_filename))

sent1 = 'I shopped at the store .'
print(list(model2.full_scores(sent1)))

print(list(model5.full_scores(sent1)))

#sys.exit(0)

# Word predictor module.
# https://github.com/kdv123/TextPredictorPython
textpredict_dir = Path.home() / \
    'Projects/braille_reading/extern/TextPredictorPython'
sys.path.append(str(textpredict_dir))
from predictor import WordPredictor

#lm_filename = 'resources/lm_word_medium.kenlm'
for lm in ['wiki_2gram', 'wiki_3gram', 'wiki_4gram', 'wiki_5gram']:
    lm_filename = Path.home() / \
        f'Languages/00English/Wiki-Text-103/{lm}.binary'
    vocab_filename = 'resources/vocab_100k'
    punct_filename = 'resources/tokens.txt'
    model = WordPredictor(\
        str(lm_filename), vocab_filename, punct_filename)
    model.verbose = False

    prefix = 's'
    context = 'i shopped at the fancy'
    most_prob_word, log_prob = model.get_most_probable_word(
        prefix, context, vocab_id='', min_log_prob=-float('inf'))

    print('Context: ' + context)
    print('Prefix: ' + prefix)
    print('Most probable word: "' + most_prob_word +
          '" with log probability: ' + str(log_prob))
    print(
        model.get_words_with_context(prefix='s',
                                     context=context,
                                     num_predictions=5))
    print()
#print(predictor.create_char_list_from_vocab(vocab_filename))

#words = predictor.get_words_with_context('s', 'abra ka dabra', '', 3, -float('inf'))
#predictor.print_suggestions(words)

# words = word_predictor.get_words_with_context('', 'hello', '', 3,
#                                               -float('inf'))
# word_predictor.print_suggestions(words)
#print(predictor.get_most_likely_word(words))
#predictor.add_vocab('vocab_100k', vocab_filename)
