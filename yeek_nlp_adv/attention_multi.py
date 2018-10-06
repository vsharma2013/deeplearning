from os import system
import numpy as np
import time
from keras.preprocessing.sequence import pad_sequences
from attention import getSeq2SeqModel

LATENT_DIM_DECODER = 256 # idea: make it different to ensure things all fit together properly!

def decode_sequence(encoder_model, decoder_model, word2idx_outputs, idx2word_trans, max_len_target, input_seq):
  # Encode the input as state vectors.
  enc_out = encoder_model.predict(input_seq)

  # Generate empty target sequence of length 1.
  target_seq = np.zeros((1, 1))
  
  # Populate the first character of target sequence with the start character.
  # NOTE: tokenizer lower-cases all words
  target_seq[0, 0] = word2idx_outputs['<sos>']

  # if we get this we break
  eos = word2idx_outputs['<eos>']


  # [s, c] will be updated in each loop iteration
  s = np.zeros((1, LATENT_DIM_DECODER))
  c = np.zeros((1, LATENT_DIM_DECODER))


  # Create the translation
  output_sentence = []
  for _ in range(max_len_target):
    o, s, c = decoder_model.predict([target_seq, enc_out, s, c])
        

    # Get next word
    idx = np.argmax(o.flatten())

    # End sentence of EOS
    if eos == idx:
      break

    word = ''
    if idx > 0:
      word = idx2word_trans[idx]
      output_sentence.append(word)

    # Update the decoder input
    # which is just the word just generated
    target_seq[0, 0] = idx

  return ' '.join(output_sentence)

print('\n')
print('\n')
print('\n')

tic = time.time()
model_fra = getSeq2SeqModel('/home/vishal/deeplearning/yeek_nlp_adv/data/translate/fra.txt')
model_deu = getSeq2SeqModel('/home/vishal/deeplearning/yeek_nlp_adv/data/translate/deu.txt')
model_mar = getSeq2SeqModel('/home/vishal/deeplearning/yeek_nlp_adv/data/translate/mar.txt')
model_spa = getSeq2SeqModel('/home/vishal/deeplearning/yeek_nlp_adv/data/translate/spa.txt')
model_ara = getSeq2SeqModel('/home/vishal/deeplearning/yeek_nlp_adv/data/translate/ara.txt')
model_cmn = getSeq2SeqModel('/home/vishal/deeplearning/yeek_nlp_adv/data/translate/cmn.txt')

toc = time.time()
time_elapsed = toc - tic
str_time = "0 ms"

if time_elapsed < 1:
    str_time = str(1000 * time_elapsed) + " ms"
elif time_elapsed < 60:
    str_time = str(time_elapsed) + " sec"
else:
    str_time = str(time_elapsed // 60.0) + " mins"

print("\n\nTime taken to complete the operation = ", str_time)
print('\n')
print('\n')

def runForSeqModeAndInput(r, inputText, lang):
	tokenizer_inputs = r['tokenizer_inputs'];
	max_len_input = r['max_len_input']
	input_texts = []
	input_texts.append(inputText)
	input_seq = tokenizer_inputs.texts_to_sequences(input_texts)
	encoder_inputs = pad_sequences(input_seq, maxlen=max_len_input)
	translation = decode_sequence(r['encoder_model'], r['decoder_model'], r['word2idx_outputs'], r['idx2word_trans'], r['max_len_target'], encoder_inputs)
	
	print('\nPredicted %s translation: %s' %(lang, translation))

while True:
	ans = input("\n\nEnter text to translate: ")
	if ans and ans.lower().startswith('q'):
		break
	if(ans.startswith(']')):
		system('clear')
	else:
		print('\n\nInput sentence:', ans)
		runForSeqModeAndInput(model_fra, ans, 'FRENCH')
		runForSeqModeAndInput(model_deu, ans, 'GERMAN')
		runForSeqModeAndInput(model_mar, ans, 'MARATHI')
		runForSeqModeAndInput(model_spa, ans, 'SPANISH')
		runForSeqModeAndInput(model_ara, ans, 'ARABIC')
		runForSeqModeAndInput(model_cmn, ans, 'CHINESE')

