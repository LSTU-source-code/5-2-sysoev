import tensorflow as tf
import numpy as np
import re
import os
import time
import nltk
from nltk.translate.bleu_score import sentence_bleu

# Установка библиотеки для BLEU-метрики
nltk.download('punkt')

# === 1. Подготовка данных ===
def preprocess_sentence(w):
    """Предобработка предложения"""
    w = w.lower().strip()
    w = re.sub(r"([?.!,])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^a-zA-Zа-яА-Я0-9?.!,]+", " ", w)
    w = w.rstrip().strip()
    return '<start> ' + w + ' <end>'

def create_dataset(path, num_examples):
    """Создание датасета из пар предложений"""
    lines = open(path, encoding='utf-8').read().strip().split('\n')
    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')[:2]] for l in lines[:num_examples]]
    return word_pairs

def create_inverse_dataset(path, num_examples):
    """Создание обратных пар (английский -> русский)"""
    lines = open(path, encoding='utf-8').read().strip().split('\n')
    inverse_pairs = [[preprocess_sentence(l.split('\t')[1]), preprocess_sentence(l.split('\t')[0])] for l in lines[:num_examples]]
    return inverse_pairs

# === 2. Создание словарей ===
class LanguageIndex:
    """Класс для создания словарей"""
    def __init__(self, lang):
        self.word2idx = {'<pad>': 0}
        self.idx2word = {0: '<pad>'}
        self.vocab = set()
        
        for phrase in lang:
            self.vocab.update(phrase.split(' '))
        
        for index, word in enumerate(sorted(self.vocab)):
            self.word2idx[word] = index + 1
        
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

# === 3. Архитектура модели ===
def gru(units):
    """Функция для создания GRU-слоя"""
    return tf.keras.layers.GRU(units, 
                               return_sequences=True, 
                               return_state=True, 
                               recurrent_initializer='glorot_uniform')

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = gru(self.enc_units)

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = gru(self.dec_units)
        self.fc = tf.keras.layers.Dense(vocab_size)

        # Внимание
        self.W1 = tf.keras.layers.Dense(dec_units)
        self.W2 = tf.keras.layers.Dense(dec_units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, x, hidden, enc_output):
        # Расчет весов внимания
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = self.V(tf.nn.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * enc_output
        context_vector = tf.reduce_sum(context_vector, axis=1)

        # Объединение контекста и входа
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, state, attention_weights

# === 4. Функции для обучения ===
def loss_function(real, pred):
    """Функция потерь"""
    mask = 1 - np.equal(real, 0)
    loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
    return tf.reduce_mean(loss_)

def train_model(encoder, decoder, dataset, vocab_tar_size, checkpoint_prefix, epochs=10):
    """Обучение модели"""
    optimizer = tf.optimizers.Adam()
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
    
    for epoch in range(epochs):
        start = time.time()
        total_loss = 0
        
        for (batch, (inp, targ)) in enumerate(dataset):
            loss = 0
            hidden = encoder.initialize_hidden_state()
            
            with tf.GradientTape() as tape:
                enc_output, enc_hidden = encoder(inp, hidden)
                dec_hidden = enc_hidden
                dec_input = tf.expand_dims([targ_lang.word2idx['<start>']] * BATCH_SIZE, 1)
                
                for t in range(1, targ.shape[1]):
                    predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
                    loss += loss_function(targ[:, t], predictions)
                    dec_input = tf.expand_dims(targ[:, t], 1)
                
                batch_loss = (loss / int(targ.shape[1]))
                total_loss += batch_loss
                
                variables = encoder.variables + decoder.variables
                gradients = tape.gradient(loss, variables)
                optimizer.apply_gradients(zip(gradients, variables))
            
            if batch % 100 == 0:
                print(f'Epoch {epoch+1} Batch {batch} Loss {batch_loss.numpy():.4f}')
        
        checkpoint.save(file_prefix=checkpoint_prefix)
        print(f'Epoch {epoch+1} Loss {total_loss/len(dataset):.4f}')
        print(f'Time taken for 1 epoch {time.time()-start:.2f} sec\n')

# === 5. Функции перевода ===
def evaluate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ):
    """Функция перевода"""
    inputs = [inp_lang.word2idx.get(word, 0) for word in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding='post')
    inputs = tf.convert_to_tensor(inputs)
    
    result = ''
    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word2idx['<start>']], 0)
    attention_plot = np.zeros((max_length_targ, max_length_inp))
    
    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)
        attention_plot[t] = attention_weights.numpy().reshape(-1)
        predicted_id = tf.argmax(predictions[0]).numpy()
        
        if targ_lang.idx2word[predicted_id] == '<end>':
            break
            
        result += targ_lang.idx2word[predicted_id] + ' '
        dec_input = tf.expand_dims([predicted_id], 0)
    
    return result.strip(), sentence, attention_plot

def double_translate(sentence, ru_en_encoder, ru_en_decoder, en_ru_encoder, en_ru_decoder, 
                     inp_ru, targ_en, inp_en, targ_ru, max_len_ru, max_len_en):
    """Двойной перевод: русский -> английский -> русский"""
    en_translation, _, _ = evaluate(sentence, ru_en_encoder, ru_en_decoder, inp_ru, targ_en, max_len_ru, max_len_en)
    ru_translation, _, _ = evaluate(en_translation, en_ru_encoder, en_ru_decoder, inp_en, targ_ru, max_len_en, max_len_ru)
    return en_translation, ru_translation

# === 6. Пример использования ===
if __name__ == "__main__":
    # Параметры
    NUM_EXAMPLES = 30000
    BATCH_SIZE = 64
    UNITS = 1024
    EMBEDDING_DIM = 256
    
    # Загрузка данных
    path_to_file = 'rus.txt'
    pairs = create_dataset(path_to_file, NUM_EXAMPLES)
    inverse_pairs = create_inverse_dataset(path_to_file, NUM_EXAMPLES)
    
    # Создание словарей
    input_tensor, target_tensor, inp_lang, targ_lang, max_length_inp, max_length_targ = load_dataset(path_to_file, NUM_EXAMPLES)
    en_input_tensor, ru_target_tensor, en_lang, ru_lang, max_en_len, max_ru_len = load_dataset(path_to_file, NUM_EXAMPLES, inverse=True)
    
    # Разделение на обучающую и валидационную выборки
    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)
    
    # Создание датасета
    BUFFER_SIZE = len(input_tensor_train)
    dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    
    # Инициализация моделей
    vocab_inp_size = len(inp_lang.word2idx)
    vocab_tar_size = len(targ_lang.word2idx)
    
    ru_en_encoder = Encoder(vocab_inp_size, EMBEDDING_DIM, UNITS, BATCH_SIZE)
    ru_en_decoder = Decoder(vocab_tar_size, EMBEDDING_DIM, UNITS, BATCH_SIZE)
    
    # Обучение модели русский -> английский
    checkpoint_dir = './ru_en_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    train_model(ru_en_encoder, ru_en_decoder, dataset, vocab_tar_size, checkpoint_prefix, epochs=10)
    
    # Аналогично создается и обучается модель английский -> русский
    
    # Пример двойного перевода
    original = "Я живу в Липецке"
    en_trans, ru_trans = double_translate(original, ru_en_encoder, ru_en_decoder, en_ru_encoder, en_ru_decoder,
                                          inp_ru=inp_lang, targ_en=targ_lang, inp_en=en_lang, targ_ru=ru_lang,
                                          max_len_ru=max_length_inp, max_len_en=max_en_len)
    
    print(f"Оригинал: {original}")
    print(f"Английский: {en_trans}")
    print(f"Обратный перевод: {ru_trans}")
    
    # Оценка качества
    reference = [original.split()]
    candidate = ru_trans.split()
    bleu_score = sentence_bleu(reference, candidate)
    print(f"BLEU-оценка: {bleu_score:.2f}")