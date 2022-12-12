import tensorflow as tf
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab
import keras_nlp

"""
## Tokenizing the data
We'll define two tokenizers - one for the source language (English), and the other
for the target language (Spanish). We'll be using
`keras_nlp.tokenizers.WordPieceTokenizer` to tokenize the text.
`keras_nlp.tokenizers.WordPieceTokenizer` takes a WordPiece vocabulary
and has functions for tokenizing the text, and detokenizing sequences of tokens.
Before we define the two tokenizers, we first need to train them on the dataset
we have. The WordPiece tokenization algorithm is a subword tokenization algorithm;
training it on a corpus gives us a vocabulary of subwords. A subword tokenizer
is a compromise between word tokenizers (word tokenizers need very large
vocabularies for good coverage of input words), and character tokenizers
(characters don't really encode meaning like words do). Luckily, TensorFlow Text
makes it very simple to train WordPiece on a corpus as described in
[this guide](https://www.tensorflow.org/text/guide/subwords_tokenizer).
"""


def train_word_piece(text_samples, vocab_size, reserved_tokens):
    bert_vocab_args = dict(
        # The target vocabulary size
        vocab_size=vocab_size,
        # Reserved tokens that must be included in the vocabulary
        reserved_tokens=reserved_tokens,
        # Arguments for `text.BertTokenizer`
        bert_tokenizer_params={"lower_case": True},
    )

    word_piece_ds = tf.data.Dataset.from_tensor_slices(text_samples)
    vocab = bert_vocab.bert_vocab_from_dataset(
        word_piece_ds.batch(1000).prefetch(2), **bert_vocab_args
    )
    return vocab


"""
## Format datasets
"""


def make_dataset(pairs, eng_tokenizer, spa_tokenizer, max_sequence_length, batch_size):
    def preprocess_batch(eng, spa):
        batch_size = tf.shape(spa)[0]

        eng = eng_tokenizer(eng)
        spa = spa_tokenizer(spa)

        # Pad `eng` to `MAX_SEQUENCE_LENGTH`.
        eng_start_end_packer = keras_nlp.layers.StartEndPacker(
            sequence_length=max_sequence_length,
            pad_value=eng_tokenizer.token_to_id("[PAD]"),
        )
        eng = eng_start_end_packer(eng)

        # Add special tokens (`"[START]"` and `"[END]"`) to `spa` and pad it as well.
        spa_start_end_packer = keras_nlp.layers.StartEndPacker(
            sequence_length=max_sequence_length + 1,
            start_value=spa_tokenizer.token_to_id("[START]"),
            end_value=spa_tokenizer.token_to_id("[END]"),
            pad_value=spa_tokenizer.token_to_id("[PAD]"),
        )
        spa = spa_start_end_packer(spa)

        return (
            {
                "encoder_inputs": eng,
                "decoder_inputs": spa[:, :-1],
            },
            spa[:, 1:],
        )
    eng_texts, spa_texts = zip(*pairs)
    eng_texts = list(eng_texts)
    spa_texts = list(spa_texts)
    dataset = tf.data.Dataset.from_tensor_slices((eng_texts, spa_texts))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(preprocess_batch, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.shuffle(2048).prefetch(16).cache()


"""
## Decoding test sentences (qualitative analysis)
"""


def decode_sequences(input_sentences, eng_tokenizer, spa_tokenizer, max_sequence_length, transformer):
    batch_size = tf.shape(input_sentences)[0]

    # Tokenize the encoder input.
    encoder_input_tokens = eng_tokenizer(input_sentences).to_tensor(
        shape=(None, max_sequence_length)
    )

    # Define a function that outputs the next token's probability given the
    # input sequence.
    def token_probability_fn(decoder_input_tokens):
        return transformer([encoder_input_tokens, decoder_input_tokens])[:, -1, :]

    # Set the prompt to the "[START]" token.
    prompt = tf.fill((batch_size, 1), spa_tokenizer.token_to_id("[START]"))

    generated_tokens = keras_nlp.utils.greedy_search(
        token_probability_fn,
        prompt,
        max_length=40,
        end_token_id=spa_tokenizer.token_to_id("[END]"),
    )
    generated_sentences = spa_tokenizer.detokenize(generated_tokens)
    return generated_sentences
