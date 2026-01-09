# Machine Translation
This repository contains an end-to-end Neural Machine Translation system that takes English text as input and returns the French translation. The project demonstrates the full pipeline, including preprocessing, vocabulary creation, model building, and evaluation using a Seq2Seq architecture with Bidirectional LSTMs.

<img src="https://www.dynamiclanguage.com/wp-content/uploads/2019/03/blog-heading-1.png" 
     alt="Machine Translation." 
     style="width: 750px; height: auto; border: 5px solid black; border-radius: 6px;">


## Content 
- [Machine Translation](#machine-translation)
  - [Content](#content)
  - [Description](#description)
  - [Dataset](#dataset)
  - [Install Prerequisites](#install-prerequisites)
  - [Notebook Explainations](#Notebook-Explanation)
  - [Network Architecture](#network-architecture)
  - [Final Model Code](#final-model-code)
  - [Authors](#authors)
  - [Contributing](#contributing)

## Description
This project takes any English text and converts it to sequences of integers based on a big enough French and english vocabularies and passes it to a model that returns a probability distribution over possible translations with accuracy > 97%.



## Dataset
In this project, we will be using [WMT](http://www.statmt.org/), The most common datasets used for machine translation.


## Install Prerequisites
This project requires **Python 3** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [TensorFlow](https://www.tensorflow.org) 1.x
- [Keras](https://keras.io) 2.x

## Notebook Explanation
The notebook included in this repository is very detailed and covers:

### Data Cleaning & Preprocessing
- Lowercasing text
- Removing punctuation
- Tokenization for English & French
- Creating word-to-index and index-to-word mappings
- Padding sequences

### Vocabulary Construction
- Counting unique words
- Building integer encodings
- Preparing input/output pairs

### Model Development
- Explanation of embedding-based Seq2Seq
- Why Bidirectional LSTMs are used
- Visualization of sequence flow
- Step-by-step model summary

### Training & Evaluation
- Loss curves
- Accuracy curves
- Sample translations
- Error analysis

### Inference Pipeline

- Preparing new English text
- Encoding input
- Model prediction
- Decoding to French text

Everything is explained clearly with comments, diagrams, and examples to make the concepts beginner-friendly.

## Network Architecture 
After tokenizing the text and make all pre-processes to it we pass it to a ```Word Embedding``` layer then to 2 ```Bidirectional LSTM``` with 256 units then ```TimeDistributed``` layer with a ```softmax``` activation function to produce probability distribution.
![](https://i.ibb.co/0sKYNHt/Screen-Shot-2020-06-20-at-5-48-59-PM.png)




## Final Model Code

After training this model for 10 epochs we were able to get accuracy of 98% over both the training and validation sets.


```python

def model_final(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    """
    Build and train a model that incorporates embedding, encoder-decoder, and bidirectional RNN on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
   
    learning_rate=5e-3
    
    model=Sequential()
    model.add(Embedding(english_vocab_size,256,
                      input_length=input_shape[1]))

    model.add(Bidirectional(LSTM(256),))

    model.add(RepeatVector(output_sequence_length))

    model.add(Bidirectional(LSTM(256,return_sequences=True)))

    model.add(TimeDistributed(Dense(french_vocab_size,
                              activation='softmax')))
    
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate),
                  metrics=['accuracy'])
    
    print(model.summary())
    return model

```
> you can find all model trials in the notebook

## Author

- **Musa Qureshi**
- [**Github**](https://github.com/Musa-Qureshi-01)
- [**LinkedIn**](https://www.linkedin.com/in/musaqureshi)
- [**X (Twitter)**](https://x.com/Musa_Qureshi_01)
  


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.
