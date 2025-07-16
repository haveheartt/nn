#makemore

makemore takes one text file as input, where each line is assumed to be one training thing, and generates more things like it. Under the hood, it is an autoregressive character-level language model, with a wide choice of models from bigrams all the way to a Transformer (exactly as seen in GPT). For example, we can feed it a database of names, and makemore will generate cool baby name ideas that all sound name-like, but are not already existing names. Or if we feed it a database of company names then we can generate new ideas for a name of a company. Or we can just feed it valid scrabble words and generate english-like babble.

let's think through our very first language model! A character level language model is predicting the next
character in a sequence given already some concrete sequence of characters before it.

we will start build a bigram language model, in the bigram we are always working with two characters at a time, we are looking for a character given and trying to predict the next one in the sequence, we are just modeling that little local structure. it's a very simple and weak language model but it's a great place to start.

we iterate through the list of words, and for each word we iterate through the string by 2, but before we create special start and end tokens to differentiate between words and to know which character is at the start and at the end. In order to learn the statistics about which characters are likely to follow other characters, the simplest way in the bigram language models is to simply do it by counting. We are going to count how often any one of these combinations occurs in the training set.

in the jupyter notebook we start keeping these informations in a dictionary, but will be significantly more convenient to keep these informations in a 2D array. The rows are going to be the first character of the bigram and the columns are going to be the second character and each entry in this two dimensional array will tell us how often that first character follows the second character in the dataset.

after we create the array we have all the information necessary for us to actually sample from this bigram character level modeel and roughly what we ar going to do is just start following the probabilities and counts. In the beggining we start with the dot (our start token) so to sample the first character of a name we are looking at the first row, we have got the raw counts, then we need to convert to probabilities

then after learning about internals of broadcast and vector normalization, we trained a bigram language model by just counting how frequently any pairing occurs and then normalizing so that we get a nice property distribution. So really the elements of array P are really the parameters of our bigram language model, giving us and summarizing the statistics. So we trained and know how to sample from the model, we iteratively just sample from next character and feed it in each time and get a next character.

now to summarize the quality of this model into a single number, how good it predicts the training set, for example. We can now evaluate the training loss, which tells us the quality of this model, just like we saw in micrograd.

