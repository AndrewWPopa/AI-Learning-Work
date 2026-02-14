def buildVocabulary(text):
    words = text.split()  # split by spaces
    
    stringtoindex = {}
    indexTostring = {}
    
    for word in words:
        if word not in stringtoindex:
            idx = len(stringtoindex)
            stringtoindex[word] = idx
            indexTostring[idx] = word

    return stringtoindex, indexTostring

# Function to encode a string of text into a list of indices using the stringtoindex mapping
def encode(text, stringtoindex):
    encoded = []
    for ch in text:
        encoded.append(stringtoindex[ch])
    return encoded

# Function to decode a list of indices back into a string of text using the indexTostring mapping
def decode(encoded, indexTostring):
	decoded = ""
	for i in encoded:
		decoded += indexTostring[i]
	return decoded