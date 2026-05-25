from . import BPE

tokenizer = BPE(vocab_size=30)
text = 'Из кузова в кузов шла перегрузка арбузов. В грозу в грязи от груза арбузов развалился кузов.'
tokenizer.fit(text)
print(tokenizer.encode(text))

# Из кузова в кузов шла перегрузка арбузов. В грозу в грязи от груза арбузов развалился кузов.
# | |
