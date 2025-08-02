Lexical Normalizer for Historical Middle High and Early New High German
This project provides a lexical normalizer for historical spellings of Middle High German and Early New High German texts created in German Order Prussia in 14th and 15th centuries. The normalization is performed using a transformer architecture (BART).
Normalization is carried out with the type-based method. Auxiliary words (such as in=ihn/in or im=ihm/im (from in dem)) can be confused. Words are converted to lowercase. The normalized forms are oriented towards modern spelling; if the word does not have modern spelling, the form the Deutschen Rechtswörterbuch is used.

WordAcc 89,60

WordAcc OOV 89,65

Levenshtein distance 0.1464

CER 0.0195

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "username/text-normalization-for-german-order-acts"

tokenizer = AutoTokenizer.from_pretrained(text-normalization-for-german-order-acts)
model = AutoModelForSeq2SeqLM.from_pretrained(text-normalization-for-german-order-acts)

inputs = tokenizer("Hiruff mir geantwert wart durch des keysers rethe und ouch durch unsern doctorem, is mochte nicht gesein uff diese czeit die weyle wir in hengendem rechte sein, sundir dornoch findet man wol rot", return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0]))
