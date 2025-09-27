Lexical Normalizer for Historical Middle High and Early New High German
This project provides a lexical normalizer for historical spellings of Middle High German and Early New High German texts created in German Order Prussia in 14th and 15th centuries. The normalization is performed using a transformer architecture (BART).
Normalization is carried out with the type-based method. Auxiliary words (such as in=ihn/in or im=ihm/im (from in dem)) can be confused. Words are converted to lowercase. The normalized forms are oriented towards modern spelling; if the word does not have modern spelling, the form the Deutschen Rechtswörterbuch is used.

You may find the model here https://huggingface.co/Antnis/text-normalization-for-german-order-acts/tree/main

WordAcc 89,60

WordAcc OOV 89,65

Levenshtein distance 0.1464

CER 0.0195

See also
Котов А.С. Дообучение модели на основе архитектуры Transformer для нормализации корпуса средневековых текстов на немецком языке XIV-XV вв. из орденской Пруссии // Историческая информатика. 2025. № 3. С. 128-140. DOI: 10.7256/2585-7797.2025.4.75275 EDN: XOHQXO URL: https://nbpublish.com/library_read_article.php?id=75275 (in Russian)
Abstract
The article is dedicated to the methods of automatic normalization of texts in Middle High German and Early New High German for the application of NLP in medieval history research. It provides an overview of existing approaches to the automatic normalization of historical texts in German. The problems of normalizing medieval German texts are identified: the peculiarities of using substitution dictionaries and replacement rules. The limitations of these approaches and the necessity of considering the goals of normalization are described. Neural language models are defined as the most promising for automatic normalization. The study compares the effectiveness of existing neural language models (NMT) with respect to texts in Middle High German and Early New High German. It demonstrates the low effectiveness of using NMT trained on texts from the New and Modern eras. Based on reviews presented in the literature, it asserts the need to prepare NMT according to specific goals and corpora. For the normalization of texts from the 14th-15th centuries created in monastic Prussia, a neural language model based on the Transformer architecture (BART) was further trained, and its effectiveness was presented in comparison with other models. The model was trained on a custom dataset of word pairs: original-normalized, consisting of 6,570 pairs. The conditions for retraining the model were: Epoch = 28; Batch = 50. For normalizing a corpus of texts in three historical forms of the German language, the DTAEC Type Normalizer model was chosen. The effectiveness of the retrained model's normalization was compared with existing models trained on German texts from the New and Modern eras based on the metrics of Accuracy, Accuracy OOV, CER, and Levenshtein distance. The retrained model shows significant effectiveness compared to other models. One normalized sentence using the model is proposed for review, and a comparison with a benchmark is conducted. Instances of "hallucinations" in the retrained model were identified. With an Accuracy OOV of 89.6, using this method is considered promising. However, the identified shortcomings in text normalization indicate the necessity of employing additional normalization methods, such as lemmatization.


```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "username/text-normalization-for-german-order-acts"

tokenizer = AutoTokenizer.from_pretrained(text-normalization-for-german-order-acts)
model = AutoModelForSeq2SeqLM.from_pretrained(text-normalization-for-german-order-acts)

inputs = tokenizer("Hiruff mir geantwert wart durch des keysers rethe und ouch durch unsern doctorem, is mochte nicht gesein uff diese czeit die weyle wir in hengendem rechte sein, sundir dornoch findet man wol rot", return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0]))



