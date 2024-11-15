import re
import numpy as np
import matplotlib.pyplot as plt
import pymorphy2
from gensim.models	import	KeyedVectors
from nltk.corpus import	stopwords
import inspect

email	=	"""
Уважаемая	Эльвира	Геннадьевна,
Как	шеф-повар,	я	должен	выразить	свою	обеспокоенность	слухами	о	появлении	плесени	вблизи	некоторых	кухонь	в	корпусе	1
Если	эти	сообщения	найдут	подтверждения,	мне	придется	закрыть	несколько	столовых.	Возможно	распределение	нагрузки	на	
№9	площадь	Гагарина,	99
№10	улица	Южная,	10
С уважением,
Алексей	Мартынов. """



def mask_proper_names(text):
    address_pattern = r'(?:(?:улица|площадь|проспект|переулок)\s+)?[А-Я][а-яА-Я\s]+,\s*\d+'
    text = re.sub(address_pattern, '[address]', text)
    text = re.sub(r'(Уважаем(?:ая|ый)\s+)([А-Я][а-я]+(?:\s+[А-Я][а-я]+)*)', r'\1[name]', text)
    full_name_pattern = r'([А-Я][а-я]+\s+[А-Я][а-я]+(?:\s+[А-Я][а-я]+)?)'
    text = re.sub(full_name_pattern, '[name]', text)
    text: str = re.sub(r'\n([А-Я][а-я]+)\.', r'\n[name].', text,)
    return text

masked_email = mask_proper_names(email)
print(masked_email)


#	Пример	входных	данных
orig_wikitext = """Стрекозы	(лат. Odonáta) — отряд древних летающих насекомых, насчитывающий в мировой фауне свыше 6650 видов
Все	представители отряда ведут амфибионтный oбразa жизни — яйца и личинки развиваются в водной среде,	а имаго	( взрослая стадия индивидуального развития насекомых и некоторых других членистоногих животных со сложным жизненным циклом.)
Стрекозы имеют большое	значение для человека.	Велика	их	роль в регуляции численности кровососущих насекомых. """

# Пример выходных данных
result_wikitext = """ Годзиллы (лат. Odonáta) — отряд древних летающих насекомых, насчитывающий в мировой фауне свыше 6650 видов
Все представители отряда ведут амфибионтный образ жизни — яйца и личинки развиваются в водной среде, а имаго (взрослы Годзиллы 
имеют большое значение для человека. Велика их роль в регуляции численности кровососущих насекомых, ряд) """


if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec

def replace_word_with_cases(text, orig_word, new_word):
    morph = pymorphy2.MorphAnalyzer()
    orig_parsed = morph.parse(orig_word)[0]
    new_parsed = morph.parse(new_word)[0]

    def replace_match(match):
        word = match.group(0)
        parse = morph.parse(word)[0]
        gram_info = parse.tag
        new_form = new_parsed.inflect(gram_info.grammemes)
        if new_form:
            result = new_form.word
            if word[0].isupper():
                result = result.capitalize()
            return result
        return word

    result = re.sub(r'\b[СсCc]трекоз[а-яё]*\b', replace_match, text)
    return result


result = replace_word_with_cases(orig_wikitext, "стрекоза", "годзилла")
print(result)


def get_analysis(text):
    morph = pymorphy2.MorphAnalyzer()

    # natasha
    segmenter = Segmenter()
    morph_vocab = MorphVocab()
    emb = NewsEmbedding()
    morph_tagger = NewsMorphTagger(emb)

    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    for token in doc.tokens:
        token.lemmatize(morph_vocab)

    different_analyses = []

    for token in doc.tokens:
        py_parse = morph.parse(token.text)[0]

        if py_parse.normal_form != token.lemma:
            different_analyses.append({
                'word': token.text,
                'pymorphy': {
                    'lemma': py_parse.normal_form,
                    'tags': str(py_parse.tag)
                },
                'natasha': {
                    'lemma': token.lemma,
                    'tags': f"{token.pos},{','.join(token.feats.values()) if token.feats else ''}"
                }
            })
            if len(different_analyses) == 3:
                break

    return different_analyses


text = """В	этом	тексте	есть	разные	сложные	слова,	включая	причастия	и	деепричастия.	
В	этом	стали	появляться	различные	проблемы.	Мы	стали	замечать	больше	несоответствий.	
Стали	больше	печь	пироги.	В	этом	году	мой	дом	стоит	дороже."""

results = get_analysis(text)

for i, result in enumerate(results, 1):
    print(f"\n{i}-	word:	{result['word']}")
    print(f"pymorphy2:	{result['pymorphy']['lemma']},	tags:	{result['pymorphy']['tags']}")
    print(f"natasha:	{result['natasha']['lemma']},	tags:	{result['natasha']['tags']}")


model = KeyedVectors.load_word2vec_format('ruwikiruscorpora-nobigrams_upos_skipgram_300_5_2018.vec', binary=False, limit=100000)

russian_stopwords = stopwords.words("russian")
words = [w for w in model.index_to_key if w.split('_')[0] not in russian_stopwords]
words = words[:100]

word_vectors	=	np.array([model[w]	for	w	in	words])
tsne	=	TSNE(n_components=2,	random_state=0,	perplexity=10)
word_vectors_tsne	=	tsne.fit_transform(word_vectors)
plt.figure(figsize=(12,	12))
plt.scatter(word_vectors_tsne[:,	0],	word_vectors_tsne[:,	1])
for	label,	x,	y	in	zip(words,	word_vectors_tsne[:,	0],	word_vectors_tsne[:,	1]):
    word,	pos_tag	=	label.split('_')
    plt.annotate(word	+	"_"	+	pos_tag,	xy=(x,	y),	xytext=(0,	0),	textcoords='offset	points')

plt.title("100	word2vec-токенов")
plt.show()


