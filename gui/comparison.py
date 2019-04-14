from luhn_summarizer import luhn_summarize
from text_rank import textrank_summarize
import os

BASE_PATH = "comparison_dataset\\articles"
LUHN = "comparison_dataset\\generated_summaries\\luhn"
TEXT_RANK = "comparison_dataset\\generated_summaries\\text_rank"


for article in os.listdir(BASE_PATH):
    filename, _ = os.path.splitext(article)
    with open(BASE_PATH + '\\' + article, 'r') as f:
        text = f.read()
        luhn_summary = luhn_summarize(text, num_sent=5)
        text_rank_summary = textrank_summarize(text, num_sent=5)

    with open(LUHN + '\\' + filename + '.txt', 'w+') as o:
        o.write(luhn_summary)

    with open(TEXT_RANK + '\\' + filename + '.txt', 'w+') as o:
        o.write(text_rank_summary)

    print(article , "done")

