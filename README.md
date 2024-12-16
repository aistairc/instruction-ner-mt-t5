# Multi-task NER-MT-T5
Fine-tuning T5 with instructions for named entity recognition and entity-aware machine translation

 * Use scripts in the `data-processing` directory to prepare training, development and evaluation data
   * `add-tags.sh` adds instruction tags to data
   * `count-tags.sh` counts all named entity (NE) tags in a file
   * `remove-non-matching-tags.php` removes tags from parallel files for machine translation where the source side tags do not match the target side
   * `remove-tags.sh` removes all NE tags
   * `spm-encode.sh` trains a sentencepiece model
   * `tag-entities.py` adds NE tags to an input file using Spacy
 * Use `run.sh` to launch fine-tuning using deepspeed
 * `t5-generate.py` can be used to generate output with a trained/fine-tuned model

# Processing Script Usage

1. The idea is to first use [tag-entities.py](https://github.com/aistairc/instruct-ner-mt-t5/blob/main/data-processing/tag-entities.py) to add the XML style named entity tags to the plain text data. There is one command line argument - the language code.
```bash
python tag-entities.py en < input.file.en.txt > output.file.ner.en.txt
python tag-entities.py de < input.file.en.txt > output.file.ner.de.txt
```
2. Then as an additional check there's the script [count-tags.sh](https://github.com/aistairc/instruct-ner-mt-t5/blob/main/data-processing/count-tags.sh) to see how many of each entity type was tagged. This is useful for machine translation where you want parallel language files to have the same entities tagged.
```bash
./count-tags.sh output.file.ner.en.txt > output.file.ner.en.tag.counts.txt
./count-tags.sh output.file.ner.de.txt > output.file.ner.de.tag.counts.txt
```
3. In case there are too many tags in one language or the other, you can use [remove-non-matching-tags.php](https://github.com/aistairc/instruct-ner-mt-t5/blob/main/data-processing/remove-non-matching-tags.php) to only keep the ones that match.
```bash
php remove-non-matching-tags.php output.file.ner.en.txt output.file.ner.de.txt
```
4. Then optionally train the SentencePiece model using [spm-encode.sh](https://github.com/aistairc/instruct-ner-mt-t5/blob/main/data-processing/spm-encode.sh) and encode the tagged files using that 
(`spm_encode --model=ende-ner.model < output.from.the.previous.step.ner.matching.txt`). This is not necessary if you only fine-tune T5, but can be useful when using the data for training a baseline Transformer model or other architectures.
5. Finally, add the instruction tags with [add-tags.sh](https://github.com/aistairc/instruct-ner-mt-t5/blob/main/data-processing/add-tags.sh) for the different tasks, concatenate and sub-sample a smaller set as needed. 
You really need to update this one according to your own data. Since I originally experimented with various model training using this data, I applied SentencePiece, but this is not at all required for T5. This is why there is a part that removes the spm encoding `| spm_decode --model=../deen-ner.model`, so this can be removed...

The [remove-tags.sh](https://github.com/aistairc/instruct-ner-mt-t5/blob/main/data-processing/remove-tags.sh) script is called from within [add-tags.sh](https://github.com/aistairc/instruct-ner-mt-t5/blob/main/data-processing/add-tags.sh), but overall this just reduces the file to its original state. So if you still have the original file before adding the NE tags with [tag-entities.py](https://github.com/aistairc/instruct-ner-mt-t5/blob/main/data-processing/tag-entities.py) from the first step, then you can just use that instead of doing the whole process back and forth.
 
		
Publications
---------

If you use this tool, please cite the following paper:

Matīss Rikters, Makoto Miwa (2024). "[Entity-aware Multi-task Training Helps Rare Word Machine Translation.](https://aclanthology.org/2024.inlg-main.5)" In Proceedings of the 17th International Natural Language Generation Conference (2024).

```bibtex
@inproceedings{rikters-miwa-2024-entity-aware,
    title = "Entity-aware Multi-task Training Helps Rare Word Machine Translation",
    author = "Rikters, Matīss  and
      Miwa, Makoto",
    editor = "Mahamood, Saad  and
      Minh, Nguyen Le  and
      Ippolito, Daphne",
    booktitle = "Proceedings of the 17th International Natural Language Generation Conference",
    month = sep,
    year = "2024",
    address = "Tokyo, Japan",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.inlg-main.5",
    pages = "47--54",
}
```