# IPAPack

Phoneme recognition is a common task in speech benchmarks (e.g. SUPERB), as one desideratum of speech models is to learn basic pronunciation units. TIMIT is one of the most famous datasets for phoneme recognition. However, TIMIT only includes American English.

Aside from the CMU Wilderness dataset (14000 hours across 700 languages), IPAPack [1] provides one of the largest multilingual phonemically transcribed datasets, with 1,000 hours across 115 languages. IPAPack is derived from:
* FLEURS
* DoReCo
* MSWC (Multilingual Spoken Word Corpus) - clips of single words padded to 1 second


As such, this recipe contains three test sets: `test_fleurs`, `test_doreco`, and `test_mswc`


* TODO: results
* TODO: explain the train/dev/test splits
* TODO: cite FLEURS, DoReCo and MSWC


Note: a 20,000 hour will be released soon.


# References

[1] Jian Zhu, Changbing Yang, Farhan Samir, and Jahurul Islam. 2024. The taste of IPA: Towards open-vocabulary keyword spotting and forced alignment in any language. In *Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies* (Volume 1: Long Papers), pages 750–772, Mexico City, Mexico. Association for Computational Linguistics.
