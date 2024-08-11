# subject_topic_modeling
Subject based topic modeling utilizing pubmed abstracts, a word_2_vec classification model, and BERTopic.
Pubmed baseline dataset can be found here: https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/
## Subsections:

**Part_0:** Data Collection/Pubmed Dataset generation
Scripts:
- parse_pubmed_xml.py: Script used for parsing downloaded baseline xmls into tsv files
- generate_embeddings.py: Script used for cleaning abstract text contained in tsv files, and generating embedding matricies for article abstracts.
**Part_1:** Word_2_vec article classification using Pubmed Dataset
Scripts:
- script_create_corpus.py: Script for generating training dataset for w2v classification model
- script_w2v_text_classify.py: Script for building and training PubMed abstract classification model
**Part_2:** Topic Modeling via BERTopic
Scripts:
- generate_topics.py: Script for topic_modeling via BERTopic.
