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
- generate within_between: Script for creating cosine_similarity matrix comparing articles within topics to each other and articles between topics to eachother.

**Tools:** Miscellaneous tools with various uses
Scripts:
- addzerostoname.py: adds leading zero to file names
- calculatena.py: can be used after generate_embeddings.py to get count of articles removed due to NA's
- cleancsv.py: deprecated
- combineembeddingdocs.py: used to combine embedding matricies into one matrix, and clean files into one file.
- combinetsv.py: used to combine tsv files into one file
- comparetopics.py: deprecated
- genheatmaptest.py: deprecated
- groupby.py: Used to group entire corpus by journal
- savetopN.py: deprecated
- savetopNdata.py: used for splitting pubmed corpus into subsets based on SJR journal rank
- saveYear.py: used for subsetting a specific year of journals
- script_8_1_impact_topic.ipynb: used for combining journal SJR rank with pubmed database
- splittsv.py: used for splitting a tsv file into partitions.
- topicsimilarity_full.py: deprecated
- topicsimilarity.py: deprecated
- writeslurmscripts.py: deprecated
