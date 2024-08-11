################################################################################
#
# By:   Shin-Han Shiu
# Modified by: Edmond Anderson
# Created: 11/15/21
# Modified: 8/31/23
# For: Screen through the pubmed baseline files and filter them to get plant
#   science records.
#    
#   Modified to screen through pubmed baseline files without filtering for plant
#   science records.
# 11/16/21: 
# - Found that texts would be trucated if there is > inside. Fixed with new
#   get_value().
# - Abstract do not always start with <AbstractText>. Some are multiple parts
#   with different tags. But all are enclosed within <Abstract> and </Abstract>.
#   Fixed.
# 11/17/21:
# - Try to make the program more efficient by moving most elif into an elif in
#   the while loop of parse_xml(), but it is not really saving much time and is
#   buggy. Revert back to just a linear if-elif.   
################################################################################

import argparse, gzip, sys, yaml
from pathlib import Path
from os import listdir

#
# For: Iterate through xml files in the target directory
# Parameters: 
#   xmls_path - The Path object to the baseline xml files from NCBI Pubmed.
#   pnames - plant names for matching later.
# Output: 
#   A tab delimited file cotanining [PMID, Date, Journal, Title, Abstract]. 
#   The file has a .parse_tsv extension. One output is generated for each 
#   gzipped pubmed XML file.
#
def iterate_xmls(xmls_path, output_dir):
    out_log = open(output_dir/ "log_error", "w")
    xmls = listdir(xmls_path)
    problem_files = []
    problem_count = 0
    count = 0
    # Go through compressed xml files
    total_q = 0
    for xml in xmls:
        if xml.endswith(".xml.gz"):
            print(xml)
            out_log.write(f'{xml}\n')
            xml_path = xmls_path / xml
            try:
                pubmed_d = parse_xml(xml_path, out_log)
                # output for each xml file
                out_path = output_dir / (xml[:-7] + ".tsv")
                with open(out_path, "w") as f:
                    f.write("PMID\tDate\tJournal\tTitle\tAbstract\n")
                    for ID in pubmed_d:
                        [TI, JO, AB, YR, MO, DY,Q] = pubmed_d[ID]
                        f.write(f"{ID}\t{YR}-{MO}-{DY}\t{JO}\t{TI}\t{AB}\n")
                count += 1
            except:
                print(f'Error loading {xml}, skipping...')
                problem_count += 1
                problem_files.append(xml)

    
    out_log.close()
    print(f'Number of TSV files generated: {count}')
    print(f'Number of XML files skipped: {problem_count}')
    print(f'List of files skipped: {problem_files}')
    
#
# For: Go through an XML file and return a dictionary with PMID, title, date,
#   abstract, and journal name.
# Parameters: 
#   xml_gz_path - The Path object to an xml baseline gzipped file
#   pnames - plant names for matching later.
#   out_log - path to the log file for documenting errors.
# Return: 
#   pubmed_d - {pmid:[TI, JO, AB, YR, MO, DY, Qualified]}. Qualified indicates
#     if the record contains plant science related keyword.
#

def get_args():
  """Get command-line arguments"""
  parser = argparse.ArgumentParser(
    description='Generate TSV tables from pubmed-formatted XML files',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  
  parser.add_argument('-c', '--config_file', 
                      type=str,
                      help='Config file path',
                      default='./config.yaml')

  args = parser.parse_args()
  
  return args

def parse_xml(xml_path, out_log, debug=0):

    # Tags: 
    # 11/17/21: In the following order of appearance. Did not followed this
    #   and resulted in problems in the while loop below.
    PM = "<PMID Version="
    #PM = "ArticleId IdType=\"pubmed\""
    # 11/15/21: The above is no good because citation field also used the same
    #   format.
    AR = "<PubmedArticle>"    # new article
    TI = "<ArticleTitle>"     # title begin tag
    # 11/17/21: Realize that a small number of journals do not ISOAbbreviation.
    #   get full title instead.
    #JO = "<ISOAbbreviation>"
    JO = "<Journal>"
    JOe= "<Title>"
    AB = "<Abstract>"
    ABe= "</Abstract>"
    DA = "<PubMedPubDate PubStatus=\"pubmed\">"
    DAe= "</PubMedPubDate>"   # Note that other PubStatus also has the
                              # same end tag.
    YR = "<Year>"
    MO = "<Month>"
    DY = "<Day>"
    
    # {pmid:[TI, JO, AB, YR, MO, DY]}
    pubmed_d = {}
    
    # read file line by line
    input_obj = gzip.open(xml_path, 'rt') # read in text mode
    L         = input_obj.readline()
    c         = 0
    PMID      = ""
    ABSTRACT  = ""
    # PMID already found or not
    flag_PMID = 0
   
    # whether DA tag is found or not before encoutering an DA end tag.
    flag_DA   = 0
    # whether AB tag is found or not.
    flag_AB   = 0
    
    # Title or Abstract contains plant-related keywords
    while L != "":
        L = L.strip()
        if debug:
            print([L])
        if L.startswith(AR) != 0:
            # This is set to make sure PMID tag is found for this article
            # and values are stored in the dictionary.
            flag_PMID = 0
            if debug:
                print("New record")
            
            if c % 1e3 == 0:
                print(f' {c/1e3} x 1000')
            c+= 1
        # 11/16/21: Found that the same PMID tag can occur multiple times for
        #   the same XML record, so use a flag to control for it.
        elif L.startswith(PM) and flag_PMID == 0:
            PMID = get_value(L)
            flag_PMID = 1
            if debug:
                print("-->",PMID)
            # This record lead to an infinite loop in get_value()
            #if PMID == "32271890": 
            #    debug = 1
            #    print(PMID)
            #else:
            #    debug = 0
            
            if PMID not in pubmed_d:
                pubmed_d[PMID] = ["","","","","","",""]
            else:
                out_log.write(f" Redun:{PMID}\n")
        # 11/17/21: Some records don't have titles or abstracts. Ignore.
        #   e.g., 31722833.
        elif L.startswith(TI):    # Title
            pubmed_d[PMID][0] = get_value(L)
            if debug:
                print("--> TI:",pubmed_d[PMID][0][:30],"...")
        elif L == JO:    # Journal 
            flag_JO = 1
            if debug:
                print("--> JO start tag")
            # have an inner loop till the end tag is found
            L = input_obj.readline()
            while L != "":
                #if debug:
                #    print("here")
                L = L.strip()
                if L.startswith(JOe):
                    pubmed_d[PMID][1] = get_value(L)
                    if debug:
                        print("--> JO:", pubmed_d[PMID][1])
                    break
                L = input_obj.readline()
            
        elif L == AB:             # Abstract
            flag_AB = 1
            if debug:
                print("--> AB start tag")
        # Populate abstract text if encountering the beginning abstract tag
        elif flag_AB and L != ABe:
            ABSTRACT += L            
            if debug:
                print("--> AB populate")
        # 11/18/21: Some abstract has XML tag that does not have end tag. E.g.,
        # in 32271890, <AbstractText Label="Conclusions"/>. Fix get_value().
        elif L == ABe and flag_AB == 1:
            if debug:
                print("--> AB end tag", flag_AB)
                print("--> AB:", [ABSTRACT])
            pubmed_d[PMID][2] = get_value(ABSTRACT)
            # reset values
            flag_AB = 0
            ABSTRACT = ""

        # Deal with date
        elif L.startswith(DA):
            if debug:
                print("--> DA start tag")
            flag_DA = 1
        elif L.startswith(YR) and flag_DA == 1:
            pubmed_d[PMID][3] = get_value(L)
            if debug:
                print("--> YR:",pubmed_d[PMID][3])
        elif L.startswith(MO) and flag_DA == 1:
            pubmed_d[PMID][4] = get_value(L)
            if len(pubmed_d[PMID][4]) == 1:
                pubmed_d[PMID][4] = "0" + pubmed_d[PMID][4]
            if debug:
                print("--> MO:",pubmed_d[PMID][4])
        elif L.startswith(DY) and flag_DA == 1:
            pubmed_d[PMID][5] = get_value(L)
            if len(pubmed_d[PMID][5]) == 1:
                pubmed_d[PMID][5] = "0" + pubmed_d[PMID][5]        
            if debug:
                print("--> MO:",pubmed_d[PMID][5])
        # Encouter Date end tag when a corresponding begin tag exists. This is
        # the end of the record for this entry. Reset values
        elif L.startswith(DAe) and flag_DA == 1:
            flag_DA = 0
            if debug:
                print("--> DA end tag")
                print("[Done]",pubmed_d[PMID],"\n")   

            # See if this is a plant science related record
            
                
        L = input_obj.readline()
               
    print("  # articles:",c)
    out_log.write(f" # articles:{c}")
    
    return pubmed_d

# Rid of the tags
def get_value(L, debug=0):
    # Get the 1st chuck of text 
    tag1_L_b = L.find("<")  # beginning html tag, Left, beginning
    tag1_L_e = L.find(">")  # beginning html tag, Left, ending
    tag1_R_b = L.find("</") # beginning html tag, Right, beginning
    
    # Text example
    # L1           L2      L3
    # blah blah <i>bleh</i> and <d>blue</d>.
    # Also work if text starts with tag.
    L1 = L[:tag1_L_b]
    L2 = L[tag1_L_e+1 : tag1_R_b]
    L3 = L[tag1_R_b:]
    tag1_R_e = L3.find(">") # beginning html tag, Right, ending
    if debug:
        print("L1:",L1)
        print("L2:",L2)
        print("L3:",L3)
        print("tag1_R_e:",tag1_R_e)
    
    # 11/18/21: This enters an infitnit loop in the recusrive part if the line
    # contain tag that is just by itself: E.g.,
    #  <AbstractText Label="Results"/>
    #  <AbstractText Label="Conclusions"/>
    # This happens in 32271890, so this if statement is find situation where L3
    # is ">", then L2 must be such tag.
    if tag1_R_e != 0:
        L3 = L3[tag1_R_e+1:]
        
        L = L1 + L2 + L3
        
        # Check if there is more tag, if so, run the function again
        if len(L.split("<")) > 1:
            if debug:
                print("--> recusive:",len(L.split("<")),"\n")
            L = get_value(L)
    else:
        L = L1
    
    
    return L

# For: Read names for the name files with both NCBI taxnomy and USDA entries.
# Parameters:
#   pnames_path - path to the plant name file.
# Return:
#   pnames - If a pname have multiple tokens, {token:{pname:0}}. If just one,
#     then {pname:1}. This is to take care of compound plant names.

    
################################################################################
if __name__== '__main__':

    print("Get config")
    args  = get_args()
    config_file = Path(args.config_file)
    print(f"  config_file: {config_file}\n")
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    xmls_path = Path(config['parse_xml']['xml_path'])
    print(f"  xmls_path: {xmls_path}")
    output_dir = Path(config['parse_xml']['output_dir'])

    #deprecated- use config file instead

    # parser = argparse.ArgumentParser()
    # parser.add_argument('-x', '--xmls_path', help='path to gzipped xmls',
    #                 required=True)
    # args = parser.parse_args()

    # xmls_path = Path(args.xmls_path)





    iterate_xmls(xmls_path, output_dir)



''' # For debugging get_value()

ABSTRACT = '<AbstractText Label="Purpose">Purpose The role of endothelial Yes-associated protein 1 (YAP) in the pathogenesis of retinal angiogenesis and the astrocyte network in the mouse oxygen-induced retinopathy (OIR) model is unknown.</AbstractText><AbstractText Label="Methods">For in vivo studies, OIR was induced in conditional endothelial YAP knockout mice and their wild-type littermates. Retinal vascularization and the astrocyte network were evaluated by whole-mount fluorescence and Western blotting. In vitro experiments were performed in astrocytes cultured with human microvascular endothelial cell-1-conditioned medium to analyze the mechanisms underlying the effect of endothelial YAP on astrocytes.</AbstractText><AbstractText Label="Results"/><AbstractText Label="Conclusions"/>'

print(get_value(ABSTRACT))
'''