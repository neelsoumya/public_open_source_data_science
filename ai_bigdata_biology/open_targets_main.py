#############################################################################################
# Query opentargets for diseases, targets and associations
#   fo autoimmune diseases
#
# Usage:
#       python3 open_targets_main.py
#
# Installation:
#       pip install opentargets
#
# Acknowledgements:
#       1) Has code from
#           http://opentargets.readthedocs.io/en/stable/tutorial.html#quick-start
#       2) Dedicated to my mother Kalyani Banerjee, my father Tarakeswar Banerjee,
#				my wife Joyeeta Ghose and my friend Irene Egli.
#
# Author:
#       Soumya Banerjee
#       https://sites.google.com/site/neelsoumya/Home
#
#
#############################################################################################


from opentargets import OpenTargetsClient
from opentargets.statistics import HarmonicSumScorer


def function_query_disease(str_disease_type, str_output_filename):
    """

    Args:
        str_disease_type: String
        str_output_filename: String

    Returns:

    """

    print('Querying repository for disease: ', str_disease_type)

    #############################
    # open client using API
    #############################
    ot = OpenTargetsClient()

    # or if you have an API key

    #from opentargets import OpenTargetsClient
    #ot = OpenTargetsClient(auth_app_name=<YOUR_APIKEY_APPNAME>, auth_secret=<YOUR_APIKEY_SECRET>,)


    ##########################
    #search for a target:
    ##########################

    #search_result = ot.search('BRAF')
    search_result = ot.search(str_disease_type)
    print (search_result[0])

    # search associations for a target:

    #a_for_target = ot.get_associations_for_target('BRAF')
    a_for_target = ot.get_associations_for_target(str_disease_type)
    for a in a_for_target:
        print(a['id'], a['association_score']['overall'])
        print(a['target'], a['association_score']['overall'])

    # ENSG00000138385', 'gene_info': {'symbol': 'SSB', 'name': 'Sjogren syndrome antigen B


    #########################################
    #search associations for a disease:
    #########################################

    a_for_disease = ot.get_associations_for_disease(str_disease_type)

    a_for_disease[1]

    #get an association by id:

    #########################################
    #get evidence for a target:
    #########################################

    e_for_target = ot.get_evidence_for_target(str_disease_type)
    for evidence_json in e_for_target.to_json():
        print(evidence_json)

    ######################################
    #get evidence for a disease:
    ######################################

    e_for_disease = ot.get_evidence_for_disease(str_disease_type)

    ######################################
    #get an evidence by id:
    ######################################

    #print(ot.get_evidence('5cf863da265c32d112ff4fc3bfc25ab3')[0])

    #get stats about the release:

    print(ot.get_stats().info)

    #############################
    #use incremental filter:
    #############################

    #from opentargets import OpenTargetsClient
    client = OpenTargetsClient()
    response = client.filter_associations()
    response
    #<opentargets.conn.IterableResult object at 0x105c32d68>
    print(response)
    #2484000 Results found
    #response.filter(target='ENSG00000157764')
    print(response)
    #865 Results found | parameters: {'target': 'ENSG00000157764'}
    response.filter(direct=True)
    print(response)
    #454 Results found | parameters: {'target': 'ENSG00000157764', 'direct': True}
    response.filter(scorevalue_min=0.2)
    print(response)
    #156 Results found | parameters: {'scorevalue_min': 0.2, 'target': 'ENSG00000157764', 'direct': True}
    response.filter(therapeutic_area='efo_0000701')
    print(response)
    #12 Results found | parameters: {'therapeutic_area': 'efo_0000701', 'scorevalue_min': 0.2, 'target': 'ENSG00000157764', 'direct': True}
    for i, r in enumerate(response):
         print(i, r['id'], r['association_score']['overall'], r['disease']['efo_info']['label'])

    #0 ENSG00000157764-EFO_0000756 1.0 melanoma
    #1 ENSG00000157764-Orphanet_1340 1.0 Cardiofaciocutaneous syndrome
    #2 ENSG00000157764-Orphanet_648 1.0 Noonan syndrome
    #3 ENSG00000157764-Orphanet_500 1.0 LEOPARD syndrome
    #4 ENSG00000157764-EFO_0002617 1.0 metastatic melanoma
    #5 ENSG00000157764-EFO_0000389 0.9975053926198617 cutaneous melanoma
    #6 ENSG00000157764-EFO_0004199 0.6733333333333333 dysplastic nevus
    #7 ENSG00000157764-EFO_0002894 0.6638888888888889 amelanotic skin melanoma
    #8 ENSG00000157764-EFO_1000080 0.5609555555555555 Anal Melanoma
    #9 ENSG00000157764-EFO_0000558 0.5602555555555556 Kaposi's sarcoma
    #10 ENSG00000157764-EFO_1000249 0.5555555555555556 Extramammary Paget Disease
    #11 ENSG00000157764-Orphanet_774 0.21793721666666668 Hereditary hemorrhagic telangiectasia


    ########################################################################################
    #export a table with association score for each datasource into an excel file:
    ########################################################################################

    #>>> from opentargets import OpenTargetsClient
    #>>> client = OpenTargetsClient()

    response = client.get_associations_for_target(str_disease_type,
                                                  fields=['association_score.datasource*',
                                                          'association_score.overall',
                                                          'target.gene_info.symbol',
                                                          'disease.efo_info.*']
                                                  )
    response
    # 865 Results found | parameters: {'target': 'ENSG00000157764', 'fields': ['association_score.datasource*', 'association_score.overall', 'target.gene_info.symbol', 'disease.efo_info.label']}
    response.to_excel(str_output_filename)




    #######################
    #If you want to change the way the associations are scored using just some datatype you might try something like this:
    #######################

    ot = OpenTargetsClient()
    r = ot.get_associations_for_target(str_disease_type)
    interesting_datatypes = ['genetic_association', 'known_drug', 'somatic_mutation']

    for i in score_with_datatype_subset(interesting_datatypes, r):
        print(i)


def score_with_datatype_subset(datatypes, results):
    """

    Args:
        datatypes:
        results:

    Returns:

    """
    for i in results:
         datatype_scores = i['association_score']['datatypes']
         filtered_scores = [datatype_scores[dt] for dt in datatypes]
         custom_score = HarmonicSumScorer.harmonic_sum(filtered_scores)
         if custom_score:
             yield (custom_score, i['disease']['id'], dict(zip(datatypes, filtered_scores))) #return some useful data




if __name__ == '__main__':

    function_query_disease(str_disease_type='lupus',
                           str_output_filename='lupus_associated_diseases_by_datasource.xls')

    function_query_disease(str_disease_type='Sjogren',
                           str_output_filename='sjogren_associated_diseases_by_datasource.xls')