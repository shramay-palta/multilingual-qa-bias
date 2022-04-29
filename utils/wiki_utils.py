import requests
from urllib.parse import urlparse, unquote

INVALID_QID = "-1"

def get_qid_from_wiki_url(wiki_url: str) -> str:
    
    u = urlparse(unquote(wiki_url))
    page_title = u.path.split('/')[-1]
    url =f'https://{u.netloc}/w/api.php'
    
    resp = requests.get(url, params={
        'action': 'query',
        'prop': 'pageprops',
        'format': 'json',
        'titles':page_title,
    }).json()
    
    for obj in resp['query']['pages'].values():
        if 'pageprops' in obj:
            return obj['pageprops']['wikibase_item']
    
    return INVALID_QID


def get_qid_from_wiki_page(page_title: str, lang: str) -> str:
    
    url =f'https://{lang}.wikipedia.org/w/api.php'
    
    resp = requests.get(url, params={
        'action': 'query',
        'prop': 'pageprops',
        'format': 'json',
        'titles':page_title,
    }).json()
    
    for obj in resp['query']['pages'].values():
        if 'pageprops' in obj:
            return obj['pageprops']['wikibase_item']
    
    return INVALID_QID
    

if __name__ == '__main__':
    page_name_en = 'Second_law_of_thermodynamics'
    page_name_fr = 'Deuxième_principe_de_la_thermodynamique'
    
    qid_en = get_qid_from_wiki_page(page_name_en, 'en')
    qid_fr = get_qid_from_wiki_page(page_name_fr, 'fr')
    
    print(f'{qid_en=} {qid_fr=} Equal: {qid_fr==qid_en}')