import requests
import json
from tqdm import tqdm

def scrap_page():
    articles = []  
    for i in tqdm(range(1, 101)):  

        url_json = f"https://bff-service.rtbf.be/oaos/v1.5/pages/en-continu?_page={i}&_limit=100"
        base_url = 'https://www.rtbf.be/article/'
        response = requests.get(url_json).text
        data = json.loads(response)["data"]["articles"]
        

        for article in data:
            article_dict = {
                'id': article['id'],
                'slug': article['slug'],
                'title': article['title'],
                'type': article['type'],
                'Theme': article['dossierLabel'],
                'summary': article['summary'],
                'readingTime': article['readingTime'],
                'Publish_Date': article['publishedFrom'],
                "url": base_url + article["slug"] + "-" + str(article["id"])}


            articles.append(article_dict)

    return articles


def save_json(articles):
    with open('data/articles.json', 'w',encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=4)


data = scrap_page()
save_json(data)


