import arxiv

client = arxiv.Client()

arxiv_titles = []
arxiv_links = []

search = arxiv.Search(query="Transformers and attention mechanisms",max_results=10)

for result in client.results(search):
    arxiv_titles.append(result.title)
    arxiv_links.append(result.entry_id)

print(arxiv_titles)
print(arxiv_links)