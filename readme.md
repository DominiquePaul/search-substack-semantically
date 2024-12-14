# Subspace

Substack has some of the best and in-depth blog posts in the world, but unfortunately their not well-searchable. This repository is a "Perplexity for Substack" where you can ask questions about culture, tech and more and get in-depth answers grounded in the research of individuals whose newsletters you can then subscribe to. 

This project scrapes all substack posts, embeds them using MixedBread and stores them in a llamaindex vector DB.



# Issues:
- Substack crawler stops on page 22


# Useful commands

`poetry export -f requirements.txt --without-hashes | sed 's/;.*$//' > requirements.txt`