# Rothko Image Mapper

Upload any photo and find the Mark Rothko painting that most closely matches its colour palette.

## How it works

1. Your image is reduced to six dominant colours using K-means clustering, which partitions all pixels into groups by proximity in colour space and takes the centroid of each group.
2. Each centroid is expressed in CIELAB colour space, a perceptually uniform model where equal numerical distances correspond to equal perceived colour differences.
3. Those six values are compared against pre-computed palettes for 100 Rothko paintings using weighted nearest-neighbour distance. The closest painting is your match.

## Stack

- [Streamlit](https://streamlit.io) for the web interface
- [Pillow](https://python-pillow.org) for image loading and compositing
- [scikit-learn](https://scikit-learn.org) for K-means clustering
- [scikit-image](https://scikit-image.org) for RGB to CIELAB conversion
- Painting data sourced from [WikiArt](https://www.wikiart.org/en/mark-rothko)

## Running locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The `data/` folder is committed to the repo and contains 100 pre-processed Rothko paintings with their colour profiles. No setup step required.

## Rebuilding the dataset

To re-download all paintings from WikiArt:

```bash
python setup_dataset.py
```

To rebuild `profiles.json` after manually deleting images you do not want:

```bash
python setup_dataset.py --reindex
```

To fetch descriptions, gallery names, and tags from WikiArt:

```bash
python setup_dataset.py --enrich
```

## Deploying

This app is designed to run on [Streamlit Community Cloud](https://share.streamlit.io) for free. Connect your GitHub repo, point it at `app.py`, and deploy.
