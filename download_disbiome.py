"""Downloads the Disbiome experiment table used for blind validation into external_data/disbiome/."""
import os, urllib.request
os.makedirs('external_data/disbiome', exist_ok=True)
urllib.request.urlretrieve('https://disbiome.ugent.be:8080/experiment', 'external_data/disbiome/disbiome_experiments.json')
print('saved external_data/disbiome/disbiome_experiments.json')
