User-friendly extensions of the DrugBank database
DOI 10.5281/zenodo.45579

DrugBank is a publicly-available resource of drug information [1]. We rely on DrugBank for our project to repurpose drugs. We are conducting this project openly on ThinkLab, and this README will reference Thinklab discussions providing greater detail.

This repository contains several code and data components:

parse.ipynb -- extracts information from the DrugBank xml download into a tsv file where each row represents a drug. A subset referred to as slim contains only drugs that are approved, small molecules, and contain an InChI structure (discussion). We also extract the interacting proteins for each drug, which include targets, enzymes, transporters, and carriers (dicussion).

similarity.ipynb -- calculates chemical similarity between drugbank compounds using extended connectivity fingerprints (dicussion). Similarities range from 0 to 1. The full similarity download is available on figshare. The subset of similarities for slim compounds is on github.

unichem-map.ipynb -- maps DrugBank compounds to 30 other compound resources using UniChem. The mapping is based on atomic connectivity and ignores differences in small molecular details. Mappings are available in a bulk download or for individual resources. Summary statistics are also available (discussion).

pubchem-map.ipynb -- DrugBank compounds were mapped to PubChem based on exact InChi string matches. The mapping is available as a tsv file.

parse-halflife.ipynb -- extracts half-life and other structural information from the Drugbank xml download into a tsv file where each row represents a drug. The half-life information was listed as free text in Drugbank. We manually extract the numeric value from free text into a xlsx file. All values were converted to hours. If the value was listed as time range (e.g. a ~ b) in DrugBank, average was calculated (e.g. (a + b)/2).

extract-curated-halflife.ipynb -- extracts subset of drugs with curated half-life into a tsv file where each row represents a drug.

predict-halflife.ipynb -- builds supervised learning models to predict half-life based on structural properties of drugs.

License
DrugBank content and derivates are licensed under CC BY-NC 4.0. Original content is released as CC0 1.0
