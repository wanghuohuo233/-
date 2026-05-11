---
configs:
- config_name: default
  data_files: "co/*.parquet"
- config_name: info
  data_files: "ds.parquet"
license: cc-by-4.0
tags:
- molecular dynamics
- mlip
- interatomic potential
pretty_name: JARVIS C2DB
---
### <details><summary>Cite this dataset </summary>Haastrup, S., Strange, M., Pandey, M., Deilmann, T., Schmidt, P. S., Hinsche, N. F., Gjerding, M. N., Torelli, D., Larsen, P. M., Riis-Jensen, A. C., Gath, J., Jacobsen, K. W., Mortensen, J. J., Olsen, T., and Thygesen, K. S. _JARVIS C2DB_. ColabFit, 2023. https://doi.org/10.60732/37c26dae</details>  
#### This dataset has been curated and formatted for the ColabFit Exchange  
#### This dataset is also available on the ColabFit Exchange:  
https://materials.colabfit.org/id/DS_8hgxhsfkcfa7_0  
#### Visit the ColabFit Exchange to search additional datasets by author, description, element content and more.  
https://materials.colabfit.org
<br><hr>  
# Dataset  Name  
JARVIS C2DB  
### Description  
The JARVIS-C2DB dataset is part of the joint automated repository for various integrated simulations (JARVIS) database. This subset contains configurations from the Computational 2D Database (C2DB), which contains a variety of properties for 2-dimensional materials across more than 30 differentcrystal structures. JARVIS is a set of tools and datasets built to meet current materials design challenges.  
### Dataset authors  
Sten Haastrup, Mikkel Strange, Mohnish Pandey, Thorsten Deilmann, Per S Schmidt, Nicki F Hinsche, Morten N Gjerding, Daniele Torelli, Peter M Larsen, Anders C Riis-Jensen, Jakob Gath, Karsten W Jacobsen, Jens Jørgen Mortensen, Thomas Olsen, Kristian S Thygesen  
### Publication  
https://doi.org/10.1088/2053-1583/aacfc1  
### Original data link  
https://ndownloader.figshare.com/files/28682010  
### License  
CC-BY-4.0  
### Number of unique molecular configurations  
3520  
### Number of atoms  
17990  
### Elements included  
Ag, Al, As, Au, B, Ba, Bi, Br, C, Ca, Cd, Cl, Co, Cr, Cs, Cu, F, Fe, Ga, Ge, H, Hf, Hg, I, In, Ir, K, Li, Mg, Mn, Mo, N, Na, Nb, Ni, O, Os, P, Pb, Pd, Pt, Rb, Re, Rh, Ru, S, Sb, Sc, Se, Si, Sn, Sr, Ta, Te, Ti, Tl, V, W, Y, Zn, Zr  
### Properties included  
energy, electronic band gap  
<br>
<hr>  

# Usage  
- `ds.parquet` : Aggregated dataset information.  
- `co/` directory: Configuration rows each include a structure, calculated properties, and metadata.  
- `cs/` directory : Configuration sets are subsets of configurations grouped by some common characteristic. If `cs/` does not exist, no configurations sets have been defined for this dataset.  
- `cs_co_map/` directory : The mapping of configurations to configuration sets (if defined).  
<br>
#### ColabFit Exchange documentation includes descriptions of content and example code for parsing parquet files:  
- [Parquet parsing: example code](https://materials.colabfit.org/docs/how_to_use_parquet)  
- [Dataset info schema](https://materials.colabfit.org/docs/dataset_schema)  
- [Configuration schema](https://materials.colabfit.org/docs/configuration_schema)  
- [Configuration set schema](https://materials.colabfit.org/docs/configuration_set_schema)  
- [Configuration set to configuration mapping schema](https://materials.colabfit.org/docs/cs_co_mapping_schema)  
