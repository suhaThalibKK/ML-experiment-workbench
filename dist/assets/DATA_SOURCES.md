# Dataset provenance

Prepared on 13 November 2026 for the eight Computing Lab 1 experiments. Dataset files were not included with the three supplied PDF records.

| File | Origin and preparation |
| --- | --- |
| `weather.csv` | Transcribed the 14 Play Tennis examples from the supplied ID3 program. Original `id`, `outlook`, `temperature`, `humidity`, `windy`, `play` schema retained. |
| `iris.csv` | Exported `sklearn.datasets.load_iris(as_frame=True)` in scikit-learn 1.7.2: 150 observations, four features, final `species` label. No synthetic observations. |
| `wine.csv` | Exported `sklearn.datasets.load_wine(as_frame=True)` in scikit-learn 1.7.2: 178 observations, 13 features, final `class` label. Labels are `class_0`, `class_1`, `class_2`. |
| `processed.cleveland.data` | Original 303-row processed Cleveland member of the official UCI Heart Disease ZIP. Headerless, with original diagnosis codes 0 through 4 and missing-value markers retained. |
| `heart.csv` | Added headers to the preceding Cleveland file; replaced final diagnosis with `target = int(num > 0)`. All original feature encodings and `?` markers retained. No imputation/discretization is done in this file. |
| `spam.csv` | Converted the official UCI `SMSSpamCollection` tab-separated file into UTF-8 CSV with `label,message` headers. All 5,574 original records retained here. The classifier performs normalization/deduplication separately before splitting. |
| `SMS_original_readme.txt` | Original attribution and usage text from the UCI SMS ZIP, with original byte encoding retained. |

Official download URLs:

- [Heart Disease archive](https://archive.ics.uci.edu/static/public/45/heart+disease.zip).
- [SMS Spam Collection archive](https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip).
- [scikit-learn toy dataset documentation](https://scikit-learn.org/stable/datasets/toy_dataset.html).

Attribution:

- Heart Disease: Janosi, A., Steinbrunn, W., Pfisterer, M., and Detrano, R. (1988). *Heart Disease*. UCI Machine Learning Repository. [DOI: 10.24432/C52P4X](https://doi.org/10.24432/C52P4X).
- SMS Spam Collection: Almeida, T., and Hidalgo, J. (2011). *SMS Spam Collection*. UCI Machine Learning Repository. [DOI: 10.24432/C5CC84](https://doi.org/10.24432/C5CC84). Associated paper: Almeida, T. A., Gomez Hidalgo, J. M., and Yamakami, A., *Contributions to the Study of SMS Spam Filtering: New Collection and Results*, ACM DocEng 2011.
- Iris: Fisher, R. A. (1936), *The use of multiple measurements in taxonomic problems*. Dataset also distributed by UCI and scikit-learn.
- Wine: Forina, M. et al., Institute of Pharmaceutical and Food Analysis and Technologies, Genoa; dataset distributed by UCI and scikit-learn.

The UCI Heart Disease and SMS dataset pages list CC BY 4.0. Keep these credits and the original SMS readme with redistributed data. See the official dataset pages for full metadata and terms.

`manifest.json` records row counts and SHA-256 checksums of the files used in the supplied test run. The CSVs are separate from the PDFs; observed model results should not be assumed to match a different CSV shown only in a screenshot.
