# ML Lab Companion

An interactive machine learning companion for learners and instructors, covering all eight Computing Lab 1 experiments.

## Features

- Switch experiments using the desktop sidebar or mobile selector.
- Review structured results, plots and the full captured Python console output.
- Try ID3, Gaussian NB, document classification, Bayesian-network inference and KNN predictions using actual exported fitted parameters.
- Explore clustering coordinates and memberships, select cross-validation folds and calculate matrix products and student-mark summaries.
- Search and paginate the complete Weather, Iris, Wine, Cleveland and SMS datasets; download each CSV.
- Read, copy and download the complete Python programs.
- Download the consolidated lab PDF, Colab notebook and original program package.
- Share a direct link to a particular experiment and tab.

## Execution model

The Output views show the verified local Python runs, clearly labelled as captured outputs. The interactive panels calculate in the browser from the exported fitted models. The app does not claim to execute scikit-learn or pgmpy training on its web host. The full Python implementation is available to execute in Colab or locally.

All entered demonstration values stay in the browser. There are no analytics, remote form submissions or additional external script dependencies. Dataset provenance is available in the app downloads.

## Development

Project folder and suggested GitHub repository name: `ml-lab-companion`.

The static application is in `dist/`. Serve it over HTTP rather than opening `index.html` as a file:

```bash
python3 -m http.server 8766 --directory dist
```

With Node installed, run:

```bash
npm run check
```

`export_lab_data.py` rebuilds `lab-data.json` using the corrected Python programs in the parent workspace and their real fitted parameters. It requires the same Python libraries as the original lab. The app has no runtime Python dependency.

## Validation

- 143 model checks, including equality of all 61 heart posteriors and 60 KNN test predictions against Python.
- All eight experiment output, interactive, dataset and source views checked in the browser.
- Dataset empty search and pagination, Wine's 13-feature input, ham/spam classification, cross-validation fold selection, matrix calculations and clustering controls checked interactively.
- Responsive desktop and mobile layouts checked.

The heart dataset explorer is educational, not clinically validated. Notebook and hardware operations specific to Colab remain in the provided notebook.

## Beginner learning mode

The first tab is now Learn step by step. All eight experiments have guided theory, worked calculations, visual stages and three understanding questions each (63 steps in total). The glossary explains features, labels, training, prediction and evaluation before the mathematics. Formula panels are optional. Each lesson ends with links to the existing interactive experiment, output and exact Python program.

ID3 includes the supplied revision guide's learning sequence, a live entropy sandbox, feature-split calculations, progressive tree construction, root-to-leaf tracing and the six question patterns plus the Studies exercise. Numeric threshold ties and multiclass entropy are handled explicitly. Calculations use the shown tables; the Competition count mismatch in the profit example is corrected. Exam-session attributions from the reference are not independently authenticated.

Small teaching datasets (document counts, a weather probability network, and 2D/1D clustering) are labelled as illustrations. Real Iris and heart calculations use exported fitted parameters. These examples do not replace or rewrite the verified experiment results.

Learning position and quiz choices are saved only in browser localStorage on the current device. No learner names or accounts are collected.

Validation: all 63 steps rendered in browser checks; entropy extremes/balance, decision-path controls, quiz feedback, numeric threshold ties, and phone-width scrolling were exercised. Additional math checks cover exact gains, K-Means center updates, normalized EM responsibilities, smoothed document scores and Gaussian score agreement.

### Previous-year question studio

The **Previous-year questions** tab contains 23 guides and 117 steps across all
8 experiments. Search by text, year or question ID and filter by experiment and
source. The source is the user's 12-page `KTU_MTech_AML_Experiment_Topics_PYQ_Dataset_Pack.pdf`;
record-level page references and evidence limitations are visible in the app.
University-paper records, secondary transcriptions, internal papers, question-bank
prompts, model questions and newly created practice remain distinguishable.

`dist/pyq-data.js` stores the transcribed questions and supplied tables.
`dist/pyq-math.js` computes the numerical walkthroughs; `dist/pyq.js` renders them.
Guides provide basics, annotated calculations, visualisations, complete conditional
answers, quizzes, device-local progress, shareable question URLs, a text download
and print styling for the entire selected guide.

Source limitations are part of the solutions: the 2022 customer NB scores both
vanish without smoothing; fruit frequencies use an explicitly selected overlapping
binary-attribute interpretation (the pack's broad inconsistency warning is audited);
Alarm CPT repairs are labelled assumptions; K-Means requires explicit initial
centres and declares its Manhattan/mean-update convention. Missing matrices,
car rows and network figures are not invented. The MACE Marketing example
explicitly reuses the reproduced 2023 employee table and does not claim to recover
the absent internal-paper dataset.

`node scripts/check-pyq.mjs` checks exact products, weighted counts, data-quality
gates, all Alarm correction profiles, 336 distinct K-Means seed combinations,
silhouette calculations, provenance coverage and all guide builders. The normal
`check` script includes this suite as well as the original model and lesson checks.
No dependency installation or build is required.
