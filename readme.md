# FairHIL

As our daily lives become increasingly reliant on decision systems, it’s more important than ever to recognise, explore and mitigate bias in the datasets that train these systems. Even if initially unnoticeable, embedded bias is exacerbated when deployed at scale, leading to potentially harmful favouritism and prejudice. This project promotes the adoption of bias exploration in the Machine Learning pipeline. We aim to present the first implementation of FairHIL: a human-in-the-loop interface for intuitively exploring bias in datasets by leveraging causality, widening able participation beyond Data Scientists to Domain Experts.

# Installation

For the Python Causal Discovery Toolbox (CDT) package to work (and hence for FairHIL to run), you must first install R on your system, run R as admin and copy & paste the commands in r_requirements.txt the into R cmd line

Then install the Python packages in python_requirements.txt via:
```bash
pip install -r FairHIL/requirements.txt
```

Once all packages are installed, run FairHIL via:
```bash
bokeh serve --show *path to FairHIL*
```

## Happy bias exploring!

## License
[MIT](https://choosealicense.com/licenses/mit/)