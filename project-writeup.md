# GitHub Issue Labeling Fine-Tune Project - What I Actually Did

I dove into fine-tuning a small model for automatic GitHub issue labeling mostly as a way to get some hands-on experience with fine-tuning.  Currently, I think this will most benefit someone getting into fine tuning to give them some resources and ideas to consider before fine tuning a model. 

# Resources
Unsloth resources were very helpful getting started. 

- https://docs.unsloth.ai/get-started/fine-tuning-llms-guide - Guide to high level understanding of the fine tuning process
- https://docs.unsloth.ai/get-started/unsloth-notebooks - Many pre-made notebooks that are a great starting point for your own fine tune.
- https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(270M).ipynb - The notebook I started with for the fine tune described here.
- https://huggingface.co/docs/datasets/index - Huggingface datasets for potential use in fine tuning


# Why Fine Tune a Model?
As I considered why I should fine tune a model, I discovered these common use cases:
- Cost Savings (compared to a large model)
- Lower Latency (compared to a large model)
- Customize Behavior (e.g. Persona Adoption i.e get LLM to act like someone else)
- Add reasoning capability to a non-reasoning model
- Improve performance on a specialized task (not in the training data)


# My Fine Tune Task - Label Github Issues
Mostly as a learning exercise, I chose to fine tune a model with LoRa adaptors to automatically label GitHub issues based on the title and description provided in the initial GitHub issue. This is a task a large LLM could perform easily, but I wanted to try to improve performance of a much smaller LLM.

The steps were:
1. generate synthetic dataset
2. create fine tuning environment (harder than expected!)
3. calculate fine tune weights (~5 min on modest consumer GPU 3060)
4. compare the fine tuned/base model performance
5. lightweight Github Actions Deployment for quick deployment


## Should we fine tune an LLM for this use case?
This is essentially a multi-class sequence classification problem as proposed here.  There are other ways to approach this problem such as: 
- Probably the most straight-forward approach - fine tune an existing sequnce classification model directly. (https://huggingface.co/docs/transformers/en/tasks/sequence_classification)
- Alternatively, we could constrain the model outputs to the allowed label set using something like [outlines](https://github.com/dottxt-ai/outlines).
- Finally, we could also modify the LLM model architecture and adding a classification head at the end.  We'd need to train that classification head from scratch.
- I'm not sure which would produce the best results for this beforehand, and they have different tradeoffs when it comes to adding a new label, etc. but for this I'm going to ignore the alternatives and simple fine tune a small LLM.


# 1. Generate Synthetic Dataset
For this example I chose to scrape the github issues on an open source data science platform called Nebari. I scraped 1567 issues. While many of those issues did have labels, the labeling system had evolved over time so I decided to use a large LLM create a synthetic dataset.  The "true" labels are somewhat subjective, and if this were more than just a learning exercise, I might consider using 3-4 large models to label each example and only keep labels with majority model consensus. I could also include human review of examples with low consensus.  

Alternatively, I could use active learning if data generation cost were prohibitive. Instead of randomly selecting 1,567 issues to label with Gemini Pro, active learning would:

  1. Start with 100-200 labeled issues
  2. Train the 270M model
  3. Run it on remaining unlabeled issues
  4. Pick the 50 issues where the model is most confused (e.g. lowest confidence)
  5. Get Gemini labels for just those 50
  6. Retrain and repeat

As it is a learning exercise, I ended up using gemini-2.5-pro exclusively to label the entire 1567 issues to generate the training dataset.  I didn't document costs strictly, but I think costs were around ~$25 when using gemini-2.5-pro.

I used the following prompt to fine tune the LLM.

```
You are an assistant that assigns GitHub issue labels.
Return ONLY a single line of comma-separated labels from the allowed set.
Format example:
bug, enhancement, documentation
Rules:
- Choose any number of labels from the allowed set (including zero).
- Do not include any extra text, code fences, or explanations. Only the CSV line.

Allowed labels with descriptions:
- "bug": A reported error or unexpected behavior in the software.
- "enhancement": A request for a new feature or an improvement to an existing one.
- "documentation": Issues related to improving or expanding the documentation.
- "question": A user question that requires clarification or guidance.
- "maintenance": Routine tasks, refactoring, and dependency updates.
- "ci/cd": Issues related to continuous integration and deployment pipelines.
- "testing": Tasks related to creating or improving tests.
- "release": Tasks and checklists related to software releases.
- "aws": Issues specific to Amazon Web Services (AWS) deployments.
- "gcp": Issues specific to Google Cloud Platform (GCP) deployments.
- "azure": Issues specific to Microsoft Azure deployments.
- "security": Issues related to security vulnerabilities or concerns.
- "performance": Issues related to performance, cost, or resource optimization.
- "ux/ui": Issues related to user experience and user interface design.
- "configuration": Issues related to setup, configuration, or deployment settings.
- "dependency-update": Tasks related to updating third-party dependencies.

Issue title: \<title\>
Issue body:
\<body\>
```

While this seems straight forward, this probably took the largest amount of time out of all the steps. It definitely seems worth it to search [huggingface datasets](https://huggingface.co/docs/datasets/index) before creating your own. The overall cost to label this dataset was ~$25.  Tiny per evaluation costs add up when run thousands of times! 

When fine tuning and eventually using the fine-tuned LLM, I re-use this prompt template. I felt the prompt was quite long, and I considered speeding up evaluation by shortening it to a proxy prompt of a short sequence special characters not widely used in the training data, and letting the LLM would learn the meaning during the fine tuning process.  I'm not sure if this would work well, but I think this would make it harder for the LLM to use it's pre-existing knowledge from the training process.  The relatively modest improvement in fine-tuning and inference time didn't seem worth it, so I set aside this idea.

I split the dataset into 80% train, 10% eval, and 10% test splits.

# 2. Create the Fine Tune environment
This turned out to be harder than expected.  I started from the environment in [this Unsloth notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(270M).ipynb), but I did try to install as many as possible of the requirements as possible from conda-forge.  This turned out to be a pain, not b/c of a limitation with conda or conda-forge, but rather b/c the community seemed to treat conda-forge as (a close) 2nd class citizen (PyPI is first) when packaging.  While I didn't record the exact issues/bugs I hit, there was a recent fix I needed and one of the latest unsloth of unsloth zoo packages was only available on PyPI.  There seems to be a lag of a day or 2 for conda-forge packages for the unslath packages at least.  Additionally, many of the ML libraries were not packaged on conda-forge at all.  Due to the trouble I was having, I resorted to moving nearly everything possible to PyPI, and eventually got a working version.  

Somehow in the process of building the environment my uv cached unloth package was corrupted leading to a the python interpreter crashing anytime I imported unsloth, but claude code was surprisingly helpful in tracking down the issue.  Clearing the uv cache fixed the issue.

# 3. Calculate Fine Tune Weights
There were some modifications needed to the Unsloth notebook I started out with, but once getting the notebook to run, the training process only took ~5 minutes for 100 steps and I saw a bump in model performance even from that.  Later I went in and increased training time, and set some early stopping conditions based on the loss improvement on the eval split of the dataset.

# 4. Compare the Base and Fine Tuned Model Performance

The base model produced 20 (out of 157) invalid outputs according to the prompt definition, while the fine tuning format was correct in all 157 test instances.  Looking at only properly formatted examples we see the following performance. 

METRIC                         BASE MODEL                FINE-TUNED                IMPROVEMENT    
------------------------------ ------------------------- ------------------------- ---------------
Jaccard Similarity             18.18% (218/1199)         48.28% (280/580)          +30.09%
Micro-Avg Precision            20.82% (218/1047)         65.27% (280/429)          +44.45%
Micro-Avg Recall               58.92% (218/370)          64.97% (280/431)          +6.05%
Micro-Avg F1                   30.77%                    65.12%                    +34.35%

Jaccard Similarity is the most stringent of the metrics listed.  It's calculated as the intersection of ground truth and predicted labels divided by the union of ground truth and predicted lablels.  It penalizes both extra and missing labels whereas precision only penalizes predicted labels that are wrong and recall only penalizes not predicting the ground truth labels.  There is a drop in recall in the fine tuned model, but not too suprising given the 7.64 labels per example predicted by the base model compared to only 2.73 for the fine-tuned model.  Overall, the performance of the fine-tuned model is much higher as expected.

Many things could be done to improve model performance further.  I wouldn't expect more data to help with the Gemma 270M model since training stopped due to early stopping criteria.  I would expect better performance by bumping up the model size since there are 1b, 4b, 12b and 27b gemma3 variants.

# 5. Deployment
For this learning exercise, I've added a github action that is triggered when an issue is opened.  LLM Fill this out a bit more.

# Learnings
### Consider model updates 

Consider what is required if I want to my model to predict a new label among the existing set.  I not only have to retrain the model, but I need to re-generate my training dataset.  Alternatively, I could have trained a model to take accept a variable list of possible labels and choose among them for the output.  I'd expect such a model to require more training data, but would also be more robust to adding new labels and making predictions on various repos rather than just a single repo.


### Consider deployment strategy
I expect the model I've trained to get very limited use as the Nebari repo gets maybe a few issues per week.  It's not worth keeping this running constantly.  As a quick validation strategy, I planned to have the github action triggered when a new issue is created.  The github action would need to pull the model from storage somewhere.  I've saved the LoRa weights and the base model weights separately currently, but even if I combined them the github action will need to pull ~300MB of weights each time it runs, and that's just with the smallest Gemma3 model.  Of course caching could help with these issues.  

Additionally, the public github action runners don't have gpus.  In a quick test, it takes ~6 seconds to load the model and ~2 seconds to make a prediction when running on cpu.  This is fine for this application, but of course the larger models will take longer and it's something to consider.

### Many options exist to increase model performance
- increase model size
- change model architecture
- gather more data
- use model in combination with RAG (turn 0 shot into few shot)
- context enhancement (include)
- many others.
