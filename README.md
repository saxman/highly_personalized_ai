# Highly Personalized Interactions with AI

This collection of notebooks and scripts are an exploration into how generative AI and supporting technologies (e.g. vector/graph databases) can be used to build highly personalized interactions with AI. The overall goal of this exploration is to build a virtual asistant that can simulate a real person by accessing knowedge about that individual. Since highly personal information is captured and used in this project, all aspects of the proejct, including large language models and storage, run locally on the user's personal computer or workstation.

The exploration is comprised of several major sub-components. First, there is an evaluation of how knowledge about an individual can be effectively extracted and stored from chat transcripts. This knowledge will be used in a later phase of the project to simulate an individual. The second phase of the project explores ways that knowledge about an individual can be elucided throught a chat interface. The third phase looks into how knowledge about an indiviual can be used to simulate an individual through a chat interface. And the last phases of the project explore ways the interaction can be made more natural, using voice input and output.

## Requirements

1. Python installed
2. PIP installed

## Getting Started

1. Enter the project directory
2. Set up and activate a Python virtual environment

```
python -m venv .venv
source .venv/bin/activate
```
   
4. Install and run Jupyter Lab

```
pip install jupyterlab   
jupyter lab
```
