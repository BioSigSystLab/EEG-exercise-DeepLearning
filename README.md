# EEG-exercise Deep Learning
Code base for the paper titled **"Unfolding the effects of acute cardiovascular exercise on neural correlates of motor learning using Convolutional Neural Networks"**

## Abstract
Cardiovascular exercise is known to promote the consolidation of newly acquired motor skills. Previous studies seeking to understand the neural correlates underlying motor memory consolidation that is modulated by exercise, have relied so far on using traditional statistical approaches for \textit{a priori} selected features from neuroimaging data, including EEG. With recent advances in machine learning, data-driven techniques such as deep learning have shown great potential for EEG data decoding for brain-computer interfaces, but have not been explored in the context of exercise. Here, we present a novel Convolutional Neural Network (CNN)-based pipeline for analysis of EEG data to study the brain areas and spectral EEG measures modulated by exercise. To the best of our knowledge, this work is the first one to demonstrate the ability of CNNs to be trained in a limited sample size setting. Our approach revealed discriminative spectral features within a refined frequency band (27--29 Hz) as compared to the wider beta bandwidth (15--30 Hz), which is commonly used in data analyses, as well as corresponding brain regions that were modulated by exercise. These results indicate the presence of finer EEG spectral features that could have been overlooked using conventional hypothesis-driven statistical approaches. Our study thus demonstrates the feasibility of using deep network architectures for neuroimaging analysis, even in small-scale studies, to identify robust brain biomarkers and investigate neuroscience-based questions.

**Authors:** Arna Ghosh, Fabien Dal Maso, Marc Roig, Georgios Mitsis & Marie-Helene Boudrias

### Overview of repository
* **Codes** -> Contains the Lua and MATLAB scripts with descriptions inside corresponding folders
* **Results** -> Contains images from the paper and some more to aid data understanding

### Useful links
The architecure of network is influenced by [Learning to See by Moving](https://arxiv.org/abs/1505.01596) paper. <br/>
