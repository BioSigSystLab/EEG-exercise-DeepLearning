# Table showing comparison of CNN performance with alternative EEG decoding methods

| Method | Average validation accuracy over 10 folds (subject-level) |
|:------:|:---------------------------------------------------------:|
| Random Forests | 59.97 ± 5.21 % |
| Random Forests with frequency bands | 60.14 ± 5.59 % |
| Random Forests with Common SpatialPatterns | 43.23 ± 4.44 % |
| CNN without baseline normalizationarchitecture | 56.59 ± 4.04 % |
| CNN without subject adversary | 59.29 ± 4.61 % |
| *CNN with subject adversary* | *74.85 ± 5.65 %* |
| Proposed CNN with labels shuffled at subject level | 62.57 ± 3.92 % |
