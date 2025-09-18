Okay, so metrics I want:
- How many extra labels were predicted?
- How many missing lables were there?
- How many times did a label outside of the allowed set get predicted?
- How many times was the formatting invalid (not comma delimited list of values)

This is a multi class classification problem:
- Accuracy = 
- Precision = Of the labels predicted, how many were in ground-truth?
- Recall = How many did we get the right labels even if we added a bunch extra?
- F1 = Average of Precision and recall