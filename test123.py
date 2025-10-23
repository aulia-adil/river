from river import tree

update_model = tree.HoeffdingTreeClassifier(
        grace_period=200,
        leaf_prediction='nba'  # Use naive Bayes for simplicity
    )

from river.datasets import synth

dataset = synth.Agrawal(classification_function=0, seed=42)
test123 = 602
for i, (x, y) in enumerate(dataset.take(test123)):
    update_model.learn_one(x, y)

# take the data for 601 to 800
list_of_train_inputs = []
for i, (x, y) in enumerate(dataset.take(test123)):
    if i >= 600:
        list_of_train_inputs.append([x, y])