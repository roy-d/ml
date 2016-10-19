from sklearn import tree

features = [[140, "smooth"], [130, "smooth"], [150, "bumpy"], [170, "bumpy"]]
labels = ["apple", "apple", "orange", "orange"]

norm_features = map( lambda x: [x[0], 1] if x[1] == 'smooth' else [x[0], 0], features)
norm_labels = map( lambda x: 0 if x == 'apple' else 1, labels)

print(norm_features)
print(norm_labels)

clf = tree.DecisionTreeClassifier()
clf.fit(norm_features, norm_labels)
prediction = clf.predict([[150,1]])

labelize = lambda x: 'apple' if x == 0 else 'orange'
print(labelize(prediction))

