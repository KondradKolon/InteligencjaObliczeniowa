import pandas as pd

df = pd.read_csv('titanic.csv')
print(df.head())


from mlxtend.preprocessing import TransactionEncoder

records = df.astype(str).values.tolist()
te = TransactionEncoder()
te_ary = te.fit(records).transform(records)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
print("xd")
from mlxtend.frequent_patterns import apriori, association_rules

frequent_itemsets = apriori(df_encoded, min_support=0.005, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.8)
rules = rules.sort_values(by='lift', ascending=False)
print("REGULY CONFIDENCE GIT I SUPPORT GIT")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

survived_rules = rules[rules['consequents'].astype(str).str.contains('Yes')]
print("BOMBOCLAT\n",survived_rules)

import matplotlib.pyplot as plt

plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Reguły asocjacyjne Titanic')
plt.savefig("titanic.png")
plt.show()


top_rules = rules.head(10)

plt.figure(figsize=(10,6))
plt.barh(
    [f"{list(a)} → {list(c)}" for a, c in zip(top_rules['antecedents'], top_rules['consequents'])],
    top_rules['confidence'],
    color='skyblue'
)
plt.xlabel('conf')
plt.title('Top 10 reguł asocjacyjnych według conf')
plt.tight_layout()
plt.savefig("top10_conf.png")
plt.show()


'''
antecedents   consequents               support     confidence      lift
53    (2nd, Yes, Adult)      (Female)  0.036347    0.851064  3.985514
47        (1st, Female)  (Yes, Adult)  0.063607    0.965517  3.249394
27          (Child, No)         (3rd)  0.023626    1.000000  3.117564
70    (Male, No, Child)         (3rd)  0.015902    1.000000  3.117564
69  (Child, No, Female)         (3rd)  0.007724    1.000000  3.117564
..                  ...           ...       ...         ...       ...
68     (Male, Yes, 3rd)       (Adult)  0.034075    0.852273  0.896679
26           (Yes, 3rd)       (Adult)  0.068605    0.848315  0.892515
64   (Female, Yes, 3rd)       (Adult)  0.034530    0.844444  0.888443
23        (Female, 3rd)       (Adult)  0.074966    0.841837  0.885699
62    (Female, No, 3rd)       (Adult)  0.040436    0.839623  0.883370
'''